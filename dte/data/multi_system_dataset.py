"""Universal multi-system dataset utilities.

This module pads physical-unit trajectory batches to the largest registered
dimensions, returns explicit masks, and exposes the per-system normalization
tables needed for shared-checkpoint universal models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import yaml
from jaxtyping import Array, PRNGKeyArray

from dte.data.dataset import TrajectoryDataset
from dte.simulators.base import SystemSpec
from dte.simulators.registry import get_system_spec

CONDITIONING_TAG_KEYS = (
    "reaction_class",
    "thermo_regime",
    "bio_model_family",
    "operating_regime",
)


@dataclass(frozen=True)
class SystemDatasetSource:
    """Config needed to load one system into a universal dataset."""

    name: str
    system_config: str
    data_dir: str
    weight: float = 1.0

    @property
    def data_path(self) -> str:
        return str(Path(self.data_dir) / "train_data.h5")


@dataclass(frozen=True)
class PreparedSystemDataset:
    """Loaded dataset plus its resolved system specification."""

    source: SystemDatasetSource
    spec: SystemSpec
    dataset: TrajectoryDataset
    system_config: dict[str, Any]


@dataclass(frozen=True)
class UniversalSystemMetadata:
    """Padded per-system tables required by a shared universal model."""

    system_names: tuple[str, ...]
    state_center: Array
    state_scale: Array
    control_center: Array
    control_scale: Array
    disturbance_center: Array
    disturbance_scale: Array
    param_scale: Array
    state_mask: Array
    control_mask: Array
    disturbance_mask: Array
    param_mask: Array
    state_dim: Array
    control_dim: Array
    disturbance_dim: Array
    param_dim: Array
    system_descriptor: Array
    state_group_kind_names: tuple[str, ...]
    state_group_mask: Array
    state_group_active: Array
    state_group_kind_id: Array
    state_role_names: tuple[str, ...]
    state_role_id: Array
    state_lower_bound: Array
    state_upper_bound: Array
    family_names: tuple[str, ...] = ()
    family_id: Array = field(
        default_factory=lambda: jnp.zeros((0,), dtype=jnp.int32)
    )
    subtype_names: tuple[str, ...] = ()
    subtype_id: Array = field(
        default_factory=lambda: jnp.zeros((0,), dtype=jnp.int32)
    )
    law_tag_names: tuple[str, ...] = ()
    law_tag_mask: Array = field(
        default_factory=lambda: jnp.zeros((0, 0), dtype=jnp.float32)
    )
    conditioning_category_names: tuple[str, ...] = ()
    conditioning_value_names: tuple[str, ...] = ()
    conditioning_value_id: Array = field(
        default_factory=lambda: jnp.zeros((0, 0), dtype=jnp.int32)
    )
    parameter_law_tag_id: Array = field(
        default_factory=lambda: jnp.zeros((0, 0), dtype=jnp.int32)
    )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe snapshot of the metadata layout."""
        return {
            "system_names": list(self.system_names),
            "state_group_kind_names": list(self.state_group_kind_names),
            "state_role_names": list(self.state_role_names),
            "family_names": list(self.family_names),
            "subtype_names": list(self.subtype_names),
            "law_tag_names": list(self.law_tag_names),
            "conditioning_category_names": list(self.conditioning_category_names),
            "conditioning_value_names": list(self.conditioning_value_names),
            "shape": {
                "state_center": list(self.state_center.shape),
                "control_center": list(self.control_center.shape),
                "disturbance_center": list(self.disturbance_center.shape),
                "param_scale": list(self.param_scale.shape),
                "system_descriptor": list(self.system_descriptor.shape),
                "state_group_mask": list(self.state_group_mask.shape),
                "state_group_active": list(self.state_group_active.shape),
                "state_role_id": list(self.state_role_id.shape),
                "family_id": list(self.family_id.shape),
                "subtype_id": list(self.subtype_id.shape),
                "law_tag_mask": list(self.law_tag_mask.shape),
                "conditioning_value_id": list(self.conditioning_value_id.shape),
                "parameter_law_tag_id": list(self.parameter_law_tag_id.shape),
            },
        }


class MultiSystemTrajectoryDataset:
    """Mixed-system padded dataset for universal-model research.

    Batches are returned in physical units with zero padding and explicit masks:
    - `states`, `controls`, `disturbances`, `params`
    - `state_mask`, `control_mask`, `disturbance_mask`, `param_mask`
    - `time_mask`
    - `system_id`
    """

    def __init__(self, entries: list[PreparedSystemDataset], seq_len: int, stride: int):
        if not entries:
            raise ValueError("MultiSystemTrajectoryDataset requires at least one system.")

        self.entries = entries
        self.seq_len = seq_len
        self.stride = stride

        self.system_names = [entry.source.name for entry in self.entries]
        self.system_ids = {name: idx for idx, name in enumerate(self.system_names)}
        self.system_weights = np.asarray(
            [max(float(entry.source.weight), 0.0) for entry in self.entries],
            dtype=np.float64,
        )
        if not np.any(self.system_weights > 0.0):
            raise ValueError("At least one system weight must be positive.")
        self.system_weights = self.system_weights / np.sum(self.system_weights)

        self.max_state_dim = max(entry.spec.state_dim for entry in self.entries)
        self.max_control_dim = max(entry.spec.control_dim for entry in self.entries)
        self.max_disturbance_dim = max(entry.spec.disturbance_dim for entry in self.entries)
        self.max_param_dim = max(entry.spec.param_dim for entry in self.entries)
        self.max_state_groups = max(len(entry.spec.state_groups) for entry in self.entries)
        self._n_samples = sum(entry.dataset.n_samples for entry in self.entries)
        self.state_group_kind_names = self._collect_state_group_kinds()
        self.state_role_names = self._collect_state_roles()
        self.family_names = self._collect_family_names()
        self.subtype_names = self._collect_subtype_names()
        self.law_tag_names = self._collect_law_tag_names()
        self.conditioning_category_names = self._collect_conditioning_category_names()
        self.conditioning_value_names = self._collect_conditioning_value_names()
        self._metadata = self._build_metadata()

    @classmethod
    def from_sources(
        cls,
        sources: list[SystemDatasetSource],
        seq_len: int,
        stride: int,
    ) -> "MultiSystemTrajectoryDataset":
        """Load a mixed dataset directly from config-like source entries."""
        entries: list[PreparedSystemDataset] = []
        for source in sources:
            with open(source.system_config, "r") as f:
                system_config = yaml.safe_load(f)
            spec = get_system_spec(system_config)
            dataset = TrajectoryDataset(source.data_path, seq_len=seq_len, stride=stride)
            entries.append(
                PreparedSystemDataset(
                    source=source,
                    spec=spec,
                    dataset=dataset,
                    system_config=system_config,
                )
            )
        return cls(entries, seq_len=seq_len, stride=stride)

    @classmethod
    def metadata_from_sources(
        cls,
        sources: list[SystemDatasetSource],
    ) -> UniversalSystemMetadata:
        """Build universal metadata without loading or indexing full datasets.

        This is useful for app/API inference paths that need to deserialize a
        shared checkpoint but do not need the heavyweight subsequence dataset.
        """

        class _MetadataOnlyDataset:
            n_samples = 0

        entries: list[PreparedSystemDataset] = []
        for source in sources:
            with open(source.system_config, "r") as f:
                system_config = yaml.safe_load(f)
            spec = get_system_spec(system_config)
            entries.append(
                PreparedSystemDataset(
                    source=source,
                    spec=spec,
                    dataset=_MetadataOnlyDataset(),
                    system_config=system_config,
                )
            )
        return cls(entries, seq_len=1, stride=1).metadata

    @property
    def n_systems(self) -> int:
        return len(self.entries)

    @property
    def n_samples(self) -> int:
        return self._n_samples

    @property
    def metadata(self) -> UniversalSystemMetadata:
        return self._metadata

    def _pad(self, values: list[float], size: int, fill: float) -> list[float]:
        values = list(values)
        if len(values) > size:
            raise ValueError(f"Cannot pad length {len(values)} into size {size}.")
        return values + [fill] * (size - len(values))

    def _descriptor_for_spec(self, spec: SystemSpec) -> list[float]:
        """Numeric summary used for optional system-spec conditioning."""
        max_state = max(self.max_state_dim, 1)
        max_control = max(self.max_control_dim, 1)
        max_disturbance = max(self.max_disturbance_dim, 1)
        max_param = max(self.max_param_dim, 1)
        max_groups = max(self.max_state_groups, 1)
        norm = spec.normalization
        return [
            spec.state_dim / max_state,
            spec.control_dim / max_control,
            spec.disturbance_dim / max_disturbance,
            spec.param_dim / max_param,
            len(spec.state_groups) / max_groups,
            *self._pad(norm.state_center, self.max_state_dim, 0.0),
            *self._pad(norm.state_scale, self.max_state_dim, 1.0),
            *self._pad(norm.control_center, self.max_control_dim, 0.0),
            *self._pad(norm.control_scale, self.max_control_dim, 1.0),
            *self._pad(norm.disturbance_center, self.max_disturbance_dim, 0.0),
            *self._pad(norm.disturbance_scale, self.max_disturbance_dim, 1.0),
            norm.param_scale,
        ]

    def _collect_state_group_kinds(self) -> tuple[str, ...]:
        kinds: list[str] = []
        for entry in self.entries:
            for group in entry.spec.state_groups:
                if group.kind not in kinds:
                    kinds.append(group.kind)
        if "generic" not in kinds:
            kinds.append("generic")
        return tuple(kinds)

    def _collect_state_roles(self) -> tuple[str, ...]:
        roles: list[str] = []
        for entry in self.entries:
            for channel in getattr(entry.spec, "state_channels", []):
                if channel.role not in roles:
                    roles.append(channel.role)
        if "generic" not in roles:
            roles.append("generic")
        return tuple(roles)

    def _collect_family_names(self) -> tuple[str, ...]:
        families = ["generic"]
        for entry in self.entries:
            family = str(getattr(entry.spec, "family", "generic") or "generic")
            if family not in families:
                families.append(family)
        return tuple(families)

    def _collect_subtype_names(self) -> tuple[str, ...]:
        subtypes = ["generic"]
        for entry in self.entries:
            subtype = getattr(entry.spec, "subtype", None) or getattr(
                entry.spec, "unit_type", None
            )
            normalized = str(subtype or "generic")
            if normalized not in subtypes:
                subtypes.append(normalized)
        return tuple(subtypes)

    def _collect_law_tag_names(self) -> tuple[str, ...]:
        law_tags = ["generic"]
        for entry in self.entries:
            for tag in getattr(entry.spec, "law_tags", []):
                normalized = str(tag or "generic")
                if normalized not in law_tags:
                    law_tags.append(normalized)
            for descriptor in getattr(entry.spec, "parameter_descriptors", []):
                normalized = str(getattr(descriptor, "law_tag", None) or "generic")
                if normalized not in law_tags:
                    law_tags.append(normalized)
        return tuple(law_tags)

    def _conditioning_tags_for_entry(self, entry: PreparedSystemDataset) -> dict[str, str]:
        raw = entry.system_config.get("system", {}).get("conditioning_tags", {}) or {}
        return {
            str(key): str(value)
            for key, value in raw.items()
        }

    def _collect_conditioning_category_names(self) -> tuple[str, ...]:
        categories = list(CONDITIONING_TAG_KEYS)
        for entry in self.entries:
            for key in self._conditioning_tags_for_entry(entry):
                if key not in categories:
                    categories.append(key)
        return tuple(categories)

    def _collect_conditioning_value_names(self) -> tuple[str, ...]:
        values = ["unknown"]
        for entry in self.entries:
            for value in self._conditioning_tags_for_entry(entry).values():
                if value not in values:
                    values.append(value)
        return tuple(values)

    def _build_metadata(self) -> UniversalSystemMetadata:
        state_center = []
        state_scale = []
        control_center = []
        control_scale = []
        disturbance_center = []
        disturbance_scale = []
        param_scale = []
        state_mask = []
        control_mask = []
        disturbance_mask = []
        param_mask = []
        state_dim = []
        control_dim = []
        disturbance_dim = []
        param_dim = []
        descriptor = []
        state_group_mask = []
        state_group_active = []
        state_group_kind_id = []
        state_role_id = []
        state_lower_bound = []
        state_upper_bound = []
        family_id = []
        subtype_id = []
        law_tag_mask = []
        conditioning_value_id = []
        parameter_law_tag_id = []
        kind_to_id = {
            name: idx for idx, name in enumerate(self.state_group_kind_names)
        }
        role_to_id = {
            name: idx for idx, name in enumerate(self.state_role_names)
        }
        family_to_id = {
            name: idx for idx, name in enumerate(self.family_names)
        }
        subtype_to_id = {
            name: idx for idx, name in enumerate(self.subtype_names)
        }
        law_tag_to_id = {
            name: idx for idx, name in enumerate(self.law_tag_names)
        }
        conditioning_value_to_id = {
            name: idx for idx, name in enumerate(self.conditioning_value_names)
        }

        for entry in self.entries:
            spec = entry.spec
            norm = spec.normalization

            state_center.append(self._pad(norm.state_center, self.max_state_dim, 0.0))
            state_scale.append(self._pad(norm.state_scale, self.max_state_dim, 1.0))
            control_center.append(self._pad(norm.control_center, self.max_control_dim, 0.0))
            control_scale.append(self._pad(norm.control_scale, self.max_control_dim, 1.0))
            disturbance_center.append(
                self._pad(norm.disturbance_center, self.max_disturbance_dim, 0.0)
            )
            disturbance_scale.append(
                self._pad(norm.disturbance_scale, self.max_disturbance_dim, 1.0)
            )
            param_scale.append(
                self._pad([norm.param_scale] * spec.param_dim, self.max_param_dim, 0.0)
            )

            state_mask.append(self._pad([1.0] * spec.state_dim, self.max_state_dim, 0.0))
            control_mask.append(
                self._pad([1.0] * spec.control_dim, self.max_control_dim, 0.0)
            )
            disturbance_mask.append(
                self._pad([1.0] * spec.disturbance_dim, self.max_disturbance_dim, 0.0)
            )
            param_mask.append(self._pad([1.0] * spec.param_dim, self.max_param_dim, 0.0))

            state_dim.append(spec.state_dim)
            control_dim.append(spec.control_dim)
            disturbance_dim.append(spec.disturbance_dim)
            param_dim.append(spec.param_dim)
            descriptor.append(self._descriptor_for_spec(spec))
            family_id.append(
                family_to_id[str(getattr(spec, "family", "generic") or "generic")]
            )
            subtype_name = getattr(spec, "subtype", None) or getattr(spec, "unit_type", None)
            subtype_id.append(subtype_to_id[str(subtype_name or "generic")])

            system_law_mask = np.zeros((len(self.law_tag_names),), dtype=np.float32)
            for tag in getattr(spec, "law_tags", []):
                system_law_mask[law_tag_to_id[str(tag or "generic")]] = 1.0
            if not np.any(system_law_mask):
                system_law_mask[law_tag_to_id["generic"]] = 1.0
            law_tag_mask.append(system_law_mask)

            raw_conditioning_tags = self._conditioning_tags_for_entry(entry)
            conditioning_value_id.append(
                [
                    conditioning_value_to_id[raw_conditioning_tags.get(category, "unknown")]
                    for category in self.conditioning_category_names
                ]
            )
            parameter_law_tag_id.append(
                self._pad(
                    [
                        law_tag_to_id[str(getattr(descriptor, "law_tag", None) or "generic")]
                        for descriptor in getattr(spec, "parameter_descriptors", [])
                    ],
                    self.max_param_dim,
                    law_tag_to_id["generic"],
                )
            )

            system_group_mask = np.zeros(
                (self.max_state_groups, self.max_state_dim),
                dtype=np.float32,
            )
            system_group_active = np.zeros((self.max_state_groups,), dtype=np.float32)
            system_group_kind_id = np.zeros((self.max_state_groups,), dtype=np.int32)
            for group_idx, group in enumerate(spec.state_groups):
                system_group_mask[group_idx, np.asarray(group.indices, dtype=np.int32)] = 1.0
                system_group_active[group_idx] = 1.0
                system_group_kind_id[group_idx] = kind_to_id[group.kind]
            state_group_mask.append(system_group_mask)
            state_group_active.append(system_group_active)
            state_group_kind_id.append(system_group_kind_id)
            state_role_id.append(
                self._pad(
                    [role_to_id[channel.role] for channel in getattr(spec, "state_channels", [])],
                    self.max_state_dim,
                    role_to_id["generic"],
                )
            )
            state_lower_bound.append(
                self._pad(
                    [
                        -np.inf if channel.lower_bound is None else float(channel.lower_bound)
                        for channel in getattr(spec, "state_channels", [])
                    ],
                    self.max_state_dim,
                    -np.inf,
                )
            )
            state_upper_bound.append(
                self._pad(
                    [
                        np.inf if channel.upper_bound is None else float(channel.upper_bound)
                        for channel in getattr(spec, "state_channels", [])
                    ],
                    self.max_state_dim,
                    np.inf,
                )
            )

        return UniversalSystemMetadata(
            system_names=tuple(self.system_names),
            state_center=jnp.asarray(np.asarray(state_center, dtype=np.float32)),
            state_scale=jnp.asarray(np.asarray(state_scale, dtype=np.float32)),
            control_center=jnp.asarray(np.asarray(control_center, dtype=np.float32)),
            control_scale=jnp.asarray(np.asarray(control_scale, dtype=np.float32)),
            disturbance_center=jnp.asarray(np.asarray(disturbance_center, dtype=np.float32)),
            disturbance_scale=jnp.asarray(np.asarray(disturbance_scale, dtype=np.float32)),
            param_scale=jnp.asarray(np.asarray(param_scale, dtype=np.float32)),
            state_mask=jnp.asarray(np.asarray(state_mask, dtype=np.float32)),
            control_mask=jnp.asarray(np.asarray(control_mask, dtype=np.float32)),
            disturbance_mask=jnp.asarray(np.asarray(disturbance_mask, dtype=np.float32)),
            param_mask=jnp.asarray(np.asarray(param_mask, dtype=np.float32)),
            state_dim=jnp.asarray(np.asarray(state_dim, dtype=np.int32)),
            control_dim=jnp.asarray(np.asarray(control_dim, dtype=np.int32)),
            disturbance_dim=jnp.asarray(np.asarray(disturbance_dim, dtype=np.int32)),
            param_dim=jnp.asarray(np.asarray(param_dim, dtype=np.int32)),
            system_descriptor=jnp.asarray(np.asarray(descriptor, dtype=np.float32)),
            state_group_kind_names=self.state_group_kind_names,
            state_group_mask=jnp.asarray(np.asarray(state_group_mask, dtype=np.float32)),
            state_group_active=jnp.asarray(np.asarray(state_group_active, dtype=np.float32)),
            state_group_kind_id=jnp.asarray(np.asarray(state_group_kind_id, dtype=np.int32)),
            state_role_names=self.state_role_names,
            state_role_id=jnp.asarray(np.asarray(state_role_id, dtype=np.int32)),
            state_lower_bound=jnp.asarray(np.asarray(state_lower_bound, dtype=np.float32)),
            state_upper_bound=jnp.asarray(np.asarray(state_upper_bound, dtype=np.float32)),
            family_names=self.family_names,
            family_id=jnp.asarray(np.asarray(family_id, dtype=np.int32)),
            subtype_names=self.subtype_names,
            subtype_id=jnp.asarray(np.asarray(subtype_id, dtype=np.int32)),
            law_tag_names=self.law_tag_names,
            law_tag_mask=jnp.asarray(np.asarray(law_tag_mask, dtype=np.float32)),
            conditioning_category_names=self.conditioning_category_names,
            conditioning_value_names=self.conditioning_value_names,
            conditioning_value_id=jnp.asarray(
                np.asarray(conditioning_value_id, dtype=np.int32)
            ),
            parameter_law_tag_id=jnp.asarray(
                np.asarray(parameter_law_tag_id, dtype=np.int32)
            ),
        )

    def split(self, val_fraction: float = 0.2) -> tuple["MultiSystemTrajectoryDataset", "MultiSystemTrajectoryDataset"]:
        """Split each constituent system dataset independently."""
        train_entries: list[PreparedSystemDataset] = []
        val_entries: list[PreparedSystemDataset] = []
        for entry in self.entries:
            train_dataset, val_dataset = entry.dataset.split(val_fraction)
            train_entries.append(
                PreparedSystemDataset(
                    source=entry.source,
                    spec=entry.spec,
                    dataset=train_dataset,
                    system_config=entry.system_config,
                )
            )
            val_entries.append(
                PreparedSystemDataset(
                    source=entry.source,
                    spec=entry.spec,
                    dataset=val_dataset,
                    system_config=entry.system_config,
                )
            )
        return (
            MultiSystemTrajectoryDataset(train_entries, seq_len=self.seq_len, stride=self.stride),
            MultiSystemTrajectoryDataset(val_entries, seq_len=self.seq_len, stride=self.stride),
        )

    def manifest(self) -> dict[str, Any]:
        """Return a JSON-serializable summary for scaffolding scripts."""
        return {
            "n_systems": self.n_systems,
            "n_samples": self.n_samples,
            "seq_len": self.seq_len,
            "stride": self.stride,
            "max_dims": {
                "state": self.max_state_dim,
                "control": self.max_control_dim,
                "disturbance": self.max_disturbance_dim,
                "param": self.max_param_dim,
            },
            "metadata": self.metadata.to_dict(),
            "systems": [
                {
                    "system_id": self.system_ids[entry.source.name],
                    "name": entry.source.name,
                    "system_config": entry.source.system_config,
                    "data_path": entry.source.data_path,
                    "weight": float(entry.source.weight),
                    "family": str(getattr(entry.spec, "family", "generic") or "generic"),
                    "subtype": str(
                        getattr(entry.spec, "subtype", None)
                        or getattr(entry.spec, "unit_type", None)
                        or "generic"
                    ),
                    "law_tags": [str(tag) for tag in getattr(entry.spec, "law_tags", [])],
                    "conditioning_tags": self._conditioning_tags_for_entry(entry),
                    "n_samples": entry.dataset.n_samples,
                    "dims": {
                        "state": entry.spec.state_dim,
                        "control": entry.spec.control_dim,
                        "disturbance": entry.spec.disturbance_dim,
                        "param": entry.spec.param_dim,
                    },
                    "state_groups": [
                        {
                            "name": group.name,
                            "kind": group.kind,
                            "indices": list(group.indices),
                        }
                        for group in entry.spec.state_groups
                    ],
                }
                for entry in self.entries
            ],
        }

    def sample_batch(
        self,
        key: PRNGKeyArray,
        batch_size: int,
        seq_len: int | None = None,
        *,
        system_probabilities: Array | None = None,
    ) -> dict[str, Array]:
        """Sample a mixed padded batch across systems."""
        target_seq_len = int(seq_len or self.seq_len)
        key_systems, key_batches = jax.random.split(key)

        probs = self.system_weights
        if system_probabilities is not None:
            probs = np.asarray(system_probabilities, dtype=np.float64)
            probs = probs / np.sum(probs)

        system_ids = np.asarray(
            jax.random.choice(
                key_systems,
                self.n_systems,
                shape=(batch_size,),
                replace=True,
                p=jnp.asarray(probs),
            )
        )

        batch = {
            "states": np.zeros((batch_size, target_seq_len, self.max_state_dim), dtype=np.float32),
            "controls": np.zeros((batch_size, target_seq_len, self.max_control_dim), dtype=np.float32),
            "disturbances": np.zeros(
                (batch_size, target_seq_len, self.max_disturbance_dim), dtype=np.float32
            ),
            "params": np.zeros((batch_size, self.max_param_dim), dtype=np.float32),
            "t": np.zeros((batch_size, target_seq_len), dtype=np.float32),
            "state_mask": np.zeros((batch_size, self.max_state_dim), dtype=bool),
            "control_mask": np.zeros((batch_size, self.max_control_dim), dtype=bool),
            "disturbance_mask": np.zeros((batch_size, self.max_disturbance_dim), dtype=bool),
            "param_mask": np.zeros((batch_size, self.max_param_dim), dtype=bool),
            "time_mask": np.zeros((batch_size, target_seq_len), dtype=bool),
            "system_id": system_ids.astype(np.int32),
        }

        unique_ids = np.unique(system_ids)
        subkeys = jax.random.split(key_batches, max(len(unique_ids), 1))

        for subkey, system_id in zip(subkeys, unique_ids.tolist()):
            positions = np.where(system_ids == system_id)[0]
            entry = self.entries[system_id]
            system_batch = entry.dataset.sample_batch(
                subkey,
                batch_size=len(positions),
                seq_len=target_seq_len,
            )

            actual_seq_len = int(system_batch["states"].shape[1])
            spec = entry.spec

            batch["states"][positions, :actual_seq_len, : spec.state_dim] = np.asarray(
                system_batch["states"]
            )
            batch["controls"][positions, :actual_seq_len, : spec.control_dim] = np.asarray(
                system_batch["controls"]
            )
            batch["disturbances"][
                positions, :actual_seq_len, : spec.disturbance_dim
            ] = np.asarray(system_batch["disturbances"])
            batch["params"][positions, : spec.param_dim] = np.asarray(system_batch["params"])
            batch["t"][positions, :actual_seq_len] = np.asarray(system_batch["t"])

            batch["state_mask"][positions, : spec.state_dim] = True
            batch["control_mask"][positions, : spec.control_dim] = True
            batch["disturbance_mask"][positions, : spec.disturbance_dim] = True
            batch["param_mask"][positions, : spec.param_dim] = True
            batch["time_mask"][positions, :actual_seq_len] = True

        return {name: jnp.asarray(value) for name, value in batch.items()}
