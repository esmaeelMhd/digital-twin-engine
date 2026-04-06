"""Universal multi-system dataset scaffolding.

This module provides the data-side plumbing needed for future shared-checkpoint
training across heterogeneous systems. It pads physical-unit trajectory batches
to the largest registered dimensions and returns explicit masks so future model
code can stay dimension-aware without hardcoding CSTR/HX/two-tank assumptions.
"""

from __future__ import annotations

from dataclasses import dataclass
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
        self._n_samples = sum(entry.dataset.n_samples for entry in self.entries)

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
            entries.append(PreparedSystemDataset(source=source, spec=spec, dataset=dataset))
        return cls(entries, seq_len=seq_len, stride=stride)

    @property
    def n_systems(self) -> int:
        return len(self.entries)

    @property
    def n_samples(self) -> int:
        return self._n_samples

    def split(self, val_fraction: float = 0.2) -> tuple["MultiSystemTrajectoryDataset", "MultiSystemTrajectoryDataset"]:
        """Split each constituent system dataset independently."""
        train_entries: list[PreparedSystemDataset] = []
        val_entries: list[PreparedSystemDataset] = []
        for entry in self.entries:
            train_dataset, val_dataset = entry.dataset.split(val_fraction)
            train_entries.append(
                PreparedSystemDataset(source=entry.source, spec=entry.spec, dataset=train_dataset)
            )
            val_entries.append(
                PreparedSystemDataset(source=entry.source, spec=entry.spec, dataset=val_dataset)
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
            "systems": [
                {
                    "system_id": self.system_ids[entry.source.name],
                    "name": entry.source.name,
                    "system_config": entry.source.system_config,
                    "data_path": entry.source.data_path,
                    "weight": float(entry.source.weight),
                    "n_samples": entry.dataset.n_samples,
                    "dims": {
                        "state": entry.spec.state_dim,
                        "control": entry.spec.control_dim,
                        "disturbance": entry.spec.disturbance_dim,
                        "param": entry.spec.param_dim,
                    },
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
