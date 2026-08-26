"""Dataset utilities for small flowsheet trajectory graphs."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

import h5py
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, PRNGKeyArray

from dte.flowsheet.schema import FlowsheetSpec
from dte.flowsheet.types import EXTERNAL_SINK, EXTERNAL_SOURCE
from dte.simulators.base import ProcessUnitSpec


def _pad(values: list[float], size: int, fill: float) -> list[float]:
    if len(values) > size:
        raise ValueError(f"cannot pad length {len(values)} into size {size}")
    return list(values) + [fill] * (size - len(values))


def _descriptor_for_spec(
    spec: ProcessUnitSpec,
    *,
    max_state_dim: int,
    max_control_dim: int,
    max_disturbance_dim: int,
    max_param_dim: int,
) -> list[float]:
    norm = spec.normalization
    return [
        spec.state_dim / max(max_state_dim, 1),
        spec.control_dim / max(max_control_dim, 1),
        spec.disturbance_dim / max(max_disturbance_dim, 1),
        spec.param_dim / max(max_param_dim, 1),
        * _pad(norm.state_center, max_state_dim, 0.0),
        * _pad(norm.state_scale, max_state_dim, 1.0),
        * _pad(norm.control_center, max_control_dim, 0.0),
        * _pad(norm.control_scale, max_control_dim, 1.0),
        * _pad(norm.disturbance_center, max_disturbance_dim, 0.0),
        * _pad(norm.disturbance_scale, max_disturbance_dim, 1.0),
        * _pad([norm.param_scale] * spec.param_dim, max_param_dim, 0.0),
    ]


@dataclass(frozen=True)
class FlowsheetGraphMetadata:
    """Static graph tables required by the Phase 3 flowsheet model."""

    flowsheet_name: str
    unit_names: tuple[str, ...]
    stream_names: tuple[str, ...]
    global_control_names: tuple[str, ...]
    global_disturbance_names: tuple[str, ...]
    unit_family_names: tuple[str, ...]
    stream_kind_names: tuple[str, ...]
    unit_state_center: Array
    unit_state_scale: Array
    unit_control_center: Array
    unit_control_scale: Array
    unit_disturbance_center: Array
    unit_disturbance_scale: Array
    unit_param_scale: Array
    unit_state_mask: Array
    unit_control_mask: Array
    unit_disturbance_mask: Array
    unit_param_mask: Array
    unit_descriptor: Array
    unit_family_id: Array
    stream_source_index: Array
    stream_target_index: Array
    stream_source_var_index: Array
    stream_target_var_index: Array
    stream_var_mask: Array
    stream_kind_id: Array
    stream_delay: Array
    unit_state_role_names: tuple[str, ...] = ()
    unit_control_role_names: tuple[str, ...] = ()
    unit_disturbance_role_names: tuple[str, ...] = ()
    unit_channel_name_names: tuple[str, ...] = ()
    unit_state_role_id: Array = field(
        default_factory=lambda: jnp.zeros((0, 0), dtype=jnp.int32)
    )
    unit_control_role_id: Array = field(
        default_factory=lambda: jnp.zeros((0, 0), dtype=jnp.int32)
    )
    unit_disturbance_role_id: Array = field(
        default_factory=lambda: jnp.zeros((0, 0), dtype=jnp.int32)
    )
    unit_state_name_id: Array = field(
        default_factory=lambda: jnp.zeros((0, 0), dtype=jnp.int32)
    )
    unit_control_name_id: Array = field(
        default_factory=lambda: jnp.zeros((0, 0), dtype=jnp.int32)
    )
    unit_disturbance_name_id: Array = field(
        default_factory=lambda: jnp.zeros((0, 0), dtype=jnp.int32)
    )
    unit_law_feature_names: tuple[str, ...] = ()
    unit_law_feature_defaults: Array = field(
        default_factory=lambda: jnp.zeros((0, 0), dtype=jnp.float32)
    )

    @classmethod
    def from_flowsheet_spec(cls, flowsheet: FlowsheetSpec) -> "FlowsheetGraphMetadata":
        unit_names = tuple(flowsheet.units.keys())
        units = list(flowsheet.units.values())
        stream_names = tuple(stream.name for stream in flowsheet.streams)
        unit_index = {name: idx for idx, name in enumerate(unit_names)}

        max_state_dim = max(unit.state_dim for unit in units)
        max_control_dim = max(unit.control_dim for unit in units)
        max_disturbance_dim = max(unit.disturbance_dim for unit in units)
        max_param_dim = max(unit.param_dim for unit in units)
        max_stream_vars = max(len(stream.variables) for stream in flowsheet.streams)

        unit_family_names = tuple(
            dict.fromkeys(["generic", *[str(unit.family or "generic") for unit in units]]).keys()
        )
        family_to_id = {name: idx for idx, name in enumerate(unit_family_names)}
        stream_kind_names = tuple(
            dict.fromkeys(["process", *[stream.kind for stream in flowsheet.streams]]).keys()
        )
        stream_kind_to_id = {name: idx for idx, name in enumerate(stream_kind_names)}
        unit_state_role_names = tuple(
            dict.fromkeys(
                [
                    "generic",
                    *[
                        channel.role
                        for unit in units
                        for channel in getattr(unit, "state_channels", [])
                    ],
                ]
            ).keys()
        )
        unit_control_role_names = tuple(
            dict.fromkeys(
                [
                    "generic",
                    *[
                        channel.role
                        for unit in units
                        for channel in getattr(unit, "control_channels", [])
                    ],
                ]
            ).keys()
        )
        unit_disturbance_role_names = tuple(
            dict.fromkeys(
                [
                    "generic",
                    *[
                        channel.role
                        for unit in units
                        for channel in getattr(unit, "disturbance_channels", [])
                    ],
                ]
            ).keys()
        )
        unit_channel_name_names = tuple(
            dict.fromkeys(
                [
                    "generic",
                    *[
                        channel.name
                        for unit in units
                        for channel in (
                            [
                                *getattr(unit, "state_channels", []),
                                *getattr(unit, "control_channels", []),
                                *getattr(unit, "disturbance_channels", []),
                            ]
                        )
                    ],
                ]
            ).keys()
        )
        unit_law_feature_names = tuple(
            dict.fromkeys(
                [
                    name
                    for unit in units
                    for name in getattr(unit, "law_feature_names", [])
                ]
            ).keys()
        )
        state_role_to_id = {name: idx for idx, name in enumerate(unit_state_role_names)}
        control_role_to_id = {name: idx for idx, name in enumerate(unit_control_role_names)}
        disturbance_role_to_id = {
            name: idx for idx, name in enumerate(unit_disturbance_role_names)
        }
        channel_name_to_id = {
            name: idx for idx, name in enumerate(unit_channel_name_names)
        }

        unit_state_center = []
        unit_state_scale = []
        unit_control_center = []
        unit_control_scale = []
        unit_disturbance_center = []
        unit_disturbance_scale = []
        unit_param_scale = []
        unit_state_mask = []
        unit_control_mask = []
        unit_disturbance_mask = []
        unit_param_mask = []
        unit_descriptor = []
        unit_family_id = []
        unit_state_role_id = []
        unit_control_role_id = []
        unit_disturbance_role_id = []
        unit_state_name_id = []
        unit_control_name_id = []
        unit_disturbance_name_id = []
        unit_law_feature_defaults = []

        for unit in units:
            norm = unit.normalization
            unit_state_center.append(_pad(norm.state_center, max_state_dim, 0.0))
            unit_state_scale.append(_pad(norm.state_scale, max_state_dim, 1.0))
            unit_control_center.append(_pad(norm.control_center, max_control_dim, 0.0))
            unit_control_scale.append(_pad(norm.control_scale, max_control_dim, 1.0))
            unit_disturbance_center.append(
                _pad(norm.disturbance_center, max_disturbance_dim, 0.0)
            )
            unit_disturbance_scale.append(
                _pad(norm.disturbance_scale, max_disturbance_dim, 1.0)
            )
            unit_param_scale.append(_pad([norm.param_scale] * unit.param_dim, max_param_dim, 0.0))
            unit_state_mask.append(_pad([1.0] * unit.state_dim, max_state_dim, 0.0))
            unit_control_mask.append(_pad([1.0] * unit.control_dim, max_control_dim, 0.0))
            unit_disturbance_mask.append(
                _pad([1.0] * unit.disturbance_dim, max_disturbance_dim, 0.0)
            )
            unit_param_mask.append(_pad([1.0] * unit.param_dim, max_param_dim, 0.0))
            unit_descriptor.append(
                _descriptor_for_spec(
                    unit,
                    max_state_dim=max_state_dim,
                    max_control_dim=max_control_dim,
                    max_disturbance_dim=max_disturbance_dim,
                    max_param_dim=max_param_dim,
                )
            )
            unit_family_id.append(family_to_id[str(unit.family or "generic")])
            unit_state_role_id.append(
                _pad(
                    [
                        state_role_to_id[channel.role]
                        for channel in getattr(unit, "state_channels", [])
                    ],
                    max_state_dim,
                    state_role_to_id["generic"],
                )
            )
            unit_control_role_id.append(
                _pad(
                    [
                        control_role_to_id[channel.role]
                        for channel in getattr(unit, "control_channels", [])
                    ],
                    max_control_dim,
                    control_role_to_id["generic"],
                )
            )
            unit_disturbance_role_id.append(
                _pad(
                    [
                        disturbance_role_to_id[channel.role]
                        for channel in getattr(unit, "disturbance_channels", [])
                    ],
                    max_disturbance_dim,
                    disturbance_role_to_id["generic"],
                )
            )
            unit_state_name_id.append(
                _pad(
                    [
                        channel_name_to_id[str(channel.name)]
                        for channel in getattr(unit, "state_channels", [])
                    ],
                    max_state_dim,
                    channel_name_to_id["generic"],
                )
            )
            unit_control_name_id.append(
                _pad(
                    [
                        channel_name_to_id[str(channel.name)]
                        for channel in getattr(unit, "control_channels", [])
                    ],
                    max_control_dim,
                    channel_name_to_id["generic"],
                )
            )
            unit_disturbance_name_id.append(
                _pad(
                    [
                        channel_name_to_id[str(channel.name)]
                        for channel in getattr(unit, "disturbance_channels", [])
                    ],
                    max_disturbance_dim,
                    channel_name_to_id["generic"],
                )
            )
            unit_law_feature_defaults.append(
                [
                    float(
                        dict(
                            zip(
                                getattr(unit, "law_feature_names", []),
                                getattr(unit, "law_feature_defaults", []),
                            )
                        ).get(feature_name, 0.0)
                    )
                    for feature_name in unit_law_feature_names
                ]
            )

        stream_source_index = []
        stream_target_index = []
        stream_source_var_index = []
        stream_target_var_index = []
        stream_var_mask = []
        stream_kind_id = []
        stream_delay = []

        for stream in flowsheet.streams:
            stream_source_index.append(
                -1 if stream.source_unit == EXTERNAL_SOURCE else unit_index[stream.source_unit]
            )
            stream_target_index.append(
                -1 if stream.target_unit == EXTERNAL_SINK else unit_index[stream.target_unit]
            )

            if stream.source_unit == EXTERNAL_SOURCE:
                source_var_idx = [-1] * len(stream.variables)
            else:
                source_spec = flowsheet.units[stream.source_unit]
                source_var_idx = [source_spec.state_names.index(name) for name in stream.variables]

            if stream.target_unit == EXTERNAL_SINK:
                target_var_idx = [-1] * len(stream.target_variables or [])
            else:
                target_spec = flowsheet.units[stream.target_unit]
                target_var_idx = [
                    target_spec.state_names.index(name)
                    for name in (stream.target_variables or [])
                ]

            stream_source_var_index.append(_pad(source_var_idx, max_stream_vars, -1))
            stream_target_var_index.append(_pad(target_var_idx, max_stream_vars, -1))
            stream_var_mask.append(_pad([1.0] * len(stream.variables), max_stream_vars, 0.0))
            stream_kind_id.append(stream_kind_to_id[stream.kind])
            stream_delay.append(float(stream.delay or 0.0))

        return cls(
            flowsheet_name=flowsheet.name,
            unit_names=unit_names,
            stream_names=stream_names,
            global_control_names=tuple(flowsheet.global_controls),
            global_disturbance_names=tuple(flowsheet.global_disturbances),
            unit_family_names=unit_family_names,
            stream_kind_names=stream_kind_names,
            unit_state_center=jnp.asarray(np.asarray(unit_state_center, dtype=np.float32)),
            unit_state_scale=jnp.asarray(np.asarray(unit_state_scale, dtype=np.float32)),
            unit_control_center=jnp.asarray(np.asarray(unit_control_center, dtype=np.float32)),
            unit_control_scale=jnp.asarray(np.asarray(unit_control_scale, dtype=np.float32)),
            unit_disturbance_center=jnp.asarray(
                np.asarray(unit_disturbance_center, dtype=np.float32)
            ),
            unit_disturbance_scale=jnp.asarray(
                np.asarray(unit_disturbance_scale, dtype=np.float32)
            ),
            unit_param_scale=jnp.asarray(np.asarray(unit_param_scale, dtype=np.float32)),
            unit_state_mask=jnp.asarray(np.asarray(unit_state_mask, dtype=np.float32)),
            unit_control_mask=jnp.asarray(np.asarray(unit_control_mask, dtype=np.float32)),
            unit_disturbance_mask=jnp.asarray(
                np.asarray(unit_disturbance_mask, dtype=np.float32)
            ),
            unit_param_mask=jnp.asarray(np.asarray(unit_param_mask, dtype=np.float32)),
            unit_descriptor=jnp.asarray(np.asarray(unit_descriptor, dtype=np.float32)),
            unit_family_id=jnp.asarray(np.asarray(unit_family_id, dtype=np.int32)),
            stream_source_index=jnp.asarray(np.asarray(stream_source_index, dtype=np.int32)),
            stream_target_index=jnp.asarray(np.asarray(stream_target_index, dtype=np.int32)),
            stream_source_var_index=jnp.asarray(
                np.asarray(stream_source_var_index, dtype=np.int32)
            ),
            stream_target_var_index=jnp.asarray(
                np.asarray(stream_target_var_index, dtype=np.int32)
            ),
            stream_var_mask=jnp.asarray(np.asarray(stream_var_mask, dtype=np.float32)),
            stream_kind_id=jnp.asarray(np.asarray(stream_kind_id, dtype=np.int32)),
            stream_delay=jnp.asarray(np.asarray(stream_delay, dtype=np.float32)),
            unit_state_role_names=unit_state_role_names,
            unit_control_role_names=unit_control_role_names,
            unit_disturbance_role_names=unit_disturbance_role_names,
            unit_channel_name_names=unit_channel_name_names,
            unit_state_role_id=jnp.asarray(np.asarray(unit_state_role_id, dtype=np.int32)),
            unit_control_role_id=jnp.asarray(np.asarray(unit_control_role_id, dtype=np.int32)),
            unit_disturbance_role_id=jnp.asarray(
                np.asarray(unit_disturbance_role_id, dtype=np.int32)
            ),
            unit_state_name_id=jnp.asarray(np.asarray(unit_state_name_id, dtype=np.int32)),
            unit_control_name_id=jnp.asarray(np.asarray(unit_control_name_id, dtype=np.int32)),
            unit_disturbance_name_id=jnp.asarray(
                np.asarray(unit_disturbance_name_id, dtype=np.int32)
            ),
            unit_law_feature_names=unit_law_feature_names,
            unit_law_feature_defaults=jnp.asarray(
                np.asarray(unit_law_feature_defaults, dtype=np.float32)
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "flowsheet_name": self.flowsheet_name,
            "unit_names": list(self.unit_names),
            "stream_names": list(self.stream_names),
            "global_control_names": list(self.global_control_names),
            "global_disturbance_names": list(self.global_disturbance_names),
            "unit_family_names": list(self.unit_family_names),
            "stream_kind_names": list(self.stream_kind_names),
            "unit_state_center": np.asarray(self.unit_state_center).tolist(),
            "unit_state_scale": np.asarray(self.unit_state_scale).tolist(),
            "unit_control_center": np.asarray(self.unit_control_center).tolist(),
            "unit_control_scale": np.asarray(self.unit_control_scale).tolist(),
            "unit_disturbance_center": np.asarray(self.unit_disturbance_center).tolist(),
            "unit_disturbance_scale": np.asarray(self.unit_disturbance_scale).tolist(),
            "unit_param_scale": np.asarray(self.unit_param_scale).tolist(),
            "unit_state_mask": np.asarray(self.unit_state_mask).tolist(),
            "unit_control_mask": np.asarray(self.unit_control_mask).tolist(),
            "unit_disturbance_mask": np.asarray(self.unit_disturbance_mask).tolist(),
            "unit_param_mask": np.asarray(self.unit_param_mask).tolist(),
            "unit_descriptor": np.asarray(self.unit_descriptor).tolist(),
            "unit_family_id": np.asarray(self.unit_family_id).tolist(),
            "stream_source_index": np.asarray(self.stream_source_index).tolist(),
            "stream_target_index": np.asarray(self.stream_target_index).tolist(),
            "stream_source_var_index": np.asarray(self.stream_source_var_index).tolist(),
            "stream_target_var_index": np.asarray(self.stream_target_var_index).tolist(),
            "stream_var_mask": np.asarray(self.stream_var_mask).tolist(),
            "stream_kind_id": np.asarray(self.stream_kind_id).tolist(),
            "stream_delay": np.asarray(self.stream_delay).tolist(),
            "unit_state_role_names": list(self.unit_state_role_names),
            "unit_control_role_names": list(self.unit_control_role_names),
            "unit_disturbance_role_names": list(self.unit_disturbance_role_names),
            "unit_channel_name_names": list(self.unit_channel_name_names),
            "unit_state_role_id": np.asarray(self.unit_state_role_id).tolist(),
            "unit_control_role_id": np.asarray(self.unit_control_role_id).tolist(),
            "unit_disturbance_role_id": np.asarray(self.unit_disturbance_role_id).tolist(),
            "unit_state_name_id": np.asarray(self.unit_state_name_id).tolist(),
            "unit_control_name_id": np.asarray(self.unit_control_name_id).tolist(),
            "unit_disturbance_name_id": np.asarray(self.unit_disturbance_name_id).tolist(),
            "unit_law_feature_names": list(self.unit_law_feature_names),
            "unit_law_feature_defaults": np.asarray(self.unit_law_feature_defaults).tolist(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FlowsheetGraphMetadata":
        return cls(
            flowsheet_name=str(payload["flowsheet_name"]),
            unit_names=tuple(payload["unit_names"]),
            stream_names=tuple(payload["stream_names"]),
            global_control_names=tuple(payload["global_control_names"]),
            global_disturbance_names=tuple(payload["global_disturbance_names"]),
            unit_family_names=tuple(payload["unit_family_names"]),
            stream_kind_names=tuple(payload["stream_kind_names"]),
            unit_state_center=jnp.asarray(payload["unit_state_center"], dtype=jnp.float32),
            unit_state_scale=jnp.asarray(payload["unit_state_scale"], dtype=jnp.float32),
            unit_control_center=jnp.asarray(payload["unit_control_center"], dtype=jnp.float32),
            unit_control_scale=jnp.asarray(payload["unit_control_scale"], dtype=jnp.float32),
            unit_disturbance_center=jnp.asarray(
                payload["unit_disturbance_center"], dtype=jnp.float32
            ),
            unit_disturbance_scale=jnp.asarray(
                payload["unit_disturbance_scale"], dtype=jnp.float32
            ),
            unit_param_scale=jnp.asarray(payload["unit_param_scale"], dtype=jnp.float32),
            unit_state_mask=jnp.asarray(payload["unit_state_mask"], dtype=jnp.float32),
            unit_control_mask=jnp.asarray(payload["unit_control_mask"], dtype=jnp.float32),
            unit_disturbance_mask=jnp.asarray(
                payload["unit_disturbance_mask"], dtype=jnp.float32
            ),
            unit_param_mask=jnp.asarray(payload["unit_param_mask"], dtype=jnp.float32),
            unit_descriptor=jnp.asarray(payload["unit_descriptor"], dtype=jnp.float32),
            unit_family_id=jnp.asarray(payload["unit_family_id"], dtype=jnp.int32),
            stream_source_index=jnp.asarray(payload["stream_source_index"], dtype=jnp.int32),
            stream_target_index=jnp.asarray(payload["stream_target_index"], dtype=jnp.int32),
            stream_source_var_index=jnp.asarray(
                payload["stream_source_var_index"], dtype=jnp.int32
            ),
            stream_target_var_index=jnp.asarray(
                payload["stream_target_var_index"], dtype=jnp.int32
            ),
            stream_var_mask=jnp.asarray(payload["stream_var_mask"], dtype=jnp.float32),
            stream_kind_id=jnp.asarray(payload["stream_kind_id"], dtype=jnp.int32),
            stream_delay=jnp.asarray(payload["stream_delay"], dtype=jnp.float32),
            unit_state_role_names=tuple(payload.get("unit_state_role_names", [])),
            unit_control_role_names=tuple(payload.get("unit_control_role_names", [])),
            unit_disturbance_role_names=tuple(
                payload.get("unit_disturbance_role_names", [])
            ),
            unit_channel_name_names=tuple(payload.get("unit_channel_name_names", [])),
            unit_state_role_id=jnp.asarray(
                payload.get("unit_state_role_id", []),
                dtype=jnp.int32,
            ),
            unit_control_role_id=jnp.asarray(
                payload.get("unit_control_role_id", []),
                dtype=jnp.int32,
            ),
            unit_disturbance_role_id=jnp.asarray(
                payload.get("unit_disturbance_role_id", []),
                dtype=jnp.int32,
            ),
            unit_state_name_id=jnp.asarray(
                payload.get("unit_state_name_id", []),
                dtype=jnp.int32,
            ),
            unit_control_name_id=jnp.asarray(
                payload.get("unit_control_name_id", []),
                dtype=jnp.int32,
            ),
            unit_disturbance_name_id=jnp.asarray(
                payload.get("unit_disturbance_name_id", []),
                dtype=jnp.int32,
            ),
            unit_law_feature_names=tuple(payload.get("unit_law_feature_names", [])),
            unit_law_feature_defaults=jnp.asarray(
                payload.get("unit_law_feature_defaults", []),
                dtype=jnp.float32,
            ),
        )

    def manifest(self) -> dict[str, Any]:
        return {
            "flowsheet_name": self.flowsheet_name,
            "unit_names": list(self.unit_names),
            "stream_names": list(self.stream_names),
            "global_control_names": list(self.global_control_names),
            "global_disturbance_names": list(self.global_disturbance_names),
            "shape": {
                "unit_state_center": list(self.unit_state_center.shape),
                "unit_control_center": list(self.unit_control_center.shape),
                "unit_disturbance_center": list(self.unit_disturbance_center.shape),
                "unit_param_scale": list(self.unit_param_scale.shape),
                "unit_descriptor": list(self.unit_descriptor.shape),
                "unit_law_feature_defaults": list(self.unit_law_feature_defaults.shape),
                "stream_source_var_index": list(self.stream_source_var_index.shape),
                "stream_var_mask": list(self.stream_var_mask.shape),
            },
        }


class FlowsheetTrajectoryDataset:
    """Dataset of graph-structured flowsheet trajectories."""

    def __init__(
        self,
        data: dict[str, Array] | str,
        metadata: FlowsheetGraphMetadata | None = None,
        *,
        seq_len: int = 50,
        stride: int = 10,
    ):
        if isinstance(data, str):
            if metadata is not None:
                raise ValueError("metadata must be omitted when loading from an HDF5 path.")
            data, metadata, seq_len, stride = self.load_hdf5(data)
        if metadata is None:
            raise ValueError("FlowsheetTrajectoryDataset requires metadata.")

        self.data = data
        self.metadata = metadata
        self.seq_len = seq_len
        self.stride = stride
        self._extract_subsequences()

    @classmethod
    def from_arrays(
        cls,
        flowsheet: FlowsheetSpec,
        data: dict[str, Array],
        *,
        seq_len: int = 50,
        stride: int = 10,
    ) -> "FlowsheetTrajectoryDataset":
        return cls(data, FlowsheetGraphMetadata.from_flowsheet_spec(flowsheet), seq_len=seq_len, stride=stride)

    def _extract_subsequences(self):
        states = np.asarray(self.data["states"])
        controls = np.asarray(self.data["controls"])
        disturbances = np.asarray(self.data["disturbances"])
        params = np.asarray(self.data["params"])
        stream_values = np.asarray(self.data["stream_values"])
        global_controls = np.asarray(self.data["global_controls"])
        global_disturbances = np.asarray(self.data["global_disturbances"])
        time = np.asarray(self.data["time"])

        n_trajectories, n_steps_per_traj, _, _ = states.shape
        effective_seq_len = min(int(self.seq_len), int(n_steps_per_traj))
        if effective_seq_len <= 0:
            raise ValueError("flowsheet trajectories must contain at least one timestep.")
        self.seq_len = effective_seq_len

        n_subseqs_per_traj = (n_steps_per_traj - self.seq_len) // self.stride + 1
        total_subseqs = n_trajectories * n_subseqs_per_traj
        start_indices = range(0, n_steps_per_traj - self.seq_len + 1, self.stride)

        def extract_windows(array: np.ndarray) -> np.ndarray:
            windows = np.stack(
                [array[:, start_idx:start_idx + self.seq_len] for start_idx in start_indices],
                axis=1,
            )
            return windows.reshape((total_subseqs, self.seq_len, *array.shape[2:]))

        time_windows = np.stack(
            [time[:, start_idx:start_idx + self.seq_len] for start_idx in start_indices],
            axis=1,
        ).reshape(total_subseqs, self.seq_len)

        self.subsequences = {
            "states": extract_windows(states),
            "controls": extract_windows(controls),
            "disturbances": extract_windows(disturbances),
            "params": np.repeat(params, n_subseqs_per_traj, axis=0),
            "stream_values": extract_windows(stream_values),
            "global_controls": extract_windows(global_controls),
            "global_disturbances": extract_windows(global_disturbances),
            "t": time_windows,
        }
        self._n_samples = total_subseqs
        print(f"Extracted {self._n_samples} flowsheet subsequences of length {self.seq_len}")

    @property
    def n_samples(self) -> int:
        return self._n_samples

    def sample_batch(
        self,
        key: PRNGKeyArray,
        batch_size: int,
        seq_len: int | None = None,
    ) -> dict[str, Array]:
        indices = np.asarray(
            jax.random.choice(key, self._n_samples, shape=(batch_size,), replace=False)
        )

        batch = {
            "states": jnp.asarray(self.subsequences["states"][indices]),
            "controls": jnp.asarray(self.subsequences["controls"][indices]),
            "disturbances": jnp.asarray(self.subsequences["disturbances"][indices]),
            "params": jnp.asarray(self.subsequences["params"][indices]),
            "stream_values": jnp.asarray(self.subsequences["stream_values"][indices]),
            "global_controls": jnp.asarray(self.subsequences["global_controls"][indices]),
            "global_disturbances": jnp.asarray(self.subsequences["global_disturbances"][indices]),
            "t": jnp.asarray(self.subsequences["t"][indices]),
        }

        if seq_len is not None:
            native_len = batch["states"].shape[1]
            effective_len = min(int(seq_len), native_len)
            for name in (
                "states",
                "controls",
                "disturbances",
                "stream_values",
                "global_controls",
                "global_disturbances",
                "t",
            ):
                batch[name] = batch[name][:, :effective_len]

        batch["time_mask"] = jnp.ones(batch["t"].shape, dtype=bool)
        return batch

    def split(self, val_fraction: float = 0.2) -> tuple["FlowsheetTrajectoryDataset", "FlowsheetTrajectoryDataset"]:
        n_trajectories = self.data["states"].shape[0]
        if n_trajectories < 2:
            raise ValueError("Need at least two trajectories to create a train/validation split.")

        n_val = int(n_trajectories * val_fraction)
        n_val = max(1, min(n_val, n_trajectories - 1))
        n_train = n_trajectories - n_val

        def slice_payload(start: int, end: int) -> dict[str, Array]:
            return {
                "states": self.data["states"][start:end],
                "controls": self.data["controls"][start:end],
                "disturbances": self.data["disturbances"][start:end],
                "params": self.data["params"][start:end],
                "stream_values": self.data["stream_values"][start:end],
                "global_controls": self.data["global_controls"][start:end],
                "global_disturbances": self.data["global_disturbances"][start:end],
                "time": self.data["time"][start:end],
            }

        train_dataset = FlowsheetTrajectoryDataset(
            slice_payload(0, n_train),
            self.metadata,
            seq_len=self.seq_len,
            stride=self.stride,
        )
        val_dataset = FlowsheetTrajectoryDataset(
            slice_payload(n_train, n_trajectories),
            self.metadata,
            seq_len=self.seq_len,
            stride=self.stride,
        )
        return train_dataset, val_dataset

    def manifest(self) -> dict[str, Any]:
        return {
            "n_samples": self.n_samples,
            "seq_len": self.seq_len,
            "stride": self.stride,
            "metadata": self.metadata.manifest(),
            "data_shape": {
                "states": list(np.asarray(self.data["states"]).shape),
                "controls": list(np.asarray(self.data["controls"]).shape),
                "disturbances": list(np.asarray(self.data["disturbances"]).shape),
                "params": list(np.asarray(self.data["params"]).shape),
                "stream_values": list(np.asarray(self.data["stream_values"]).shape),
                "global_controls": list(np.asarray(self.data["global_controls"]).shape),
                "global_disturbances": list(np.asarray(self.data["global_disturbances"]).shape),
            },
        }

    def save_hdf5(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(path, "w") as handle:
            for name, value in self.data.items():
                handle.create_dataset(name, data=np.asarray(value))
            handle.attrs["metadata_json"] = json.dumps(self.metadata.to_dict())
            handle.attrs["seq_len"] = int(self.seq_len)
            handle.attrs["stride"] = int(self.stride)

    @staticmethod
    def load_hdf5(
        path: str | Path,
    ) -> tuple[dict[str, Array], FlowsheetGraphMetadata, int, int]:
        with h5py.File(path, "r") as handle:
            data = {
                "states": np.asarray(handle["states"]),
                "controls": np.asarray(handle["controls"]),
                "disturbances": np.asarray(handle["disturbances"]),
                "params": np.asarray(handle["params"]),
                "stream_values": np.asarray(handle["stream_values"]),
                "global_controls": np.asarray(handle["global_controls"]),
                "global_disturbances": np.asarray(handle["global_disturbances"]),
                "time": np.asarray(handle["time"]),
            }
            metadata = FlowsheetGraphMetadata.from_dict(
                json.loads(handle.attrs["metadata_json"])
            )
            seq_len = int(handle.attrs.get("seq_len", data["states"].shape[1]))
            stride = int(handle.attrs.get("stride", 10))
        return data, metadata, seq_len, stride
