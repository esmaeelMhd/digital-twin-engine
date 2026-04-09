"""Typed process-channel schemas used by Phase 1 unit-model abstractions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


CANONICAL_STATE_ROLES = (
    "inventory",
    "temperature",
    "concentration",
    "pressure",
    "flow",
    "actuator_state",
    "biological",
    "energy",
    "generic",
)

_STATE_ROLE_ALIASES = {
    "inventory_level": "inventory",
    "level": "inventory",
    "hold_up": "inventory",
    "thermal": "temperature",
    "temp": "temperature",
    "temperature": "temperature",
    "concentration": "concentration",
    "composition": "concentration",
    "species": "concentration",
    "pressure": "pressure",
    "flow": "flow",
    "actuator": "actuator_state",
    "actuator_internal_state": "actuator_state",
    "actuator_state": "actuator_state",
    "biology": "biological",
    "biological_state": "biological",
    "biomass": "biological",
    "energy_like": "energy",
    "energy": "energy",
    "generic": "generic",
}


def canonicalize_state_role(role: str | None) -> str:
    """Map free-form role labels to the canonical Phase 1 role set."""
    if role is None:
        return "generic"
    normalized = str(role).strip().lower().replace("/", "_").replace("-", "_").replace(" ", "_")
    return _STATE_ROLE_ALIASES.get(normalized, normalized if normalized in CANONICAL_STATE_ROLES else "generic")


def infer_signal_role(name: str) -> str:
    """Infer a lightweight role label for control/disturbance channels."""
    token = str(name).strip().lower()
    if "temp" in token or token.startswith("t"):
        return "temperature"
    if "press" in token or token.startswith("p_"):
        return "pressure"
    if "flow" in token or token.startswith("f") or token.startswith("q"):
        return "flow"
    if "valve" in token or "actuator" in token:
        return "actuator_state"
    if "level" in token or token.startswith("h"):
        return "inventory"
    return "generic"


@dataclass
class StateChannel:
    """Ordered metadata for one state channel."""

    name: str
    role: str
    unit: str | None = None
    lower_bound: float | None = None
    upper_bound: float | None = None
    conserved_group: str | None = None
    description: str | None = None

    def __post_init__(self):
        self.role = canonicalize_state_role(self.role)


@dataclass
class SignalChannel:
    """Metadata for one control or disturbance channel."""

    name: str
    role: str = "generic"
    unit: str | None = None
    lower_bound: float | None = None
    upper_bound: float | None = None
    description: str | None = None

    def __post_init__(self):
        self.role = canonicalize_state_role(self.role)


@dataclass
class ParameterDescriptor:
    """Metadata for one physical parameter slot."""

    name: str
    unit: str | None = None
    default: float | None = None
    law_tag: str | None = None
    description: str | None = None


@dataclass
class TopologyPort:
    """Optional topology-facing port metadata for future flowsheet work."""

    name: str
    kind: str
    direction: str = "bidirectional"
    description: str | None = None


def bounds_from_decoder_constraints(
    n_channels: int,
    constraints: Sequence[object],
) -> tuple[list[float | None], list[float | None]]:
    """Infer coarse channel bounds from decoder constraints."""
    lower_bounds: list[float | None] = [None] * n_channels
    upper_bounds: list[float | None] = [None] * n_channels
    for constraint in constraints:
        constraint_type = str(getattr(constraint, "type", "none"))
        indices = [int(idx) for idx in getattr(constraint, "indices", [])]
        if constraint_type == "softplus":
            for idx in indices:
                current = lower_bounds[idx]
                lower_bounds[idx] = 0.0 if current is None else max(current, 0.0)
        elif constraint_type == "sigmoid_range":
            low = float(getattr(constraint, "low", 0.0))
            high = float(getattr(constraint, "high", 1.0))
            for idx in indices:
                lower_bounds[idx] = low
                upper_bounds[idx] = high
    return lower_bounds, upper_bounds


def infer_state_channels(
    state_names: Sequence[str],
    state_groups: Sequence[object],
    decoder_constraints: Sequence[object],
) -> list[StateChannel]:
    """Build ordered state-channel metadata from legacy names/groups."""
    lower_bounds, upper_bounds = bounds_from_decoder_constraints(
        len(state_names),
        decoder_constraints,
    )
    role_by_index = ["generic"] * len(state_names)
    group_name_by_index = [None] * len(state_names)
    for group in state_groups:
        role = canonicalize_state_role(getattr(group, "kind", "generic"))
        group_name = getattr(group, "name", None)
        for idx in getattr(group, "indices", []):
            role_by_index[int(idx)] = role
            group_name_by_index[int(idx)] = None if group_name is None else str(group_name)

    channels: list[StateChannel] = []
    for idx, name in enumerate(state_names):
        role = role_by_index[idx]
        if role == "generic":
            role = infer_signal_role(name)
        channels.append(
            StateChannel(
                name=str(name),
                role=role,
                lower_bound=lower_bounds[idx],
                upper_bound=upper_bounds[idx],
                conserved_group=group_name_by_index[idx],
            )
        )
    return channels


def _range_bounds(
    name: str,
    ranges: dict[str, Sequence[float]] | None,
) -> tuple[float | None, float | None]:
    if not ranges:
        return None, None
    raw = ranges.get(name)
    if raw is None or len(raw) < 2:
        return None, None
    return float(raw[0]), float(raw[1])


def infer_signal_channels(
    names: Iterable[str],
    ranges: dict[str, Sequence[float]] | None = None,
) -> list[SignalChannel]:
    """Infer control/disturbance metadata from names and operating ranges."""
    channels: list[SignalChannel] = []
    for name in names:
        lower, upper = _range_bounds(str(name), ranges)
        channels.append(
            SignalChannel(
                name=str(name),
                role=infer_signal_role(str(name)),
                lower_bound=lower,
                upper_bound=upper,
            )
        )
    return channels
