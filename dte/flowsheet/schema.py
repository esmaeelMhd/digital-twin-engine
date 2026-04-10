"""Schema definitions for small process flowsheets."""

from __future__ import annotations

from dataclasses import dataclass, field

from dte.flowsheet.types import EXTERNAL_SINK, EXTERNAL_SOURCE, STREAM_KINDS, is_external_node
from dte.simulators.base import ProcessUnitSpec


@dataclass(frozen=True)
class StreamSpec:
    """Directed stream between two units or an external node and a unit."""

    name: str
    source_unit: str
    target_unit: str
    variables: list[str]
    target_variables: list[str] | None = None
    kind: str = "process"
    delay: float | None = None
    description: str | None = None

    def __post_init__(self):
        if not self.variables:
            raise ValueError(f"stream '{self.name}' must define at least one variable.")
        if self.target_variables is None:
            object.__setattr__(self, "target_variables", list(self.variables))
        if len(self.target_variables or []) != len(self.variables):
            raise ValueError(
                f"stream '{self.name}' target_variables must match variables length."
            )
        if self.kind not in STREAM_KINDS:
            raise ValueError(
                f"stream '{self.name}' kind '{self.kind}' is invalid. "
                f"Expected one of {STREAM_KINDS}."
            )
        if self.delay is not None and float(self.delay) < 0.0:
            raise ValueError(f"stream '{self.name}' delay must be non-negative.")
        if self.source_unit == EXTERNAL_SINK:
            raise ValueError(f"stream '{self.name}' cannot originate from {EXTERNAL_SINK}.")
        if self.target_unit == EXTERNAL_SOURCE:
            raise ValueError(f"stream '{self.name}' cannot terminate at {EXTERNAL_SOURCE}.")


@dataclass(frozen=True)
class FlowsheetSpec:
    """Validated graph specification for a small process section."""

    name: str
    units: dict[str, ProcessUnitSpec]
    streams: list[StreamSpec]
    global_controls: list[str] = field(default_factory=list)
    global_disturbances: list[str] = field(default_factory=list)
    description: str | None = None

    def __post_init__(self):
        if not self.units:
            raise ValueError("FlowsheetSpec requires at least one unit.")

        unit_names = list(self.units.keys())
        if len(unit_names) != len(set(unit_names)):
            raise ValueError("FlowsheetSpec unit names must be unique.")

        stream_names = [stream.name for stream in self.streams]
        if len(stream_names) != len(set(stream_names)):
            raise ValueError("FlowsheetSpec stream names must be unique.")

        for stream in self.streams:
            self._validate_endpoint(stream.source_unit, stream.name, endpoint="source")
            self._validate_endpoint(stream.target_unit, stream.name, endpoint="target")
            self._validate_stream_variables(stream)

    @property
    def unit_names(self) -> tuple[str, ...]:
        return tuple(self.units.keys())

    @property
    def stream_names(self) -> tuple[str, ...]:
        return tuple(stream.name for stream in self.streams)

    def incoming_streams(self, unit_name: str) -> list[StreamSpec]:
        return [stream for stream in self.streams if stream.target_unit == unit_name]

    def outgoing_streams(self, unit_name: str) -> list[StreamSpec]:
        return [stream for stream in self.streams if stream.source_unit == unit_name]

    def has_recycle_loops(self) -> bool:
        """Return whether the internal unit graph contains a cycle."""

        adjacency = {
            unit_name: [
                stream.target_unit
                for stream in self.outgoing_streams(unit_name)
                if not is_external_node(stream.target_unit)
            ]
            for unit_name in self.unit_names
        }
        visiting: set[str] = set()
        visited: set[str] = set()

        def dfs(node: str) -> bool:
            if node in visiting:
                return True
            if node in visited:
                return False
            visiting.add(node)
            for nxt in adjacency.get(node, []):
                if dfs(nxt):
                    return True
            visiting.remove(node)
            visited.add(node)
            return False

        return any(dfs(node) for node in self.unit_names)

    def _validate_endpoint(self, node_name: str, stream_name: str, *, endpoint: str) -> None:
        if is_external_node(node_name):
            return
        if node_name not in self.units:
            raise ValueError(
                f"stream '{stream_name}' {endpoint} unit '{node_name}' is not defined."
            )

    def _validate_stream_variables(self, stream: StreamSpec) -> None:
        if not is_external_node(stream.source_unit):
            source_state_names = set(self.units[stream.source_unit].state_names)
            missing = [name for name in stream.variables if name not in source_state_names]
            if missing:
                raise ValueError(
                    f"stream '{stream.name}' source variables {missing} are not present in "
                    f"unit '{stream.source_unit}'."
                )
        if not is_external_node(stream.target_unit):
            target_state_names = set(self.units[stream.target_unit].state_names)
            missing = [
                name for name in (stream.target_variables or [])
                if name not in target_state_names
            ]
            if missing:
                raise ValueError(
                    f"stream '{stream.name}' target variables {missing} are not present in "
                    f"unit '{stream.target_unit}'."
                )
