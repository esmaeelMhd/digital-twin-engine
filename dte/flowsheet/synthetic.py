"""Small synthetic flowsheet trajectory generator for demos and tests."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dte.data.flowsheet_dataset import FlowsheetTrajectoryDataset
from dte.flowsheet.schema import FlowsheetSpec
from dte.flowsheet.types import EXTERNAL_SINK, EXTERNAL_SOURCE


@dataclass(frozen=True)
class _UnitTables:
    default_state: np.ndarray
    control_center: np.ndarray
    disturbance_center: np.ndarray
    lower: np.ndarray
    upper: np.ndarray


def _stream_lag_steps(delay: float | None, dt: float) -> int:
    if delay is None or delay <= 0.0:
        return 0
    return max(int(round(float(delay) / max(float(dt), 1e-6))), 1)


def _pad(values: np.ndarray, size: int) -> np.ndarray:
    padded = np.zeros((size,), dtype=np.float32)
    padded[: values.shape[0]] = values.astype(np.float32)
    return padded


def _build_unit_tables(flowsheet: FlowsheetSpec) -> dict[str, _UnitTables]:
    tables = {}
    for name, spec in flowsheet.units.items():
        lower = np.asarray(spec.state_lower_bounds(), dtype=np.float32)
        upper = np.asarray(spec.state_upper_bounds(), dtype=np.float32)
        tables[name] = _UnitTables(
            default_state=np.asarray(spec.default_initial_state, dtype=np.float32),
            control_center=np.asarray(spec.normalization.control_center, dtype=np.float32),
            disturbance_center=np.asarray(spec.normalization.disturbance_center, dtype=np.float32),
            lower=lower,
            upper=upper,
        )
    return tables


def _build_signal_sequence(
    rng: np.random.Generator,
    center: np.ndarray,
    n_steps: int,
    scale: float,
) -> np.ndarray:
    if center.size == 0:
        return np.zeros((n_steps, 0), dtype=np.float32)
    t = np.linspace(0.0, 1.0, n_steps, dtype=np.float32)
    phases = rng.uniform(0.0, 2.0 * np.pi, size=center.shape[0]).astype(np.float32)
    amplitudes = scale * (0.5 + rng.random(center.shape[0], dtype=np.float32))
    trend = rng.normal(0.0, scale * 0.1, size=(center.shape[0],)).astype(np.float32)
    seq = center[None, :] + amplitudes[None, :] * np.sin(
        2.0 * np.pi * t[:, None] + phases[None, :]
    )
    seq = seq + trend[None, :] * t[:, None]
    return seq.astype(np.float32)


def _extract_internal_stream_values(
    flowsheet: FlowsheetSpec,
    unit_states: dict[str, np.ndarray],
    max_stream_vars: int,
) -> dict[str, np.ndarray]:
    values = {}
    for stream in flowsheet.streams:
        padded = np.zeros((max_stream_vars,), dtype=np.float32)
        if stream.source_unit not in {EXTERNAL_SOURCE, EXTERNAL_SINK}:
            source_spec = flowsheet.units[stream.source_unit]
            source_state = unit_states[stream.source_unit]
            idx = [source_spec.state_names.index(name) for name in stream.variables]
            padded[: len(idx)] = source_state[idx]
        values[stream.name] = padded
    return values


def generate_synthetic_flowsheet_data(
    flowsheet: FlowsheetSpec,
    *,
    n_trajectories: int = 16,
    n_steps: int = 24,
    dt: float = 0.1,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Generate a coherent toy dataset for the example flowsheet graphs."""

    rng = np.random.default_rng(seed)
    unit_names = list(flowsheet.units.keys())
    stream_names = [stream.name for stream in flowsheet.streams]
    unit_index = {name: idx for idx, name in enumerate(unit_names)}
    stream_index = {name: idx for idx, name in enumerate(stream_names)}
    unit_tables = _build_unit_tables(flowsheet)

    max_state_dim = max(spec.state_dim for spec in flowsheet.units.values())
    max_control_dim = max(spec.control_dim for spec in flowsheet.units.values())
    max_disturbance_dim = max(spec.disturbance_dim for spec in flowsheet.units.values())
    max_param_dim = max(spec.param_dim for spec in flowsheet.units.values())
    max_stream_vars = max(len(stream.variables) for stream in flowsheet.streams)

    states = np.zeros(
        (n_trajectories, n_steps, len(unit_names), max_state_dim),
        dtype=np.float32,
    )
    controls = np.zeros(
        (n_trajectories, n_steps, len(unit_names), max_control_dim),
        dtype=np.float32,
    )
    disturbances = np.zeros(
        (n_trajectories, n_steps, len(unit_names), max_disturbance_dim),
        dtype=np.float32,
    )
    params = np.zeros((n_trajectories, len(unit_names), max_param_dim), dtype=np.float32)
    stream_values = np.zeros(
        (n_trajectories, n_steps, len(stream_names), max_stream_vars),
        dtype=np.float32,
    )
    global_controls = np.zeros(
        (n_trajectories, n_steps, len(flowsheet.global_controls)),
        dtype=np.float32,
    )
    global_disturbances = np.zeros(
        (n_trajectories, n_steps, len(flowsheet.global_disturbances)),
        dtype=np.float32,
    )

    time = np.tile(
        np.linspace(0.0, dt * (n_steps - 1), n_steps, dtype=np.float32),
        (n_trajectories, 1),
    )

    lag_steps = {
        stream.name: _stream_lag_steps(stream.delay, dt)
        for stream in flowsheet.streams
    }

    for traj_idx in range(n_trajectories):
        external_sequences: dict[str, np.ndarray] = {}
        for stream in flowsheet.streams:
            if stream.source_unit != EXTERNAL_SOURCE:
                continue
            target_spec = flowsheet.units[stream.target_unit]
            centers = np.asarray(
                [
                    target_spec.default_initial_state[
                        target_spec.state_names.index(name)
                    ]
                    for name in (stream.target_variables or stream.variables)
                ],
                dtype=np.float32,
            )
            external_sequences[stream.name] = np.stack(
                [
                    _pad(values, max_stream_vars)
                    for values in _build_signal_sequence(rng, centers, n_steps, 0.05)
                ],
                axis=0,
            )

        global_controls[traj_idx] = _build_signal_sequence(
            rng,
            np.full((len(flowsheet.global_controls),), 0.5, dtype=np.float32),
            n_steps,
            0.15,
        )
        global_disturbances[traj_idx] = _build_signal_sequence(
            rng,
            np.full((len(flowsheet.global_disturbances),), 0.25, dtype=np.float32),
            n_steps,
            0.1,
        )

        unit_states = {}
        for unit_name, spec in flowsheet.units.items():
            unit_idx = unit_index[unit_name]
            tables = unit_tables[unit_name]
            initial = tables.default_state + rng.normal(
                0.0,
                0.03,
                size=(spec.state_dim,),
            ).astype(np.float32)
            initial = np.clip(initial, tables.lower, tables.upper)
            states[traj_idx, 0, unit_idx, : spec.state_dim] = initial
            unit_states[unit_name] = initial.astype(np.float32)

            control_seq = _build_signal_sequence(rng, tables.control_center, n_steps, 0.08)
            disturbance_seq = _build_signal_sequence(
                rng,
                tables.disturbance_center,
                n_steps,
                0.06,
            )
            controls[traj_idx, :, unit_idx, : spec.control_dim] = control_seq
            disturbances[traj_idx, :, unit_idx, : spec.disturbance_dim] = disturbance_seq
            params[traj_idx, unit_idx, : spec.param_dim] = (
                1.0 + 0.1 * rng.normal(size=(spec.param_dim,)).astype(np.float32)
            )

        raw_internal_history = [None] * n_steps
        raw_internal_history[0] = _extract_internal_stream_values(
            flowsheet,
            unit_states,
            max_stream_vars,
        )

        for step_idx in range(n_steps):
            for stream in flowsheet.streams:
                stream_idx = stream_index[stream.name]
                if stream.source_unit == EXTERNAL_SOURCE:
                    stream_values[traj_idx, step_idx, stream_idx] = external_sequences[stream.name][
                        step_idx
                    ]
                else:
                    history_idx = max(0, step_idx - lag_steps[stream.name])
                    stream_values[traj_idx, step_idx, stream_idx] = raw_internal_history[
                        history_idx
                    ][stream.name]

            if step_idx == n_steps - 1:
                break

            incoming = {
                unit_name: np.zeros((flowsheet.units[unit_name].state_dim,), dtype=np.float32)
                for unit_name in unit_names
            }
            for stream in flowsheet.streams:
                if stream.target_unit in {EXTERNAL_SOURCE, EXTERNAL_SINK}:
                    continue
                target_spec = flowsheet.units[stream.target_unit]
                current_stream = stream_values[traj_idx, step_idx, stream_index[stream.name]]
                for value_idx, target_name in enumerate(stream.target_variables or stream.variables):
                    target_idx = target_spec.state_names.index(target_name)
                    incoming[stream.target_unit][target_idx] += current_stream[value_idx]

            next_unit_states = {}
            global_control_mean = float(global_controls[traj_idx, step_idx].mean()) if flowsheet.global_controls else 0.0
            global_disturbance_mean = (
                float(global_disturbances[traj_idx, step_idx].mean())
                if flowsheet.global_disturbances
                else 0.0
            )

            for unit_name, spec in flowsheet.units.items():
                unit_idx = unit_index[unit_name]
                tables = unit_tables[unit_name]
                current_state = states[traj_idx, step_idx, unit_idx, : spec.state_dim]
                control = controls[traj_idx, step_idx, unit_idx, : spec.control_dim]
                disturbance = disturbances[traj_idx, step_idx, unit_idx, : spec.disturbance_dim]
                param = params[traj_idx, unit_idx, : spec.param_dim]

                control_effect = np.zeros((spec.state_dim,), dtype=np.float32)
                disturbance_effect = np.zeros((spec.state_dim,), dtype=np.float32)
                param_effect = np.zeros((spec.state_dim,), dtype=np.float32)

                control_width = min(spec.control_dim, spec.state_dim)
                disturbance_width = min(spec.disturbance_dim, spec.state_dim)
                param_width = min(spec.param_dim, spec.state_dim)
                if control_width > 0:
                    control_effect[:control_width] = (
                        control[:control_width] - tables.control_center[:control_width]
                    )
                if disturbance_width > 0:
                    disturbance_effect[:disturbance_width] = (
                        disturbance[:disturbance_width]
                        - tables.disturbance_center[:disturbance_width]
                    )
                if param_width > 0:
                    param_effect[:param_width] = param[:param_width] - 1.0

                coupling = 0.1 * (np.roll(current_state, 1) - current_state)
                next_state = current_state + dt * (
                    -0.3 * (current_state - tables.default_state)
                    + 0.12 * (incoming[unit_name] - current_state)
                    + 0.08 * control_effect
                    + 0.05 * disturbance_effect
                    + 0.03 * param_effect
                    + 0.02 * global_control_mean
                    - 0.01 * global_disturbance_mean
                    + coupling
                )
                next_state = np.clip(next_state, tables.lower, tables.upper).astype(np.float32)
                states[traj_idx, step_idx + 1, unit_idx, : spec.state_dim] = next_state
                next_unit_states[unit_name] = next_state

            raw_internal_history[step_idx + 1] = _extract_internal_stream_values(
                flowsheet,
                next_unit_states,
                max_stream_vars,
            )

    return {
        "states": states,
        "controls": controls,
        "disturbances": disturbances,
        "params": params,
        "stream_values": stream_values,
        "global_controls": global_controls,
        "global_disturbances": global_disturbances,
        "time": time,
    }


def build_synthetic_flowsheet_dataset(
    flowsheet: FlowsheetSpec,
    *,
    n_trajectories: int = 16,
    n_steps: int = 24,
    dt: float = 0.1,
    seed: int = 0,
    seq_len: int = 12,
    stride: int = 4,
) -> FlowsheetTrajectoryDataset:
    data = generate_synthetic_flowsheet_data(
        flowsheet,
        n_trajectories=n_trajectories,
        n_steps=n_steps,
        dt=dt,
        seed=seed,
    )
    return FlowsheetTrajectoryDataset.from_arrays(
        flowsheet,
        data,
        seq_len=seq_len,
        stride=stride,
    )
