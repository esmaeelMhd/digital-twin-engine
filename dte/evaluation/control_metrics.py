"""Control-oriented metrics for closed-loop evaluation and robustness checks."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from dte.simulators.base import ProcessUnitSpec


def _weight_vector(size: int, weights: Sequence[float] | None) -> np.ndarray:
    if weights is None:
        return np.ones(size, dtype=np.float32)
    arr = np.asarray(weights, dtype=np.float32)
    if arr.shape != (size,):
        raise ValueError(f"Expected weight vector of shape ({size},), got {arr.shape}.")
    return arr


def _state_bounds(spec: ProcessUnitSpec) -> tuple[np.ndarray, np.ndarray]:
    lower = np.full(spec.state_dim, -np.inf, dtype=np.float32)
    upper = np.full(spec.state_dim, np.inf, dtype=np.float32)
    for idx, channel in enumerate(getattr(spec, "state_channels", [])):
        if channel.lower_bound is not None:
            lower[idx] = float(channel.lower_bound)
        if channel.upper_bound is not None:
            upper[idx] = float(channel.upper_bound)
    return lower, upper


def _control_bounds(spec: ProcessUnitSpec) -> tuple[np.ndarray, np.ndarray]:
    lower = np.asarray(
        [float(spec.control_ranges[name][0]) for name in spec.control_names],
        dtype=np.float32,
    )
    upper = np.asarray(
        [float(spec.control_ranges[name][1]) for name in spec.control_names],
        dtype=np.float32,
    )
    return lower, upper


def tracking_cost(
    states: np.ndarray,
    target_state: np.ndarray,
    *,
    state_weights: Sequence[float] | None = None,
) -> float:
    """Quadratic tracking cost over a trajectory."""

    states_arr = np.asarray(states, dtype=np.float32)
    target_arr = np.asarray(target_state, dtype=np.float32)
    weights = _weight_vector(states_arr.shape[-1], state_weights)
    errors = states_arr - target_arr[None, :]
    return float(np.mean((errors**2) * weights[None, :]))


def control_effort_cost(
    controls: np.ndarray,
    *,
    control_weights: Sequence[float] | None = None,
    previous_control: np.ndarray | None = None,
) -> float:
    """Quadratic penalty on control movement."""

    controls_arr = np.asarray(controls, dtype=np.float32)
    weights = _weight_vector(controls_arr.shape[-1], control_weights)
    if controls_arr.shape[0] == 0:
        return 0.0
    if previous_control is None:
        deltas = np.diff(controls_arr, axis=0)
    else:
        previous_arr = np.asarray(previous_control, dtype=np.float32).reshape(1, -1)
        deltas = np.diff(np.concatenate([previous_arr, controls_arr], axis=0), axis=0)
    if deltas.size == 0:
        return 0.0
    return float(np.mean((deltas**2) * weights[None, :]))


def summarize_constraint_violations(
    spec: ProcessUnitSpec,
    states: np.ndarray,
    controls: np.ndarray | None = None,
) -> dict[str, float]:
    """Summarize state/control bound violations and positivity breaches."""

    states_arr = np.asarray(states, dtype=np.float32)
    lower_state, upper_state = _state_bounds(spec)
    positivity_mask = np.isfinite(lower_state) & (lower_state >= 0.0)

    summary = {
        "state_below_lower_rate": 0.0,
        "state_above_upper_rate": 0.0,
        "state_worst_lower_violation": 0.0,
        "state_worst_upper_violation": 0.0,
        "state_positivity_violation_rate": 0.0,
        "control_below_lower_rate": 0.0,
        "control_above_upper_rate": 0.0,
        "control_worst_lower_violation": 0.0,
        "control_worst_upper_violation": 0.0,
    }

    if states_arr.size > 0:
        lower_mask = np.isfinite(lower_state)
        upper_mask = np.isfinite(upper_state)
        if np.any(lower_mask):
            lower_violation = np.maximum(lower_state[None, :] - states_arr, 0.0)
            summary["state_below_lower_rate"] = float(
                np.mean((states_arr[:, lower_mask] < lower_state[lower_mask][None, :]).astype(np.float32))
            )
            summary["state_worst_lower_violation"] = float(np.max(lower_violation[:, lower_mask]))
        if np.any(upper_mask):
            upper_violation = np.maximum(states_arr - upper_state[None, :], 0.0)
            summary["state_above_upper_rate"] = float(
                np.mean((states_arr[:, upper_mask] > upper_state[upper_mask][None, :]).astype(np.float32))
            )
            summary["state_worst_upper_violation"] = float(np.max(upper_violation[:, upper_mask]))
        if np.any(positivity_mask):
            summary["state_positivity_violation_rate"] = float(
                np.mean((states_arr[:, positivity_mask] < 0.0).astype(np.float32))
            )

    if controls is not None:
        controls_arr = np.asarray(controls, dtype=np.float32)
        lower_control, upper_control = _control_bounds(spec)
        lower_violation = np.maximum(lower_control[None, :] - controls_arr, 0.0)
        upper_violation = np.maximum(controls_arr - upper_control[None, :], 0.0)
        summary["control_below_lower_rate"] = float(
            np.mean((controls_arr < lower_control[None, :]).astype(np.float32))
        )
        summary["control_above_upper_rate"] = float(
            np.mean((controls_arr > upper_control[None, :]).astype(np.float32))
        )
        summary["control_worst_lower_violation"] = float(np.max(lower_violation))
        summary["control_worst_upper_violation"] = float(np.max(upper_violation))

    return summary


def disturbance_sensitivity(
    nominal_states: np.ndarray,
    perturbed_states: np.ndarray,
) -> dict[str, float]:
    """Measure trajectory sensitivity to disturbance changes."""

    nominal = np.asarray(nominal_states, dtype=np.float32)
    perturbed = np.asarray(perturbed_states, dtype=np.float32)
    delta = perturbed - nominal
    return {
        "mean_abs_state_delta": float(np.mean(np.abs(delta))),
        "max_abs_state_delta": float(np.max(np.abs(delta))),
        "final_state_delta_norm": float(np.linalg.norm(delta[-1])),
    }


def mismatch_robustness(
    reference_states: np.ndarray,
    candidate_states: np.ndarray,
) -> dict[str, float]:
    """Measure robustness to plant/model mismatch via trajectory disagreement."""

    reference = np.asarray(reference_states, dtype=np.float32)
    candidate = np.asarray(candidate_states, dtype=np.float32)
    diff = candidate - reference
    rmse = float(np.sqrt(np.mean(diff**2)))
    denom = float(np.sqrt(np.mean(reference**2)) + 1e-6)
    return {
        "rmse": rmse,
        "normalized_rmse": rmse / denom,
        "final_state_delta_norm": float(np.linalg.norm(diff[-1])),
    }


def closed_loop_metrics(
    spec: ProcessUnitSpec,
    states: np.ndarray,
    controls: np.ndarray,
    target_state: np.ndarray,
    *,
    state_weights: Sequence[float] | None = None,
    control_weights: Sequence[float] | None = None,
    previous_control: np.ndarray | None = None,
    constraint_penalty: float = 10.0,
) -> dict[str, float]:
    """Aggregate standard control-facing metrics into one summary."""

    tracking = tracking_cost(states, target_state, state_weights=state_weights)
    effort = control_effort_cost(
        controls,
        control_weights=control_weights,
        previous_control=previous_control,
    )
    constraints = summarize_constraint_violations(spec, states, controls)
    violation_mass = sum(
        float(value)
        for key, value in constraints.items()
        if "rate" in key or "worst" in key
    )
    total = tracking + 0.1 * effort + float(constraint_penalty) * violation_mass
    return {
        "tracking_cost": tracking,
        "control_effort_cost": effort,
        "constraint_penalty": float(constraint_penalty) * violation_mass,
        "total_cost": total,
        **constraints,
    }
