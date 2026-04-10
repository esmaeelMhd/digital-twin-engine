"""Shared demo-engine utilities for the Phase 6 API and frontend."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import yaml

from dte.flowsheet.examples import (
    build_exchanger_reactor_tank_flowsheet,
    build_reactor_separator_recycle_flowsheet,
)
from dte.models.digital_twin import DigitalTwin
from dte.simulators.base import ProcessSimulator, ProcessUnitSpec
from dte.simulators.registry import get_system_spec

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DEMO_CONFIG_PATH = PROJECT_ROOT / "configs" / "demo_app.yaml"


def load_demo_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load the demo website configuration."""

    config_path = DEFAULT_DEMO_CONFIG_PATH if path is None else Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


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


def _disturbance_bounds(spec: ProcessUnitSpec) -> tuple[np.ndarray, np.ndarray]:
    lower = np.asarray(
        [float(spec.disturbance_ranges[name][0]) for name in spec.disturbance_names],
        dtype=np.float32,
    )
    upper = np.asarray(
        [float(spec.disturbance_ranges[name][1]) for name in spec.disturbance_names],
        dtype=np.float32,
    )
    return lower, upper


def _clip_controls(spec: ProcessUnitSpec, controls: np.ndarray) -> np.ndarray:
    lower, upper = _control_bounds(spec)
    return np.clip(controls, lower[None, :], upper[None, :])


def _clip_disturbances(spec: ProcessUnitSpec, disturbances: np.ndarray) -> np.ndarray:
    lower, upper = _disturbance_bounds(spec)
    return np.clip(disturbances, lower[None, :], upper[None, :])


def default_control_sequence(spec: ProcessUnitSpec, n_steps: int) -> np.ndarray:
    """Return a constant nominal control trajectory."""

    midpoint = np.asarray(
        [0.5 * sum(spec.control_ranges[name]) for name in spec.control_names],
        dtype=np.float32,
    )
    return np.tile(midpoint[None, :], (n_steps, 1))


def default_disturbance_sequence(spec: ProcessUnitSpec, n_steps: int) -> np.ndarray:
    """Return a constant nominal disturbance trajectory."""

    base = np.asarray(spec.default_nominal_disturbance, dtype=np.float32)
    return np.tile(base[None, :], (n_steps, 1))


def time_axis(n_steps: int, dt: float) -> np.ndarray:
    return np.linspace(0.0, max(n_steps - 1, 0) * dt, n_steps, dtype=np.float32)


def simulate_open_loop(
    spec: ProcessUnitSpec,
    simulator: ProcessSimulator,
    initial_state: np.ndarray,
    controls: np.ndarray,
    disturbances: np.ndarray,
    dt: float,
) -> np.ndarray:
    """Run a deterministic simulator rollout using simple Euler integration."""

    n_steps = controls.shape[0]
    if disturbances.shape[0] != n_steps:
        raise ValueError("controls and disturbances must have the same horizon length.")

    states = np.zeros((n_steps, spec.state_dim), dtype=np.float32)
    states[0] = np.asarray(initial_state, dtype=np.float32)
    if n_steps == 1:
        return states

    controls = _clip_controls(spec, np.asarray(controls, dtype=np.float32))
    disturbances = _clip_disturbances(spec, np.asarray(disturbances, dtype=np.float32))
    for step in range(1, n_steps):
        prev = states[step - 1]
        control = controls[step - 1]
        disturbance = disturbances[step - 1]
        deriv = np.asarray(
            simulator.dynamics((step - 1) * dt, prev, control, disturbance),
            dtype=np.float32,
        )
        states[step] = prev + float(dt) * deriv
    return states


def constraint_summary(spec: ProcessUnitSpec, states: np.ndarray) -> dict[str, float]:
    """Summarize bound and positivity violations over a trajectory."""

    lower, upper = _state_bounds(spec)
    lower_mask = np.isfinite(lower)
    upper_mask = np.isfinite(upper)
    positivity_mask = np.logical_and(np.isfinite(lower), lower >= 0.0)

    summary = {
        "below_lower_bound_rate": 0.0,
        "above_upper_bound_rate": 0.0,
        "positivity_violation_rate": 0.0,
        "worst_lower_violation": 0.0,
        "worst_upper_violation": 0.0,
    }
    if states.size == 0:
        return summary

    if np.any(lower_mask):
        lower_diff = np.maximum(lower[None, :] - states, 0.0)
        summary["below_lower_bound_rate"] = float(
            np.mean((states[:, lower_mask] < lower[lower_mask][None, :]).astype(np.float32))
        )
        summary["worst_lower_violation"] = float(np.max(lower_diff[:, lower_mask]))
    if np.any(upper_mask):
        upper_diff = np.maximum(states - upper[None, :], 0.0)
        summary["above_upper_bound_rate"] = float(
            np.mean((states[:, upper_mask] > upper[upper_mask][None, :]).astype(np.float32))
        )
        summary["worst_upper_violation"] = float(np.max(upper_diff[:, upper_mask]))
    if np.any(positivity_mask):
        summary["positivity_violation_rate"] = float(
            np.mean((states[:, positivity_mask] < 0.0).astype(np.float32))
        )
    return summary


def rollout_with_model(
    model: DigitalTwin,
    spec: ProcessUnitSpec,
    initial_state: np.ndarray,
    controls: np.ndarray,
    disturbances: np.ndarray,
    params: np.ndarray,
    dt: float,
    *,
    n_samples: int = 16,
    seed: int = 0,
) -> dict[str, Any]:
    """Generate mean and uncertainty bands from a trained DigitalTwin."""

    ts = jnp.asarray(time_axis(controls.shape[0], dt), dtype=jnp.float32)
    result = model.predict_ensemble(
        jnp.asarray(initial_state, dtype=jnp.float32),
        jnp.asarray(controls, dtype=jnp.float32),
        jnp.asarray(disturbances, dtype=jnp.float32),
        jnp.asarray(params, dtype=jnp.float32),
        ts,
        jax.random.PRNGKey(seed),
        n_samples=n_samples,
    )
    samples = np.asarray(result["states_samples"])
    mean = np.asarray(result["states_mean"])
    std = np.asarray(result["states_std"])
    p05 = np.percentile(samples, 5.0, axis=0)
    p95 = np.percentile(samples, 95.0, axis=0)
    return {
        "source": "model",
        "times": np.asarray(ts),
        "mean": mean,
        "std": std,
        "p05": p05,
        "p95": p95,
        "samples": samples,
        "constraint_summary": constraint_summary(spec, mean),
    }


def rollout_with_simulator_ensemble(
    spec: ProcessUnitSpec,
    simulator: ProcessSimulator,
    initial_state: np.ndarray,
    controls: np.ndarray,
    disturbances: np.ndarray,
    dt: float,
    *,
    n_samples: int = 16,
    seed: int = 0,
) -> dict[str, Any]:
    """Approximate uncertainty with disturbance/initial-state perturbations."""

    rng = np.random.default_rng(seed)
    controls = _clip_controls(spec, np.asarray(controls, dtype=np.float32))
    disturbances = _clip_disturbances(spec, np.asarray(disturbances, dtype=np.float32))
    disturbance_scale = np.maximum(np.std(disturbances, axis=0), 1e-3)
    lower_d, upper_d = _disturbance_bounds(spec)
    samples = []
    for _ in range(max(int(n_samples), 2)):
        init = np.asarray(initial_state, dtype=np.float32) + rng.normal(
            loc=0.0,
            scale=0.01 * np.maximum(np.abs(initial_state), 1.0),
            size=(spec.state_dim,),
        ).astype(np.float32)
        disturbed = disturbances + rng.normal(
            loc=0.0,
            scale=0.05 * disturbance_scale,
            size=disturbances.shape,
        ).astype(np.float32)
        disturbed = np.clip(disturbed, lower_d[None, :], upper_d[None, :])
        samples.append(simulate_open_loop(spec, simulator, init, controls, disturbed, dt))
    samples_arr = np.asarray(samples, dtype=np.float32)
    mean = np.mean(samples_arr, axis=0)
    std = np.std(samples_arr, axis=0)
    return {
        "source": "simulator_ensemble",
        "times": time_axis(controls.shape[0], dt),
        "mean": mean,
        "std": std,
        "p05": np.percentile(samples_arr, 5.0, axis=0),
        "p95": np.percentile(samples_arr, 95.0, axis=0),
        "samples": samples_arr,
        "constraint_summary": constraint_summary(spec, mean),
    }


def rollout_scenario(
    spec: ProcessUnitSpec,
    simulator: ProcessSimulator,
    *,
    initial_state: np.ndarray,
    controls: np.ndarray,
    disturbances: np.ndarray,
    dt: float,
    model: DigitalTwin | None = None,
    params: np.ndarray | None = None,
    n_samples: int = 16,
    seed: int = 0,
) -> dict[str, Any]:
    """Run a demo rollout using a model if available, else simulator ensemble."""

    controls = _clip_controls(spec, np.asarray(controls, dtype=np.float32))
    disturbances = _clip_disturbances(spec, np.asarray(disturbances, dtype=np.float32))
    if model is not None:
        params_arr = (
            np.asarray(params, dtype=np.float32)
            if params is not None
            else np.ones(spec.param_dim, dtype=np.float32)
        )
        return rollout_with_model(
            model,
            spec,
            np.asarray(initial_state, dtype=np.float32),
            controls,
            disturbances,
            params_arr,
            dt,
            n_samples=n_samples,
            seed=seed,
        )
    return rollout_with_simulator_ensemble(
        spec,
        simulator,
        np.asarray(initial_state, dtype=np.float32),
        controls,
        disturbances,
        dt,
        n_samples=n_samples,
        seed=seed,
    )


def compare_scenarios(
    spec: ProcessUnitSpec,
    simulator: ProcessSimulator,
    *,
    initial_state: np.ndarray,
    baseline_controls: np.ndarray,
    candidate_controls: np.ndarray,
    disturbances: np.ndarray,
    dt: float,
    model: DigitalTwin | None = None,
    params: np.ndarray | None = None,
    n_samples: int = 16,
    seed: int = 0,
) -> dict[str, Any]:
    """Compare baseline and candidate control schedules."""

    baseline = rollout_scenario(
        spec,
        simulator,
        initial_state=initial_state,
        controls=baseline_controls,
        disturbances=disturbances,
        dt=dt,
        model=model,
        params=params,
        n_samples=n_samples,
        seed=seed,
    )
    candidate = rollout_scenario(
        spec,
        simulator,
        initial_state=initial_state,
        controls=candidate_controls,
        disturbances=disturbances,
        dt=dt,
        model=model,
        params=params,
        n_samples=n_samples,
        seed=seed + 1,
    )
    final_delta = candidate["mean"][-1] - baseline["mean"][-1]
    mean_abs_delta = np.mean(np.abs(candidate["mean"] - baseline["mean"]), axis=0)
    return {
        "baseline": baseline,
        "candidate": candidate,
        "summary": {
            "final_state_delta_norm": float(np.linalg.norm(final_delta)),
            "mean_abs_delta": {
                name: float(value)
                for name, value in zip(spec.state_names, mean_abs_delta.tolist())
            },
            "candidate_advantage": {
                name: float(delta)
                for name, delta in zip(spec.state_names, final_delta.tolist())
            },
        },
    }


def optimize_control_sequence(
    spec: ProcessUnitSpec,
    simulator: ProcessSimulator,
    *,
    initial_state: np.ndarray,
    disturbances: np.ndarray,
    dt: float,
    target_state: np.ndarray,
    tracked_state_names: list[str] | None = None,
    n_candidates: int = 48,
    seed: int = 0,
) -> dict[str, Any]:
    """Simple random-shooting demo optimizer over ramp control sequences."""

    disturbances = _clip_disturbances(spec, np.asarray(disturbances, dtype=np.float32))
    n_steps = disturbances.shape[0]
    lower, upper = _control_bounds(spec)
    tracked_state_names = tracked_state_names or list(spec.state_names)
    tracked_indices = [spec.state_names.index(name) for name in tracked_state_names]
    target_state = np.asarray(target_state, dtype=np.float32)

    rng = np.random.default_rng(seed)
    best_cost = math.inf
    best_controls = default_control_sequence(spec, n_steps)
    best_states = simulate_open_loop(
        spec,
        simulator,
        np.asarray(initial_state, dtype=np.float32),
        best_controls,
        disturbances,
        dt,
    )

    for _ in range(max(int(n_candidates), 1)):
        start = rng.uniform(lower, upper).astype(np.float32)
        end = rng.uniform(lower, upper).astype(np.float32)
        controls = np.stack(
            [
                np.linspace(start[idx], end[idx], n_steps, dtype=np.float32)
                for idx in range(spec.control_dim)
            ],
            axis=-1,
        )
        states = simulate_open_loop(
            spec,
            simulator,
            np.asarray(initial_state, dtype=np.float32),
            controls,
            disturbances,
            dt,
        )
        tracked_error = states[:, tracked_indices] - target_state[tracked_indices][None, :]
        terminal_error = states[-1, tracked_indices] - target_state[tracked_indices]
        smoothness = np.diff(controls, axis=0) if n_steps > 1 else np.zeros_like(controls)
        cost = (
            float(np.mean(tracked_error ** 2))
            + 2.0 * float(np.mean(terminal_error ** 2))
            + 0.05 * float(np.mean(smoothness ** 2))
        )
        if cost < best_cost:
            best_cost = cost
            best_controls = controls
            best_states = states

    return {
        "control_sequence": best_controls,
        "predicted_states": best_states,
        "objective": float(best_cost),
        "tracked_state_names": tracked_state_names,
        "constraint_summary": constraint_summary(spec, best_states),
    }


def flowsheet_preview_catalog() -> list[dict[str, Any]]:
    """Return small flowsheet previews for the demo site."""

    previews = []
    for spec in (
        build_exchanger_reactor_tank_flowsheet(),
        build_reactor_separator_recycle_flowsheet(),
    ):
        previews.append(
            {
                "id": spec.name,
                "title": spec.name.replace("_", " ").title(),
                "description": spec.description,
                "units": [
                    {
                        "name": unit_name,
                        "family": getattr(unit_spec, "family", "generic"),
                    }
                    for unit_name, unit_spec in spec.units.items()
                ],
                "streams": [
                    {
                        "name": stream.name,
                        "source": stream.source_unit,
                        "target": stream.target_unit,
                        "kind": stream.kind,
                    }
                    for stream in spec.streams
                ],
            }
        )
    return previews


def demo_catalog_from_config(
    config: dict[str, Any],
    system_configs: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Build a JSON-friendly demo catalog from config and loaded specs."""

    demos = []
    for demo in config.get("demos", []):
        system_name = demo["system"]
        system_config = system_configs.get(system_name)
        if system_config is None:
            continue
        spec = get_system_spec(system_config)
        demos.append(
            {
                "id": demo["id"],
                "title": demo["title"],
                "system": system_name,
                "kind": demo.get("kind", "unit_demo"),
                "description": demo.get("description", ""),
                "controls": list(spec.control_names),
                "disturbances": list(spec.disturbance_names),
                "states": list(spec.state_names),
                "dt": float(demo.get("dt", 0.1)),
                "n_steps": int(demo.get("n_steps", 25)),
                "highlight_states": list(demo.get("highlight_states", spec.state_names[:2])),
            }
        )
    return {
        "product_name": config.get("theme", {}).get("product_name", "Digital Twin Engine"),
        "headline": config.get("theme", {}).get("headline", "Industrial dynamics you can steer."),
        "summary": config.get("theme", {}).get("summary", ""),
        "demos": demos,
        "flowsheets": flowsheet_preview_catalog(),
    }
