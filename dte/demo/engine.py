"""Shared demo-engine utilities for the Phase 6 API and frontend."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import yaml

from dte.data.datasets.universal_unit_dataset import (
    MultiSystemTrajectoryDataset,
    SystemDatasetSource,
)
from dte.flowsheet.examples import (
    build_exchanger_reactor_tank_flowsheet,
    build_reactor_separator_recycle_flowsheet,
)
from dte.models.unit.digital_twin import DigitalTwin
from dte.models.universal.digital_twin import UniversalDigitalTwin
from dte.simulators.base import ProcessSimulator, ProcessUnitSpec
from dte.simulators.registry import get_system_spec

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DEMO_CONFIG_PATH = PROJECT_ROOT / "configs" / "demo_app.yaml"


@dataclass(frozen=True)
class UniversalDemoRuntime:
    """Loaded universal checkpoint plus the metadata required for inference."""

    model: UniversalDigitalTwin
    system_ids: dict[str, int]
    model_path: str
    config_path: str


def load_demo_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load the demo website configuration."""

    config_path = DEFAULT_DEMO_CONFIG_PATH if path is None else Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _resolve_configured_path(
    value: str | Path | None,
    *,
    anchor: str | Path | None = None,
) -> Path | None:
    if value is None:
        return None
    text = os.path.expandvars(str(value)).strip()
    if not text:
        return None
    raw_path = Path(text)
    if raw_path.is_absolute():
        return raw_path
    candidates: list[Path] = []
    if anchor is not None:
        candidates.append(Path(anchor).resolve().parent / raw_path)
    candidates.append(PROJECT_ROOT / raw_path)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _load_json_if_exists(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _load_text_if_exists(path: Path | None) -> str | None:
    if path is None or not path.exists():
        return None
    return path.read_text(encoding="utf-8")


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(numeric):
        return None
    return numeric


def _load_universal_sources(
    universal_config: dict[str, Any],
    *,
    anchor: str | Path | None,
) -> list[SystemDatasetSource]:
    systems = universal_config.get("data", {}).get("systems", [])
    sources: list[SystemDatasetSource] = []
    for item in systems:
        system_config_path = _resolve_configured_path(item.get("system_config"), anchor=anchor)
        data_dir_path = _resolve_configured_path(item.get("data_dir"), anchor=anchor)
        if system_config_path is None or data_dir_path is None:
            continue
        sources.append(
            SystemDatasetSource(
                name=str(item["name"]),
                system_config=str(system_config_path),
                data_dir=str(data_dir_path),
                weight=float(item.get("weight", 1.0)),
            )
        )
    return sources


def load_demo_release_snapshot(
    config: dict[str, Any],
    *,
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    """Return a condensed release snapshot for the presentation UI."""

    runtime_cfg = config.get("runtime", {})
    anchor = DEFAULT_DEMO_CONFIG_PATH if config_path is None else Path(config_path)
    model_path = _resolve_configured_path(runtime_cfg.get("model_path"), anchor=anchor)
    universal_config_path = _resolve_configured_path(runtime_cfg.get("config_path"), anchor=anchor)
    train_summary_path = _resolve_configured_path(
        runtime_cfg.get("train_summary_path"),
        anchor=anchor,
    )
    eval_summary_path = _resolve_configured_path(
        runtime_cfg.get("eval_summary_path"),
        anchor=anchor,
    )
    milestone_summary_path = _resolve_configured_path(
        runtime_cfg.get("milestone_summary_path"),
        anchor=anchor,
    )
    customer_summary_path = _resolve_configured_path(
        runtime_cfg.get("customer_pilot_summary_path"),
        anchor=anchor,
    )
    customer_report_path = _resolve_configured_path(
        runtime_cfg.get("customer_report_path"),
        anchor=anchor,
    )

    train_summary = _load_json_if_exists(train_summary_path)
    eval_summary = _load_json_if_exists(eval_summary_path)
    milestone_summary = _load_json_if_exists(milestone_summary_path)
    customer_summary = _load_json_if_exists(customer_summary_path)
    customer_report_markdown = _load_text_if_exists(customer_report_path)

    per_system_total_loss: dict[str, float] = {}
    for system_name, metrics in (eval_summary or {}).get("per_system_val_losses", {}).items():
        if isinstance(metrics, dict):
            total = _safe_float(metrics.get("total"))
            if total is not None:
                per_system_total_loss[system_name] = total

    return {
        "release_label": runtime_cfg.get("release_label", "V1 milestone release"),
        "model_available": bool(model_path and model_path.exists()),
        "config_available": bool(universal_config_path and universal_config_path.exists()),
        "runtime_samples": int(runtime_cfg.get("n_samples", 24)),
        "model_path": str(model_path) if model_path is not None else None,
        "config_path": str(universal_config_path) if universal_config_path is not None else None,
        "train_best_val_loss": _safe_float((train_summary or {}).get("best_val_loss")),
        "eval_metric_name": (eval_summary or {}).get(
            "aggregate_metric_name",
            (eval_summary or {}).get("aggregate_method"),
        ),
        "eval_metric_value": _safe_float((eval_summary or {}).get("aggregate_metric_value")),
        "per_system_total_loss": per_system_total_loss,
        "milestone_status": (milestone_summary or {}).get("status"),
        "customer_status": (customer_summary or {}).get("adaptation_status", (customer_summary or {}).get("status")),
        "customer_best_unit_template": (customer_summary or {}).get("best_unit_template"),
        "customer_best_val_loss": _safe_float((customer_summary or {}).get("best_val_loss")),
        "customer_forecast_rmse": _safe_float((customer_summary or {}).get("forecast_rmse")),
        "customer_rollout_rmse": _safe_float((customer_summary or {}).get("rollout_rmse")),
        "customer_report_path": str(customer_report_path) if customer_report_path is not None else None,
        "customer_report_exists": bool(customer_report_path and customer_report_path.exists()),
        "customer_report_markdown": customer_report_markdown,
    }


def load_demo_model_runtime(
    config: dict[str, Any],
    *,
    config_path: str | Path | None = None,
) -> UniversalDemoRuntime | None:
    """Load the blessed universal checkpoint used by the V1 demo app."""

    runtime_cfg = config.get("runtime", {})
    anchor = DEFAULT_DEMO_CONFIG_PATH if config_path is None else Path(config_path)
    model_path = _resolve_configured_path(runtime_cfg.get("model_path"), anchor=anchor)
    universal_config_path = _resolve_configured_path(runtime_cfg.get("config_path"), anchor=anchor)
    if model_path is None or universal_config_path is None:
        return None
    if not model_path.exists() or not universal_config_path.exists():
        return None

    with universal_config_path.open("r", encoding="utf-8") as handle:
        universal_config = yaml.safe_load(handle) or {}
    sources = _load_universal_sources(universal_config, anchor=universal_config_path)
    if not sources:
        return None
    metadata = MultiSystemTrajectoryDataset.metadata_from_sources(sources)
    model = UniversalDigitalTwin.load(str(model_path), universal_config, metadata)
    system_ids = {name: idx for idx, name in enumerate(metadata.system_names)}
    return UniversalDemoRuntime(
        model=model,
        system_ids=system_ids,
        model_path=str(model_path),
        config_path=str(universal_config_path),
    )


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


def _apply_named_updates(
    names: list[str],
    base_vector: np.ndarray,
    updates: dict[str, Any] | None,
) -> np.ndarray:
    vector = np.asarray(base_vector, dtype=np.float32).copy()
    if not updates:
        return vector
    name_to_idx = {name: idx for idx, name in enumerate(names)}
    for name, value in updates.items():
        idx = name_to_idx.get(name)
        if idx is not None:
            vector[idx] = float(value)
    return vector


def build_signal_sequence(
    spec: ProcessUnitSpec,
    n_steps: int,
    *,
    signal_kind: str,
    profile: dict[str, Any] | None = None,
) -> np.ndarray:
    """Build a preset control or disturbance sequence from demo config."""

    if signal_kind == "control":
        names = list(spec.control_names)
        sequence = default_control_sequence(spec, n_steps)
        clip_fn = lambda arr: _clip_controls(spec, arr)
    elif signal_kind == "disturbance":
        names = list(spec.disturbance_names)
        sequence = default_disturbance_sequence(spec, n_steps)
        clip_fn = lambda arr: _clip_disturbances(spec, arr)
    else:
        raise ValueError(f"Unsupported signal kind: {signal_kind}")

    if profile is None:
        return clip_fn(sequence)

    profile_type = str(profile.get("type", "constant")).lower()
    if profile_type == "constant":
        vector = _apply_named_updates(
            names,
            sequence[0],
            profile.get("channels") or profile.get("values"),
        )
        sequence = np.tile(vector[None, :], (n_steps, 1))
    elif profile_type == "ramp":
        start = _apply_named_updates(
            names,
            sequence[0],
            profile.get("start") or profile.get("channels"),
        )
        end = _apply_named_updates(names, start, profile.get("end"))
        sequence = np.stack(
            [
                np.linspace(start[idx], end[idx], n_steps, dtype=np.float32)
                for idx in range(len(names))
            ],
            axis=-1,
        )
    elif profile_type == "pulse":
        base = _apply_named_updates(names, sequence[0], profile.get("base"))
        pulse = _apply_named_updates(
            names,
            base,
            profile.get("pulse") or profile.get("channels"),
        )
        start_step = int(profile.get("start_step", max(n_steps // 3, 1)))
        duration = int(profile.get("duration", max(n_steps // 5, 1)))
        end_step = min(max(start_step, 0) + max(duration, 1), n_steps)
        sequence = np.tile(base[None, :], (n_steps, 1))
        sequence[max(start_step, 0):end_step] = pulse
    else:
        raise ValueError(f"Unsupported demo profile type: {profile_type}")

    return clip_fn(sequence.astype(np.float32))


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
    lower_state, upper_state = _state_bounds(spec)
    finite_lower = np.where(np.isfinite(lower_state), lower_state, -np.inf)
    finite_upper = np.where(np.isfinite(upper_state), upper_state, np.inf)
    for step in range(1, n_steps):
        prev = states[step - 1]
        if not np.all(np.isfinite(prev)):
            states[step:] = prev
            break
        control = controls[step - 1]
        disturbance = disturbances[step - 1]
        deriv = np.asarray(
            simulator.dynamics((step - 1) * dt, prev, control, disturbance),
            dtype=np.float32,
        )
        if not np.all(np.isfinite(deriv)):
            states[step:] = prev
            break
        next_state = prev + float(dt) * deriv
        next_state = np.clip(next_state, finite_lower, finite_upper)
        if not np.all(np.isfinite(next_state)):
            states[step:] = prev
            break
        states[step] = next_state
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


def nominal_parameter_vector(
    spec: ProcessUnitSpec,
    simulator: ProcessSimulator,
    *,
    seed: int = 0,
) -> np.ndarray:
    """Return a pragmatic parameter vector for demo inference."""

    try:
        params = np.asarray(
            simulator.sample_data_generation_params(jax.random.PRNGKey(seed)),
            dtype=np.float32,
        ).reshape(-1)
    except Exception:
        params = np.ones(spec.param_dim, dtype=np.float32)
    if params.shape[0] < spec.param_dim:
        padded = np.ones(spec.param_dim, dtype=np.float32)
        padded[: params.shape[0]] = params
        return padded
    return params[: spec.param_dim]


def rollout_with_universal_model(
    runtime: UniversalDemoRuntime,
    spec: ProcessUnitSpec,
    simulator: ProcessSimulator,
    initial_state: np.ndarray,
    controls: np.ndarray,
    disturbances: np.ndarray,
    params: np.ndarray,
    dt: float,
    *,
    n_samples: int = 16,
    seed: int = 0,
) -> dict[str, Any]:
    """Generate mean and uncertainty bands from the blessed universal checkpoint."""

    system_id = runtime.system_ids.get(spec.name)
    if system_id is None:
        raise ValueError(f"System '{spec.name}' is not available in the universal demo runtime.")

    model = runtime.model
    n_steps = int(controls.shape[0])
    system_id_arr = jnp.asarray(system_id, dtype=jnp.int32)
    state_mask = model.state_mask_table[system_id_arr]
    control_mask = model.control_mask_table[system_id_arr]
    disturbance_mask = model.disturbance_mask_table[system_id_arr]
    param_mask = model.param_mask_table[system_id_arr]

    padded_state = np.zeros(model.max_state_dim, dtype=np.float32)
    padded_state[: spec.state_dim] = np.asarray(initial_state, dtype=np.float32)
    padded_controls = np.zeros((n_steps, model.max_control_dim), dtype=np.float32)
    padded_controls[:, : spec.control_dim] = np.asarray(controls, dtype=np.float32)
    padded_disturbances = np.zeros((n_steps, model.max_disturbance_dim), dtype=np.float32)
    padded_disturbances[:, : spec.disturbance_dim] = np.asarray(disturbances, dtype=np.float32)
    padded_params = np.zeros(model.max_param_dim, dtype=np.float32)
    active_params = np.asarray(params, dtype=np.float32).reshape(-1)
    padded_params[: min(active_params.shape[0], spec.param_dim)] = active_params[: spec.param_dim]

    ts = jnp.asarray(time_axis(n_steps, dt), dtype=jnp.float32)
    state_norm = model.normalize_states(jnp.asarray(padded_state), system_id_arr) * state_mask
    controls_norm = model.normalize_controls(
        jnp.asarray(padded_controls),
        system_id_arr,
    ) * control_mask
    disturbances_norm = model.normalize_disturbances(
        jnp.asarray(padded_disturbances),
        system_id_arr,
    ) * disturbance_mask
    params_scaled = model.scale_params(jnp.asarray(padded_params), system_id_arr) * param_mask

    sample_keys = jax.random.split(jax.random.PRNGKey(seed), max(int(n_samples), 2))
    trajectories = []
    for sample_key in sample_keys:
        z0, _, _ = model.encode(
            state_norm,
            params_scaled,
            controls_norm[0],
            state_mask,
            control_mask,
            param_mask,
            system_id_arr,
            sample_key,
        )
        z_traj = model.rollout_latent(
            ts,
            z0,
            controls_norm,
            disturbances_norm,
            params_scaled,
            control_mask,
            disturbance_mask,
            param_mask,
            system_id_arr,
        )
        pred_norm = jax.vmap(
            lambda z_t, control_t: model.decode(
                z_t,
                params_scaled,
                control_t,
                state_mask,
                control_mask,
                param_mask,
                system_id_arr,
            )
        )(z_traj, controls_norm)
        pred_states = model.denormalize_states(pred_norm, system_id_arr)
        trajectories.append(np.asarray(pred_states[:, : spec.state_dim], dtype=np.float32))

    samples = np.asarray(trajectories, dtype=np.float32)
    mean = np.mean(samples, axis=0)
    std = np.std(samples, axis=0)
    return {
        "source": "universal_model",
        "times": np.asarray(ts),
        "mean": mean,
        "std": std,
        "p05": np.percentile(samples, 5.0, axis=0),
        "p95": np.percentile(samples, 95.0, axis=0),
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
    model: DigitalTwin | UniversalDemoRuntime | None = None,
    params: np.ndarray | None = None,
    n_samples: int = 16,
    seed: int = 0,
) -> dict[str, Any]:
    """Run a demo rollout using a model if available, else simulator ensemble."""

    controls = _clip_controls(spec, np.asarray(controls, dtype=np.float32))
    disturbances = _clip_disturbances(spec, np.asarray(disturbances, dtype=np.float32))
    if isinstance(model, UniversalDemoRuntime):
        params_arr = (
            np.asarray(params, dtype=np.float32)
            if params is not None
            else nominal_parameter_vector(spec, simulator, seed=seed)
        )
        return rollout_with_universal_model(
            model,
            spec,
            simulator,
            np.asarray(initial_state, dtype=np.float32),
            controls,
            disturbances,
            params_arr,
            dt,
            n_samples=n_samples,
            seed=seed,
        )
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
    model: DigitalTwin | UniversalDemoRuntime | None = None,
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
    reference_controls: np.ndarray | None = None,
    active_control_names: list[str] | None = None,
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
    reference_controls = (
        None
        if reference_controls is None
        else _clip_controls(spec, np.asarray(reference_controls, dtype=np.float32))
    )
    active_control_names = active_control_names or list(spec.control_names)
    active_indices = [
        spec.control_names.index(name)
        for name in active_control_names
        if name in spec.control_names
    ]
    if not active_indices:
        active_indices = list(range(spec.control_dim))

    rng = np.random.default_rng(seed)
    tracked_scale = np.maximum(
        np.maximum(
            np.abs(np.asarray(initial_state, dtype=np.float32)[tracked_indices]),
            np.abs(target_state[tracked_indices]),
        ),
        1.0,
    )
    tracked_guard = 20.0 * tracked_scale
    terminal_guard = 25.0 * tracked_scale
    huge_cost = float(1e12)
    local_span = 0.15 * (upper - lower)
    fixed_controls = (
        reference_controls.copy()
        if reference_controls is not None
        else default_control_sequence(spec, n_steps)
    )

    def _evaluate_controls(controls: np.ndarray) -> tuple[float, np.ndarray]:
        states = simulate_open_loop(
            spec,
            simulator,
            np.asarray(initial_state, dtype=np.float32),
            controls,
            disturbances,
            dt,
        )
        tracked_states = states[:, tracked_indices]
        terminal_state = states[-1, tracked_indices]
        if not np.all(np.isfinite(tracked_states)) or not np.all(np.isfinite(terminal_state)):
            return huge_cost, states
        if np.any(np.abs(tracked_states) > tracked_guard[None, :]):
            return huge_cost, states
        if np.any(np.abs(terminal_state) > terminal_guard):
            return huge_cost, states
        tracked_error = states[:, tracked_indices] - target_state[tracked_indices][None, :]
        terminal_error = states[-1, tracked_indices] - target_state[tracked_indices]
        smoothness = np.diff(controls, axis=0) if n_steps > 1 else np.zeros_like(controls)
        tracked_error = np.clip(tracked_error, -tracked_guard[None, :], tracked_guard[None, :])
        terminal_error = np.clip(terminal_error, -terminal_guard, terminal_guard)
        cost = (
            float(np.mean(tracked_error ** 2))
            + 2.0 * float(np.mean(terminal_error ** 2))
            + 0.05 * float(np.mean(smoothness ** 2))
        )
        if not np.isfinite(cost):
            return huge_cost, states
        return cost, states

    best_controls = (
        fixed_controls.copy()
    )
    best_cost, best_states = _evaluate_controls(best_controls)
    if not np.isfinite(best_cost):
        best_cost = huge_cost

    for _ in range(max(int(n_candidates), 1)):
        controls = fixed_controls.copy()
        if reference_controls is not None and rng.random() < 0.7:
            reference_start = reference_controls[0]
            reference_end = reference_controls[-1]
            start = np.clip(
                reference_start + rng.normal(loc=0.0, scale=local_span, size=spec.control_dim),
                lower,
                upper,
            ).astype(np.float32)
            end = np.clip(
                reference_end + rng.normal(loc=0.0, scale=local_span, size=spec.control_dim),
                lower,
                upper,
            ).astype(np.float32)
        else:
            start = rng.uniform(lower, upper).astype(np.float32)
            end = rng.uniform(lower, upper).astype(np.float32)
        for idx in active_indices:
            controls[:, idx] = np.linspace(start[idx], end[idx], n_steps, dtype=np.float32)
        cost, states = _evaluate_controls(controls)
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


def _serialize_profile(profile: dict[str, Any] | None) -> dict[str, Any] | None:
    if profile is None:
        return None
    return {
        key: value
        for key, value in profile.items()
        if value is not None
    }


def _serialize_preset(preset: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": str(preset["id"]),
        "title": str(preset.get("title", preset["id"])),
        "description": str(preset.get("description", "")),
        "profile": _serialize_profile(preset.get("profile")),
    }


def _serialize_channel(channel: Any) -> dict[str, Any]:
    lower = getattr(channel, "lower_bound", None)
    upper = getattr(channel, "upper_bound", None)
    return {
        "name": str(getattr(channel, "name", "")),
        "lower_bound": None if lower is None else float(lower),
        "upper_bound": None if upper is None else float(upper),
        "unit": getattr(channel, "unit", None),
        "description": getattr(channel, "description", None),
        "role": getattr(channel, "role", None),
    }


def _named_state_dict(
    names: list[str],
    base_vector: list[float],
    overrides: dict[str, Any] | None,
) -> dict[str, float]:
    vector = _apply_named_updates(names, np.asarray(base_vector, dtype=np.float32), overrides)
    return {
        name: float(vector[idx])
        for idx, name in enumerate(names)
    }


def serialize_system_spec(spec: ProcessUnitSpec) -> dict[str, Any]:
    """Return frontend-friendly metadata for one registered system."""

    return {
        "name": spec.name,
        "state_dim": int(spec.state_dim),
        "control_dim": int(spec.control_dim),
        "disturbance_dim": int(spec.disturbance_dim),
        "param_dim": int(spec.param_dim),
        "state_names": list(spec.state_names),
        "control_names": list(spec.control_names),
        "disturbance_names": list(spec.disturbance_names),
        "default_initial_state": [float(value) for value in spec.default_initial_state],
        "default_nominal_disturbance": [
            float(value) for value in spec.default_nominal_disturbance
        ],
        "control_ranges": {
            name: [float(bounds[0]), float(bounds[1])]
            for name, bounds in spec.control_ranges.items()
        },
        "disturbance_ranges": {
            name: [float(bounds[0]), float(bounds[1])]
            for name, bounds in spec.disturbance_ranges.items()
        },
        "state_channels": [
            _serialize_channel(channel)
            for channel in getattr(spec, "state_channels", [])
        ],
        "control_channels": [
            _serialize_channel(channel)
            for channel in getattr(spec, "control_channels", [])
        ],
        "disturbance_channels": [
            _serialize_channel(channel)
            for channel in getattr(spec, "disturbance_channels", [])
        ],
    }


def serialize_demo_definition(
    demo: dict[str, Any],
    spec: ProcessUnitSpec,
) -> dict[str, Any]:
    """Return the full browser-facing definition for one demo workspace."""

    baseline_profile = demo.get("baseline_control_profile")
    disturbance_presets = demo.get("disturbance_presets") or [
        {
            "id": "nominal",
            "title": "Nominal operation",
            "description": "Default disturbance path from the system specification.",
            "profile": None,
        }
    ]
    candidate_profiles = demo.get("candidate_profiles") or [
        {
            "id": "baseline",
            "title": "Baseline policy",
            "description": "No alternate candidate profile configured.",
            "profile": baseline_profile,
        }
    ]

    return {
        "id": str(demo["id"]),
        "title": str(demo["title"]),
        "system": str(demo["system"]),
        "kind": str(demo.get("kind", "unit_demo")),
        "description": str(demo.get("description", "")),
        "operator_goal": demo.get("operator_goal"),
        "dt": float(demo.get("dt", 0.1)),
        "n_steps": int(demo.get("n_steps", 25)),
        "highlight_states": list(demo.get("highlight_states", spec.state_names[:2])),
        "target_state": _named_state_dict(
            list(spec.state_names),
            spec.default_initial_state,
            demo.get("target_state"),
        ),
        "initial_state": _named_state_dict(
            list(spec.state_names),
            spec.default_initial_state,
            demo.get("initial_state"),
        ),
        "baseline_control_profile": _serialize_profile(baseline_profile),
        "disturbance_presets": [
            _serialize_preset(preset) for preset in disturbance_presets
        ],
        "candidate_profiles": [
            _serialize_preset(profile) for profile in candidate_profiles
        ],
        "optimization": {
            "n_candidates": int(demo.get("optimization", {}).get("n_candidates", 48)),
            "seed": int(demo.get("optimization", {}).get("seed", 0)),
        },
        "run_button_label": str(demo.get("run_button_label", "Run Scenario")),
        "optimize_button_label": str(
            demo.get("optimize_button_label", "Recommend Control Sequence")
        ),
        "editable_control_names": demo.get("editable_control_names"),
        "system_spec": serialize_system_spec(spec),
    }


def demo_page_from_config(
    config: dict[str, Any],
    system_configs: dict[str, dict[str, Any]],
    *,
    config_path: str | Path | None = None,
    runtime_loaded: bool = False,
) -> dict[str, Any]:
    """Build the full browser bootstrap payload from demo config and specs."""

    demos = []
    for demo in config.get("demos", []):
        system_name = demo["system"]
        system_config = system_configs.get(system_name)
        if system_config is None:
            continue
        spec = get_system_spec(system_config)
        demos.append(serialize_demo_definition(demo, spec))

    release = load_demo_release_snapshot(config, config_path=config_path)
    release["runtime_loaded"] = bool(runtime_loaded)
    return {
        "product_name": config.get("theme", {}).get("product_name", "Digital Twin Engine"),
        "headline": config.get("theme", {}).get(
            "headline",
            "Industrial dynamics you can steer.",
        ),
        "summary": config.get("theme", {}).get("summary", ""),
        "release": release,
        "demos": demos,
        "flowsheets": flowsheet_preview_catalog(),
    }


def demo_catalog_from_config(
    config: dict[str, Any],
    system_configs: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Build a JSON-friendly demo catalog from config and loaded specs."""

    page = demo_page_from_config(config, system_configs)
    demos = [
        {
            "id": demo["id"],
            "title": demo["title"],
            "system": demo["system"],
            "kind": demo["kind"],
            "description": demo["description"],
            "controls": list(demo["system_spec"]["control_names"]),
            "disturbances": list(demo["system_spec"]["disturbance_names"]),
            "states": list(demo["system_spec"]["state_names"]),
            "dt": float(demo["dt"]),
            "n_steps": int(demo["n_steps"]),
            "highlight_states": list(demo["highlight_states"]),
        }
        for demo in page["demos"]
    ]
    return {
        "product_name": page["product_name"],
        "headline": page["headline"],
        "summary": page["summary"],
        "demos": demos,
        "flowsheets": page["flowsheets"],
    }
