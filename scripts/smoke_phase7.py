"""Run a reusable Phase 7 smoke-test matrix for control interfaces."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class SmokeError(RuntimeError):
    """Raised when the smoke runner cannot complete successfully."""


def resolve_workspace_dir(raw_workspace: str | None) -> Path:
    if raw_workspace is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return PROJECT_ROOT / "outputs" / "phase7_smoke" / stamp
    path = Path(raw_workspace)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    return value


def _step_summary(name: str, started_at: float, succeeded: bool, **extra: Any) -> dict[str, Any]:
    return {
        "name": name,
        "started_at": datetime.fromtimestamp(started_at).isoformat(),
        "duration_seconds": time.time() - started_at,
        "succeeded": bool(succeeded),
        **{key: _json_safe(value) for key, value in extra.items()},
    }


def _ensure_finite(name: str, value: np.ndarray) -> None:
    arr = np.asarray(value)
    if not np.all(np.isfinite(arr)):
        raise SmokeError(f"{name} contains non-finite values.")


def _load_system(system_name: str):
    from dte.simulators.registry import get_simulator, get_system_spec

    config_path = PROJECT_ROOT / "configs" / f"{system_name}_default.yaml"
    system_config = load_yaml(config_path)
    spec = get_system_spec(system_config)
    simulator = get_simulator(system_name, system_config)
    return spec, simulator, system_config


def _tiny_model_config() -> dict[str, Any]:
    return {
        "model": {
            "latent_dim": 8,
            "hidden_dim": 16,
            "n_layers": 1,
            "drift_layers": 1,
            "diffusion_layers": 1,
            "diffusion_hidden_dim": 8,
            "initial_diffusion_scale": 0.05,
            "simulator_prior": {"enabled": False},
            "learned_solver": {"enabled": False},
            "self_correcting_policy": {"enabled": False},
            "neural_cde": {"enabled": False},
            "grouped_encoder": {"enabled": False},
        }
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Phase 7 control smoke matrix.")
    parser.add_argument("--workspace_dir", type=str, default=None)
    parser.add_argument("--jax_platform", type=str, default="cpu")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--control_steps", type=int, default=5)
    parser.add_argument("--mpc_horizon", type=int, default=10)
    parser.add_argument("--mpc_candidates", type=int, default=12)
    args = parser.parse_args()

    os.environ.setdefault("JAX_PLATFORMS", args.jax_platform)
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    workspace_dir = resolve_workspace_dir(args.workspace_dir)
    outputs_dir = workspace_dir / "outputs"
    workspace_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "status": "pending",
        "workspace_dir": str(workspace_dir),
        "jax_platform": args.jax_platform,
        "seed": int(args.seed),
        "steps": [],
        "artifacts": {},
    }

    if args.dry_run:
        summary["status"] = "dry_run"
        summary["steps"] = [
            {"name": "measurement_correction", "system": "cstr"},
            {"name": "model_rollout", "system": "cstr"},
            {"name": "simulator_mpc_loop", "system": "two_tank"},
            {"name": "rl_env_rollout", "system": "two_tank"},
            {"name": "control_metrics", "system": "heat_exchanger"},
        ]
        write_json(workspace_dir / "summary.json", summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    import jax

    from dte.control.mpc_interface import MPCInterfaceConfig, ProcessMPCInterface
    from dte.control.rl_env import ProcessControlEnv, ProcessControlEnvConfig
    from dte.control.state_correction import (
        StateCorrectionConfig,
        StateCorrectionHook,
    )
    from dte.evaluation.control_metrics import (
        closed_loop_metrics,
        disturbance_sensitivity,
        mismatch_robustness,
    )
    from dte.models.digital_twin import DigitalTwin

    try:
        # Step 1: measurement correction with a tiny model.
        started_at = time.time()
        cstr_spec, cstr_simulator, _ = _load_system("cstr")
        tiny_model = DigitalTwin.from_config(
            _tiny_model_config(),
            jax.random.PRNGKey(args.seed),
            system_spec=cstr_spec,
        )
        correction_hook = StateCorrectionHook(
            cstr_spec,
            model=tiny_model,
            config=StateCorrectionConfig(
                assimilation_gain=0.5,
                filter_alpha=1.0,
                clip_to_state_bounds=True,
                default_dt=0.05,
            ),
        )
        nominal_control = np.asarray(
            [0.5 * sum(cstr_spec.control_ranges[name]) for name in cstr_spec.control_names],
            dtype=np.float32,
        )
        prior_state = np.asarray(cstr_spec.default_initial_state, dtype=np.float32)
        measured_state = prior_state + np.asarray([-0.08, 0.07, -3.0, 1.5], dtype=np.float32)
        correction = correction_hook.correct(
            prior_state=prior_state,
            measurement=measured_state,
            control=nominal_control,
            params=np.ones(cstr_spec.param_dim, dtype=np.float32),
            timestamp=0.05,
            seed=args.seed + 1,
        )
        predicted_state = correction_hook.predict(
            control=nominal_control,
            disturbance=np.asarray(cstr_spec.default_nominal_disturbance, dtype=np.float32),
            params=np.ones(cstr_spec.param_dim, dtype=np.float32),
            dt=0.05,
        )
        _ensure_finite("correction.corrected_state", correction.corrected_state)
        _ensure_finite("correction.latent_mean", correction.latent_mean)
        _ensure_finite("correction.predicted_state", predicted_state)
        correction_payload = {
            "prior_state": prior_state.tolist(),
            "measured_state": measured_state.tolist(),
            "corrected_state": correction.corrected_state.tolist(),
            "innovation_norm": float(np.linalg.norm(correction.innovation)),
            "latent_dim": int(np.asarray(correction.latent_mean).shape[0]),
            "predicted_state": np.asarray(predicted_state).tolist(),
        }
        correction_path = outputs_dir / "measurement_correction.json"
        write_json(correction_path, correction_payload)
        summary["steps"].append(
            _step_summary(
                "measurement_correction",
                started_at,
                succeeded=True,
                output_path=str(correction_path),
                innovation_norm=correction_payload["innovation_norm"],
                latent_dim=correction_payload["latent_dim"],
            )
        )
        summary["artifacts"]["measurement_correction"] = str(correction_path)

        # Step 2: model-backed rollout path.
        started_at = time.time()
        cstr_runtime = ProcessMPCInterface(
            cstr_spec,
            cstr_simulator,
            model=tiny_model,
            config=MPCInterfaceConfig(dt=0.05, horizon=6, constraint_penalty=1.0, rollout_samples=3),
        )
        cstr_controls = np.tile(nominal_control[None, :], (6, 1))
        cstr_disturbances = np.tile(
            np.asarray(cstr_spec.default_nominal_disturbance, dtype=np.float32)[None, :],
            (6, 1),
        )
        model_rollout = cstr_runtime.rollout_candidate(
            cstr_controls,
            disturbances=cstr_disturbances,
            use_model=True,
            n_samples=3,
            seed=args.seed + 2,
        )
        _ensure_finite("model_rollout.states", np.asarray(model_rollout["states"]))
        model_rollout_payload = {
            "source": str(model_rollout["source"]),
            "states": np.asarray(model_rollout["states"]).tolist(),
            "std": np.asarray(model_rollout["std"]).tolist(),
        }
        model_rollout_path = outputs_dir / "model_rollout_cstr.json"
        write_json(model_rollout_path, model_rollout_payload)
        summary["steps"].append(
            _step_summary(
                "model_rollout",
                started_at,
                succeeded=True,
                output_path=str(model_rollout_path),
                source=model_rollout_payload["source"],
            )
        )
        summary["artifacts"]["model_rollout"] = str(model_rollout_path)

        # Step 3: run a short simulator-backed MPC loop with state assimilation.
        started_at = time.time()
        tank_spec, tank_simulator, _ = _load_system("two_tank")
        tank_hook = StateCorrectionHook(
            tank_spec,
            model=None,
            config=StateCorrectionConfig(
                assimilation_gain=0.8,
                filter_alpha=0.6,
                clip_to_state_bounds=True,
                default_dt=0.1,
            ),
        )
        tank_runtime = ProcessMPCInterface(
            tank_spec,
            tank_simulator,
            model=None,
            config=MPCInterfaceConfig(
                dt=0.1,
                horizon=args.mpc_horizon,
                constraint_penalty=2.0,
                rollout_samples=1,
            ),
            state_correction=tank_hook,
        )
        current_state = tank_runtime.reset()
        target_state = current_state + np.asarray([0.2, 0.1], dtype=np.float32)
        nominal_disturbance = np.asarray(tank_spec.default_nominal_disturbance, dtype=np.float32)
        closed_loop_states = []
        applied_controls = []
        objectives = []
        for step in range(args.control_steps):
            disturbance_forecast = np.tile(nominal_disturbance[None, :], (args.mpc_horizon, 1))
            best = tank_runtime.optimize_random_shooting(
                target_state=target_state,
                disturbances=disturbance_forecast,
                horizon=args.mpc_horizon,
                n_candidates=args.mpc_candidates,
                seed=args.seed + 20 + step,
            )
            action = np.asarray(best["controls"])[0]
            derivative = np.asarray(
                tank_simulator.dynamics(
                    step * 0.1,
                    current_state,
                    action,
                    nominal_disturbance,
                ),
                dtype=np.float32,
            )
            next_state = current_state + 0.1 * derivative
            update = tank_runtime.assimilate_measurement(
                next_state,
                control=action,
                timestamp=(step + 1) * 0.1,
            )
            current_state = np.asarray(update["corrected_state"], dtype=np.float32)
            _ensure_finite("simulator_mpc.current_state", current_state)
            closed_loop_states.append(current_state.copy())
            applied_controls.append(action.copy())
            objectives.append(float(best["objective"]))

        closed_loop_states_arr = np.asarray(closed_loop_states, dtype=np.float32)
        applied_controls_arr = np.asarray(applied_controls, dtype=np.float32)
        loop_metrics = closed_loop_metrics(
            tank_spec,
            closed_loop_states_arr,
            applied_controls_arr,
            target_state,
            previous_control=np.asarray(
                [0.5 * sum(tank_spec.control_ranges[name]) for name in tank_spec.control_names],
                dtype=np.float32,
            ),
            constraint_penalty=2.0,
        )
        mpc_loop_payload = {
            "target_state": target_state.tolist(),
            "states": closed_loop_states_arr.tolist(),
            "controls": applied_controls_arr.tolist(),
            "objectives": objectives,
            "metrics": loop_metrics,
        }
        mpc_loop_path = outputs_dir / "simulator_mpc_loop_two_tank.json"
        write_json(mpc_loop_path, mpc_loop_payload)
        summary["steps"].append(
            _step_summary(
                "simulator_mpc_loop",
                started_at,
                succeeded=True,
                output_path=str(mpc_loop_path),
                total_cost=float(loop_metrics["total_cost"]),
                final_objective=float(objectives[-1]),
            )
        )
        summary["artifacts"]["simulator_mpc_loop"] = str(mpc_loop_path)

        # Step 4: RL-style rollout.
        started_at = time.time()
        env = ProcessControlEnv(
            tank_spec,
            tank_simulator,
            target_state=target_state,
            config=ProcessControlEnvConfig(horizon=8, dt=0.1, terminate_on_violation=False),
        )
        midpoint_action = np.asarray(
            [0.5 * sum(tank_spec.control_ranges[name]) for name in tank_spec.control_names],
            dtype=np.float32,
        )
        env_rollout = env.rollout(
            lambda observation, info: midpoint_action,
            seed=args.seed + 30,
            initial_state=np.asarray(tank_spec.default_initial_state, dtype=np.float32),
        )
        _ensure_finite("env_rollout.observations", env_rollout["observations"])
        _ensure_finite("env_rollout.rewards", env_rollout["rewards"])
        rl_payload = {
            "observations": env_rollout["observations"].tolist(),
            "actions": env_rollout["actions"].tolist(),
            "rewards": env_rollout["rewards"].tolist(),
            "total_reward": float(np.sum(env_rollout["rewards"])),
            "n_steps": int(env_rollout["actions"].shape[0]),
        }
        rl_path = outputs_dir / "rl_env_rollout_two_tank.json"
        write_json(rl_path, rl_payload)
        summary["steps"].append(
            _step_summary(
                "rl_env_rollout",
                started_at,
                succeeded=True,
                output_path=str(rl_path),
                total_reward=rl_payload["total_reward"],
                n_steps=rl_payload["n_steps"],
            )
        )
        summary["artifacts"]["rl_env_rollout"] = str(rl_path)

        # Step 5: control metrics under disturbance mismatch.
        started_at = time.time()
        hx_spec, hx_simulator, _ = _load_system("heat_exchanger")
        hx_runtime = ProcessMPCInterface(
            hx_spec,
            hx_simulator,
            model=None,
            config=MPCInterfaceConfig(dt=0.1, horizon=8, constraint_penalty=1.0),
        )
        hx_controls = np.tile(
            np.asarray(
                [0.5 * sum(hx_spec.control_ranges[name]) for name in hx_spec.control_names],
                dtype=np.float32,
            )[None, :],
            (8, 1),
        )
        hx_nominal_disturbances = np.tile(
            np.asarray(hx_spec.default_nominal_disturbance, dtype=np.float32)[None, :],
            (8, 1),
        )
        hx_shifted_disturbances = hx_nominal_disturbances.copy()
        hx_shifted_disturbances[:, 0] += 8.0
        nominal_rollout = hx_runtime.rollout_candidate(hx_controls, disturbances=hx_nominal_disturbances)
        shifted_rollout = hx_runtime.rollout_candidate(hx_controls, disturbances=hx_shifted_disturbances)
        sensitivity = disturbance_sensitivity(
            nominal_rollout["states"],
            shifted_rollout["states"],
        )
        robustness = mismatch_robustness(
            nominal_rollout["states"],
            shifted_rollout["states"],
        )
        metrics_payload = {
            "sensitivity": sensitivity,
            "robustness": robustness,
        }
        metrics_path = outputs_dir / "control_metrics_heat_exchanger.json"
        write_json(metrics_path, metrics_payload)
        summary["steps"].append(
            _step_summary(
                "control_metrics",
                started_at,
                succeeded=True,
                output_path=str(metrics_path),
                sensitivity=sensitivity,
                robustness=robustness,
            )
        )
        summary["artifacts"]["control_metrics"] = str(metrics_path)

        summary["status"] = "ok"
        summary["mpc_total_cost"] = float(loop_metrics["total_cost"])
        summary["rl_total_reward"] = float(rl_payload["total_reward"])
        summary["sensitivity_mean_abs_state_delta"] = float(sensitivity["mean_abs_state_delta"])
        summary["robustness_normalized_rmse"] = float(robustness["normalized_rmse"])
        write_json(workspace_dir / "summary.json", summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        summary["status"] = "failed"
        summary["error"] = str(exc)
        write_json(workspace_dir / "summary.json", summary)
        print(json.dumps(summary, indent=2, sort_keys=True), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
