"""Run a reusable Phase 4 smoke-test matrix for modular law layers."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXAMPLES = ("chemistry", "biology")


class SmokeError(RuntimeError):
    """Raised when the smoke runner cannot complete successfully."""


def _json_safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not (float("-inf") < numeric < float("inf")):
        return None
    return numeric


def _jsonify(payload: Any) -> Any:
    try:
        import numpy as np
    except Exception:  # pragma: no cover
        np = None

    if isinstance(payload, dict):
        return {str(key): _jsonify(value) for key, value in payload.items()}
    if isinstance(payload, (list, tuple)):
        return [_jsonify(value) for value in payload]
    if np is not None and isinstance(payload, np.ndarray):
        return payload.tolist()

    try:
        import jax.numpy as jnp

        if isinstance(payload, jnp.ndarray):
            return payload.tolist()
    except Exception:  # pragma: no cover
        pass

    if hasattr(payload, "tolist"):
        try:
            return payload.tolist()
        except Exception:
            pass
    return payload


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def resolve_workspace_dir(raw_workspace: str | None) -> Path:
    if raw_workspace is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return PROJECT_ROOT / "outputs" / "phase4_smoke" / stamp
    path = Path(raw_workspace)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Phase 4 smoke-test matrix")
    parser.add_argument(
        "--workspace_dir",
        type=str,
        default=None,
        help="Workspace directory for configs, logs, outputs, and summary",
    )
    parser.add_argument(
        "--examples",
        nargs="+",
        default=list(DEFAULT_EXAMPLES),
        choices=list(DEFAULT_EXAMPLES),
        help="Phase 4 examples to execute",
    )
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument(
        "--n_steps",
        type=int,
        default=6,
        help="Trajectory length used for residual-series smoke evaluation",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Batch size used for physics-loss smoke evaluation",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.1,
        help="Timestep used when evaluating mechanistic residuals",
    )
    parser.add_argument(
        "--jax_platforms",
        type=str,
        default="cpu",
        help="Value exported to JAX_PLATFORMS before importing JAX",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Write configs and planned steps without executing them",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    examples = list(dict.fromkeys(args.examples))
    workspace = resolve_workspace_dir(args.workspace_dir)
    configs_dir = workspace / "configs"
    logs_dir = workspace / "logs"
    outputs_dir = workspace / "outputs"
    summary_path = workspace / "summary.json"

    workspace.mkdir(parents=True, exist_ok=True)

    if args.jax_platforms:
        os.environ["JAX_PLATFORMS"] = args.jax_platforms

    import jax.numpy as jnp

    from dte.laws.examples import (
        build_bioreactor_law_bundle_example,
        build_bioreactor_law_example_config,
        build_cstr_law_bundle_example,
        build_cstr_law_example_config,
    )
    from dte.physics.registry import get_physics_diagnostic_fn, get_physics_loss
    from dte.training.losses import LossComputer

    def load_example_configs() -> dict[str, dict[str, Any]]:
        return {
            "chemistry": build_cstr_law_example_config(),
            "biology": build_bioreactor_law_example_config(),
        }

    configs = load_example_configs()
    for name, payload in configs.items():
        write_yaml(configs_dir / f"{name}_example.yaml", payload)

    summary: dict[str, Any] = {
        "status": "running",
        "workspace_dir": str(workspace),
        "started_at": datetime.now().isoformat(),
        "settings": {
            "examples": examples,
            "n_steps": args.n_steps,
            "batch_size": args.batch_size,
            "dt": args.dt,
            "dry_run": args.dry_run,
            "jax_platforms": args.jax_platforms,
        },
        "configs": {
            name: str(configs_dir / f"{name}_example.yaml")
            for name in configs
        },
        "steps": [],
        "artifacts": {
            "examples": {
                name: {
                    "result_path": str(outputs_dir / name / "result.json"),
                }
                for name in examples
            }
        },
        "failure": None,
    }
    write_json(summary_path, summary)

    def write_summary() -> None:
        write_json(summary_path, summary)

    def run_internal_step(
        *,
        name: str,
        log_path: Path,
        dry_run: bool,
        fn: Callable[[], dict[str, Any] | None],
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        step_summary = {
            "name": name,
            "mode": "internal",
            "log_path": str(log_path),
            "started_at": datetime.now().isoformat(),
            "returncode": None,
            "duration_seconds": None,
            "succeeded": False,
        }
        print("\n" + "=" * 60)
        print(f"STEP: {name}")
        print("=" * 60, flush=True)
        print("Log:", log_path, flush=True)

        if dry_run:
            log_path.write_text("[dry-run] step not executed\n", encoding="utf-8")
            step_summary["returncode"] = 0
            step_summary["duration_seconds"] = 0.0
            step_summary["succeeded"] = True
            return step_summary, None

        import time

        start = time.perf_counter()
        result: dict[str, Any] | None = None
        with log_path.open("w", encoding="utf-8") as log_handle:
            try:
                with contextlib.redirect_stdout(log_handle), contextlib.redirect_stderr(log_handle):
                    result = fn()
                step_summary["returncode"] = 0
                step_summary["succeeded"] = True
            except Exception as exc:  # pragma: no cover - exercised by smoke failures
                traceback.print_exc(file=log_handle)
                step_summary["returncode"] = 1
                step_summary["succeeded"] = False
                step_summary["error"] = str(exc)
        step_summary["duration_seconds"] = time.perf_counter() - start
        return step_summary, result

    def chemistry_smoke() -> dict[str, Any]:
        spec, config, bundle = build_cstr_law_bundle_example()
        physics_loss = get_physics_loss(spec.name, config)
        diagnostic_fn = get_physics_diagnostic_fn(spec.name, config)
        if diagnostic_fn is None:
            raise SmokeError("expected chemistry example to expose a diagnostic function")

        base_state = jnp.asarray([0.8, 0.2, 330.0, 300.0], dtype=jnp.float32)
        state_delta = jnp.asarray([-0.01, 0.01, 0.15, 0.05], dtype=jnp.float32)
        states = jnp.stack(
            [base_state + idx * state_delta for idx in range(args.n_steps)],
            axis=0,
        )
        control = jnp.asarray([50.0, 300.0], dtype=jnp.float32)
        disturbance = jnp.asarray([1.0, 320.0], dtype=jnp.float32)
        controls = jnp.tile(control[None, :], (args.n_steps, 1))
        disturbances = jnp.tile(disturbance[None, :], (args.n_steps, 1))
        params = jnp.asarray(
            [100.0, 8750.0, -50000.0, 50000.0, 15.0, 0.239],
            dtype=jnp.float32,
        )
        batch_states = jnp.stack(
            [states, states + jnp.asarray([0.02, -0.02, -1.0, -0.5], dtype=jnp.float32)],
            axis=0,
        )[: args.batch_size]
        batch_controls = jnp.tile(controls[None, :, :], (batch_states.shape[0], 1, 1))
        batch_disturbances = jnp.tile(disturbances[None, :, :], (batch_states.shape[0], 1, 1))

        features = bundle.feature_vector(states[0], control, disturbance, params, args.dt)
        mechanistic_delta = bundle.mechanistic_delta(states[0], control, disturbance, params, args.dt)
        residual_series = bundle.trajectory_residual_series(
            states=states,
            controls=controls,
            disturbances=disturbances,
            dt=args.dt,
            params=params,
        )
        residuals = physics_loss.compute_residuals(
            batch_states,
            batch_controls,
            batch_disturbances,
            args.dt,
        )
        diagnostic_residuals = diagnostic_fn(states, controls, disturbances, args.dt)

        lw = {
            "reconstruction": 1.0,
            "trajectory": 1.0,
            "one_step": 0.0,
            "mass": 0.1,
            "species_mass": 0.1,
            "energy": 0.1,
        }
        for name in physics_loss.residual_names():
            lw.setdefault(name, 0.1)
        loss_computer = LossComputer(
            config={"loss_weights": lw},
            normalization_stats={
                "state_mean": jnp.zeros((spec.state_dim,), dtype=jnp.float32),
                "state_std": jnp.ones((spec.state_dim,), dtype=jnp.float32),
                "control_mean": jnp.zeros((spec.control_dim,), dtype=jnp.float32),
                "control_std": jnp.ones((spec.control_dim,), dtype=jnp.float32),
                "disturbance_mean": jnp.zeros((spec.disturbance_dim,), dtype=jnp.float32),
                "disturbance_std": jnp.ones((spec.disturbance_dim,), dtype=jnp.float32),
            },
            physics_loss=physics_loss,
            state_names=spec.state_names,
        )
        total_physics_loss, loss_terms = loss_computer.physics_losses(
            batch_states,
            batch_controls,
            batch_disturbances,
            args.dt,
        )

        return {
            "example": "chemistry",
            "spec_name": spec.name,
            "module_names": [module.module_name for module in bundle.modules],
            "feature_names": list(bundle.feature_names()),
            "residual_names": list(bundle.residual_names()),
            "feature_vector": _jsonify(features),
            "mechanistic_delta": _jsonify(mechanistic_delta),
            "bundle_residual_series_mean": {
                name: _json_safe_float(jnp.mean(values))
                for name, values in residual_series.items()
            },
            "physics_residuals": {
                name: _json_safe_float(value)
                for name, value in residuals.items()
            },
            "diagnostic_residual_series_mean": {
                name: _json_safe_float(jnp.mean(values))
                for name, values in diagnostic_residuals.items()
            },
            "loss_computer_total_physics_loss": _json_safe_float(total_physics_loss),
            "loss_computer_terms": {
                name: _json_safe_float(value)
                for name, value in loss_terms.items()
            },
        }

    def biology_smoke() -> dict[str, Any]:
        spec, _, bundle = build_bioreactor_law_bundle_example()

        base_state = jnp.asarray([1.1, 0.4, 0.6], dtype=jnp.float32)
        state_delta = jnp.asarray([-0.03, 0.015, -0.005], dtype=jnp.float32)
        states = jnp.stack(
            [base_state + idx * state_delta for idx in range(args.n_steps)],
            axis=0,
        )
        control = jnp.asarray([0.6], dtype=jnp.float32)
        disturbance = jnp.asarray([1.0], dtype=jnp.float32)
        controls = jnp.tile(control[None, :], (args.n_steps, 1))
        disturbances = jnp.tile(disturbance[None, :], (args.n_steps, 1))

        features = bundle.feature_vector(states[0], control, disturbance, None, args.dt)
        mechanistic_delta = bundle.mechanistic_delta(states[0], control, disturbance, None, args.dt)
        residual_series = bundle.trajectory_residual_series(
            states=states,
            controls=controls,
            disturbances=disturbances,
            dt=args.dt,
            params=None,
        )
        batch_states = jnp.stack(
            [states, states + jnp.asarray([0.1, 0.0, 0.05], dtype=jnp.float32)],
            axis=0,
        )[: args.batch_size]
        batch_controls = jnp.tile(controls[None, :, :], (batch_states.shape[0], 1, 1))
        batch_disturbances = jnp.tile(disturbances[None, :, :], (batch_states.shape[0], 1, 1))
        residuals = bundle.compute_residuals(
            batch_states,
            batch_controls,
            batch_disturbances,
            args.dt,
            params_batch=None,
        )

        return {
            "example": "biology",
            "spec_name": spec.name,
            "module_names": [module.module_name for module in bundle.modules],
            "feature_names": list(bundle.feature_names()),
            "residual_names": list(bundle.residual_names()),
            "feature_vector": _jsonify(features),
            "mechanistic_delta": _jsonify(mechanistic_delta),
            "bundle_residual_series_mean": {
                name: _json_safe_float(jnp.mean(values))
                for name, values in residual_series.items()
            },
            "bundle_batch_residuals": {
                name: _json_safe_float(value)
                for name, value in residuals.items()
            },
        }

    example_fns: dict[str, Callable[[], dict[str, Any]]] = {
        "chemistry": chemistry_smoke,
        "biology": biology_smoke,
    }

    try:
        for idx, name in enumerate(examples):
            del idx
            step, result = run_internal_step(
                name=f"example:{name}",
                log_path=logs_dir / f"{name}.log",
                dry_run=args.dry_run,
                fn=example_fns[name],
            )
            summary["steps"].append(step)
            if result is not None:
                result_path = outputs_dir / name / "result.json"
                write_json(result_path, _jsonify(result))
                summary["artifacts"]["examples"][name]["result"] = _jsonify(result)
            write_summary()
            if not step["succeeded"]:
                raise SmokeError(
                    f"step '{step['name']}' failed with return code {step['returncode']}"
                )

        if not args.dry_run:
            example_metrics: dict[str, Any] = {}
            chemistry = summary["artifacts"]["examples"].get("chemistry", {}).get("result")
            if chemistry is not None:
                example_metrics["chemistry_total_physics_loss"] = chemistry.get(
                    "loss_computer_total_physics_loss"
                )
            biology = summary["artifacts"]["examples"].get("biology", {}).get("result")
            if biology is not None:
                example_metrics["biology_biomass_delta"] = (
                    biology.get("mechanistic_delta", [None, None])[1]
                    if isinstance(biology.get("mechanistic_delta"), list)
                    else None
                )
            summary["aggregate"] = example_metrics

    except SmokeError as exc:
        summary["status"] = "failed"
        summary["failure"] = {"message": str(exc)}
        summary["finished_at"] = datetime.now().isoformat()
        write_summary()
        print(f"\nSmoke run failed. Summary saved to {summary_path}")
        print(str(exc))
        return 1

    summary["status"] = "ok"
    summary["finished_at"] = datetime.now().isoformat()
    write_summary()
    print(f"\nPhase 4 smoke run complete. Summary saved to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
