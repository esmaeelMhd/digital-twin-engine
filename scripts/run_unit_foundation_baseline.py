"""Run the canonical unit-foundation baseline from corpus generation to control gate."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class BaselineError(RuntimeError):
    """Raised when the canonical baseline cannot complete successfully."""


def resolve_workspace_dir(raw_workspace: str | None) -> Path:
    if raw_workspace is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return PROJECT_ROOT / "outputs" / "unit_foundation_baseline" / stamp
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
        return {str(key): _json_safe(item) for key, item in value.items()}
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


def _run_command(
    *,
    name: str,
    command: list[str],
    log_path: Path,
    env_updates: dict[str, str],
    dry_run: bool,
) -> dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("\n" + "=" * 80)
    print(f"STEP: {name}")
    print("=" * 80)
    print("Command:", " ".join(command), flush=True)
    print("Log:", log_path, flush=True)

    started_at = time.time()
    if dry_run:
        log_path.write_text("[dry-run] command not executed\n", encoding="utf-8")
        return _step_summary(name, started_at, True, command=command, log_path=log_path)

    env = os.environ.copy()
    env.update(env_updates)
    env["PYTHONUNBUFFERED"] = "1"
    with log_path.open("wb") as log_handle:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
        )
        assert process.stdout is not None
        while True:
            chunk = process.stdout.read(8192)
            if not chunk:
                break
            sys.stdout.buffer.write(chunk)
            sys.stdout.buffer.flush()
            log_handle.write(chunk)
        returncode = process.wait()
    if returncode != 0:
        raise BaselineError(f"step '{name}' failed with exit code {returncode}")
    return _step_summary(name, started_at, True, command=command, log_path=log_path)


def _load_universal_sources(config: dict[str, Any]):
    from dte.data.datasets.universal_unit_dataset import SystemDatasetSource

    systems = config.get("data", {}).get("systems", [])
    if not systems:
        raise BaselineError("Universal training config must define data.systems.")
    return [
        SystemDatasetSource(
            name=str(item["name"]),
            system_config=str(item["system_config"]),
            data_dir=str(item["data_dir"]),
            weight=float(item.get("weight", 1.0)),
        )
        for item in systems
    ]


def _resolve_checkpoint(run_dir: Path) -> Path:
    for candidate in (run_dir / "best_model.eqx", run_dir / "final_model.eqx"):
        if candidate.exists():
            return candidate
    raise BaselineError(f"No checkpoint found in {run_dir}")


def _state_bounds(spec) -> tuple[np.ndarray, np.ndarray]:
    lower = np.full(spec.state_dim, -np.inf, dtype=np.float32)
    upper = np.full(spec.state_dim, np.inf, dtype=np.float32)
    for idx, channel in enumerate(getattr(spec, "state_channels", [])):
        if channel.lower_bound is not None:
            lower[idx] = float(channel.lower_bound)
        if channel.upper_bound is not None:
            upper[idx] = float(channel.upper_bound)
    return lower, upper


def _target_state(spec) -> np.ndarray:
    initial = np.asarray(spec.default_initial_state, dtype=np.float32)
    base_scale = np.asarray(
        getattr(spec.normalization, "state_scale", [1.0] * spec.state_dim),
        dtype=np.float32,
    )
    shift = np.maximum(0.05 * base_scale[: spec.state_dim], 1e-3)
    direction = np.where(np.arange(spec.state_dim) % 2 == 0, 1.0, -1.0).astype(np.float32)
    target = initial + direction * shift
    lower, upper = _state_bounds(spec)
    finite_lower = np.where(np.isfinite(lower), lower, -np.inf)
    finite_upper = np.where(np.isfinite(upper), upper, np.inf)
    return np.clip(target, finite_lower, finite_upper).astype(np.float32)


def _run_control_gate(
    *,
    training_config_path: Path,
    checkpoint_path: Path,
    output_dir: Path,
    control_systems: list[str],
    horizon: int,
    n_candidates: int,
    rollout_samples: int,
    seed: int,
) -> dict[str, Any]:
    from dte.control.mpc_interface import MPCInterfaceConfig, ProcessMPCInterface
    from dte.data.datasets.universal_unit_dataset import MultiSystemTrajectoryDataset
    from dte.evaluation.control_metrics import mismatch_robustness
    from dte.models.universal.digital_twin import UniversalDigitalTwin
    from dte.simulators.registry import get_simulator, get_system_spec

    training_config = load_yaml(training_config_path)
    sources = _load_universal_sources(training_config)
    source_by_name = {source.name: source for source in sources}
    metadata = MultiSystemTrajectoryDataset.metadata_from_sources(sources)
    model = UniversalDigitalTwin.load(str(checkpoint_path), training_config, metadata)

    output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "status": "ok",
        "model_path": str(checkpoint_path.resolve()),
        "training_config_path": str(training_config_path.resolve()),
        "systems": {},
    }
    mismatch_values: list[float] = []

    for idx, system_name in enumerate(control_systems):
        if system_name not in source_by_name:
            raise BaselineError(
                f"Control gate system '{system_name}' is not present in the universal training config."
            )

        source = source_by_name[system_name]
        system_config = load_yaml(PROJECT_ROOT / source.system_config)
        spec = get_system_spec(system_config)
        simulator = get_simulator(spec.name, system_config)
        dt = float(system_config.get("simulation", {}).get("dt", 0.1))
        runtime = ProcessMPCInterface(
            spec,
            simulator,
            model=model,
            config=MPCInterfaceConfig(
                dt=dt,
                horizon=horizon,
                rollout_samples=rollout_samples,
            ),
        )

        initial_state = np.asarray(spec.default_initial_state, dtype=np.float32)
        target_state = _target_state(spec)
        runtime.reset(initial_state=initial_state)
        best = runtime.optimize_random_shooting(
            target_state=target_state,
            horizon=horizon,
            n_candidates=n_candidates,
            use_model=True,
            seed=seed + idx,
        )
        simulator_eval = runtime.evaluate_candidate(
            np.asarray(best["controls"], dtype=np.float32),
            target_state=target_state,
            use_model=False,
            seed=seed + 100 + idx,
        )
        mismatch = mismatch_robustness(simulator_eval["states"], best["states"])
        mismatch_values.append(float(mismatch["normalized_rmse"]))

        system_summary = {
            "target_state": target_state,
            "model_objective": float(best["objective"]),
            "simulator_objective": float(simulator_eval["objective"]),
            "model_metrics": best["metrics"],
            "simulator_metrics": simulator_eval["metrics"],
            "mismatch": mismatch,
            "source": str(best["source"]),
        }
        write_json(output_dir / f"{system_name}.json", _json_safe(system_summary))
        summary["systems"][system_name] = _json_safe(system_summary)

    summary["aggregate"] = {
        "mean_normalized_mismatch_rmse": float(np.mean(mismatch_values)) if mismatch_values else None,
        "max_normalized_mismatch_rmse": float(np.max(mismatch_values)) if mismatch_values else None,
        "n_systems": len(control_systems),
    }
    write_json(output_dir / "summary.json", _json_safe(summary))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the canonical unit-foundation baseline: regime corpus -> universal training -> evaluation -> control gate."
    )
    parser.add_argument(
        "--generation_config",
        type=str,
        default="configs/generation_phase1_regime.yaml",
        help="Corpus generation manifest for the canonical unit-foundation run.",
    )
    parser.add_argument(
        "--training_config",
        type=str,
        default="configs/training_universal_phase1_regime.yaml",
        help="Universal training config for the canonical unit-foundation run.",
    )
    parser.add_argument("--workspace_dir", type=str, default=None)
    parser.add_argument("--jax_platform", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--control_systems",
        nargs="+",
        default=["cstr", "heat_exchanger", "two_tank"],
        help="Systems used for the control-readiness gate after universal evaluation.",
    )
    parser.add_argument("--control_horizon", type=int, default=12)
    parser.add_argument("--control_candidates", type=int, default=16)
    parser.add_argument("--control_rollout_samples", type=int, default=4)
    parser.add_argument("--skip_generation", action="store_true")
    parser.add_argument("--skip_training", action="store_true")
    parser.add_argument("--skip_evaluation", action="store_true")
    parser.add_argument("--skip_control", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    workspace_dir = resolve_workspace_dir(args.workspace_dir)
    logs_dir = workspace_dir / "logs"
    outputs_dir = workspace_dir / "outputs"
    run_dir = outputs_dir / "unit_foundation"
    eval_dir = run_dir / "eval"
    control_dir = run_dir / "control_gate"
    summary_path = workspace_dir / "summary.json"

    workspace_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "status": "running",
        "workspace_dir": str(workspace_dir.resolve()),
        "generation_config": str((PROJECT_ROOT / args.generation_config).resolve()),
        "training_config": str((PROJECT_ROOT / args.training_config).resolve()),
        "steps": [],
        "artifacts": {
            "run_dir": str(run_dir.resolve()),
            "eval_dir": str(eval_dir.resolve()),
            "control_dir": str(control_dir.resolve()),
        },
    }
    write_json(summary_path, summary)

    env_updates = {
        "JAX_PLATFORMS": args.jax_platform,
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
    }

    try:
        if not args.skip_generation:
            summary["steps"].append(
                _run_command(
                    name="generate_regime_corpus",
                    command=[
                        sys.executable,
                        "scripts/generate_corpus.py",
                        "--config",
                        args.generation_config,
                    ],
                    log_path=logs_dir / "generate_regime_corpus.log",
                    env_updates=env_updates,
                    dry_run=args.dry_run,
                )
            )
            write_json(summary_path, summary)

        if not args.skip_training:
            summary["steps"].append(
                _run_command(
                    name="train_unit_foundation",
                    command=[
                        sys.executable,
                        "scripts/train_universal.py",
                        "--config",
                        args.training_config,
                        "--output_dir",
                        str(run_dir),
                        "--seed",
                        str(args.seed),
                    ],
                    log_path=logs_dir / "train_unit_foundation.log",
                    env_updates=env_updates,
                    dry_run=args.dry_run,
                )
            )
            write_json(summary_path, summary)

        if not args.skip_evaluation:
            checkpoint_path = run_dir / "best_model.eqx" if args.dry_run else _resolve_checkpoint(run_dir)
            summary["steps"].append(
                _run_command(
                    name="evaluate_unit_foundation",
                    command=[
                        sys.executable,
                        "scripts/evaluate_universal.py",
                        "--model_path",
                        str(checkpoint_path),
                        "--config",
                        str(run_dir / "config.yaml"),
                        "--output_dir",
                        str(eval_dir),
                        "--seed",
                        str(args.seed),
                    ],
                    log_path=logs_dir / "evaluate_unit_foundation.log",
                    env_updates=env_updates,
                    dry_run=args.dry_run,
                )
            )
            write_json(summary_path, summary)

        if not args.skip_control:
            started_at = time.time()
            if args.dry_run:
                summary["steps"].append(
                    _step_summary(
                        "control_gate",
                        started_at,
                        True,
                        control_systems=args.control_systems,
                        control_horizon=args.control_horizon,
                        control_candidates=args.control_candidates,
                        control_rollout_samples=args.control_rollout_samples,
                    )
                )
            else:
                checkpoint_path = _resolve_checkpoint(run_dir)
                control_summary = _run_control_gate(
                    training_config_path=run_dir / "config.yaml",
                    checkpoint_path=checkpoint_path,
                    output_dir=control_dir,
                    control_systems=list(args.control_systems),
                    horizon=int(args.control_horizon),
                    n_candidates=int(args.control_candidates),
                    rollout_samples=int(args.control_rollout_samples),
                    seed=int(args.seed),
                )
                summary["artifacts"]["control_summary"] = str((control_dir / "summary.json").resolve())
                summary["steps"].append(
                    _step_summary(
                        "control_gate",
                        started_at,
                        True,
                        aggregate=control_summary.get("aggregate", {}),
                    )
                )
            write_json(summary_path, summary)

        summary["status"] = "dry_run" if args.dry_run else "ok"
        write_json(summary_path, summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        summary["status"] = "failed"
        summary["error"] = str(exc)
        write_json(summary_path, summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
