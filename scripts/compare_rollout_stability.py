"""Compare rollout stability between the current model and an ablated baseline."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SYSTEM_TRAINING_CONFIGS = {
    "cstr": PROJECT_ROOT / "configs" / "training_default.yaml",
    "heat_exchanger": PROJECT_ROOT / "configs" / "heat_exchanger_training.yaml",
    "two_tank": PROJECT_ROOT / "configs" / "two_tank_training.yaml",
}


class ComparisonError(RuntimeError):
    """Raised when the rollout comparison cannot complete."""


def _json_safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_workspace_dir(raw_workspace: str | None) -> Path:
    if raw_workspace is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return PROJECT_ROOT / "outputs" / "rollout_stability" / stamp
    path = Path(raw_workspace)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def resolve_training_config(system_name: str) -> Path:
    try:
        path = SYSTEM_TRAINING_CONFIGS[system_name]
    except KeyError as exc:
        raise ComparisonError(
            f"no default training config is registered for system '{system_name}'"
        ) from exc
    if not path.exists():
        raise ComparisonError(f"training config not found: {path}")
    return path


def resolve_system_config(system_name: str) -> Path:
    path = PROJECT_ROOT / "configs" / f"{system_name}_default.yaml"
    if not path.exists():
        raise ComparisonError(f"system config not found: {path}")
    return path


def resolve_data_dir(system_name: str) -> Path:
    path = PROJECT_ROOT / "data" / system_name
    if not (path / "train_data.h5").exists():
        raise ComparisonError(f"dataset not found for system '{system_name}': {path}")
    return path


def build_training_config(
    *,
    system_name: str,
    variant: str,
    base_config_path: Path,
    n_epochs: int,
    max_batches_per_epoch: int,
    max_val_batches: int,
    disable_neural_cde_in_old: bool,
) -> dict[str, Any]:
    config = deepcopy(load_yaml(base_config_path))
    model_cfg = config.setdefault("model", {})
    training_cfg = config.setdefault("training", {})
    optimizer_cfg = config.setdefault("optimizer", {})
    checkpointing_cfg = config.setdefault("checkpointing", {})

    training_cfg["n_epochs"] = int(n_epochs)
    training_cfg["max_batches_per_epoch"] = int(max_batches_per_epoch)

    checkpointing_cfg["val_every"] = 1
    checkpointing_cfg["save_every"] = int(n_epochs)
    checkpointing_cfg["max_val_batches"] = int(max_val_batches)
    checkpointing_cfg["save_best"] = True

    warmup_steps = min(int(optimizer_cfg.get("warmup_steps", 200)), 10)
    optimizer_cfg["warmup_steps"] = warmup_steps
    optimizer_cfg["total_steps"] = max(
        warmup_steps + 1,
        int(n_epochs) * int(max_batches_per_epoch) + 10,
    )

    if variant == "old_style":
        model_cfg.setdefault("simulator_prior", {})["enabled"] = False
        model_cfg.setdefault("learned_solver", {})["enabled"] = False
        model_cfg.setdefault("self_correcting_policy", {})["enabled"] = False
        if disable_neural_cde_in_old:
            model_cfg.setdefault("neural_cde", {})["enabled"] = False
    elif variant == "new_architecture":
        pass
    else:
        raise ComparisonError(f"unknown variant '{variant}' for {system_name}")

    return config


def run_command(
    *,
    name: str,
    command: list[str],
    log_path: Path,
    dry_run: bool,
    env_updates: dict[str, str],
) -> dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    step = {
        "name": name,
        "command": command,
        "log_path": str(log_path),
        "started_at": datetime.now().isoformat(),
        "returncode": None,
        "duration_seconds": None,
        "succeeded": False,
    }

    print("\n" + "=" * 72)
    print(f"STEP: {name}")
    print("=" * 72)
    print("Command:", " ".join(command), flush=True)
    print("Log:", log_path, flush=True)

    if dry_run:
        log_path.write_text("[dry-run] command not executed\n", encoding="utf-8")
        step["returncode"] = 0
        step["duration_seconds"] = 0.0
        step["succeeded"] = True
        return step

    env = os.environ.copy()
    env.update(env_updates)
    env["PYTHONUNBUFFERED"] = "1"

    start = time.perf_counter()
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

    step["returncode"] = returncode
    step["duration_seconds"] = time.perf_counter() - start
    step["succeeded"] = returncode == 0
    return step


def extract_variant_metrics(
    *,
    training_summary_path: Path,
    evaluation_summary_path: Path,
) -> dict[str, Any]:
    training_summary = load_json(training_summary_path)
    evaluation_summary = load_json(evaluation_summary_path)
    model_metrics = evaluation_summary["model_metrics"]
    return {
        "training_summary_path": str(training_summary_path.resolve()),
        "evaluation_summary_path": str(evaluation_summary_path.resolve()),
        "best_val_loss": _json_safe_float(training_summary.get("best_val_loss")),
        "non_finite_detected": bool(training_summary.get("non_finite_detected", False)),
        "timed_out": bool(training_summary.get("timed_out", False)),
        "mse_1step": _json_safe_float(model_metrics["mse_1step"]["mean"]),
        "mse_10step": _json_safe_float(model_metrics["mse_10step"]["mean"]),
        "mse_laststep": _json_safe_float(model_metrics["mse_laststep"]["mean"]),
        "mse_fullseq": _json_safe_float(model_metrics["mse_fullseq"]["mean"]),
        "mass_violation_max": _json_safe_float(model_metrics["mass_violation_max"]["mean"]),
        "energy_violation_max": _json_safe_float(model_metrics["energy_violation_max"]["mean"]),
    }


def _ratio(new_value: float | None, old_value: float | None) -> float | None:
    if new_value is None or old_value is None or old_value == 0.0:
        return None
    return float(new_value / old_value)


def _improvement_percent(new_value: float | None, old_value: float | None) -> float | None:
    ratio = _ratio(new_value, old_value)
    if ratio is None:
        return None
    return float((1.0 - ratio) * 100.0)


def compare_system_metrics(
    *,
    system_name: str,
    old_metrics: dict[str, Any],
    new_metrics: dict[str, Any],
    min_improvement_ratio: float,
) -> dict[str, Any]:
    comparison = {
        "system": system_name,
        "old_style": old_metrics,
        "new_architecture": new_metrics,
        "ratios": {},
        "improvement_percent": {},
        "pass": False,
        "reason": None,
    }

    for metric_name in ("mse_fullseq", "mse_laststep", "mse_10step", "mse_1step"):
        comparison["ratios"][metric_name] = _ratio(
            new_metrics.get(metric_name), old_metrics.get(metric_name)
        )
        comparison["improvement_percent"][metric_name] = _improvement_percent(
            new_metrics.get(metric_name), old_metrics.get(metric_name)
        )

    if old_metrics["non_finite_detected"] or new_metrics["non_finite_detected"]:
        comparison["reason"] = "non-finite loss detected"
        return comparison

    fullseq_ratio = comparison["ratios"]["mse_fullseq"]
    laststep_ratio = comparison["ratios"]["mse_laststep"]
    tenstep_ratio = comparison["ratios"]["mse_10step"]

    if fullseq_ratio is None or laststep_ratio is None or tenstep_ratio is None:
        comparison["reason"] = "missing rollout metrics"
        return comparison

    if (
        fullseq_ratio <= min_improvement_ratio
        and laststep_ratio <= min_improvement_ratio
        and tenstep_ratio <= min_improvement_ratio
    ):
        comparison["pass"] = True
        comparison["reason"] = "rollout metrics improved versus ablated baseline"
        return comparison

    comparison["reason"] = (
        "insufficient rollout improvement: "
        f"fullseq={fullseq_ratio:.3f}, "
        f"laststep={laststep_ratio:.3f}, "
        f"tenstep={tenstep_ratio:.3f}"
    )
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare rollout stability against an ablated old-style baseline"
    )
    parser.add_argument(
        "--workspace_dir",
        type=str,
        default=None,
        help="Workspace directory for configs, logs, outputs, and summary.json",
    )
    parser.add_argument(
        "--systems",
        nargs="+",
        default=["heat_exchanger"],
        help=(
            "Systems to compare. Defaults to the representative heat-exchanger "
            "milestone proof; add systems explicitly for broader ablations."
        ),
    )
    parser.add_argument(
        "--n_epochs",
        type=int,
        default=4,
        help="Epochs for each training run",
    )
    parser.add_argument(
        "--max_batches_per_epoch",
        type=int,
        default=24,
        help="Maximum train batches per epoch",
    )
    parser.add_argument(
        "--max_val_batches",
        type=int,
        default=4,
        help="Maximum validation batches during training",
    )
    parser.add_argument(
        "--n_trajectories",
        type=int,
        default=24,
        help="Validation subsequences to evaluate",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for both variants",
    )
    parser.add_argument(
        "--jax_platform",
        type=str,
        default="cpu",
        help="JAX platform for subprocesses",
    )
    parser.add_argument(
        "--min_improvement_ratio",
        type=float,
        default=0.9,
        help="Required new/old ratio threshold for fullseq, laststep, and tenstep MSE",
    )
    parser.add_argument(
        "--disable_neural_cde_in_old",
        action="store_true",
        help="Also disable the neural CDE block in the ablated baseline",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Write configs and commands without executing them",
    )
    args = parser.parse_args()

    workspace_dir = resolve_workspace_dir(args.workspace_dir)
    configs_dir = workspace_dir / "configs"
    logs_dir = workspace_dir / "logs"
    outputs_dir = workspace_dir / "outputs"
    summary_path = workspace_dir / "summary.json"

    workspace_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "status": "running",
        "workspace_dir": str(workspace_dir),
        "systems": args.systems,
        "n_epochs": args.n_epochs,
        "max_batches_per_epoch": args.max_batches_per_epoch,
        "max_val_batches": args.max_val_batches,
        "n_trajectories": args.n_trajectories,
        "seed": args.seed,
        "jax_platform": args.jax_platform,
        "min_improvement_ratio": args.min_improvement_ratio,
        "disable_neural_cde_in_old": bool(args.disable_neural_cde_in_old),
        "steps": [],
        "comparisons": [],
        "overall_pass": False,
    }
    write_json(summary_path, summary)

    env_updates = {"JAX_PLATFORMS": args.jax_platform}

    try:
        for system_name in args.systems:
            print("\n" + "#" * 72)
            print(f"SYSTEM: {system_name}")
            print("#" * 72)

            base_training_config = resolve_training_config(system_name)
            system_config = resolve_system_config(system_name)
            data_dir = resolve_data_dir(system_name)

            for variant in ("old_style", "new_architecture"):
                training_config = build_training_config(
                    system_name=system_name,
                    variant=variant,
                    base_config_path=base_training_config,
                    n_epochs=args.n_epochs,
                    max_batches_per_epoch=args.max_batches_per_epoch,
                    max_val_batches=args.max_val_batches,
                    disable_neural_cde_in_old=args.disable_neural_cde_in_old,
                )
                config_path = configs_dir / f"{system_name}_{variant}.yaml"
                output_dir = outputs_dir / system_name / variant
                write_yaml(config_path, training_config)

                train_step = run_command(
                    name=f"train_{system_name}_{variant}",
                    command=[
                        sys.executable,
                        str(PROJECT_ROOT / "scripts" / "train.py"),
                        "--config",
                        str(config_path),
                        "--system_config",
                        str(system_config),
                        "--data_dir",
                        str(data_dir),
                        "--output_dir",
                        str(output_dir),
                        "--seed",
                        str(args.seed),
                    ],
                    log_path=logs_dir / f"train_{system_name}_{variant}.log",
                    dry_run=args.dry_run,
                    env_updates=env_updates,
                )
                summary["steps"].append(train_step)
                write_json(summary_path, summary)
                if not train_step["succeeded"]:
                    raise ComparisonError(
                        f"training failed for {system_name}/{variant}; see {train_step['log_path']}"
                    )

                eval_step = run_command(
                    name=f"evaluate_{system_name}_{variant}",
                    command=[
                        sys.executable,
                        str(PROJECT_ROOT / "scripts" / "evaluate.py"),
                        "--model_path",
                        str(output_dir / "best_model.eqx"),
                        "--config",
                        str(output_dir / "config.yaml"),
                        "--system_config",
                        str(output_dir / "system_config.yaml"),
                        "--data_dir",
                        str(data_dir),
                        "--output_dir",
                        str(output_dir / "eval"),
                        "--predict_mode",
                        "deterministic",
                        "--n_trajectories",
                        str(args.n_trajectories),
                        "--plot_count",
                        "1",
                        "--skip_ensemble",
                    ],
                    log_path=logs_dir / f"evaluate_{system_name}_{variant}.log",
                    dry_run=args.dry_run,
                    env_updates=env_updates,
                )
                summary["steps"].append(eval_step)
                write_json(summary_path, summary)
                if not eval_step["succeeded"]:
                    raise ComparisonError(
                        f"evaluation failed for {system_name}/{variant}; see {eval_step['log_path']}"
                    )

            if args.dry_run:
                continue

            old_metrics = extract_variant_metrics(
                training_summary_path=outputs_dir / system_name / "old_style" / "training_summary.json",
                evaluation_summary_path=outputs_dir / system_name / "old_style" / "eval" / "evaluation_summary.json",
            )
            new_metrics = extract_variant_metrics(
                training_summary_path=outputs_dir / system_name / "new_architecture" / "training_summary.json",
                evaluation_summary_path=outputs_dir / system_name / "new_architecture" / "eval" / "evaluation_summary.json",
            )
            comparison = compare_system_metrics(
                system_name=system_name,
                old_metrics=old_metrics,
                new_metrics=new_metrics,
                min_improvement_ratio=args.min_improvement_ratio,
            )
            summary["comparisons"].append(comparison)
            write_json(summary_path, summary)

        if args.dry_run:
            summary["status"] = "dry_run"
            summary["overall_pass"] = True
        else:
            summary["overall_pass"] = bool(summary["comparisons"]) and all(
                bool(item.get("pass")) for item in summary["comparisons"]
            )
            summary["status"] = "ok" if summary["overall_pass"] else "failed"

        write_json(summary_path, summary)
    except Exception as exc:
        summary["status"] = "failed"
        summary["error"] = str(exc)
        write_json(summary_path, summary)
        raise

    print("\n" + "=" * 72)
    print("ROLLOUT STABILITY COMPARISON COMPLETE")
    print("=" * 72)
    print(f"Summary: {summary_path}")
    if args.dry_run:
        print("Status: dry_run")
    else:
        print(f"Overall pass: {summary['overall_pass']}")
        for item in summary["comparisons"]:
            ratio = item["ratios"]["mse_fullseq"]
            ratio_text = f"{ratio:.3f}" if ratio is not None else "n/a"
            print(
                f"  {item['system']}: pass={item['pass']} "
                f"fullseq_ratio={ratio_text} reason={item['reason']}"
            )
    print("=" * 72)


if __name__ == "__main__":
    main()
