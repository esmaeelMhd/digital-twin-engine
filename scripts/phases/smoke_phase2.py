"""Run a reusable Phase 2 smoke-test matrix for adapters and calibration."""

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

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SYSTEMS = ("cstr", "heat_exchanger", "two_tank")


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


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def resolve_workspace_dir(raw_workspace: str | None) -> Path:
    if raw_workspace is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return PROJECT_ROOT / "outputs" / "phase2_smoke" / stamp
    path = Path(raw_workspace)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def resolve_system_config(system_name: str) -> Path:
    config_path = PROJECT_ROOT / "configs" / f"{system_name}_default.yaml"
    if not config_path.exists():
        raise SmokeError(f"system config not found for '{system_name}': {config_path}")
    return config_path


def build_universal_phase2_smoke_config(
    *,
    systems: list[str],
    data_root: Path,
    n_epochs: int,
    batch_size: int,
    seq_len: int,
    stride: int,
    max_batches_per_epoch: int,
) -> dict[str, Any]:
    return {
        "model": {
            "family": "universal_backbone",
            "latent_dim": 16,
            "shared_hidden_dim": 48,
            "system_embedding_dim": 12,
            "state_group_token_dim": 24,
            "state_group_kind_dim": 8,
            "state_group_encoder_layers": 2,
            "state_group_coupling_layers": 2,
            "encoder_layers": 2,
            "decoder_layers": 2,
            "drift_layers": 2,
            "use_system_spec_embedding": True,
            "use_variational_encoder": True,
            "adapters": {
                "enabled": True,
                "bottleneck_dim": 8,
                "residual_scale": 0.1,
                "encoder": True,
                "drift": True,
                "decoder": True,
            },
            "neural_cde": {"enabled": True, "hidden_dim": 24, "n_layers": 2},
        },
        "training": {
            "batch_size": batch_size,
            "seq_len": seq_len,
            "stride": stride,
            "val_split": 0.2,
            "n_epochs": n_epochs,
            "mixed_batching": "weighted",
            "max_batches_per_epoch": max_batches_per_epoch,
        },
        "loss_weights": {
            "reconstruction": 1.0,
            "trajectory": 1.0,
            "one_step": 0.5,
            "k_step": 0.25,
            "kl": 0.0001,
            "state_bounds": 0.0,
            "positivity": 0.0,
        },
        "multi_horizon": {"k_steps": [2, 3]},
        "optimizer": {
            "peak_lr": 0.0005,
            "end_lr": 0.00001,
            "warmup_steps": 1,
            "total_steps": max(10, n_epochs * max_batches_per_epoch + 5),
            "gradient_clip": 1.0,
        },
        "checkpointing": {
            "val_every": 1,
            "save_every": 1,
            "max_val_batches": 1,
        },
        "data": {
            "systems": [
                {
                    "name": name,
                    "system_config": str(resolve_system_config(name)),
                    "data_dir": str((data_root / name).resolve()),
                    "weight": 1.0,
                }
                for name in systems
            ]
        },
        "evaluation": {
            "per_system_batches": 1,
            "aggregate_method": "geometric_mean",
            "metric_name": "geometric_mean_per_system_total_loss",
            "uncertainty_samples": 4,
            "uncertainty_batches": 1,
            "sensitivity_batches": 1,
        },
    }


def build_target_variant_config(
    *,
    base_system: str,
    target_system_name: str,
) -> dict[str, Any]:
    """Build a small held-out customer variant config.

    The registry still requires the canonical `system.name`, so this keeps the
    canonical system identity while nudging simulator parameters and operating
    conditions to produce a distinct target dataset. The external dataset name
    is handled separately by the smoke script and calibration CLI.
    """

    config = load_yaml(resolve_system_config(base_system))
    config.setdefault("system", {})
    config["system"].setdefault("conditioning_tags", {})
    config["system"]["conditioning_tags"]["operating_regime"] = "customer_shifted"

    if base_system == "cstr":
        config["cstr"]["Ea_over_R"] = float(config["cstr"]["Ea_over_R"]) * 0.97
        config["cstr"]["UA"] = float(config["cstr"]["UA"]) * 0.90
        config["cstr"]["Fc"] = float(config["cstr"]["Fc"]) * 1.05
        config["initial_conditions"]["Ca"] = 0.6
        config["initial_conditions"]["Cb"] = 0.4
        config["initial_conditions"]["T"] = 345.0
        config["initial_conditions"]["Tc"] = 302.0
        config["operating_ranges"]["F_in"] = [12.0, 95.0]
        config["operating_ranges"]["Tc_in"] = [282.0, 318.0]
        config["operating_ranges"]["Ca_in"] = [0.6, 1.8]
        config["operating_ranges"]["T_in"] = [295.0, 345.0]
        config["system"]["normalization"]["state_center"] = [1.0, 1.0, 330.0, 320.0]
    elif base_system == "heat_exchanger":
        config["heat_exchanger"]["UA"] = float(config["heat_exchanger"]["UA"]) * 0.88
        config["heat_exchanger"]["V_hot"] = float(config["heat_exchanger"]["V_hot"]) * 1.10
        config["heat_exchanger"]["V_cold"] = float(config["heat_exchanger"]["V_cold"]) * 0.95
        config["initial_conditions"]["T_hot"] = 342.0
        config["initial_conditions"]["T_cold"] = 304.0
        config["operating_ranges"]["F_hot"] = [1.5, 9.0]
        config["operating_ranges"]["F_cold"] = [1.5, 9.0]
        config["system"]["normalization"]["state_center"] = [342.0, 304.0]
    elif base_system == "two_tank":
        config["two_tank"]["A1"] = float(config["two_tank"]["A1"]) * 1.08
        config["two_tank"]["A2"] = float(config["two_tank"]["A2"]) * 0.95
        config["two_tank"]["k12"] = float(config["two_tank"]["k12"]) * 0.96
        config["two_tank"]["kout"] = float(config["two_tank"]["kout"]) * 1.04
        config["initial_conditions"]["h1"] = 1.6
        config["initial_conditions"]["h2"] = 0.95
        config["operating_ranges"]["q_in"] = [0.35, 1.05]
        config["operating_ranges"]["valve"] = [0.68, 0.98]
        config["system"]["normalization"]["state_center"] = [1.4, 0.95]
    else:
        raise SmokeError(f"unsupported target base system for variant smoke: {base_system}")

    # Keep the canonical registry name in the YAML.
    del target_system_name
    return config


def run_command(
    *,
    name: str,
    command: list[str],
    log_path: Path,
    dry_run: bool,
) -> dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    step_summary = {
        "name": name,
        "command": command,
        "log_path": str(log_path),
        "started_at": datetime.now().isoformat(),
        "returncode": None,
        "duration_seconds": None,
        "succeeded": False,
    }
    print("\n" + "=" * 60)
    print(f"STEP: {name}")
    print("=" * 60)
    print("Command:", " ".join(command), flush=True)
    print("Log:", log_path, flush=True)

    if dry_run:
        log_path.write_text("[dry-run] command not executed\n", encoding="utf-8")
        step_summary["returncode"] = 0
        step_summary["duration_seconds"] = 0.0
        step_summary["succeeded"] = True
        return step_summary

    start = time.perf_counter()
    env = os.environ.copy()
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

    duration = time.perf_counter() - start
    step_summary["returncode"] = returncode
    step_summary["duration_seconds"] = duration
    step_summary["succeeded"] = returncode == 0
    return step_summary


def load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_summary(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Phase 2 smoke-test matrix")
    parser.add_argument(
        "--workspace_dir",
        type=str,
        default=None,
        help="Workspace directory for generated data, configs, logs, and outputs",
    )
    parser.add_argument(
        "--systems",
        nargs="+",
        default=list(DEFAULT_SYSTEMS),
        help="Systems used for universal pretraining smoke",
    )
    parser.add_argument(
        "--target_base_system",
        type=str,
        default="cstr",
        choices=list(DEFAULT_SYSTEMS),
        help="Canonical system used to synthesize a held-out target variant",
    )
    parser.add_argument(
        "--target_system_name",
        type=str,
        default="cstr_variant",
        help="External dataset name used for the held-out target variant",
    )
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument(
        "--n_trajectories",
        type=int,
        default=24,
        help="Trajectories to generate per pretraining system",
    )
    parser.add_argument(
        "--target_n_trajectories",
        type=int,
        default=16,
        help="Trajectories to generate for the held-out target variant",
    )
    parser.add_argument("--n_steps", type=int, default=30, help="Steps per generated trajectory")
    parser.add_argument(
        "--generation_batch_size",
        type=int,
        default=8,
        help="Dataset-generation batch size",
    )
    parser.add_argument(
        "--simulation_mode",
        type=str,
        default="dataset",
        choices=["dataset", "reference"],
        help="Generation rollout mode",
    )
    parser.add_argument(
        "--universal_n_epochs",
        type=int,
        default=1,
        help="Epochs for universal Phase 2 smoke training",
    )
    parser.add_argument(
        "--universal_batch_size",
        type=int,
        default=8,
        help="Batch size for universal Phase 2 smoke training",
    )
    parser.add_argument(
        "--universal_seq_len",
        type=int,
        default=10,
        help="Sequence length for universal Phase 2 smoke training",
    )
    parser.add_argument(
        "--universal_stride",
        type=int,
        default=5,
        help="Stride for universal Phase 2 smoke training",
    )
    parser.add_argument(
        "--universal_max_batches_per_epoch",
        type=int,
        default=2,
        help="Batch cap per epoch for universal Phase 2 smoke training",
    )
    parser.add_argument(
        "--calibration_n_epochs",
        type=int,
        default=1,
        help="Epochs for target-unit calibration",
    )
    parser.add_argument(
        "--calibration_batch_size",
        type=int,
        default=8,
        help="Batch size for target-unit calibration",
    )
    parser.add_argument(
        "--calibration_trainable_mode",
        type=str,
        default="adapters",
        choices=["adapters", "full"],
        help="Calibration parameter subset",
    )
    parser.add_argument(
        "--calibration_param_indices",
        type=str,
        default="1,3",
        help="Comma-separated physical parameter indices to calibrate",
    )
    parser.add_argument(
        "--skip_data_generation",
        action="store_true",
        help="Skip data generation and reuse existing workspace datasets",
    )
    parser.add_argument(
        "--skip_universal",
        action="store_true",
        help="Skip universal train/eval smoke",
    )
    parser.add_argument(
        "--skip_calibration",
        action="store_true",
        help="Skip target-unit calibration smoke",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Write configs and print commands without executing subprocesses",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    systems = list(dict.fromkeys(args.systems))
    if args.target_base_system not in systems and not args.skip_universal:
        raise SmokeError(
            f"--target_base_system {args.target_base_system!r} must be included in --systems "
            f"for the default held-out-variant smoke."
        )

    workspace = resolve_workspace_dir(args.workspace_dir)
    configs_dir = workspace / "configs"
    logs_dir = workspace / "logs"
    data_root = workspace / "data"
    outputs_root = workspace / "outputs"
    summary_path = workspace / "summary.json"

    workspace.mkdir(parents=True, exist_ok=True)

    universal_config_path = configs_dir / "training_universal_phase2_smoke.yaml"
    target_system_config_path = configs_dir / f"{args.target_system_name}_system.yaml"

    universal_config = build_universal_phase2_smoke_config(
        systems=systems,
        data_root=data_root,
        n_epochs=args.universal_n_epochs,
        batch_size=args.universal_batch_size,
        seq_len=args.universal_seq_len,
        stride=args.universal_stride,
        max_batches_per_epoch=args.universal_max_batches_per_epoch,
    )
    target_variant_config = build_target_variant_config(
        base_system=args.target_base_system,
        target_system_name=args.target_system_name,
    )
    write_yaml(universal_config_path, universal_config)
    write_yaml(target_system_config_path, target_variant_config)

    summary: dict[str, Any] = {
        "status": "running",
        "workspace_dir": str(workspace),
        "started_at": datetime.now().isoformat(),
        "settings": {
            "systems": systems,
            "target_base_system": args.target_base_system,
            "target_system_name": args.target_system_name,
            "n_trajectories": args.n_trajectories,
            "target_n_trajectories": args.target_n_trajectories,
            "n_steps": args.n_steps,
            "dry_run": args.dry_run,
        },
        "configs": {
            "universal": str(universal_config_path),
            "target_system": str(target_system_config_path),
        },
        "steps": [],
        "artifacts": {
            "data": {name: str(data_root / name / "train_data.h5") for name in systems},
            "target_variant": {
                "system_config_path": str(target_system_config_path),
                "data_path": str(data_root / args.target_system_name / "train_data.h5"),
            },
            "universal": {},
            "calibration": {},
        },
        "failure": None,
    }
    write_summary(summary_path, summary)

    try:
        if not args.skip_data_generation:
            for idx, system_name in enumerate(systems):
                data_dir = data_root / system_name
                data_dir.mkdir(parents=True, exist_ok=True)
                step = run_command(
                    name=f"generate_data:{system_name}",
                    command=[
                        sys.executable,
                        str(PROJECT_ROOT / "scripts" / "generate_data.py"),
                        "--config",
                        str(resolve_system_config(system_name)),
                        "--n_trajectories",
                        str(args.n_trajectories),
                        "--n_steps",
                        str(args.n_steps),
                        "--batch_size",
                        str(args.generation_batch_size),
                        "--simulation_mode",
                        args.simulation_mode,
                        "--seed",
                        str(args.seed + idx),
                        "--output_dir",
                        str(data_dir),
                    ],
                    log_path=logs_dir / f"generate_data_{system_name}.log",
                    dry_run=args.dry_run,
                )
                summary["steps"].append(step)
                write_summary(summary_path, summary)
                if not step["succeeded"]:
                    raise SmokeError(
                        f"step '{step['name']}' failed with exit code {step['returncode']}"
                    )

            target_data_dir = data_root / args.target_system_name
            target_data_dir.mkdir(parents=True, exist_ok=True)
            step = run_command(
                name=f"generate_data:{args.target_system_name}",
                command=[
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / "generate_data.py"),
                    "--config",
                    str(target_system_config_path),
                    "--n_trajectories",
                    str(args.target_n_trajectories),
                    "--n_steps",
                    str(args.n_steps),
                    "--batch_size",
                    str(args.generation_batch_size),
                    "--simulation_mode",
                    args.simulation_mode,
                    "--seed",
                    str(args.seed + len(systems)),
                    "--output_dir",
                    str(target_data_dir),
                ],
                log_path=logs_dir / f"generate_data_{args.target_system_name}.log",
                dry_run=args.dry_run,
            )
            summary["steps"].append(step)
            write_summary(summary_path, summary)
            if not step["succeeded"]:
                raise SmokeError(
                    f"step '{step['name']}' failed with exit code {step['returncode']}"
                )

        if not args.skip_universal:
            universal_output_dir = outputs_root / "universal_phase2"
            universal_train_summary = universal_output_dir / "train_summary.json"
            step = run_command(
                name="train:universal_phase2",
                command=[
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / "train_universal.py"),
                    "--config",
                    str(universal_config_path),
                    "--output_dir",
                    str(universal_output_dir),
                    "--summary_path",
                    str(universal_train_summary),
                ],
                log_path=logs_dir / "train_universal_phase2.log",
                dry_run=args.dry_run,
            )
            summary["steps"].append(step)
            write_summary(summary_path, summary)
            if not step["succeeded"]:
                raise SmokeError(f"step '{step['name']}' failed with exit code {step['returncode']}")
            summary["artifacts"]["universal"] = {
                "output_dir": str(universal_output_dir),
                "train_summary_path": str(universal_train_summary),
                "model_path": str(universal_output_dir / "best_model.eqx"),
                "eval_summary_path": str(universal_output_dir / "eval" / "summary.json"),
            }
            write_summary(summary_path, summary)

            step = run_command(
                name="evaluate:universal_phase2",
                command=[
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / "evaluate_universal.py"),
                    "--config",
                    str(universal_config_path),
                    "--model_path",
                    str(universal_output_dir / "best_model.eqx"),
                    "--output_dir",
                    str(universal_output_dir / "eval"),
                ],
                log_path=logs_dir / "evaluate_universal_phase2.log",
                dry_run=args.dry_run,
            )
            summary["steps"].append(step)
            write_summary(summary_path, summary)
            if not step["succeeded"]:
                raise SmokeError(f"step '{step['name']}' failed with exit code {step['returncode']}")
            summary["artifacts"]["universal"]["train_summary"] = load_json_if_exists(
                universal_train_summary
            )
            summary["artifacts"]["universal"]["eval_summary"] = load_json_if_exists(
                universal_output_dir / "eval" / "summary.json"
            )
            write_summary(summary_path, summary)

        if not args.skip_calibration:
            if args.skip_universal:
                raise SmokeError("calibration smoke requires the universal training step.")
            calibration_output_dir = outputs_root / "calibration_phase2"
            calibration_summary = calibration_output_dir / "summary.json"
            step = run_command(
                name="calibrate:target_variant",
                command=[
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / "calibrate_unit.py"),
                    "--model_path",
                    str(outputs_root / "universal_phase2" / "best_model.eqx"),
                    "--config",
                    str(universal_config_path),
                    "--system_config",
                    str(target_system_config_path),
                    "--data_dir",
                    str(data_root / args.target_system_name),
                    "--system_name",
                    args.target_system_name,
                    "--output_dir",
                    str(calibration_output_dir),
                    "--trainable_mode",
                    args.calibration_trainable_mode,
                    "--tune_normalization",
                    "--tune_physics_params",
                    "--param_indices",
                    args.calibration_param_indices,
                    "--n_epochs",
                    str(args.calibration_n_epochs),
                    "--batch_size",
                    str(args.calibration_batch_size),
                    "--summary_path",
                    str(calibration_summary),
                ],
                log_path=logs_dir / "calibrate_target_variant.log",
                dry_run=args.dry_run,
            )
            summary["steps"].append(step)
            write_summary(summary_path, summary)
            if not step["succeeded"]:
                raise SmokeError(f"step '{step['name']}' failed with exit code {step['returncode']}")
            summary["artifacts"]["calibration"] = {
                "output_dir": str(calibration_output_dir),
                "summary_path": str(calibration_summary),
                "model_path": str(calibration_output_dir / "best_model.eqx"),
                "summary": load_json_if_exists(calibration_summary),
            }
            write_summary(summary_path, summary)

            calibration_summary_data = summary["artifacts"]["calibration"].get("summary") or {}
            trainable_count = calibration_summary_data.get("trainable_parameter_count")
            total_count = (calibration_summary_data.get("parameter_counts") or {}).get("total")
            if (
                trainable_count is not None
                and total_count is not None
                and int(trainable_count) >= int(total_count)
            ):
                raise SmokeError(
                    "adapter calibration did not reduce trainable parameter count "
                    f"({trainable_count} >= {total_count})"
                )

    except SmokeError as exc:
        summary["status"] = "failed"
        summary["failure"] = {"message": str(exc)}
        summary["finished_at"] = datetime.now().isoformat()
        write_summary(summary_path, summary)
        print(f"\nSmoke run failed. Summary saved to {summary_path}")
        print(str(exc))
        return 1

    summary["status"] = "ok"
    summary["finished_at"] = datetime.now().isoformat()
    summary["universal_best_val_loss"] = _json_safe_float(
        (summary["artifacts"].get("universal") or {}).get("train_summary", {}).get("best_val_loss")
        if isinstance((summary["artifacts"].get("universal") or {}).get("train_summary"), dict)
        else None
    )
    summary["universal_eval_metric"] = _json_safe_float(
        (summary["artifacts"].get("universal") or {}).get("eval_summary", {}).get("aggregate_metric_value")
        if isinstance((summary["artifacts"].get("universal") or {}).get("eval_summary"), dict)
        else None
    )
    calibration_summary_data = (summary["artifacts"].get("calibration") or {}).get("summary")
    summary["calibration_best_val_loss"] = _json_safe_float(
        calibration_summary_data.get("best_val_loss")
        if isinstance(calibration_summary_data, dict)
        else None
    )
    summary["calibration_trainable_parameter_count"] = (
        int(calibration_summary_data.get("trainable_parameter_count"))
        if isinstance(calibration_summary_data, dict)
        and calibration_summary_data.get("trainable_parameter_count") is not None
        else None
    )
    summary["calibration_total_parameter_count"] = (
        int((calibration_summary_data.get("parameter_counts") or {}).get("total"))
        if isinstance(calibration_summary_data, dict)
        and (calibration_summary_data.get("parameter_counts") or {}).get("total") is not None
        else None
    )
    write_summary(summary_path, summary)

    print("\n" + "=" * 60)
    print("PHASE 2 SMOKE COMPLETE")
    print("=" * 60)
    print(f"Workspace: {workspace}")
    print(f"Summary: {summary_path}")
    if summary["universal_best_val_loss"] is not None:
        print(f"Universal best val loss: {summary['universal_best_val_loss']:.4f}")
    if summary["universal_eval_metric"] is not None:
        print(f"Universal eval metric: {summary['universal_eval_metric']:.4f}")
    if summary["calibration_best_val_loss"] is not None:
        print(f"Calibration best val loss: {summary['calibration_best_val_loss']:.4f}")
    if summary["calibration_trainable_parameter_count"] is not None:
        print(
            "Calibration trainable params: "
            f"{summary['calibration_trainable_parameter_count']:,}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
