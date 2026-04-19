"""Run a reusable Phase 1 smoke-test matrix for data, train, and eval paths."""

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
        return PROJECT_ROOT / "outputs" / "phase1_smoke" / stamp
    path = Path(raw_workspace)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def resolve_system_config(system_name: str) -> Path:
    config_path = PROJECT_ROOT / "configs" / f"{system_name}_default.yaml"
    if not config_path.exists():
        raise SmokeError(f"system config not found for '{system_name}': {config_path}")
    return config_path


def build_single_system_smoke_config(
    *,
    base_config_path: Path,
    n_epochs: int,
    batch_size: int,
    seq_len: int,
    stride: int,
    max_batches_per_epoch: int,
    seed: int,
) -> dict[str, Any]:
    del seed
    config = load_yaml(base_config_path)
    model_cfg = config.setdefault("model", {})
    training_cfg = config.setdefault("training", {})
    optimizer_cfg = config.setdefault("optimizer", {})
    loss_cfg = config.setdefault("loss_weights", {})

    model_cfg["latent_dim"] = int(model_cfg.get("latent_dim", 32) // 2)
    model_cfg["hidden_dim"] = min(int(model_cfg.get("hidden_dim", 128)), 64)
    model_cfg["n_layers"] = 2
    model_cfg["drift_layers"] = 2
    model_cfg["diffusion_layers"] = 2
    model_cfg["diffusion_hidden_dim"] = 32
    model_cfg.setdefault("simulator_prior", {})["enabled"] = False
    model_cfg.setdefault("learned_solver", {})["enabled"] = False
    model_cfg.setdefault("self_correcting_policy", {})["enabled"] = False
    model_cfg.setdefault("neural_cde", {})["enabled"] = False
    model_cfg["grouped_encoder"] = {
        "enabled": True,
        "group_token_dim": 32,
        "group_kind_dim": 8,
        "group_encoder_layers": 2,
        "group_mixer_layers": 2,
    }

    training_cfg["n_epochs"] = n_epochs
    training_cfg["batch_size"] = batch_size
    training_cfg["seq_len"] = seq_len
    training_cfg["stride"] = stride
    training_cfg["max_batches_per_epoch"] = max_batches_per_epoch

    warmup_steps = min(int(optimizer_cfg.get("warmup_steps", 200)), 5)
    total_steps = max(warmup_steps + 1, n_epochs * max_batches_per_epoch + 5)
    optimizer_cfg["warmup_steps"] = warmup_steps
    optimizer_cfg["total_steps"] = total_steps

    loss_cfg["trajectory"] = 2.0
    loss_cfg["one_step"] = 0.5
    loss_cfg["mass_balance"] = 0.0
    loss_cfg["species_mass_balance"] = 0.0
    loss_cfg["energy_balance"] = 0.0
    return config


def build_universal_smoke_config(
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
    parser = argparse.ArgumentParser(description="Run the Phase 1 smoke-test matrix")
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
        help="Systems to include in data generation and universal smoke testing",
    )
    parser.add_argument(
        "--single_system",
        type=str,
        default="cstr",
        help="System used for the grouped single-system train/eval smoke",
    )
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument(
        "--n_trajectories",
        type=int,
        default=24,
        help="Trajectories to generate per system",
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
        "--single_n_epochs",
        type=int,
        default=1,
        help="Epochs for grouped single-system smoke training",
    )
    parser.add_argument(
        "--single_batch_size",
        type=int,
        default=8,
        help="Batch size for grouped single-system smoke training",
    )
    parser.add_argument(
        "--single_seq_len",
        type=int,
        default=10,
        help="Sequence length for grouped single-system smoke training",
    )
    parser.add_argument(
        "--single_stride",
        type=int,
        default=5,
        help="Stride for grouped single-system smoke training",
    )
    parser.add_argument(
        "--single_max_batches_per_epoch",
        type=int,
        default=3,
        help="Batch cap per epoch for grouped single-system smoke training",
    )
    parser.add_argument(
        "--single_eval_trajectories",
        type=int,
        default=4,
        help="Validation subsequences for grouped single-system evaluation",
    )
    parser.add_argument(
        "--single_plot_count",
        type=int,
        default=2,
        help="Saved plot count for grouped single-system evaluation",
    )
    parser.add_argument(
        "--universal_n_epochs",
        type=int,
        default=1,
        help="Epochs for universal smoke training",
    )
    parser.add_argument(
        "--universal_batch_size",
        type=int,
        default=8,
        help="Batch size for universal smoke training",
    )
    parser.add_argument(
        "--universal_seq_len",
        type=int,
        default=10,
        help="Sequence length for universal smoke training",
    )
    parser.add_argument(
        "--universal_stride",
        type=int,
        default=5,
        help="Stride for universal smoke training",
    )
    parser.add_argument(
        "--universal_max_batches_per_epoch",
        type=int,
        default=2,
        help="Batch cap per epoch for universal smoke training",
    )
    parser.add_argument(
        "--skip_data_generation",
        action="store_true",
        help="Skip data generation and reuse existing workspace datasets",
    )
    parser.add_argument(
        "--skip_single",
        action="store_true",
        help="Skip the grouped single-system train/eval smoke",
    )
    parser.add_argument(
        "--skip_universal",
        action="store_true",
        help="Skip the universal train/eval smoke",
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
    if args.single_system not in systems and not args.skip_single:
        raise SmokeError(
            f"--single_system {args.single_system!r} must be included in --systems "
            f"when single-system smoke testing is enabled."
        )

    workspace = resolve_workspace_dir(args.workspace_dir)
    configs_dir = workspace / "configs"
    logs_dir = workspace / "logs"
    data_root = workspace / "data"
    outputs_root = workspace / "outputs"
    summary_path = workspace / "summary.json"

    workspace.mkdir(parents=True, exist_ok=True)

    single_config_path = configs_dir / "training_grouped_single.yaml"
    universal_config_path = configs_dir / "training_universal_phase1_smoke.yaml"

    single_config = build_single_system_smoke_config(
        base_config_path=PROJECT_ROOT / "configs" / "training_default.yaml",
        n_epochs=args.single_n_epochs,
        batch_size=args.single_batch_size,
        seq_len=args.single_seq_len,
        stride=args.single_stride,
        max_batches_per_epoch=args.single_max_batches_per_epoch,
        seed=args.seed,
    )
    universal_config = build_universal_smoke_config(
        systems=systems,
        data_root=data_root,
        n_epochs=args.universal_n_epochs,
        batch_size=args.universal_batch_size,
        seq_len=args.universal_seq_len,
        stride=args.universal_stride,
        max_batches_per_epoch=args.universal_max_batches_per_epoch,
    )
    write_yaml(single_config_path, single_config)
    write_yaml(universal_config_path, universal_config)

    summary: dict[str, Any] = {
        "status": "running",
        "workspace_dir": str(workspace),
        "started_at": datetime.now().isoformat(),
        "settings": {
            "systems": systems,
            "single_system": args.single_system,
            "n_trajectories": args.n_trajectories,
            "n_steps": args.n_steps,
            "dry_run": args.dry_run,
        },
        "configs": {
            "single": str(single_config_path),
            "universal": str(universal_config_path),
        },
        "steps": [],
        "artifacts": {
            "data": {name: str(data_root / name / "train_data.h5") for name in systems},
            "single": {},
            "universal": {},
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
                    raise SmokeError(f"step '{step['name']}' failed with exit code {step['returncode']}")

        if not args.skip_single:
            single_output_dir = outputs_root / "grouped_single"
            single_train_summary = single_output_dir / "train_summary.json"
            step = run_command(
                name="train:grouped_single",
                command=[
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / "train.py"),
                    "--config",
                    str(single_config_path),
                    "--system_config",
                    str(resolve_system_config(args.single_system)),
                    "--data_dir",
                    str(data_root / args.single_system),
                    "--output_dir",
                    str(single_output_dir),
                    "--summary_path",
                    str(single_train_summary),
                ],
                log_path=logs_dir / "train_grouped_single.log",
                dry_run=args.dry_run,
            )
            summary["steps"].append(step)
            write_summary(summary_path, summary)
            if not step["succeeded"]:
                raise SmokeError(f"step '{step['name']}' failed with exit code {step['returncode']}")
            summary["artifacts"]["single"] = {
                "output_dir": str(single_output_dir),
                "train_summary_path": str(single_train_summary),
                "model_path": str(single_output_dir / "best_model.eqx"),
                "eval_summary_path": str(single_output_dir / "eval" / "summary.json"),
            }
            write_summary(summary_path, summary)

            step = run_command(
                name="evaluate:grouped_single",
                command=[
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / "evaluate.py"),
                    "--model_path",
                    str(single_output_dir / "best_model.eqx"),
                    "--config",
                    str(single_output_dir / "config.yaml"),
                    "--system_config",
                    str(single_output_dir / "system_config.yaml"),
                    "--data_dir",
                    str(data_root / args.single_system),
                    "--output_dir",
                    str(single_output_dir / "eval"),
                    "--n_trajectories",
                    str(args.single_eval_trajectories),
                    "--plot_count",
                    str(args.single_plot_count),
                    "--summary_path",
                    str(single_output_dir / "eval" / "summary.json"),
                    "--skip_ensemble",
                ],
                log_path=logs_dir / "evaluate_grouped_single.log",
                dry_run=args.dry_run,
            )
            summary["steps"].append(step)
            write_summary(summary_path, summary)
            if not step["succeeded"]:
                raise SmokeError(f"step '{step['name']}' failed with exit code {step['returncode']}")
            summary["artifacts"]["single"]["train_summary"] = load_json_if_exists(single_train_summary)
            summary["artifacts"]["single"]["eval_summary"] = load_json_if_exists(
                single_output_dir / "eval" / "summary.json"
            )
            write_summary(summary_path, summary)

        if not args.skip_universal:
            universal_output_dir = outputs_root / "universal_phase1"
            universal_train_summary = universal_output_dir / "train_summary.json"
            step = run_command(
                name="train:universal_phase1",
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
                log_path=logs_dir / "train_universal_phase1.log",
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
                name="evaluate:universal_phase1",
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
                log_path=logs_dir / "evaluate_universal_phase1.log",
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
    summary["single_best_val_loss"] = _json_safe_float(
        (summary["artifacts"].get("single") or {}).get("train_summary", {}).get("best_val_loss")
        if isinstance((summary["artifacts"].get("single") or {}).get("train_summary"), dict)
        else None
    )
    summary["universal_best_val_loss"] = _json_safe_float(
        (summary["artifacts"].get("universal") or {}).get("train_summary", {}).get("best_val_loss")
        if isinstance((summary["artifacts"].get("universal") or {}).get("train_summary"), dict)
        else None
    )
    write_summary(summary_path, summary)

    print("\n" + "=" * 60)
    print("PHASE 1 SMOKE COMPLETE")
    print("=" * 60)
    print(f"Workspace: {workspace}")
    print(f"Summary: {summary_path}")
    if summary["single_best_val_loss"] is not None:
        print(f"Single-system best val loss: {summary['single_best_val_loss']:.4f}")
    if summary["universal_best_val_loss"] is not None:
        print(f"Universal best val loss: {summary['universal_best_val_loss']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
