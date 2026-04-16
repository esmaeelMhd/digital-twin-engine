"""Run a reusable Phase 5 smoke-test matrix for customer adaptation."""

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

from dte.simulators.registry import get_system_spec


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
        return PROJECT_ROOT / "outputs" / "phase5_smoke" / stamp
    path = Path(raw_workspace)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def resolve_system_config(system_name: str) -> Path:
    config_path = PROJECT_ROOT / "configs" / f"{system_name}_default.yaml"
    if not config_path.exists():
        raise SmokeError(f"system config not found for '{system_name}': {config_path}")
    return config_path


def build_universal_phase5_smoke_config(
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
) -> dict[str, Any]:
    """Build a small held-out customer variant config."""

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
    return config


def build_customer_onboarding(
    *,
    target_system_name: str,
    target_config: dict[str, Any],
) -> dict[str, Any]:
    spec = get_system_spec(target_config)
    unit_name = f"{target_system_name}_section"
    return {
        "name": target_system_name,
        "asset_kind": "unit",
        "units": [
            {
                "name": unit_name,
                "family": getattr(spec, "family", "generic"),
                "subtype": getattr(spec, "subtype", None) or getattr(spec, "unit_type", None),
                "unit_type": getattr(spec, "unit_type", None),
                "controls": list(spec.control_names),
                "disturbances": list(spec.disturbance_names),
                "measurements": list(spec.state_names),
                "known_laws": list(getattr(spec, "law_tags", [])),
            }
        ],
        "controls": [
            {
                "name": channel.name,
                "role": channel.role,
                "unit": channel.unit,
                "unit_name": unit_name,
                "lower_bound": channel.lower_bound,
                "upper_bound": channel.upper_bound,
            }
            for channel in getattr(spec, "control_channels", [])
        ],
        "disturbances": [
            {
                "name": channel.name,
                "role": channel.role,
                "unit": channel.unit,
                "unit_name": unit_name,
                "lower_bound": channel.lower_bound,
                "upper_bound": channel.upper_bound,
            }
            for channel in getattr(spec, "disturbance_channels", [])
        ],
        "measurements": [
            {
                "name": channel.name,
                "role": channel.role,
                "unit": channel.unit,
                "unit_name": unit_name,
                "source": "state",
                "lower_bound": channel.lower_bound,
                "upper_bound": channel.upper_bound,
            }
            for channel in getattr(spec, "state_channels", [])
        ],
        "known_laws": list(getattr(spec, "law_tags", [])),
    }


def run_command(
    *,
    name: str,
    command: list[str],
    log_path: Path,
    dry_run: bool,
    extra_env: dict[str, str] | None = None,
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
    if extra_env:
        env.update(extra_env)
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

    if returncode != 0:
        raise SmokeError(f"step '{name}' failed with exit code {returncode}")
    return step_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the reusable Phase 5 smoke matrix")
    parser.add_argument("--workspace_dir", type=str, default=None, help="Output workspace directory")
    parser.add_argument(
        "--systems",
        nargs="+",
        default=list(DEFAULT_SYSTEMS),
        help="Source systems to include in the small universal pretraining run",
    )
    parser.add_argument(
        "--target_base_system",
        choices=list(DEFAULT_SYSTEMS),
        default="cstr",
        help="Base system family to shift into a held-out customer variant",
    )
    parser.add_argument(
        "--target_system_name",
        type=str,
        default="customer_cstr_variant",
        help="External dataset/customer name for the held-out target variant",
    )
    parser.add_argument("--n_trajectories", type=int, default=18, help="Source trajectories per system")
    parser.add_argument("--target_n_trajectories", type=int, default=16, help="Target variant trajectories")
    parser.add_argument("--n_steps", type=int, default=24, help="Steps per generated trajectory")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--n_epochs", type=int, default=1, help="Universal pretraining epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Training/calibration batch size")
    parser.add_argument("--seq_len", type=int, default=16, help="Sequence length for the smoke config")
    parser.add_argument("--stride", type=int, default=4, help="Sequence stride for the smoke config")
    parser.add_argument(
        "--max_batches_per_epoch",
        type=int,
        default=2,
        help="Cap training batches per epoch for the smoke run",
    )
    parser.add_argument(
        "--trainable_mode",
        choices=["adapters", "full"],
        default="adapters",
        help="Customer calibration mode",
    )
    parser.add_argument(
        "--skip_data_generation",
        action="store_true",
        help="Reuse existing datasets under the workspace",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Write configs and print planned commands without executing them",
    )
    parser.add_argument(
        "--jax_platform",
        choices=["cpu", "gpu"],
        default="cpu",
        help="JAX platform to force for smoke subprocesses",
    )
    args = parser.parse_args()

    workspace_dir = resolve_workspace_dir(args.workspace_dir)
    config_dir = workspace_dir / "configs"
    data_root = workspace_dir / "data"
    logs_dir = workspace_dir / "logs"
    outputs_root = workspace_dir / "outputs"
    target_config_path = config_dir / f"{args.target_system_name}_system.yaml"
    onboarding_path = config_dir / f"{args.target_system_name}_onboarding.yaml"
    universal_config_path = config_dir / "training_universal_phase5_smoke.yaml"
    target_data_dir = data_root / args.target_system_name
    universal_output_dir = outputs_root / "universal_pretrain"
    adaptation_output_dir = outputs_root / "customer_adaptation"
    summary_path = workspace_dir / "summary.json"
    runtime_env = {
        "JAX_PLATFORMS": args.jax_platform,
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
    }

    workspace_dir.mkdir(parents=True, exist_ok=True)
    write_yaml(
        universal_config_path,
        build_universal_phase5_smoke_config(
            systems=list(args.systems),
            data_root=data_root,
            n_epochs=args.n_epochs,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            stride=args.stride,
            max_batches_per_epoch=args.max_batches_per_epoch,
        ),
    )
    target_variant_config = build_target_variant_config(base_system=args.target_base_system)
    write_yaml(target_config_path, target_variant_config)
    write_yaml(
        onboarding_path,
        build_customer_onboarding(
            target_system_name=args.target_system_name,
            target_config=target_variant_config,
        ),
    )

    step_summaries: list[dict[str, Any]] = []

    if not args.skip_data_generation:
        for index, system_name in enumerate(args.systems):
            step_summaries.append(
                run_command(
                    name=f"generate_{system_name}_data",
                    command=[
                        sys.executable,
                        "scripts/generate_data.py",
                        "--config",
                        str(resolve_system_config(system_name)),
                        "--n_trajectories",
                        str(args.n_trajectories),
                        "--n_steps",
                        str(args.n_steps),
                        "--seed",
                        str(args.seed + index),
                        "--output_dir",
                        str(data_root / system_name),
                    ],
                    log_path=logs_dir / f"generate_{system_name}.log",
                    dry_run=args.dry_run,
                    extra_env=runtime_env,
                )
            )
        step_summaries.append(
            run_command(
                name=f"generate_{args.target_system_name}_data",
                command=[
                    sys.executable,
                    "scripts/generate_data.py",
                    "--config",
                    str(target_config_path),
                    "--n_trajectories",
                    str(args.target_n_trajectories),
                    "--n_steps",
                    str(args.n_steps),
                    "--seed",
                    str(args.seed + 100),
                    "--output_dir",
                    str(target_data_dir),
                ],
                log_path=logs_dir / f"generate_{args.target_system_name}.log",
                dry_run=args.dry_run,
                extra_env=runtime_env,
            )
        )

    step_summaries.append(
        run_command(
            name="train_universal_pretrain",
            command=[
                sys.executable,
                "scripts/train_universal.py",
                "--config",
                str(universal_config_path),
                "--output_dir",
                str(universal_output_dir),
                "--seed",
                str(args.seed),
                "--summary_path",
                str(universal_output_dir / "summary.json"),
            ],
            log_path=logs_dir / "train_universal.log",
            dry_run=args.dry_run,
            extra_env=runtime_env,
        )
    )

    step_summaries.append(
        run_command(
            name="adapt_customer",
            command=[
                sys.executable,
                "scripts/adapt_customer.py",
                "--onboarding",
                str(onboarding_path),
                "--model_path",
                str(universal_output_dir / "best_model.eqx"),
                "--config",
                str(universal_config_path),
                "--system_config",
                str(target_config_path),
                "--data_dir",
                str(target_data_dir),
                "--system_name",
                str(args.target_system_name),
                "--output_dir",
                str(adaptation_output_dir),
                "--trainable_mode",
                str(args.trainable_mode),
                "--tune_normalization",
                "--seed",
                str(args.seed + 1),
                "--summary_path",
                str(adaptation_output_dir / "summary.json"),
            ],
            log_path=logs_dir / "adapt_customer.log",
            dry_run=args.dry_run,
            extra_env=runtime_env,
        )
    )

    summary: dict[str, Any] = {
        "status": "ok",
        "workspace_dir": str(workspace_dir.resolve()),
        "systems": list(args.systems),
        "target_base_system": args.target_base_system,
        "target_system_name": args.target_system_name,
        "jax_platform": args.jax_platform,
        "steps": step_summaries,
        "universal_config_path": str(universal_config_path.resolve()),
        "target_config_path": str(target_config_path.resolve()),
        "onboarding_path": str(onboarding_path.resolve()),
    }

    if not args.dry_run:
        universal_summary = json.loads((universal_output_dir / "summary.json").read_text(encoding="utf-8"))
        adaptation_summary = json.loads((adaptation_output_dir / "summary.json").read_text(encoding="utf-8"))
        report = adaptation_summary["validation_report"]
        best_unit_match = report["template_matching"]["best_unit_match"]["name"]
        if best_unit_match != args.target_base_system:
            raise SmokeError(
                f"expected best unit template '{args.target_base_system}' but got '{best_unit_match}'"
            )
        if not Path(adaptation_summary["report_json_path"]).exists():
            raise SmokeError("customer adaptation did not produce validation_report.json")
        if not Path(adaptation_summary["report_markdown_path"]).exists():
            raise SmokeError("customer adaptation did not produce validation_report.md")

        summary.update(
            {
                "universal_summary_path": str((universal_output_dir / "summary.json").resolve()),
                "adaptation_summary_path": str((adaptation_output_dir / "summary.json").resolve()),
                "universal_best_val_loss": _json_safe_float(universal_summary.get("best_val_loss")),
                "adaptation_best_val_loss": _json_safe_float(adaptation_summary.get("best_val_loss")),
                "report_json_path": adaptation_summary["report_json_path"],
                "report_markdown_path": adaptation_summary["report_markdown_path"],
                "best_unit_template": best_unit_match,
                "forecast_rmse": _json_safe_float(report["forecast_metrics"].get("rmse")),
                "rollout_rmse": _json_safe_float(report["rollout_metrics"].get("rmse")),
                "uncertainty_calibration_gap": _json_safe_float(
                    report["uncertainty_summary"].get("calibration_gap")
                ),
            }
        )

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nPhase 5 smoke summary: {summary_path}")


if __name__ == "__main__":
    main()
