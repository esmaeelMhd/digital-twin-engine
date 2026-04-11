"""Run an ingestion-backed customer adaptation pilot."""

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
import pandas as pd
import yaml

from dte.data.generation import load_dataset
from dte.simulators.registry import get_system_spec


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_SYSTEMS = ("cstr", "heat_exchanger", "two_tank")


class PilotError(RuntimeError):
    """Raised when the ingestion-backed pilot cannot complete."""


def _json_safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(numeric):
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


def resolve_workspace_dir(raw_workspace: str | None) -> Path:
    if raw_workspace is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return PROJECT_ROOT / "outputs" / "real_data_adaptation_pilot" / stamp
    path = Path(raw_workspace)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def resolve_system_config(system_name: str) -> Path:
    path = PROJECT_ROOT / "configs" / f"{system_name}_default.yaml"
    if not path.exists():
        raise PilotError(f"system config not found for '{system_name}': {path}")
    return path


def build_target_variant_config(base_system: str) -> dict[str, Any]:
    config = load_yaml(resolve_system_config(base_system))
    config.setdefault("system", {})
    config["system"].setdefault("conditioning_tags", {})
    config["system"]["conditioning_tags"]["operating_regime"] = "real_data_pilot"

    if base_system == "cstr":
        config["cstr"]["Ea_over_R"] = float(config["cstr"]["Ea_over_R"]) * 0.97
        config["cstr"]["UA"] = float(config["cstr"]["UA"]) * 0.9
        config["cstr"]["Fc"] = float(config["cstr"]["Fc"]) * 1.05
        config["initial_conditions"]["Ca"] = 0.6
        config["initial_conditions"]["Cb"] = 0.4
        config["initial_conditions"]["T"] = 344.0
        config["initial_conditions"]["Tc"] = 302.0
        config["operating_ranges"]["F_in"] = [12.0, 95.0]
        config["operating_ranges"]["Tc_in"] = [282.0, 318.0]
        config["operating_ranges"]["Ca_in"] = [0.6, 1.8]
        config["operating_ranges"]["T_in"] = [295.0, 345.0]
    elif base_system == "heat_exchanger":
        config["heat_exchanger"]["UA"] = float(config["heat_exchanger"]["UA"]) * 0.88
        config["heat_exchanger"]["V_hot"] = float(config["heat_exchanger"]["V_hot"]) * 1.10
        config["heat_exchanger"]["V_cold"] = float(config["heat_exchanger"]["V_cold"]) * 0.95
        config["initial_conditions"]["T_hot"] = 342.0
        config["initial_conditions"]["T_cold"] = 304.0
        config["operating_ranges"]["F_hot"] = [1.5, 9.0]
        config["operating_ranges"]["F_cold"] = [1.5, 9.0]
    elif base_system == "two_tank":
        config["two_tank"]["A1"] = float(config["two_tank"]["A1"]) * 1.08
        config["two_tank"]["A2"] = float(config["two_tank"]["A2"]) * 0.95
        config["two_tank"]["k12"] = float(config["two_tank"]["k12"]) * 0.96
        config["two_tank"]["kout"] = float(config["two_tank"]["kout"]) * 1.04
        config["initial_conditions"]["h1"] = 1.6
        config["initial_conditions"]["h2"] = 0.95
        config["operating_ranges"]["q_in"] = [0.35, 1.05]
        config["operating_ranges"]["valve"] = [0.68, 0.98]
    else:
        raise PilotError(f"unsupported target base system '{base_system}'")

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
    env_updates: dict[str, str],
    dry_run: bool,
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
    if returncode != 0:
        raise PilotError(f"step '{name}' failed with exit code {returncode}")
    return step


def write_pseudo_historian_csv(
    *,
    generated_dataset_path: Path,
    target_config: dict[str, Any],
    csv_path: Path,
    seed: int,
) -> dict[str, Any]:
    dataset = load_dataset(str(generated_dataset_path))
    spec = get_system_spec(target_config)
    rng = np.random.default_rng(seed)

    states = np.asarray(dataset["states"])
    controls = np.asarray(dataset["controls"])
    disturbances = np.asarray(dataset["disturbances"])
    time_grid = np.asarray(dataset["time"])

    rows: list[dict[str, float]] = []
    timestamp_offset = 0.0
    gap_seconds = 3.0 * float(np.median(np.diff(time_grid[0])))
    used_trajectories = min(states.shape[0], 6)
    for traj_idx in range(used_trajectories):
        traj_time = time_grid[traj_idx] + timestamp_offset
        for step_idx, timestamp in enumerate(traj_time):
            row = {"timestamp": float(timestamp)}
            for channel_idx, name in enumerate(spec.state_names):
                row[name] = float(states[traj_idx, step_idx, channel_idx])
            for channel_idx, name in enumerate(spec.control_names):
                row[name] = float(controls[traj_idx, step_idx, channel_idx])
            for channel_idx, name in enumerate(spec.disturbance_names):
                row[name] = float(disturbances[traj_idx, step_idx, channel_idx])
            rows.append(row)
        timestamp_offset = float(traj_time[-1] + gap_seconds)

    frame = pd.DataFrame(rows)

    # Introduce light historian-like imperfections without making the pilot fragile.
    frame = frame.drop(index=frame.index[::17]).reset_index(drop=True)
    if len(frame) > 12:
        state_col = spec.state_names[0]
        frame.loc[5, state_col] = np.nan
        frame.loc[11, state_col] = float(frame.loc[11, state_col] * 1.1)
        control_col = spec.control_names[0]
        frame.loc[9, control_col] = np.nan
    frame["timestamp"] += rng.normal(loc=0.0, scale=0.01, size=len(frame))
    frame = frame.sort_values("timestamp").reset_index(drop=True)

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(csv_path, index=False)
    return {
        "csv_path": str(csv_path.resolve()),
        "rows_written": int(len(frame)),
        "used_trajectories": int(used_trajectories),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a real-data-style customer adaptation pilot")
    parser.add_argument("--workspace_dir", type=str, default=None, help="Workspace directory")
    parser.add_argument("--model_path", type=str, required=True, help="Pretrained universal checkpoint")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Universal training config used to reconstruct the checkpoint",
    )
    parser.add_argument(
        "--target_base_system",
        choices=list(DEFAULT_SOURCE_SYSTEMS),
        default="cstr",
        help="Base system family to shift into an ingestion-backed target variant",
    )
    parser.add_argument(
        "--target_system_name",
        type=str,
        default="customer_cstr_ingested",
        help="External customer/dataset name for the ingested target variant",
    )
    parser.add_argument("--target_n_trajectories", type=int, default=18, help="Generated raw trajectories")
    parser.add_argument("--n_steps", type=int, default=36, help="Steps per raw generated trajectory")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument(
        "--trainable_mode",
        choices=["adapters", "full"],
        default="adapters",
        help="Calibration mode for adaptation",
    )
    parser.add_argument(
        "--jax_platform",
        choices=["cpu", "gpu"],
        default="cpu",
        help="JAX platform for subprocesses",
    )
    parser.add_argument("--dry_run", action="store_true", help="Plan commands without executing them")
    args = parser.parse_args()

    workspace_dir = resolve_workspace_dir(args.workspace_dir)
    configs_dir = workspace_dir / "configs"
    data_root = workspace_dir / "data"
    logs_dir = workspace_dir / "logs"
    outputs_dir = workspace_dir / "outputs"
    summary_path = workspace_dir / "summary.json"

    workspace_dir.mkdir(parents=True, exist_ok=True)

    target_config = build_target_variant_config(args.target_base_system)
    target_config_path = configs_dir / f"{args.target_system_name}_system.yaml"
    onboarding_path = configs_dir / f"{args.target_system_name}_onboarding.yaml"
    write_yaml(target_config_path, target_config)
    write_yaml(
        onboarding_path,
        build_customer_onboarding(
            target_system_name=args.target_system_name,
            target_config=target_config,
        ),
    )

    runtime_env = {
        "JAX_PLATFORMS": args.jax_platform,
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
    }
    raw_generated_dir = data_root / "raw_generated"
    raw_csv_path = data_root / "historian_export.csv"
    ingested_data_dir = data_root / "ingested_target"
    adaptation_output_dir = outputs_dir / "customer_adaptation"
    ingest_summary_path = outputs_dir / "ingest_summary.json"
    adaptation_summary_path = adaptation_output_dir / "summary.json"

    steps: list[dict[str, Any]] = []
    steps.append(
        run_command(
            name="generate_target_variant_data",
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
                str(args.seed),
                "--output_dir",
                str(raw_generated_dir),
            ],
            log_path=logs_dir / "generate_target_variant.log",
            env_updates=runtime_env,
            dry_run=args.dry_run,
        )
    )

    csv_summary: dict[str, Any] | None = None
    if not args.dry_run:
        csv_summary = write_pseudo_historian_csv(
            generated_dataset_path=raw_generated_dir / "train_data.h5",
            target_config=target_config,
            csv_path=raw_csv_path,
            seed=args.seed + 1,
        )

    spec = get_system_spec(target_config)
    simulation_dt = float(target_config.get("simulation", {}).get("dt", 1.0))
    raw_duration = simulation_dt * float(args.n_steps)
    trajectory_duration = max(simulation_dt * 8.0, raw_duration * 0.5)
    trajectory_stride = max(simulation_dt * 4.0, trajectory_duration * 0.5)

    steps.append(
        run_command(
            name="ingest_historian_export",
            command=[
                sys.executable,
                "scripts/ingest_real_data.py",
                "--source",
                str(raw_csv_path),
                "--output",
                str(ingested_data_dir / "train_data.h5"),
                "--system_config",
                str(target_config_path),
                "--state_columns",
                *spec.state_names,
                "--control_columns",
                *spec.control_names,
                "--disturbance_columns",
                *spec.disturbance_names,
                "--timestamp_column",
                "timestamp",
                "--dt",
                str(simulation_dt),
                "--trajectory_duration",
                str(trajectory_duration),
                "--trajectory_stride",
                str(trajectory_stride),
                "--drop_large_gaps",
                "--save_summary",
                str(ingest_summary_path),
            ],
            log_path=logs_dir / "ingest_real_data.log",
            env_updates=runtime_env,
            dry_run=args.dry_run,
        )
    )

    steps.append(
        run_command(
            name="adapt_customer_from_ingested_data",
            command=[
                sys.executable,
                "scripts/adapt_customer.py",
                "--onboarding",
                str(onboarding_path),
                "--model_path",
                str(Path(args.model_path).resolve()),
                "--config",
                str(Path(args.config).resolve()),
                "--system_config",
                str(target_config_path),
                "--data_dir",
                str(ingested_data_dir),
                "--system_name",
                str(args.target_system_name),
                "--output_dir",
                str(adaptation_output_dir),
                "--trainable_mode",
                args.trainable_mode,
                "--tune_normalization",
                "--seed",
                str(args.seed + 2),
                "--summary_path",
                str(adaptation_summary_path),
            ],
            log_path=logs_dir / "adapt_customer.log",
            env_updates=runtime_env,
            dry_run=args.dry_run,
        )
    )

    summary: dict[str, Any] = {
        "status": "dry_run" if args.dry_run else "ok",
        "workspace_dir": str(workspace_dir.resolve()),
        "target_base_system": args.target_base_system,
        "target_system_name": args.target_system_name,
        "model_path": str(Path(args.model_path).resolve()),
        "config_path": str(Path(args.config).resolve()),
        "steps": steps,
        "target_config_path": str(target_config_path.resolve()),
        "onboarding_path": str(onboarding_path.resolve()),
    }

    if csv_summary is not None:
        summary["historian_csv"] = csv_summary

    if not args.dry_run:
        ingest_summary = json.loads(ingest_summary_path.read_text(encoding="utf-8"))
        adaptation_summary = json.loads(adaptation_summary_path.read_text(encoding="utf-8"))
        report = adaptation_summary["validation_report"]
        best_unit_match = report["template_matching"]["best_unit_match"]["name"]
        if best_unit_match != args.target_base_system:
            raise PilotError(
                f"expected best unit match '{args.target_base_system}' but got '{best_unit_match}'"
            )
        summary.update(
            {
                "ingest_summary_path": str(ingest_summary_path.resolve()),
                "adaptation_summary_path": str(adaptation_summary_path.resolve()),
                "ingested_trajectories": int(ingest_summary["n_trajectories"]),
                "ingested_steps_per_trajectory": int(ingest_summary["n_steps_per_trajectory"]),
                "best_unit_template": best_unit_match,
                "adaptation_status": adaptation_summary["status"],
                "best_val_loss": _json_safe_float(adaptation_summary.get("best_val_loss")),
                "report_json_path": adaptation_summary["report_json_path"],
                "report_markdown_path": adaptation_summary["report_markdown_path"],
                "forecast_rmse": _json_safe_float(report["forecast_metrics"].get("rmse")),
                "rollout_rmse": _json_safe_float(report["rollout_metrics"].get("rmse")),
                "uncertainty_calibration_gap": _json_safe_float(
                    report["uncertainty_summary"].get("calibration_gap")
                ),
            }
        )

    write_json(summary_path, summary)
    print(f"\nReal-data adaptation pilot summary: {summary_path}")


if __name__ == "__main__":
    main()
