"""Run the full first-version foundation-stack validation sequence."""

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


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class MilestoneError(RuntimeError):
    """Raised when the milestone runner cannot complete successfully."""


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
        return PROJECT_ROOT / "outputs" / "v1_milestone" / stamp
    path = Path(raw_workspace)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def _json_safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not (-float("inf") < numeric < float("inf")):
        return None
    return numeric


def _extract_universal_per_system_losses(summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Return the per-system loss mapping across older and newer eval summaries."""

    mapping = summary.get("per_system_val_losses")
    if isinstance(mapping, dict):
        return mapping
    mapping = summary.get("per_system_train_fallback")
    if isinstance(mapping, dict):
        return mapping
    mapping = summary.get("per_system_metrics")
    if isinstance(mapping, dict):
        return mapping
    return {}


def _extract_universal_total_loss(entry: dict[str, Any]) -> float | None:
    """Return the primary per-system total-loss field across eval summary variants."""

    total = entry.get("total")
    if total is not None:
        return _json_safe_float(total)
    total = entry.get("total_loss")
    if total is not None:
        return _json_safe_float(total)
    return None


def _extract_phase6_demo_count(summary: dict[str, Any]) -> int:
    """Return the number of demo routes exercised by the Phase 6 smoke runner."""

    top_level = summary.get("n_demos")
    if top_level is not None:
        return int(top_level)

    for step in summary.get("steps", []):
        if step.get("name") == "demo_catalog":
            try:
                return int(step.get("n_demos", 0))
            except (TypeError, ValueError):
                return 0
    return 0


def run_command(
    *,
    name: str,
    command: list[str],
    log_path: Path,
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

    print("\n" + "=" * 80)
    print(f"STEP: {name}")
    print("=" * 80)
    print("Command:", " ".join(command), flush=True)
    print("Log:", log_path, flush=True)

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
        raise MilestoneError(f"step '{name}' failed with exit code {returncode}")
    return step


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full v1 foundation-stack milestone validation")
    parser.add_argument("--workspace_dir", type=str, default=None, help="Workspace directory")
    parser.add_argument("--jax_platform", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument(
        "--compare_systems",
        nargs="+",
        default=["heat_exchanger"],
        help=(
            "Systems to include in the rollout-stability A/B benchmark. "
            "Defaults to the representative heat-exchanger release proof."
        ),
    )
    parser.add_argument(
        "--skip_pytest",
        action="store_true",
        help="Skip the full pytest suite",
    )
    parser.add_argument(
        "--skip_smokes",
        action="store_true",
        help="Skip phase smoke runners",
    )
    parser.add_argument(
        "--finalize_existing",
        action="store_true",
        help="Reuse existing artifacts in the workspace and only recompute milestone pass conditions",
    )
    args = parser.parse_args()

    workspace_dir = resolve_workspace_dir(args.workspace_dir)
    logs_dir = workspace_dir / "logs"
    smoke_root = workspace_dir / "smokes"
    outputs_root = workspace_dir / "outputs"
    summary_path = workspace_dir / "summary.json"

    workspace_dir.mkdir(parents=True, exist_ok=True)
    env_updates = {
        "JAX_PLATFORMS": args.jax_platform,
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
    }

    if args.finalize_existing and summary_path.exists():
        summary = load_json(summary_path)
        summary["status"] = "running"
        summary.pop("error", None)
    else:
        summary = {
            "status": "running",
            "workspace_dir": str(workspace_dir.resolve()),
            "jax_platform": args.jax_platform,
            "steps": [],
            "pass_conditions": {},
        }
    write_json(summary_path, summary)

    try:
        if not args.finalize_existing:
            summary["steps"].append(
                run_command(
                    name="verify_install",
                    command=[sys.executable, "scripts/verify_install.py"],
                    log_path=logs_dir / "verify_install.log",
                    env_updates=env_updates,
                )
            )
            write_json(summary_path, summary)

            if not args.skip_pytest:
                summary["steps"].append(
                    run_command(
                        name="pytest_full",
                        command=[sys.executable, "-m", "pytest", "tests/", "-v"],
                        log_path=logs_dir / "pytest_full.log",
                        env_updates=env_updates,
                    )
                )
                write_json(summary_path, summary)

            if not args.skip_smokes:
                smoke_commands = [
                    (
                        "smoke_phase1",
                        [
                            sys.executable,
                            "scripts/phases/smoke_phase1.py",
                            "--workspace_dir",
                            str(smoke_root / "phase1"),
                        ],
                        logs_dir / "smoke_phase1.log",
                    ),
                    (
                        "smoke_phase2",
                        [
                            sys.executable,
                            "scripts/phases/smoke_phase2.py",
                            "--workspace_dir",
                            str(smoke_root / "phase2"),
                        ],
                        logs_dir / "smoke_phase2.log",
                    ),
                    (
                        "smoke_phase3",
                        [
                            sys.executable,
                            "scripts/phases/smoke_phase3.py",
                            "--workspace_dir",
                            str(smoke_root / "phase3"),
                            "--jax_platforms",
                            args.jax_platform,
                        ],
                        logs_dir / "smoke_phase3.log",
                    ),
                    (
                        "smoke_phase4",
                        [
                            sys.executable,
                            "scripts/phases/smoke_phase4.py",
                            "--workspace_dir",
                            str(smoke_root / "phase4"),
                            "--jax_platforms",
                            args.jax_platform,
                        ],
                        logs_dir / "smoke_phase4.log",
                    ),
                    (
                        "smoke_phase5",
                        [
                            sys.executable,
                            "scripts/phases/smoke_phase5.py",
                            "--workspace_dir",
                            str(smoke_root / "phase5"),
                            "--jax_platform",
                            args.jax_platform,
                        ],
                        logs_dir / "smoke_phase5.log",
                    ),
                    (
                        "smoke_phase6",
                        [
                            sys.executable,
                            "scripts/phases/smoke_phase6.py",
                            "--workspace_dir",
                            str(smoke_root / "phase6"),
                            "--jax_platform",
                            args.jax_platform,
                        ],
                        logs_dir / "smoke_phase6.log",
                    ),
                    (
                        "smoke_phase7",
                        [
                            sys.executable,
                            "scripts/phases/smoke_phase7.py",
                            "--workspace_dir",
                            str(smoke_root / "phase7"),
                            "--jax_platform",
                            args.jax_platform,
                        ],
                        logs_dir / "smoke_phase7.log",
                    ),
                ]
                for name, command, log_path in smoke_commands:
                    summary["steps"].append(
                        run_command(
                            name=name,
                            command=command,
                            log_path=log_path,
                            env_updates=env_updates,
                        )
                    )
                    write_json(summary_path, summary)

            universal_output_dir = outputs_root / "universal_fast_baseline"
            summary["steps"].append(
                run_command(
                    name="train_universal_fast_baseline",
                    command=[
                        sys.executable,
                        "scripts/train_universal.py",
                        "--config",
                        "configs/training_universal_baseline_fast.yaml",
                        "--output_dir",
                        str(universal_output_dir),
                        "--seed",
                        "42",
                    ],
                    log_path=logs_dir / "train_universal_fast_baseline.log",
                    env_updates=env_updates,
                )
            )
            write_json(summary_path, summary)

            summary["steps"].append(
                run_command(
                    name="evaluate_universal_fast_baseline",
                    command=[
                        sys.executable,
                        "scripts/evaluate_universal.py",
                        "--model_path",
                        str(universal_output_dir / "best_model.eqx"),
                        "--config",
                        str(universal_output_dir / "config.yaml"),
                        "--output_dir",
                        str(universal_output_dir / "eval"),
                    ],
                    log_path=logs_dir / "evaluate_universal_fast_baseline.log",
                    env_updates=env_updates,
                )
            )
            write_json(summary_path, summary)

            summary["steps"].append(
                run_command(
                    name="pilot_real_data_adaptation",
                    command=[
                        sys.executable,
                        "scripts/pilot_real_data_adaptation.py",
                        "--workspace_dir",
                        str(outputs_root / "real_data_pilot"),
                        "--model_path",
                        str(universal_output_dir / "best_model.eqx"),
                        "--config",
                        str(universal_output_dir / "config.yaml"),
                        "--target_base_system",
                        "cstr",
                        "--target_system_name",
                        "customer_cstr_ingested",
                        "--jax_platform",
                        args.jax_platform,
                    ],
                    log_path=logs_dir / "pilot_real_data_adaptation.log",
                    env_updates=env_updates,
                )
            )
            write_json(summary_path, summary)

            summary["steps"].append(
                run_command(
                    name="compare_rollout_stability",
                    command=[
                        sys.executable,
                        "scripts/compare_rollout_stability.py",
                        "--workspace_dir",
                        str(outputs_root / "rollout_stability"),
                        "--jax_platform",
                        args.jax_platform,
                        "--systems",
                        *args.compare_systems,
                    ],
                    log_path=logs_dir / "compare_rollout_stability.log",
                    env_updates=env_updates,
                )
            )
            write_json(summary_path, summary)

        universal_output_dir = outputs_root / "universal_fast_baseline"
        universal_train_summary = load_json(universal_output_dir / "summary.json")
        universal_eval_summary = load_json(universal_output_dir / "eval" / "summary.json")
        rollout_summary = load_json(outputs_root / "rollout_stability" / "summary.json")
        real_data_pilot_summary = load_json(outputs_root / "real_data_pilot" / "summary.json")

        universal_per_system_losses = _extract_universal_per_system_losses(universal_eval_summary)
        phase3_summary = load_json(smoke_root / "phase3" / "summary.json") if not args.skip_smokes else {}
        phase4_summary = load_json(smoke_root / "phase4" / "summary.json") if not args.skip_smokes else {}
        phase6_summary = load_json(smoke_root / "phase6" / "summary.json") if not args.skip_smokes else {}
        phase7_summary = load_json(smoke_root / "phase7" / "summary.json") if not args.skip_smokes else {}

        pass_conditions = {
            "pretrained_shared_unit_model_multi_family": universal_eval_summary.get("status") == "ok"
            and len(universal_per_system_losses) >= 3
            and all(
                _extract_universal_total_loss(item) is not None
                for item in universal_per_system_losses.values()
            ),
            "small_flowsheet_graph_simulation": phase3_summary.get("status") == "ok"
            if not args.skip_smokes
            else None,
            "chemistry_or_biology_law_hooks": phase4_summary.get("status") == "ok"
            if not args.skip_smokes
            else None,
            "customer_adaptation_script_exists_and_runs": real_data_pilot_summary.get("adaptation_status") == "ok",
            "at_least_three_website_demos_run": phase6_summary.get("status") == "ok"
            and _extract_phase6_demo_count(phase6_summary) >= 3
            if not args.skip_smokes
            else None,
            "simulator_usable_by_mpc_and_rl_wrappers": phase7_summary.get("status") == "ok"
            if not args.skip_smokes
            else None,
            "rollout_stability_significantly_improved": bool(rollout_summary.get("overall_pass")),
        }

        summary.update(
            {
                "status": "ok" if all(value is True or value is None for value in pass_conditions.values()) else "failed",
                "pass_conditions": pass_conditions,
                "universal_train_summary_path": str((universal_output_dir / "summary.json").resolve()),
                "universal_eval_summary_path": str((universal_output_dir / "eval" / "summary.json").resolve()),
                "real_data_pilot_summary_path": str((outputs_root / "real_data_pilot" / "summary.json").resolve()),
                "rollout_stability_summary_path": str((outputs_root / "rollout_stability" / "summary.json").resolve()),
                "universal_best_val_loss": _json_safe_float(universal_train_summary.get("best_val_loss")),
                "universal_eval_metric": _json_safe_float(universal_eval_summary.get("aggregate_metric_value"))
                or _json_safe_float(universal_eval_summary.get("geometric_mean_per_system_total_loss")),
                "real_data_adaptation_best_val_loss": _json_safe_float(real_data_pilot_summary.get("best_val_loss")),
            }
        )
        write_json(summary_path, summary)
        if summary["status"] != "ok":
            raise MilestoneError("One or more milestone pass conditions failed")
    except Exception as exc:
        summary["status"] = "failed"
        summary["error"] = str(exc)
        write_json(summary_path, summary)
        raise

    print(f"\nV1 milestone summary: {summary_path}")


if __name__ == "__main__":
    main()
