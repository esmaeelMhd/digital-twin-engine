"""Run a bounded autoresearch experiment for Digital Twin Engine."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

import yaml

from dte.autoresearch.workflow import (
    append_result_row,
    current_git_commit,
    load_baseline_state,
    make_run_id,
    metric_improved,
    promote_baseline,
    read_json,
    resolve_metric_value,
    write_json,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def resolve_repo_path(path_value: str) -> Path:
    """Resolve a repo-relative path."""

    path = Path(path_value)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def load_settings(path: Path) -> dict:
    """Load autoresearch settings."""

    with path.open("r", encoding="utf-8") as handle:
        settings = yaml.safe_load(handle) or {}

    settings.setdefault("research", {})
    settings.setdefault("train", {})
    return settings


def format_optional_float(value: float | None, precision: int = 6) -> str:
    """Format optional floats for logging."""

    if value is None:
        return ""
    return f"{value:.{precision}f}"


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description="Run a bounded autoresearch experiment")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/autoresearch_default.yaml",
        help="Path to autoresearch config",
    )
    parser.add_argument(
        "--description",
        type=str,
        default="experiment",
        help="Short freeform description of the experimental change",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Training dataset directory (overrides autoresearch config)",
    )
    parser.add_argument(
        "--workspace_dir",
        type=str,
        default=None,
        help="Workspace for run logs, results, and promoted baseline",
    )
    parser.add_argument(
        "--train_config",
        type=str,
        default=None,
        help="Training config path override",
    )
    parser.add_argument(
        "--cstr_config",
        type=str,
        default=None,
        help="CSTR config path override",
    )
    parser.add_argument(
        "--time_budget_minutes",
        type=float,
        default=None,
        help="Wall-clock training budget in minutes",
    )
    parser.add_argument(
        "--n_epochs",
        type=int,
        default=None,
        help="Epoch limit passed through to scripts/train.py",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size override passed through to scripts/train.py",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed override passed through to scripts/train.py",
    )
    parser.add_argument(
        "--val_every",
        type=int,
        default=None,
        help="Validation cadence in epochs",
    )
    parser.add_argument(
        "--metric_name",
        type=str,
        default=None,
        help="Metric in summary.json to compare across runs",
    )
    parser.add_argument(
        "--metric_mode",
        type=str,
        choices=("min", "max"),
        default=None,
        help="Whether smaller or larger metric values are better",
    )
    parser.add_argument(
        "--no_promote",
        action="store_true",
        help="Do not update the promoted baseline even when the run improves",
    )
    return parser.parse_args()


def main() -> int:
    """Run one experiment and optionally promote it."""

    args = parse_args()
    description = args.description.replace("\t", " ").replace("\n", " ").strip()
    if not description:
        description = "experiment"
    settings = load_settings(resolve_repo_path(args.config))
    research_cfg = settings["research"]
    train_cfg = settings["train"]

    workspace_dir = resolve_repo_path(
        args.workspace_dir or research_cfg.get("workspace_dir", "outputs/autoresearch")
    )
    data_dir = resolve_repo_path(args.data_dir or train_cfg["data_dir"])
    train_config = resolve_repo_path(args.train_config or train_cfg["config"])
    cstr_config = resolve_repo_path(args.cstr_config or train_cfg["cstr_config"])

    time_budget_minutes = args.time_budget_minutes
    if time_budget_minutes is None:
        time_budget_minutes = float(research_cfg.get("time_budget_minutes", 5))

    n_epochs = args.n_epochs if args.n_epochs is not None else train_cfg.get("n_epochs")
    batch_size = args.batch_size if args.batch_size is not None else train_cfg.get("batch_size")
    seed = args.seed if args.seed is not None else int(train_cfg.get("seed", 42))
    val_every = args.val_every if args.val_every is not None else int(train_cfg.get("val_every", 1))
    metric_name = args.metric_name or research_cfg.get("metric_name", "best_val_loss")
    metric_mode = args.metric_mode or research_cfg.get("metric_mode", "min")
    promote_if_improved = not args.no_promote and bool(
        research_cfg.get("promote_if_improved", True)
    )
    timeout_buffer_minutes = float(research_cfg.get("hard_timeout_buffer_minutes", 5))

    run_id = make_run_id(description)
    run_dir = workspace_dir / "runs" / run_id
    artifacts_dir = run_dir / "artifacts"
    summary_path = run_dir / "summary.json"
    log_path = run_dir / "train.log"
    result_path = run_dir / "result.json"
    results_path = workspace_dir / "results.tsv"
    baseline_dir = workspace_dir / "baseline"
    run_dir.mkdir(parents=True, exist_ok=True)

    baseline_state = load_baseline_state(baseline_dir, metric_name)
    baseline_before = baseline_state.metric_value if baseline_state is not None else None
    commit = current_git_commit(REPO_ROOT)

    command = [
        sys.executable,
        str(REPO_ROOT / "scripts/train.py"),
        "--config",
        str(train_config),
        "--cstr_config",
        str(cstr_config),
        "--data_dir",
        str(data_dir),
        "--output_dir",
        str(artifacts_dir),
        "--summary_path",
        str(summary_path),
        "--seed",
        str(seed),
        "--val_every",
        str(val_every),
    ]
    if n_epochs is not None:
        command.extend(["--n_epochs", str(n_epochs)])
    if batch_size is not None:
        command.extend(["--batch_size", str(batch_size)])
    if time_budget_minutes is not None:
        command.extend(["--time_budget_minutes", str(time_budget_minutes)])

    timeout_seconds = None
    if time_budget_minutes is not None:
        timeout_seconds = int((time_budget_minutes + timeout_buffer_minutes) * 60)

    returncode = None
    timeout_error = None
    with log_path.open("w", encoding="utf-8") as log_handle:
        log_handle.write(f"run_id: {run_id}\n")
        log_handle.write(f"description: {description}\n")
        log_handle.write(f"commit: {commit}\n")
        log_handle.write(f"command: {' '.join(shlex.quote(part) for part in command)}\n\n")
        try:
            completed = subprocess.run(
                command,
                cwd=REPO_ROOT,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                check=False,
                timeout=timeout_seconds,
                text=True,
            )
            returncode = completed.returncode
        except subprocess.TimeoutExpired:
            timeout_error = f"Experiment exceeded hard timeout ({timeout_seconds}s)."
            log_handle.write(timeout_error + "\n")

    if summary_path.exists():
        try:
            summary = read_json(summary_path)
        except json.JSONDecodeError:
            summary = {}
    else:
        summary = {}
    metric_value = resolve_metric_value(summary, metric_name)
    training_seconds = resolve_metric_value(summary, "training_seconds")
    baseline_promoted = False

    if timeout_error is not None or returncode not in (0, None) or metric_value is None:
        status = "crash"
    else:
        improved = metric_improved(metric_value, baseline_before, mode=metric_mode)
        status = "keep" if improved else "discard"
        if improved and promote_if_improved:
            promote_baseline(
                run_dir=run_dir,
                baseline_dir=baseline_dir,
                summary=summary,
                metric_name=metric_name,
                description=description,
                commit=commit,
            )
            baseline_promoted = True

    result_payload = {
        "run_id": run_id,
        "description": description,
        "commit": commit,
        "metric_name": metric_name,
        "metric_value": metric_value,
        "baseline_before": baseline_before,
        "status": status,
        "baseline_promoted": baseline_promoted,
        "training_seconds": training_seconds,
        "summary_path": str(summary_path),
        "log_path": str(log_path),
        "artifacts_dir": str(artifacts_dir),
        "returncode": returncode,
        "timeout_error": timeout_error,
    }
    write_json(result_path, result_payload)

    append_result_row(
        results_path,
        {
            "timestamp": run_id[:15],
            "run_id": run_id,
            "commit": commit,
            "metric_name": metric_name,
            "metric_value": format_optional_float(metric_value),
            "baseline_before": format_optional_float(baseline_before),
            "training_seconds": format_optional_float(training_seconds, precision=2),
            "status": status,
            "description": description,
        },
    )

    print("=" * 60)
    print("AUTORESEARCH EXPERIMENT")
    print("=" * 60)
    print(f"Run ID: {run_id}")
    print(f"Commit: {commit}")
    print(f"Metric ({metric_name}): {format_optional_float(metric_value)}")
    print(f"Baseline before: {format_optional_float(baseline_before)}")
    print(f"Status: {status}")
    print(f"Baseline promoted: {'yes' if baseline_promoted else 'no'}")
    print(f"Results ledger: {results_path}")
    print(f"Run log: {log_path}")
    print(f"Run summary: {summary_path}")
    if timeout_error is not None:
        print(timeout_error)

    return 0 if status != "crash" else 1


if __name__ == "__main__":
    raise SystemExit(main())
