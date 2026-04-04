"""Run a bounded autoresearch experiment for Digital Twin Engine."""

from __future__ import annotations

import argparse
import json
import math
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

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
DEFAULT_MULTI_TARGET_METRIC = "aggregate_relative_best_val_loss"
DEFAULT_AGGREGATE_METHOD = "weighted_geometric_mean_ratio"
DEFAULT_TIME_BUDGET_MODE = "split_total"


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


def _config_defines_targets(train_cfg: dict[str, Any]) -> bool:
    """Return True when the config declares explicit benchmark targets."""

    raw_targets = train_cfg.get("targets")
    return isinstance(raw_targets, list) and len(raw_targets) > 0


def _slugify_target_name(text: str) -> str:
    """Create a stable filesystem-safe target name."""

    slug = re.sub(r"[^a-z0-9]+", "-", str(text).strip().lower()).strip("-")
    return slug or "target"


def _derive_target_name(target_cfg: dict[str, Any], system_config: Path, index: int) -> str:
    """Resolve a readable benchmark name."""

    explicit_name = target_cfg.get("name")
    if explicit_name:
        return _slugify_target_name(str(explicit_name))

    stem = system_config.stem
    for suffix in ("_default", "_training", "_config"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    stem = _slugify_target_name(stem)
    return stem or f"target-{index + 1}"


def _make_unique_target_name(name: str, seen: set[str]) -> str:
    """Ensure target names are unique within one benchmark set."""

    candidate = name
    suffix = 2
    while candidate in seen:
        candidate = f"{name}-{suffix}"
        suffix += 1
    seen.add(candidate)
    return candidate


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
        "--system_config",
        "--cstr_config",  # backward-compatible alias
        type=str,
        default=None,
        dest="system_config",
        help="System config path override (CSTR, heat exchanger, etc.)",
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


def _resolve_single_target(
    args: argparse.Namespace,
    train_cfg: dict[str, Any],
    default_train_config: Path,
) -> list[dict[str, Any]]:
    """Resolve the legacy single-target training configuration."""

    data_dir_value = args.data_dir or train_cfg.get("data_dir")
    if not data_dir_value:
        raise ValueError("Single-target autoresearch config must define train.data_dir.")

    system_config_value = (
        args.system_config
        or train_cfg.get("system_config")
        or train_cfg.get("cstr_config")
        or "configs/cstr_default.yaml"
    )
    system_config = resolve_repo_path(str(system_config_value))
    return [
        {
            "name": _derive_target_name({}, system_config, 0),
            "weight": 1.0,
            "data_dir": resolve_repo_path(str(data_dir_value)),
            "system_config": system_config,
            "train_config": default_train_config,
        }
    ]


def resolve_train_targets(
    args: argparse.Namespace,
    train_cfg: dict[str, Any],
    default_train_config: Path,
) -> list[dict[str, Any]]:
    """Resolve one or more benchmark targets from config or CLI overrides."""

    # CLI overrides preserve the old single-target behavior.
    if args.data_dir is not None or args.system_config is not None:
        return _resolve_single_target(args, train_cfg, default_train_config)

    raw_targets = train_cfg.get("targets")
    if not isinstance(raw_targets, list) or not raw_targets:
        return _resolve_single_target(args, train_cfg, default_train_config)

    resolved_targets: list[dict[str, Any]] = []
    seen_names: set[str] = set()

    for index, raw_target in enumerate(raw_targets):
        if not isinstance(raw_target, dict):
            raise ValueError(f"train.targets[{index}] must be a mapping.")

        data_dir_value = raw_target.get("data_dir")
        system_config_value = raw_target.get("system_config") or raw_target.get("cstr_config")
        if not data_dir_value or not system_config_value:
            raise ValueError(
                f"train.targets[{index}] must define both data_dir and system_config."
            )

        system_config = resolve_repo_path(str(system_config_value))
        name = _make_unique_target_name(
            _derive_target_name(raw_target, system_config, index),
            seen_names,
        )
        weight = float(raw_target.get("weight", 1.0))
        if weight <= 0.0:
            raise ValueError(f"train.targets[{index}].weight must be > 0.")

        train_config_value = (
            raw_target.get("config")
            or raw_target.get("train_config")
            or str(default_train_config)
        )
        resolved_targets.append(
            {
                "name": name,
                "weight": weight,
                "data_dir": resolve_repo_path(str(data_dir_value)),
                "system_config": system_config,
                "train_config": resolve_repo_path(str(train_config_value)),
            }
        )

    return resolved_targets


def _allocate_target_time_budgets(
    total_minutes: float | None,
    targets: list[dict[str, Any]],
    mode: str,
) -> list[float | None]:
    """Allocate wall-clock budgets across benchmark targets."""

    if total_minutes is None:
        return [None] * len(targets)

    if len(targets) <= 1 or mode == "per_target":
        return [total_minutes] * len(targets)

    if mode != "split_total":
        raise ValueError(
            f"Unsupported autoresearch time_budget_mode '{mode}'. "
            "Expected 'split_total' or 'per_target'."
        )

    total_weight = sum(float(target["weight"]) for target in targets)
    if total_weight <= 0.0:
        raise ValueError("Total target weight must be > 0.")
    return [
        total_minutes * float(target["weight"]) / total_weight
        for target in targets
    ]


def _reference_metrics_path(workspace_dir: Path) -> Path:
    """Return the fixed cross-system reference metrics path."""

    return workspace_dir / "reference_metrics.json"


def _load_reference_metrics(workspace_dir: Path) -> dict[str, Any] | None:
    """Load the fixed aggregate-score reference metrics, if present."""

    path = _reference_metrics_path(workspace_dir)
    if not path.exists():
        return None
    try:
        payload = read_json(path)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _write_reference_metrics(workspace_dir: Path, payload: dict[str, Any]) -> None:
    """Persist the fixed aggregate-score reference metrics."""

    write_json(_reference_metrics_path(workspace_dir), payload)


def _initialize_reference_metrics(
    workspace_dir: Path,
    run_id: str,
    metric_name: str,
    benchmark_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Create the fixed reference metrics from the first successful multi-system run."""

    payload = {
        "created_from_run_id": run_id,
        "aggregate_metric_name": metric_name,
        "targets": [
            {
                "name": benchmark["name"],
                "weight": benchmark["weight"],
                "system_config": benchmark["system_config"],
                "data_dir": benchmark["data_dir"],
                "reference_metric_name": benchmark["metric_name"],
                "reference_metric": benchmark["metric_value"],
            }
            for benchmark in benchmark_results
        ],
    }
    _write_reference_metrics(workspace_dir, payload)
    return payload


def _validate_reference_metrics(
    reference_payload: dict[str, Any],
    benchmark_results: list[dict[str, Any]],
) -> str | None:
    """Check that the fixed reference matches the configured benchmark targets."""

    reference_targets = reference_payload.get("targets")
    if not isinstance(reference_targets, list) or not reference_targets:
        return "reference_metrics.json is missing its target list."

    current_names = {benchmark["name"] for benchmark in benchmark_results}
    reference_names = {
        str(target.get("name", "")).strip()
        for target in reference_targets
        if str(target.get("name", "")).strip()
    }
    if current_names != reference_names:
        return (
            "Configured benchmark targets do not match reference_metrics.json. "
            "Start a new workspace or refresh the reference metrics."
        )

    for target in reference_targets:
        reference_metric = resolve_metric_value(target, "reference_metric")
        if reference_metric is None or reference_metric <= 0.0:
            return (
                f"reference_metrics.json has an invalid reference metric for target "
                f"'{target.get('name', 'unknown')}'."
            )
    return None


def _compute_aggregate_relative_metric(
    benchmark_results: list[dict[str, Any]],
    reference_payload: dict[str, Any],
    aggregate_method: str,
) -> tuple[float, list[dict[str, Any]]]:
    """Aggregate per-system benchmark metrics into one fixed-reference score."""

    if aggregate_method != DEFAULT_AGGREGATE_METHOD:
        raise ValueError(
            f"Unsupported aggregate_method '{aggregate_method}'. "
            f"Expected '{DEFAULT_AGGREGATE_METHOD}'."
        )

    reference_map = {
        str(target["name"]): target
        for target in reference_payload.get("targets", [])
        if isinstance(target, dict) and target.get("name")
    }

    weighted_log_sum = 0.0
    total_weight = 0.0
    enriched_results: list[dict[str, Any]] = []

    for benchmark in benchmark_results:
        target_name = benchmark["name"]
        reference_target = reference_map[target_name]
        reference_metric = float(reference_target["reference_metric"])
        candidate_metric = float(benchmark["metric_value"])
        relative_metric = candidate_metric / reference_metric
        weight = float(benchmark["weight"])
        weighted_log_sum += weight * math.log(max(relative_metric, 1e-12))
        total_weight += weight

        benchmark_enriched = dict(benchmark)
        benchmark_enriched["reference_metric"] = reference_metric
        benchmark_enriched["relative_metric"] = relative_metric
        enriched_results.append(benchmark_enriched)

    aggregate_metric = math.exp(weighted_log_sum / total_weight)
    return aggregate_metric, enriched_results


def _run_train_for_target(
    *,
    target: dict[str, Any],
    run_dir: Path,
    single_target_mode: bool,
    log_handle,
    seed: int,
    val_every: int,
    n_epochs: int | None,
    batch_size: int | None,
    time_budget_minutes: float | None,
    timeout_buffer_minutes: float,
) -> dict[str, Any]:
    """Run one target benchmark and return its raw result payload."""

    target_name = target["name"]
    if single_target_mode:
        artifacts_dir = run_dir / "artifacts"
        summary_path = run_dir / "summary.json"
    else:
        artifacts_dir = run_dir / "artifacts" / target_name
        summary_path = run_dir / "benchmarks" / target_name / "summary.json"

    command = [
        sys.executable,
        str(REPO_ROOT / "scripts/train.py"),
        "--config",
        str(target["train_config"]),
        "--system_config",
        str(target["system_config"]),
        "--data_dir",
        str(target["data_dir"]),
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

    log_handle.write("\n" + "-" * 60 + "\n")
    log_handle.write(f"benchmark: {target_name}\n")
    log_handle.write(f"system_config: {target['system_config']}\n")
    log_handle.write(f"data_dir: {target['data_dir']}\n")
    log_handle.write(f"train_config: {target['train_config']}\n")
    log_handle.write(f"time_budget_minutes: {format_optional_float(time_budget_minutes, precision=3)}\n")
    log_handle.write(f"command: {' '.join(shlex.quote(part) for part in command)}\n\n")
    log_handle.flush()

    returncode = None
    timeout_error = None
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
        timeout_error = f"Benchmark '{target_name}' exceeded hard timeout ({timeout_seconds}s)."
        log_handle.write(timeout_error + "\n")
        log_handle.flush()

    if summary_path.exists():
        try:
            summary = read_json(summary_path)
        except json.JSONDecodeError:
            summary = {}
    else:
        summary = {}

    metric_name = "best_val_loss"
    metric_value = resolve_metric_value(summary, metric_name)
    training_seconds = resolve_metric_value(summary, "training_seconds")
    benchmark_status = (
        "crash"
        if timeout_error is not None or returncode not in (0, None) or metric_value is None
        else "ok"
    )

    return {
        "name": target_name,
        "weight": float(target["weight"]),
        "metric_name": metric_name,
        "metric_value": metric_value,
        "training_seconds": training_seconds,
        "status": benchmark_status,
        "summary": summary,
        "summary_path": str(summary_path),
        "artifacts_dir": str(artifacts_dir),
        "system_config": str(target["system_config"]),
        "data_dir": str(target["data_dir"]),
        "train_config": str(target["train_config"]),
        "time_budget_minutes": time_budget_minutes,
        "returncode": returncode,
        "timeout_error": timeout_error,
    }


def _build_multi_target_summary(
    *,
    metric_name: str,
    aggregate_metric: float | None,
    aggregate_method: str,
    benchmark_results: list[dict[str, Any]],
    time_budget_minutes: float | None,
    time_budget_mode: str,
    failure_reason: str | None,
) -> dict[str, Any]:
    """Build the aggregate summary written to run_dir/summary.json."""

    total_training_seconds = 0.0
    have_training_seconds = False
    for benchmark in benchmark_results:
        training_seconds = benchmark.get("training_seconds")
        if isinstance(training_seconds, (int, float)):
            total_training_seconds += float(training_seconds)
            have_training_seconds = True

    benchmark_payloads = []
    for benchmark in benchmark_results:
        summary = benchmark.get("summary") if isinstance(benchmark.get("summary"), dict) else {}
        benchmark_payloads.append(
            {
                "name": benchmark["name"],
                "weight": benchmark["weight"],
                "metric_name": benchmark["metric_name"],
                "metric_value": benchmark["metric_value"],
                "reference_metric": benchmark.get("reference_metric"),
                "relative_metric": benchmark.get("relative_metric"),
                "training_seconds": benchmark["training_seconds"],
                "status": benchmark["status"],
                "summary_path": benchmark["summary_path"],
                "artifacts_dir": benchmark["artifacts_dir"],
                "system_config": benchmark["system_config"],
                "data_dir": benchmark["data_dir"],
                "train_config": benchmark["train_config"],
                "time_budget_minutes": benchmark.get("time_budget_minutes"),
                "epochs_completed": summary.get("epochs_completed"),
                "steps_completed": summary.get("steps_completed"),
                "timed_out": summary.get("timed_out"),
                "failure_reason": summary.get("failure_reason"),
                "non_finite_detected": summary.get("non_finite_detected"),
                "final_train_loss": summary.get("final_train_loss"),
                "final_val_loss": summary.get("final_val_loss"),
            }
        )

    summary = {
        metric_name: aggregate_metric,
        DEFAULT_MULTI_TARGET_METRIC: aggregate_metric,
        "aggregate_method": aggregate_method,
        "benchmark_count": len(benchmark_payloads),
        "training_seconds": total_training_seconds if have_training_seconds else None,
        "time_budget_minutes": time_budget_minutes,
        "time_budget_mode": time_budget_mode,
        "failure_reason": failure_reason,
        "benchmarks": benchmark_payloads,
    }
    return summary


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
    default_train_config = resolve_repo_path(args.train_config or train_cfg["config"])
    targets = resolve_train_targets(args, train_cfg, default_train_config)
    single_target_mode = len(targets) == 1

    time_budget_minutes = args.time_budget_minutes
    if time_budget_minutes is None:
        time_budget_minutes = float(research_cfg.get("time_budget_minutes", 5))

    n_epochs = args.n_epochs if args.n_epochs is not None else train_cfg.get("n_epochs")
    batch_size = args.batch_size if args.batch_size is not None else train_cfg.get("batch_size")
    seed = args.seed if args.seed is not None else int(train_cfg.get("seed", 42))
    val_every = args.val_every if args.val_every is not None else int(train_cfg.get("val_every", 1))

    if args.metric_name is not None:
        metric_name = args.metric_name
    elif single_target_mode and _config_defines_targets(train_cfg):
        # CLI overrides can force legacy single-target mode even when the default
        # config is multi-target. In that case the aggregate metric is invalid.
        metric_name = "best_val_loss"
    else:
        metric_name = research_cfg.get(
            "metric_name",
            "best_val_loss" if single_target_mode else DEFAULT_MULTI_TARGET_METRIC,
        )
    metric_mode = args.metric_mode or research_cfg.get("metric_mode", "min")
    promote_if_improved = not args.no_promote and bool(
        research_cfg.get("promote_if_improved", True)
    )
    timeout_buffer_minutes = float(research_cfg.get("hard_timeout_buffer_minutes", 5))

    time_budget_mode = str(
        research_cfg.get(
            "time_budget_mode",
            "per_target" if single_target_mode else DEFAULT_TIME_BUDGET_MODE,
        )
    ).strip().lower()
    aggregate_method = str(
        research_cfg.get("aggregate_method", DEFAULT_AGGREGATE_METHOD)
    ).strip().lower()
    target_budgets = _allocate_target_time_budgets(
        time_budget_minutes,
        targets,
        time_budget_mode,
    )

    run_id = make_run_id(description)
    run_dir = workspace_dir / "runs" / run_id
    summary_path = run_dir / "summary.json"
    log_path = run_dir / "train.log"
    result_path = run_dir / "result.json"
    results_path = workspace_dir / "results.tsv"
    baseline_dir = workspace_dir / "baseline"
    run_dir.mkdir(parents=True, exist_ok=True)

    reference_payload = None if single_target_mode else _load_reference_metrics(workspace_dir)
    if single_target_mode:
        baseline_state = load_baseline_state(baseline_dir, metric_name)
        baseline_before = baseline_state.metric_value if baseline_state is not None else None
    else:
        # Aggregate metrics are only comparable once the fixed reference exists.
        baseline_state = (
            load_baseline_state(baseline_dir, metric_name) if reference_payload is not None else None
        )
        baseline_before = baseline_state.metric_value if baseline_state is not None else None

    commit = current_git_commit(REPO_ROOT)

    benchmark_results: list[dict[str, Any]] = []
    with log_path.open("w", encoding="utf-8") as log_handle:
        log_handle.write(f"run_id: {run_id}\n")
        log_handle.write(f"description: {description}\n")
        log_handle.write(f"commit: {commit}\n")
        log_handle.write(f"metric_name: {metric_name}\n")
        log_handle.write(f"metric_mode: {metric_mode}\n")
        log_handle.write(f"time_budget_mode: {time_budget_mode}\n")
        log_handle.write(f"aggregate_method: {aggregate_method}\n")
        log_handle.write(f"targets: {json.dumps([target['name'] for target in targets])}\n")
        log_handle.flush()

        for target, target_budget in zip(targets, target_budgets):
            benchmark_results.append(
                _run_train_for_target(
                    target=target,
                    run_dir=run_dir,
                    single_target_mode=single_target_mode,
                    log_handle=log_handle,
                    seed=seed,
                    val_every=val_every,
                    n_epochs=n_epochs,
                    batch_size=batch_size,
                    time_budget_minutes=target_budget,
                    timeout_buffer_minutes=timeout_buffer_minutes,
                )
            )

    failure_reason = None
    metric_value = None
    training_seconds = None
    summary: dict[str, Any]

    if single_target_mode:
        benchmark = benchmark_results[0]
        summary = benchmark["summary"] if isinstance(benchmark["summary"], dict) else {}
        if summary:
            write_json(summary_path, summary)
        metric_value = resolve_metric_value(summary, metric_name)
        training_seconds = resolve_metric_value(summary, "training_seconds")
        if benchmark["status"] == "crash" or metric_value is None:
            failure_reason = (
                benchmark["timeout_error"]
                or summary.get("failure_reason")
                or "single-target benchmark failed"
            )
    else:
        crashed_benchmarks = [benchmark for benchmark in benchmark_results if benchmark["status"] != "ok"]
        if crashed_benchmarks:
            failed_names = ", ".join(benchmark["name"] for benchmark in crashed_benchmarks)
            failure_reason = f"One or more benchmark targets failed: {failed_names}"
        elif reference_payload is not None:
            reference_error = _validate_reference_metrics(reference_payload, benchmark_results)
            if reference_error is not None:
                failure_reason = reference_error

        if failure_reason is None:
            if reference_payload is None:
                reference_payload = _initialize_reference_metrics(
                    workspace_dir,
                    run_id,
                    metric_name,
                    benchmark_results,
                )
            metric_value, benchmark_results = _compute_aggregate_relative_metric(
                benchmark_results,
                reference_payload,
                aggregate_method,
            )

        summary = _build_multi_target_summary(
            metric_name=metric_name,
            aggregate_metric=metric_value,
            aggregate_method=aggregate_method,
            benchmark_results=benchmark_results,
            time_budget_minutes=time_budget_minutes,
            time_budget_mode=time_budget_mode,
            failure_reason=failure_reason,
        )
        write_json(summary_path, summary)
        training_seconds = resolve_metric_value(summary, "training_seconds")

    baseline_promoted = False
    if failure_reason is not None or metric_value is None:
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
        "artifacts_dir": str(run_dir / "artifacts"),
        "failure_reason": failure_reason,
        "benchmarks": benchmark_results,
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
    if failure_reason:
        print(f"failure_reason: {failure_reason}")

    if not single_target_mode:
        print("Benchmark breakdown:")
        for benchmark in benchmark_results:
            metric_text = format_optional_float(benchmark.get("metric_value"))
            rel_text = format_optional_float(benchmark.get("relative_metric"))
            print(
                f"  - {benchmark['name']}: best_val_loss={metric_text} "
                f"relative={rel_text} status={benchmark['status']}"
            )

    return 0 if status != "crash" else 1


if __name__ == "__main__":
    raise SystemExit(main())
