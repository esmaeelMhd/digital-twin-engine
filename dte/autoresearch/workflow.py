"""Helpers for bounded autoresearch runs and baseline promotion."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any, Mapping


RESULTS_COLUMNS = (
    "timestamp",
    "run_id",
    "commit",
    "metric_name",
    "metric_value",
    "baseline_before",
    "training_seconds",
    "status",
    "description",
)

MAX_RUN_SLUG_LENGTH = 96


@dataclass(frozen=True)
class BaselineState:
    """Tracked state for the current promoted baseline."""

    metric_name: str
    metric_value: float
    run_id: str | None = None
    commit: str | None = None
    description: str | None = None


def slugify(text: str) -> str:
    """Create a filesystem-safe slug from freeform text."""

    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    if len(slug) > MAX_RUN_SLUG_LENGTH:
        slug = slug[:MAX_RUN_SLUG_LENGTH].rstrip("-")
    return slug or "experiment"


def make_run_id(description: str, now: datetime | None = None) -> str:
    """Create a timestamped run id."""

    timestamp = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
    return f"{timestamp}-{slugify(description)}"


def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON file."""

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write a JSON file with stable formatting."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def ensure_results_file(path: Path) -> None:
    """Create the results TSV with a header if it does not exist."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return

    with path.open("w", encoding="utf-8") as handle:
        handle.write("\t".join(RESULTS_COLUMNS) + "\n")


def append_result_row(path: Path, row: Mapping[str, str]) -> None:
    """Append a row to the results TSV."""

    ensure_results_file(path)
    values = [row.get(column, "") for column in RESULTS_COLUMNS]
    with path.open("a", encoding="utf-8") as handle:
        handle.write("\t".join(values) + "\n")


def resolve_metric_value(summary: Mapping[str, Any], metric_name: str) -> float | None:
    """Extract a numeric metric value from a summary payload."""

    raw_value = summary.get(metric_name)
    if raw_value is None:
        return None

    try:
        value = float(raw_value)
    except (TypeError, ValueError):
        return None

    if math.isnan(value) or math.isinf(value):
        return None

    return value


def load_baseline_state(baseline_dir: Path, metric_name: str) -> BaselineState | None:
    """Load the current baseline if one has been promoted."""

    metadata_path = baseline_dir / "metadata.json"
    if metadata_path.exists():
        metadata = read_json(metadata_path)
        metric_value = resolve_metric_value(metadata, "metric_value")
        if metric_value is None:
            return None

        return BaselineState(
            metric_name=metadata.get("metric_name", metric_name),
            metric_value=metric_value,
            run_id=metadata.get("run_id"),
            commit=metadata.get("commit"),
            description=metadata.get("description"),
        )

    summary_path = baseline_dir / "summary.json"
    if not summary_path.exists():
        return None

    summary = read_json(summary_path)
    metric_value = resolve_metric_value(summary, metric_name)
    if metric_value is None:
        return None

    return BaselineState(metric_name=metric_name, metric_value=metric_value)


def metric_improved(
    candidate: float,
    baseline: float | None,
    mode: str = "min",
    atol: float = 0.0,
) -> bool:
    """Return True when a candidate beats the current baseline."""

    if baseline is None:
        return True

    if mode == "min":
        return candidate < (baseline - atol)
    if mode == "max":
        return candidate > (baseline + atol)

    raise ValueError(f"Unsupported metric mode: {mode}")


def promote_baseline(
    run_dir: Path,
    baseline_dir: Path,
    summary: Mapping[str, Any],
    metric_name: str,
    description: str,
    commit: str,
) -> dict[str, Any]:
    """Promote a run into the tracked baseline directory."""

    artifacts_src = run_dir / "artifacts"
    summary_src = run_dir / "summary.json"
    log_src = run_dir / "train.log"

    if not artifacts_src.exists():
        raise FileNotFoundError(f"Missing run artifacts: {artifacts_src}")
    if not summary_src.exists():
        raise FileNotFoundError(f"Missing run summary: {summary_src}")

    baseline_dir.mkdir(parents=True, exist_ok=True)

    artifacts_dst = baseline_dir / "artifacts"
    if artifacts_dst.exists():
        shutil.rmtree(artifacts_dst)
    shutil.copytree(artifacts_src, artifacts_dst)
    shutil.copy2(summary_src, baseline_dir / "summary.json")
    if log_src.exists():
        shutil.copy2(log_src, baseline_dir / "train.log")

    metric_value = resolve_metric_value(summary, metric_name)
    if metric_value is None:
        raise ValueError(f"Summary is missing metric '{metric_name}'")

    metadata = {
        "promoted_at": datetime.now().isoformat(timespec="seconds"),
        "metric_name": metric_name,
        "metric_value": metric_value,
        "run_id": run_dir.name,
        "commit": commit,
        "description": description,
    }
    write_json(baseline_dir / "metadata.json", metadata)
    return metadata


def current_git_commit(repo_root: Path) -> str:
    """Resolve the current short git commit hash."""

    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return "unknown"

    return result.stdout.strip() or "unknown"
