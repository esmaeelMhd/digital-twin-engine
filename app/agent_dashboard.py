"""Streamlit dashboard for monitoring autoresearch agent runs."""

from __future__ import annotations

import os
import re
import shlex
import subprocess
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
import json

import plotly.graph_objects as go
import streamlit as st
import yaml

from app._theme import inject_theme


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AUTORESEARCH_CONFIG = PROJECT_ROOT / "configs" / "autoresearch_default.yaml"
STATE_FILE = PROJECT_ROOT / "agent_state.json"
LOG_FILE = PROJECT_ROOT / "outputs" / "autoresearch_logs" / "agent.log"
LEGACY_LOG_FILE = PROJECT_ROOT / "agent.log"
CAMPAIGN_RUNNER_DIR = PROJECT_ROOT / "outputs" / "campaign_runner"
IDEA_MARKER_RE = re.compile(r"\[idea:([a-zA-Z0-9_.-]+)\]")

_STATUS_COLOR = {
    "keep": "#16a34a",
    "crash": "#dc2626",
    "discard": "#6b7280",
    "timeout": "#d97706",
}


# ---------------------------------------------------------------------------
# Data helpers (all cached)
# ---------------------------------------------------------------------------

def _list_config_options() -> list[Path]:
    config_dir = PROJECT_ROOT / "configs"
    candidates = sorted(config_dir.glob("autoresearch*.yaml"))
    if DEFAULT_AUTORESEARCH_CONFIG not in candidates and DEFAULT_AUTORESEARCH_CONFIG.exists():
        candidates.insert(0, DEFAULT_AUTORESEARCH_CONFIG)
    return candidates


def _resolve_config_path() -> Path:
    state = _read_json_file(STATE_FILE) or {}

    requested = st.query_params.get("config") or os.environ.get("AUTORESEARCH_DASHBOARD_CONFIG")
    if requested:
        candidate = Path(str(requested))
        if not candidate.is_absolute():
            candidate = PROJECT_ROOT / candidate
        if candidate.exists():
            return candidate

    state_config = state.get("config_path")
    if state_config:
        candidate = Path(str(state_config))
        if not candidate.is_absolute():
            candidate = PROJECT_ROOT / candidate
        if candidate.exists():
            return candidate

    for process in _agent_processes():
        candidate = _config_from_agent_command(process.get("command", ""))
        if candidate and candidate.exists():
            return candidate

    config_options = _list_config_options()
    for candidate in reversed(config_options):
        try:
            config = _load_config(candidate)
            if _results_path(config).exists():
                return candidate
        except Exception:
            continue

    return DEFAULT_AUTORESEARCH_CONFIG


@st.cache_data(ttl=30)
def _load_config(config_path: Path) -> dict:
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}
    except Exception:
        return {}


def _workspace_dir(config: dict) -> Path:
    workspace_value = config.get("research", {}).get("workspace_dir", "outputs/autoresearch")
    workspace_dir = Path(workspace_value)
    if not workspace_dir.is_absolute():
        workspace_dir = PROJECT_ROOT / workspace_dir
    return workspace_dir


def _results_path(config: dict) -> Path:
    return _workspace_dir(config) / "results.tsv"


def _baseline_metadata_path(config: dict) -> Path:
    return _workspace_dir(config) / "baseline" / "metadata.json"


def _read_json_file(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return None


def _parse_iso_datetime(raw_value: str | None) -> datetime | None:
    if not raw_value:
        return None
    try:
        return datetime.fromisoformat(str(raw_value))
    except ValueError:
        return None


def _parse_results_timestamp(raw_value: str | None) -> datetime | None:
    if not raw_value:
        return None
    try:
        return datetime.strptime(str(raw_value), "%Y%m%d-%H%M%S")
    except ValueError:
        return None


def _parse_metric(raw_value: str) -> float | None:
    raw_value = (raw_value or "").strip()
    if raw_value in ("", "0", "0.0", "0.000000"):
        return None
    try:
        value = float(raw_value)
    except ValueError:
        return None
    if value <= 0.0:
        return None
    return value


def _decode_description(description: str) -> tuple[str, str]:
    description = (description or "").strip()
    if description.startswith("[") and "] " in description:
        file_changed, remainder = description[1:].split("] ", 1)
        return file_changed, remainder
    return "", description


def _idea_label(description: str) -> str | None:
    match = IDEA_MARKER_RE.search(description or "")
    return match.group(1) if match else None


def _display_description(description: str) -> str:
    idea_id = _idea_label(description)
    if idea_id:
        return idea_id
    return " ".join((description or "").split())


@st.cache_data(ttl=5)
def _load_results(results_path_str: str) -> list[dict]:
    path = Path(results_path_str)
    if not path.exists():
        return []

    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        header = handle.readline().rstrip("\n").split("\t")
        column_index = {name: idx for idx, name in enumerate(header)}
        new_format = (
            "metric_value" in column_index
            and "status" in column_index
            and "description" in column_index
        )

        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if not parts or all(not part for part in parts):
                continue

            if new_format:
                def get_col(name: str, default: str = "") -> str:
                    idx = column_index.get(name)
                    if idx is None or idx >= len(parts):
                        return default
                    return parts[idx]

                file_changed, description = _decode_description(get_col("description"))
                rows.append(
                    {
                        "timestamp": get_col("timestamp"),
                        "run_id": get_col("run_id"),
                        "commit": get_col("commit", "-------") or "-------",
                        "status": get_col("status", "crash") or "crash",
                        "file": file_changed,
                        "description": description,
                        "val_loss": _parse_metric(get_col("metric_value")),
                        "training_seconds": get_col("training_seconds"),
                    }
                )
                continue

            if len(parts) >= 5:
                rows.append(
                    {
                        "timestamp": "",
                        "run_id": "",
                        "commit": parts[0],
                        "status": parts[2],
                        "file": parts[3],
                        "description": parts[4],
                        "val_loss": _parse_metric(parts[1]),
                        "training_seconds": "",
                    }
                )

    return rows


def _campaign_runner_summary_paths() -> list[Path]:
    if not CAMPAIGN_RUNNER_DIR.exists():
        return []
    return sorted(CAMPAIGN_RUNNER_DIR.glob("*/summary.json"))


@st.cache_data(ttl=10)
def _load_campaign_runner_session(summary_path_str: str) -> dict | None:
    summary_path = Path(summary_path_str)
    summary = _read_json_file(summary_path)
    if not summary:
        return None

    aggregated_rows: list[dict] = []
    campaign_rows: list[dict] = []
    for campaign in summary.get("campaigns", []):
        config_path = Path(str(campaign.get("config_path", "")))
        if not config_path.is_absolute():
            config_path = PROJECT_ROOT / config_path
        config = _load_config(config_path)
        rows = _load_results(str(_results_path(config)))
        started_at = _parse_iso_datetime(campaign.get("started_at"))
        finished_at = _parse_iso_datetime(campaign.get("finished_at"))

        session_rows: list[dict] = []
        for row in rows:
            row_ts = _parse_results_timestamp(row.get("timestamp"))
            if started_at and row_ts and row_ts < started_at:
                continue
            if finished_at and row_ts and row_ts > finished_at:
                continue
            enriched = dict(row)
            enriched["campaign"] = str(campaign.get("campaign", ""))
            enriched["stage"] = str(campaign.get("stage", ""))
            enriched["branch"] = str(campaign.get("branch", ""))
            enriched["merged_into_main"] = bool(campaign.get("merged_into_main", False))
            enriched["_parsed_timestamp"] = row_ts
            session_rows.append(enriched)

        session_rows.sort(
            key=lambda row: (
                row.get("_parsed_timestamp") or datetime.min,
                str(row.get("run_id", "")),
            )
        )
        aggregated_rows.extend(session_rows)

        valid_losses = [row["val_loss"] for row in session_rows if row["val_loss"] is not None]
        campaign_rows.append(
            {
                "campaign": campaign.get("campaign", ""),
                "stage": campaign.get("stage", ""),
                "runs": len(session_rows),
                "keeps": sum(1 for row in session_rows if row["status"] == "keep"),
                "crashes": sum(1 for row in session_rows if row["status"] == "crash"),
                "best_val_loss": min(valid_losses) if valid_losses else None,
                "merged_into_main": bool(campaign.get("merged_into_main", False)),
                "branch": campaign.get("branch", ""),
            }
        )

    aggregated_rows.sort(
        key=lambda row: (
            row.get("_parsed_timestamp") or datetime.min,
            str(row.get("campaign", "")),
            str(row.get("run_id", "")),
        )
    )
    for row in aggregated_rows:
        row.pop("_parsed_timestamp", None)

    return {
        "summary": summary,
        "rows": aggregated_rows,
        "campaign_rows": campaign_rows,
    }


def _tail_lines(path: Path, limit: int = 120) -> list[str]:
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            return list(deque(handle, maxlen=limit))
    except Exception:
        return []


@st.cache_data(ttl=5)
def _tail_log_lines(limit: int = 80) -> list[str]:
    lines = _tail_lines(LOG_FILE, limit=limit)
    if lines:
        return lines
    return _tail_lines(LEGACY_LOG_FILE, limit=limit)


@st.cache_data(ttl=5)
def _agent_processes() -> list[dict]:
    try:
        result = subprocess.run(
            ["pgrep", "-af", "scripts/agent.py"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception:
        return []

    processes: list[dict] = []
    current_pid = os.getpid()

    for line in result.stdout.splitlines():
        parts = line.strip().split(maxsplit=1)
        if not parts:
            continue
        try:
            pid = int(parts[0])
        except ValueError:
            continue
        command = parts[1] if len(parts) > 1 else ""
        if pid == current_pid:
            continue
        if "pgrep -af scripts/agent.py" in command:
            continue
        processes.append({"pid": pid, "command": command})

    return processes


def _config_from_agent_command(command: str) -> Path | None:
    try:
        argv = shlex.split(command)
    except ValueError:
        argv = command.split()

    for index, token in enumerate(argv):
        if token == "--config" and index + 1 < len(argv):
            candidate = Path(argv[index + 1])
            if not candidate.is_absolute():
                candidate = PROJECT_ROOT / candidate
            return candidate
        if token.startswith("--config="):
            candidate = Path(token.split("=", 1)[1])
            if not candidate.is_absolute():
                candidate = PROJECT_ROOT / candidate
            return candidate
    return None


def _max_runs_from_agent_command(command: str) -> int | None:
    try:
        argv = shlex.split(command)
    except ValueError:
        argv = command.split()

    for index, token in enumerate(argv):
        raw_value = None
        if token == "--max-runs" and index + 1 < len(argv):
            raw_value = argv[index + 1]
        elif token.startswith("--max-runs="):
            raw_value = token.split("=", 1)[1]
        if raw_value is None:
            continue
        try:
            return int(raw_value)
        except ValueError:
            return None
    return None


@st.cache_data(ttl=5)
def _gpu_stats() -> dict:
    try:
        from pynvml import (
            NVML_TEMPERATURE_GPU,
            nvmlDeviceGetHandleByIndex,
            nvmlDeviceGetMemoryInfo,
            nvmlDeviceGetName,
            nvmlDeviceGetTemperature,
            nvmlDeviceGetUtilizationRates,
            nvmlInit,
        )

        nvmlInit()
        handle = nvmlDeviceGetHandleByIndex(0)
        mem = nvmlDeviceGetMemoryInfo(handle)
        util = nvmlDeviceGetUtilizationRates(handle)
        raw_name = nvmlDeviceGetName(handle)
        gpu_name = raw_name.decode() if isinstance(raw_name, bytes) else str(raw_name)
        return {
            "name": gpu_name,
            "temp": nvmlDeviceGetTemperature(handle, NVML_TEMPERATURE_GPU),
            "gpu_util": util.gpu,
            "vram_used_mb": mem.used / (1024 * 1024),
            "vram_total_mb": mem.total / (1024 * 1024),
        }
    except Exception:
        return {
            "name": "GPU unavailable",
            "temp": None,
            "gpu_util": 0,
            "vram_used_mb": 0.0,
            "vram_total_mb": 0.0,
        }


# ---------------------------------------------------------------------------
# Chart helpers
# ---------------------------------------------------------------------------

def _loss_chart_series(results: list[dict]) -> tuple[list[float], list[float]]:
    valid_losses = [row["val_loss"] for row in results if row["val_loss"] is not None]
    best_so_far: list[float] = []
    running_best = float("inf")
    for value in valid_losses:
        running_best = min(running_best, value)
        best_so_far.append(running_best)
    return valid_losses, best_so_far


def _metric_points(results: list[dict]) -> list[dict]:
    points: list[dict] = []
    running_best = float("inf")
    for index, row in enumerate(results, start=1):
        value = row["val_loss"]
        if value is None:
            continue
        running_best = min(running_best, value)
        points.append(
            {
                "x": index,
                "val_loss": value,
                "best_so_far": running_best,
                "status": row["status"],
                "description": row["description"],
            }
        )
    return points


def _quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = max(0.0, min(1.0, q)) * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _high_outlier_cutoff(values: list[float]) -> float | None:
    """Return a robust display cutoff for unusually large validation losses."""
    if len(values) < 5:
        return None
    q1 = _quantile(values, 0.25)
    q3 = _quantile(values, 0.75)
    if q1 is None or q3 is None or q3 < q1:
        return None
    iqr = q3 - q1
    if iqr <= 0.0:
        return None
    return q3 + 3.0 * iqr


def _short_label(text: str, limit: int = 40) -> str:
    clean = _display_description(text)
    if len(clean) <= limit:
        return clean
    return clean[: limit - 3].rstrip() + "..."


def _render_validation_trend_plotly(
    results: list[dict],
    *,
    title_prefix: str = "Validation Trend",
    keep_only: bool | None = None,
    hide_high_outliers: bool = True,
) -> None:
    points = _metric_points(results)
    if not points:
        st.info("No validation-loss points logged yet.")
        return

    keep_only_points = [point for point in points if point["status"] == "keep"]
    showing_keep_only = bool(keep_only_points) if keep_only is None else (keep_only and bool(keep_only_points))
    visible_points = keep_only_points if showing_keep_only else points
    hidden_points: list[dict] = []
    cutoff = None
    if hide_high_outliers and not showing_keep_only:
        cutoff = _high_outlier_cutoff([point["val_loss"] for point in visible_points])
    if cutoff is not None:
        visible_points = [point for point in points if point["val_loss"] <= cutoff]
        hidden_points = [point for point in points if point["val_loss"] > cutoff]
        if not visible_points:
            visible_points = keep_only_points if showing_keep_only else points
            hidden_points = []
            cutoff = None

    x_vals = [point["x"] for point in visible_points]
    val_losses = [point["val_loss"] for point in visible_points]
    best_losses = [point["best_so_far"] for point in visible_points]
    keep_points = [point for point in visible_points if point["status"] == "keep"]

    fig = go.Figure()

    line_label = "Kept val loss" if showing_keep_only else "Val loss"
    fig.add_trace(go.Scatter(
        x=x_vals,
        y=val_losses,
        mode="lines+markers",
        name=line_label,
        line=dict(color="#4C78A8", width=2),
        marker=dict(size=5),
        hovertemplate="Exp %{x}<br>Val loss: %{y:.6f}<extra></extra>",
    ))

    if not showing_keep_only:
        fig.add_trace(go.Scatter(
            x=x_vals,
            y=best_losses,
            mode="lines",
            name="Best so far",
            line=dict(color="#54A24B", width=2, dash="dash"),
            hovertemplate="Exp %{x}<br>Best so far: %{y:.6f}<extra></extra>",
        ))

    if keep_points:
        fig.add_trace(go.Scatter(
            x=[p["x"] for p in keep_points],
            y=[p["val_loss"] for p in keep_points],
            mode="markers+text",
            name="Kept improvement",
            marker=dict(color="#E45756", size=10, symbol="star"),
            text=[_short_label(p["description"]) for p in keep_points],
            textposition="top center",
            textfont=dict(size=9),
            hovertemplate="Exp %{x}<br>Val loss: %{y:.6f}<br>%{text}<extra>kept</extra>",
        ))

    title = title_prefix
    if showing_keep_only:
        title += " (kept changes only)"
    if hidden_points:
        title += f" · {len(hidden_points)} outlier{'s' if len(hidden_points) != 1 else ''} hidden"

    fig.update_layout(
        title=title,
        template="plotly_white",
        height=400,
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        xaxis_title="Experiment #",
        yaxis_title="Validation Loss",
        hovermode="x unified",
    )

    st.plotly_chart(fig, use_container_width=True)

    caption_bits: list[str] = []
    if showing_keep_only:
        caption_bits.append("Showing baseline plus kept improvements only.")
    if keep_points:
        caption_bits.append("Stars mark kept improvements.")
    if hidden_points and cutoff is not None:
        caption_bits.append(f"High outliers above {cutoff:.3f} are hidden.")
    if caption_bits:
        st.caption(" ".join(caption_bits))


def _render_baseline_details(baseline_payload: dict | None) -> None:
    if not baseline_payload:
        st.info("No baseline metadata found yet.")
        return

    metric_value = baseline_payload.get("metric_value", baseline_payload.get("best_val_loss"))
    commit = str(baseline_payload.get("commit", "—"))[:7]
    run_id = str(baseline_payload.get("run_id", "—"))
    promoted_at = str(baseline_payload.get("promoted_at", "—"))
    description = str(baseline_payload.get("description", "")).strip()

    col1, col2 = st.columns(2)
    col1.metric("Baseline Val Loss", f"{metric_value:.6f}" if isinstance(metric_value, (int, float)) else "—")
    col2.metric("Commit", commit or "—")
    st.caption(f"Run ID: `{run_id}`")
    st.caption(f"Promoted: `{promoted_at}`")
    if description:
        st.write(description)
    with st.expander("Full baseline details"):
        st.json(baseline_payload)


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Autoresearch Monitor",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_theme()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

config_path = _resolve_config_path()
campaign_runner_summaries = _campaign_runner_summary_paths()

with st.sidebar:
    config_options = _list_config_options()
    option_labels = [path.stem for path in config_options]
    full_labels = [str(path.relative_to(PROJECT_ROOT)) for path in config_options]
    current_label_short = config_path.stem
    default_idx = option_labels.index(current_label_short) if current_label_short in option_labels else 0

    selected_short = st.selectbox(
        "Autoresearch config",
        option_labels,
        index=default_idx,
        help="Full path shown in workspace info below",
    )
    selected_idx = option_labels.index(selected_short)
    selected_config_path = config_options[selected_idx]
    if selected_config_path != config_path:
        st.query_params["config"] = full_labels[selected_idx]
        st.rerun()

    selected_session_summary = None
    selected_session_label = ""
    query_session = str(st.query_params.get("session") or "")
    query_view_mode = str(st.query_params.get("view") or "Active config")

    if campaign_runner_summaries:
        session_options = list(reversed(campaign_runner_summaries))
        session_labels = [path.parent.name for path in session_options]
        default_session_index = session_labels.index(query_session) if query_session in session_labels else 0
        selected_session_label = st.selectbox(
            "Campaign runner session",
            session_labels,
            index=default_session_index,
            key="campaign_runner_session",
        )
        selected_session_summary = session_options[session_labels.index(selected_session_label)]
    else:
        selected_session_summary = None

    if selected_session_summary is not None:
        default_view_index = 1 if query_view_mode == "Campaign runner session" else 0
        view_mode = st.radio(
            "Primary view",
            ["Active config", "Campaign runner session"],
            index=default_view_index,
            key="primary_view_mode",
        )
    else:
        view_mode = "Active config"

    auto_refresh = st.toggle("Auto-refresh", value=True)
    refresh_seconds = st.slider("Refresh every (seconds)", min_value=5, max_value=60, value=30)
    st.divider()

    with st.expander("Launch commands"):
        st.code(
            "streamlit run app/agent_dashboard.py "
            "--server.address 127.0.0.1 --server.port 8502",
            language="bash",
        )
        st.code(
            "streamlit run app/agent_dashboard.py "
            "--server.address 0.0.0.0 --server.port 8502",
            language="bash",
        )
        st.caption("Use `127.0.0.1` for local-only access. `0.0.0.0` for LAN access from trusted devices.")

# ---------------------------------------------------------------------------
# Load all data (cached)
# ---------------------------------------------------------------------------

config = _load_config(config_path)
workspace_dir = _workspace_dir(config)
results = _load_results(str(_results_path(config)))
baseline_metadata = _read_json_file(_baseline_metadata_path(config))
state = _read_json_file(STATE_FILE)
processes = _agent_processes()
gpu = _gpu_stats()
log_tail = _tail_log_lines()

if selected_session_summary is not None:
    st.query_params["session"] = selected_session_label
    st.query_params["view"] = view_mode
else:
    st.query_params.pop("session", None)
    st.query_params.pop("view", None)

if not state and processes:
    for process in processes:
        process_config = _config_from_agent_command(process.get("command", ""))
        if process_config != config_path:
            continue
        estimated_experiment = len(results) + 1 if results else 1
        state = {
            "phase": "RUNNING",
            "experiment_num": estimated_experiment,
            "max_runs": _max_runs_from_agent_command(process.get("command", "")),
            "description": "Live text-mode run detected; fine-grained phase tracking starts on the next agent launch.",
            "config_path": str(config_path.relative_to(PROJECT_ROOT)),
            "workspace_dir": str(workspace_dir.relative_to(PROJECT_ROOT)),
            "command": process.get("command", ""),
        }
        break

campaign_runner_session = (
    _load_campaign_runner_session(str(selected_session_summary))
    if selected_session_summary is not None
    else None
)

using_campaign_session = (
    view_mode == "Campaign runner session"
    and campaign_runner_session is not None
)

if using_campaign_session:
    session_summary = campaign_runner_session["summary"]
    display_results = campaign_runner_session["rows"]
    display_best_val_loss = min(
        [row["val_loss"] for row in display_results if row["val_loss"] is not None],
        default=None,
    )
    display_phase_label = f"{session_summary.get('stage', 'session')} (campaign session)"
    display_experiment = len(display_results)
    display_workspace_text = f"outputs/campaign_runner/{session_summary.get('session_tag', 'unknown')}"
    display_config_text = "aggregated campaign-runner session"
    display_results_text = f"aggregated across {len(campaign_runner_session['campaign_rows'])} campaign workspaces"
    display_keep_count = sum(1 for row in display_results if row["status"] == "keep")
    display_crash_count = sum(1 for row in display_results if row["status"] == "crash")
    display_state_payload = session_summary
    display_state_header = "Campaign Session Summary"
    display_baseline_payload = None
    display_baseline_header = "Session Details"
else:
    display_results = results
    display_best_val_loss = None
    if baseline_metadata:
        display_best_val_loss = baseline_metadata.get("metric_value", baseline_metadata.get("best_val_loss"))
    else:
        valid_losses = [row["val_loss"] for row in display_results if row["val_loss"] is not None]
        if valid_losses:
            display_best_val_loss = min(valid_losses)
    phase = state.get("phase") if state else "IDLE"
    display_phase_label = phase
    if state and not processes:
        display_phase_label = f"{phase} (stale state)"
    display_experiment = state.get("experiment_num", "—") if state else "—"
    display_workspace_text = str(workspace_dir)
    display_config_text = str(config_path.relative_to(PROJECT_ROOT))
    display_results_text = str(_results_path(config))
    display_keep_count = sum(1 for row in display_results if row["status"] == "keep")
    display_crash_count = sum(1 for row in display_results if row["status"] == "crash")
    display_state_payload = state
    display_state_header = "Current Agent State"
    display_baseline_payload = baseline_metadata
    display_baseline_header = "Promoted Baseline"

# ---------------------------------------------------------------------------
# Auto-refresh via st.rerun (native, state-preserving)
# ---------------------------------------------------------------------------

if auto_refresh:
    last_refresh = st.session_state.get("_last_refresh", 0.0)
    import time as _time
    now = _time.time()
    if now - last_refresh >= refresh_seconds:
        st.session_state["_last_refresh"] = now
        st.rerun()

# ---------------------------------------------------------------------------
# Page header
# ---------------------------------------------------------------------------

st.title("🧪 Autoresearch Monitor")
st.caption("Autonomous experiment loop — live view")

# ---------------------------------------------------------------------------
# Status banner
# ---------------------------------------------------------------------------

is_running = bool(processes)
if is_running:
    st.success(f"Agent running · {len(processes)} process(es) active", icon="✅")
else:
    st.warning("No active agent process detected. The state below may be stale.", icon="⚠️")

# ---------------------------------------------------------------------------
# Top metrics
# ---------------------------------------------------------------------------

status_col, phase_col, exp_col, best_col, keep_col, crash_col = st.columns(6)
status_col.metric("Agent", "Running" if is_running else "Stopped")
phase_col.metric("Phase", display_phase_label)
exp_col.metric("Experiment", display_experiment)
best_col.metric(
    "Best Val Loss",
    f"{display_best_val_loss:.6f}" if isinstance(display_best_val_loss, (int, float)) else "—",
)
keep_col.metric("Keeps", display_keep_count)
crash_col.metric("Crashes", display_crash_count)

with st.expander("Workspace info", expanded=False):
    st.caption(f"Workspace: `{display_workspace_text}`")
    st.caption(f"Config: `{display_config_text}`")
    st.caption(f"Results source: `{display_results_text}`")

# ---------------------------------------------------------------------------
# Main tabs
# ---------------------------------------------------------------------------

overview_tab, results_tab, trend_tab, hardware_tab, logs_tab = st.tabs(
    ["Overview", "Results", "Validation Trend", "Hardware", "Activity Log"]
)

# --- Overview tab ---
with overview_tab:
    left, right = st.columns([3, 2])

    with left:
        st.subheader(display_state_header)
        if display_state_payload:
            # Extract key fields for prominent display
            key_fields: dict[str, Any] = {}
            if isinstance(display_state_payload, dict):
                for key in ("phase", "experiment_num", "max_runs", "description"):
                    if key in display_state_payload:
                        key_fields[key] = display_state_payload[key]

            if key_fields:
                kf_cols = st.columns(min(len(key_fields), 3))
                for i, (k, v) in enumerate(key_fields.items()):
                    kf_cols[i % len(kf_cols)].metric(k.replace("_", " ").title(), str(v)[:60])
                with st.expander("Full agent state JSON"):
                    st.json(display_state_payload)
            else:
                st.json(display_state_payload)
        else:
            st.info("No agent state found yet. Start the agent with `python scripts/agent.py`.")
            st.code("python scripts/agent.py --config configs/autoresearch_default.yaml", language="bash")

    with right:
        st.subheader(display_baseline_header)
        if display_baseline_payload:
            _render_baseline_details(display_baseline_payload)
        elif using_campaign_session and campaign_runner_session:
            st.dataframe(
                [
                    {
                        "campaign": row["campaign"],
                        "stage": row["stage"],
                        "runs": row["runs"],
                        "keeps": row["keeps"],
                        "crashes": row["crashes"],
                        "best_val_loss": row["best_val_loss"],
                        "merged_into_main": row["merged_into_main"],
                    }
                    for row in campaign_runner_session["campaign_rows"]
                ],
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No baseline metadata found yet.")

# --- Results tab ---
with results_tab:
    st.subheader("Recent Results" if not using_campaign_session else "Campaign Results")
    display_rows = []
    table_rows = display_results if using_campaign_session else display_results[-20:]
    table_start = 1 if using_campaign_session else max(1, len(display_results) - 19)

    if not table_rows:
        st.info("No results logged yet. The agent hasn't completed any experiments.")
        st.code("python scripts/agent.py --config configs/autoresearch_default.yaml", language="bash")
    else:
        for index, row in enumerate(table_rows, start=table_start):
            status = row["status"]
            status_badge = f"🟢 {status}" if status == "keep" else (
                f"🔴 {status}" if status == "crash" else f"⚪ {status}"
            )
            display_row = {
                "#": index,
                "status": status_badge,
                "val_loss": row["val_loss"],
                "file": row["file"],
                "description": row["description"],
                "commit": row["commit"][:7] if row["commit"] else "—",
            }
            if using_campaign_session:
                display_row["campaign"] = row.get("campaign", "")
                display_row["stage"] = row.get("stage", "")
            display_rows.append(display_row)

        st.dataframe(display_rows, use_container_width=True, hide_index=True)

    if campaign_runner_session and campaign_runner_session["rows"] and not using_campaign_session:
        st.divider()
        st.subheader("Campaign Runner History")
        session_summary = campaign_runner_session["summary"]
        session_rows = campaign_runner_session["rows"]
        session_campaign_rows = campaign_runner_session["campaign_rows"]
        session_valid_losses = [row["val_loss"] for row in session_rows if row["val_loss"] is not None]
        session_best = min(session_valid_losses) if session_valid_losses else None
        sess_col1, sess_col2, sess_col3, sess_col4 = st.columns(4)
        sess_col1.metric("Session", str(session_summary.get("session_tag", "—")))
        sess_col2.metric("Campaigns", len(session_campaign_rows))
        sess_col3.metric("Session Best", f"{session_best:.6f}" if isinstance(session_best, (int, float)) else "—")
        sess_col4.metric("Merged Campaigns", sum(1 for row in session_campaign_rows if row["merged_into_main"]))

        st.subheader("Campaign Summary")
        st.dataframe(
            [
                {
                    "campaign": row["campaign"],
                    "stage": row["stage"],
                    "runs": row["runs"],
                    "keeps": row["keeps"],
                    "crashes": row["crashes"],
                    "best_val_loss": row["best_val_loss"],
                    "merged_into_main": row["merged_into_main"],
                    "branch": row["branch"],
                }
                for row in session_campaign_rows
            ],
            use_container_width=True,
            hide_index=True,
        )

        st.subheader("Aggregated Recent Runs")
        st.dataframe(
            [
                {
                    "#": index,
                    "campaign": row["campaign"],
                    "status": row["status"],
                    "val_loss": row["val_loss"],
                    "file": row["file"],
                    "description": row["description"],
                    "commit": row["commit"][:7] if row["commit"] else "—",
                }
                for index, row in enumerate(session_rows[-30:], start=max(1, len(session_rows) - 29))
            ],
            use_container_width=True,
            hide_index=True,
        )

# --- Validation Trend tab ---
with trend_tab:
    if any(row["val_loss"] is not None for row in display_results):
        _render_validation_trend_plotly(
            display_results,
            title_prefix="Campaign Runner Validation Trend" if using_campaign_session else "Validation Trend",
            keep_only=False if using_campaign_session else None,
            hide_high_outliers=not using_campaign_session,
        )

        if campaign_runner_session and campaign_runner_session["rows"] and not using_campaign_session:
            st.divider()
            st.subheader("Campaign Runner Validation Trend")
            _render_validation_trend_plotly(
                campaign_runner_session["rows"],
                title_prefix="Campaign Runner",
                keep_only=False,
                hide_high_outliers=False,
            )
            st.caption("Aggregated from all campaign workspaces, filtered to each campaign's time window.")
    else:
        st.info("No validation-loss data yet. Run an experiment first.")

# --- Hardware tab ---
with hardware_tab:
    st.subheader("Hardware")
    gpu_temp = "N/A" if gpu["temp"] is None else f"{gpu['temp']}°C"
    vram_text = "N/A"
    if gpu["vram_total_mb"] > 0:
        vram_text = f"{gpu['vram_used_mb']:.0f}/{gpu['vram_total_mb']:.0f} MB"
    gpu_col1, gpu_col2, gpu_col3, gpu_col4 = st.columns(4)
    gpu_col1.metric("GPU", gpu["name"])
    gpu_col2.metric("Temperature", gpu_temp)
    gpu_col3.metric("Utilization", f"{gpu['gpu_util']}%")
    gpu_col4.metric("VRAM", vram_text)

    st.subheader("Agent Processes")
    if processes:
        st.code("\n".join(f"{proc['pid']}  {proc['command']}" for proc in processes), language="text")
    else:
        st.info("No live `scripts/agent.py` process detected.")

# --- Logs tab ---
with logs_tab:
    st.subheader("Activity Log")
    if log_tail:
        st.code("".join(log_tail), language="text")
    else:
        st.info("No `agent.log` entries found yet. Logs appear here once the agent starts writing.")
