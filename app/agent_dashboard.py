"""Streamlit dashboard for monitoring autoresearch agent runs."""

from __future__ import annotations

import json
import os
import subprocess
from collections import deque
from pathlib import Path

import matplotlib.pyplot as plt
import streamlit as st
import streamlit.components.v1 as components
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AUTORESEARCH_CONFIG = PROJECT_ROOT / "configs" / "autoresearch_default.yaml"
STATE_FILE = PROJECT_ROOT / "agent_state.json"
LOG_FILE = PROJECT_ROOT / "agent.log"


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

    config_options = _list_config_options()
    for candidate in reversed(config_options):
        try:
            config = _load_config(candidate)
            if _results_path(config).exists():
                return candidate
        except Exception:
            continue

    return DEFAULT_AUTORESEARCH_CONFIG


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


def _load_results(config: dict) -> list[dict]:
    path = _results_path(config)
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


def _tail_lines(path: Path, limit: int = 120) -> list[str]:
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            return list(deque(handle, maxlen=limit))
    except Exception:
        return []


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
                "commit": row["commit"][:7] if row["commit"] else "—",
            }
        )
    return points


def _short_label(text: str, limit: int = 60) -> str:
    clean = " ".join((text or "").split())
    if len(clean) <= limit:
        return clean
    return clean[: limit - 3].rstrip() + "..."


def _render_validation_trend(results: list[dict]) -> None:
    points = _metric_points(results)
    if not points:
        st.info("No validation-loss points logged yet.")
        return

    x_vals = [point["x"] for point in points]
    val_losses = [point["val_loss"] for point in points]
    best_losses = [point["best_so_far"] for point in points]
    keep_points = [point for point in points if point["status"] == "keep"]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(x_vals, val_losses, color="#4C78A8", marker="o", linewidth=1.8, markersize=4, label="val_loss")
    ax.plot(x_vals, best_losses, color="#54A24B", linewidth=2.0, linestyle="--", label="best_so_far")

    if keep_points:
        keep_x = [point["x"] for point in keep_points]
        keep_y = [point["val_loss"] for point in keep_points]
        ax.scatter(keep_x, keep_y, color="#E45756", s=50, zorder=3, label="kept improvement")

        for idx, point in enumerate(keep_points):
            y_offset = 14 if idx % 2 == 0 else -18
            label = f"{point['commit']}: {_short_label(point['description'])}"
            ax.annotate(
                label,
                xy=(point["x"], point["val_loss"]),
                xytext=(6, y_offset),
                textcoords="offset points",
                fontsize=8,
                color="#222222",
                bbox={"boxstyle": "round,pad=0.25", "fc": "#FFF7D6", "ec": "#D9C98E", "alpha": 0.95},
                arrowprops={"arrowstyle": "-", "color": "#D9C98E", "lw": 1.0},
            )

    ax.set_title("Validation Trend")
    ax.set_xlabel("Experiment")
    ax.set_ylabel("best_val_loss")
    ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.5)
    ax.legend(loc="best")
    fig.tight_layout()
    st.pyplot(fig, width="stretch")
    plt.close(fig)

    if keep_points:
        st.caption("Kept points are annotated with commit and truncated description.")


st.set_page_config(
    page_title="Autoresearch Monitor",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

config_path = _resolve_config_path()
config = _load_config(config_path)
workspace_dir = _workspace_dir(config)
results = _load_results(config)
baseline_metadata = _read_json_file(_baseline_metadata_path(config))
state = _read_json_file(STATE_FILE)
processes = _agent_processes()
gpu = _gpu_stats()
log_tail = _tail_lines(LOG_FILE)

st.title("🧪 Autoresearch Monitor")
st.caption("Web dashboard for the autonomous experiment loop.")

with st.sidebar:
    st.header("Monitor")
    config_options = _list_config_options()
    option_labels = [str(path.relative_to(PROJECT_ROOT)) for path in config_options]
    selected_label = st.selectbox(
        "Autoresearch config",
        option_labels,
        index=option_labels.index(str(config_path.relative_to(PROJECT_ROOT)))
        if str(config_path.relative_to(PROJECT_ROOT)) in option_labels else 0,
    )
    selected_config_path = PROJECT_ROOT / selected_label
    if selected_config_path != config_path:
        st.query_params["config"] = selected_label
        st.rerun()

    auto_refresh = st.toggle("Auto-refresh", value=True)
    refresh_seconds = st.slider("Refresh every (seconds)", min_value=2, max_value=30, value=30)
    st.divider()
    st.subheader("Launch URLs")
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
    st.caption(
        "Use `127.0.0.1` for local-only access. Use `0.0.0.0` for LAN access from trusted devices, "
        "then open `http://<your-machine-ip>:8502`."
    )
    st.warning("This dashboard has no authentication. Do not expose it to the public internet.")

if auto_refresh:
    components.html(
        f"""
        <script>
        setTimeout(function() {{
            window.parent.location.reload();
        }}, {refresh_seconds * 1000});
        </script>
        """,
        height=0,
    )

phase = state.get("phase") if state else "IDLE"
phase_label = phase
if state and not processes:
    phase_label = f"{phase} (stale state)"

best_val_loss = None
if baseline_metadata:
    best_val_loss = baseline_metadata.get("best_val_loss")
else:
    valid_losses = [row["val_loss"] for row in results if row["val_loss"] is not None]
    if valid_losses:
        best_val_loss = min(valid_losses)

status_col, phase_col, exp_col, best_col, keep_col, crash_col = st.columns(6)
status_col.metric("Agent Process", "Running" if processes else "Stopped")
phase_col.metric("Phase", phase_label)
exp_col.metric("Experiment", state.get("experiment_num", "—") if state else "—")
best_col.metric("Best Val Loss", f"{best_val_loss:.6f}" if isinstance(best_val_loss, (int, float)) else "—")
keep_col.metric("Keeps", sum(1 for row in results if row["status"] == "keep"))
crash_col.metric("Crashes", sum(1 for row in results if row["status"] == "crash"))

st.caption(f"Workspace: `{workspace_dir}`")
st.caption(f"Config: `{config_path.relative_to(PROJECT_ROOT)}`")
st.caption(f"Results ledger: `{_results_path(config)}`")

left, right = st.columns([3, 2])

with left:
    st.subheader("Current Agent State")
    if state:
        st.json(state)
    else:
        st.info("No `agent_state.json` found.")

    st.subheader("Recent Results")
    display_rows = []
    for index, row in enumerate(results[-20:], start=max(1, len(results) - 19)):
        display_rows.append(
            {
                "#": index,
                "status": row["status"],
                "val_loss": row["val_loss"],
                "file": row["file"],
                "description": row["description"],
                "commit": row["commit"][:7] if row["commit"] else "—",
            }
        )
    if display_rows:
        st.dataframe(display_rows, width="stretch", hide_index=True)
    else:
        st.info("No results logged yet.")

with right:
    st.subheader("Hardware")
    gpu_temp = "N/A" if gpu["temp"] is None else f"{gpu['temp']} C"
    vram_text = "N/A"
    if gpu["vram_total_mb"] > 0:
        vram_text = f"{gpu['vram_used_mb']:.0f}/{gpu['vram_total_mb']:.0f} MB"
    gpu_col1, gpu_col2 = st.columns(2)
    gpu_col1.metric("GPU", gpu["name"])
    gpu_col2.metric("Temp", gpu_temp)
    gpu_col1.metric("Utilization", f"{gpu['gpu_util']}%")
    gpu_col2.metric("VRAM", vram_text)

    st.subheader("Processes")
    if processes:
        st.code("\n".join(f"{proc['pid']}  {proc['command']}" for proc in processes), language="text")
    else:
        st.info("No live `scripts/agent.py` process detected.")

    st.subheader("Promoted Baseline")
    if baseline_metadata:
        st.json(baseline_metadata)
    else:
        st.info("No baseline metadata found yet.")

if any(row["val_loss"] is not None for row in results):
    st.subheader("Validation Trend")
    _render_validation_trend(results)

st.subheader("Activity Log")
if log_tail:
    st.code("".join(log_tail[-80:]), language="text")
else:
    st.info("No `agent.log` entries yet.")
