"""Autonomous research agent for Digital Twin Engine.

Loops indefinitely: asks an LLM to propose an experiment patch,
applies it, runs the autoresearch harness (scripts/autoresearch.py),
keeps improvements, reverts failures, and shows a Rich TUI dashboard.

Usage:
    python scripts/agent.py                    # Provider from autoresearch config
    python scripts/agent.py --config configs/autoresearch_stage1.yaml
    python scripts/agent.py --rebaseline       # Refresh workspace baseline from current code
    python scripts/agent.py --deepseek         # DeepSeek reasoning model
    python scripts/agent.py --claude           # Claude Sonnet 4.6
    python scripts/agent.py --opus             # Claude Opus 4.6 with 32k thinking
    python scripts/agent.py --openai o3        # OpenAI o3
    python scripts/agent.py --grok             # xAI Grok 3
    python scripts/agent.py --local            # Local LM Studio
    python scripts/agent.py --max-runs 50      # Cap total experiments
    python scripts/agent.py --resume           # Continue from existing branch
    python scripts/agent.py --tag mar26        # Named branch tag
    python scripts/agent.py --no-dashboard     # Text-only mode
    python scripts/agent.py --file dte/training/trainer.py  # Restrict to one file

Required environment variables (for the chosen provider):
    DEEPSEEK_API_KEY    -- DeepSeek
    GEMINI_API_KEY      -- Google Gemini
    ANTHROPIC_API_KEY   -- Claude
    OPENAI_API_KEY      -- OpenAI
    XAI_API_KEY         -- xAI Grok

The agent also auto-loads `.env` and `.env.local` from the project root
when present.
"""

from __future__ import annotations

import ast
import difflib
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import yaml

from dte.autoresearch.workflow import append_result_row, ensure_results_file, make_run_id

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUTORESEARCH_SCRIPT = PROJECT_ROOT / "scripts" / "autoresearch.py"
DEFAULT_AUTORESEARCH_CONFIG = PROJECT_ROOT / "configs" / "autoresearch_default.yaml"
ACTIVE_AUTORESEARCH_CONFIG = DEFAULT_AUTORESEARCH_CONFIG
DEFAULT_WORKSPACE_DIR = PROJECT_ROOT / "outputs" / "autoresearch"
DEFAULT_AGENT_CONTEXT_FILE = PROJECT_ROOT / "auto_research.md"
LOG_DIR = PROJECT_ROOT / "outputs" / "autoresearch_logs"
LOG_FILE = LOG_DIR / "agent.log"
STATE_FILE = PROJECT_ROOT / "agent_state.json"


# ---------------------------------------------------------------------------
# Environment loading
# ---------------------------------------------------------------------------

def load_env_file(path: Path, override: bool = False) -> None:
    """Load simple KEY=VALUE pairs from a .env-style file."""

    if not path.exists():
        return

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception as exc:
        print(f"WARNING env load failed: {path} ({exc})", file=sys.stderr)
        return

    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue

        if value and len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]

        if override or key not in os.environ:
            os.environ[key] = value


load_env_file(PROJECT_ROOT / ".env", override=False)
load_env_file(PROJECT_ROOT / ".env.local", override=True)


# ---------------------------------------------------------------------------
# GPU monitoring (optional – graceful fallback on Mac / CPU machines)
# ---------------------------------------------------------------------------

_nvml_available = False
_nvml_handle = None
GPU_NAME = "No GPU"
VRAM_TOTAL_MB = 0
VRAM_LIMIT_MB = 0

try:
    from pynvml import (
        nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo,
        nvmlDeviceGetName, nvmlDeviceGetTemperature, nvmlDeviceGetUtilizationRates,
        NVML_TEMPERATURE_GPU,
    )
    nvmlInit()
    _nvml_handle = nvmlDeviceGetHandleByIndex(0)
    _nvml_available = True
    mem = nvmlDeviceGetMemoryInfo(_nvml_handle)
    VRAM_TOTAL_MB = mem.total // (1024 * 1024)
    VRAM_LIMIT_MB = VRAM_TOTAL_MB - 500
    raw_name = nvmlDeviceGetName(_nvml_handle)
    GPU_NAME = raw_name.decode() if isinstance(raw_name, bytes) else raw_name
except Exception:
    # Graceful fallback on CPU-only machines (Mac, etc.)
    pass

GPU_TEMP_MAX_START = 70
GPU_TEMP_ABORT = 85


def get_gpu_stats() -> dict:
    if not _nvml_available:
        return {"temp": None, "vram_used_mb": 0, "vram_total_mb": VRAM_TOTAL_MB, "gpu_util": 0}
    try:
        temp = nvmlDeviceGetTemperature(_nvml_handle, NVML_TEMPERATURE_GPU)
        mem = nvmlDeviceGetMemoryInfo(_nvml_handle)
        util = nvmlDeviceGetUtilizationRates(_nvml_handle)
        return {
            "temp": temp,
            "vram_used_mb": mem.used / 1024 / 1024,
            "vram_total_mb": mem.total / 1024 / 1024,
            "gpu_util": util.gpu,
        }
    except Exception:
        return {"temp": None, "vram_used_mb": 0, "vram_total_mb": VRAM_TOTAL_MB, "gpu_util": 0}


def wait_for_cool_gpu(max_wait: int = 120) -> bool:
    """Wait up to max_wait seconds for GPU to cool below threshold."""
    if not _nvml_available:
        return True
    start = time.time()
    while time.time() - start < max_wait:
        try:
            temp = nvmlDeviceGetTemperature(_nvml_handle, NVML_TEMPERATURE_GPU)
        except Exception:
            return True
        if temp is None or temp <= GPU_TEMP_MAX_START:
            return True
        if temp >= GPU_TEMP_ABORT:
            return False
        time.sleep(5)
    return True  # best effort – don't block indefinitely on CPU machines


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def log_to_file(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(f"[{ts}] {msg}\n")
        f.flush()
        os.fsync(f.fileno())


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def write_state(exp_num: int, description: str, phase: str, extra: dict | None = None) -> None:
    payload = {
        "timestamp": datetime.now().isoformat(),
        "experiment_num": exp_num,
        "description": description,
        "phase": phase,
        "config_path": _display_path(ACTIVE_AUTORESEARCH_CONFIG),
        "workspace_dir": _display_path(get_workspace_dir()),
    }
    if extra:
        payload.update(extra)
    with STATE_FILE.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def clear_state() -> None:
    if STATE_FILE.exists():
        STATE_FILE.unlink()


def read_state() -> dict | None:
    if not STATE_FILE.exists():
        return None
    try:
        with STATE_FILE.open("r") as f:
            return json.load(f)
    except Exception:
        return None


def other_agent_process_running() -> bool:
    """Return True when another scripts/agent.py process is still running."""

    try:
        result = subprocess.run(
            ["pgrep", "-af", "scripts/agent.py"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception:
        return False

    current_pid = os.getpid()
    for line in result.stdout.splitlines():
        parts = line.strip().split(maxsplit=1)
        if not parts:
            continue
        try:
            pid = int(parts[0])
        except ValueError:
            continue
        if pid != current_pid:
            return True
    return False


def set_autoresearch_config(path_value: str | Path) -> Path:
    """Set the active autoresearch config path for this agent session."""

    global ACTIVE_AUTORESEARCH_CONFIG

    path = Path(path_value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    ACTIVE_AUTORESEARCH_CONFIG = path
    return ACTIVE_AUTORESEARCH_CONFIG


def get_workspace_dir() -> Path:
    """Resolve the active autoresearch workspace from config."""

    config = load_autoresearch_config()
    if not config:
        return DEFAULT_WORKSPACE_DIR

    workspace_value = config.get("research", {}).get("workspace_dir", "outputs/autoresearch")
    workspace_dir = Path(workspace_value)
    if not workspace_dir.is_absolute():
        workspace_dir = PROJECT_ROOT / workspace_dir
    return workspace_dir


def get_results_tsv_path() -> Path:
    """Resolve the active results ledger path."""

    return get_workspace_dir() / "results.tsv"


def resolve_repo_path(path_value: str) -> Path:
    """Resolve repo-relative paths from config values."""

    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_agent_context(config: dict | None = None) -> str:
    """Load repo-specific agent guidance from disk."""

    agent_cfg = (config or {}).get("agent", {})
    context_file = agent_cfg.get("context_file")
    context_path = resolve_repo_path(context_file) if context_file else DEFAULT_AGENT_CONTEXT_FILE

    try:
        text = context_path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return ""
    except Exception as exc:
        log_to_file(f"WARNING agent context load failed: {context_path} ({exc})")
        return ""

    if not text:
        return ""

    # Keep prompt bloat under control if the file grows too much.
    if len(text) > 12000:
        text = text[:12000].rstrip() + "\n\n[Context truncated]"
    return text


def get_execution_mode(config: dict | None = None) -> str:
    """Resolve the agent execution mode from config."""

    agent_cfg = (config or {}).get("agent", {})
    mode = str(agent_cfg.get("execution_mode", DEFAULT_EXECUTION_MODE)).strip().lower()
    return "multi_file" if mode == "multi_file" else "single_file"


def _normalize_repo_relative_path(raw_path: str) -> str | None:
    """Return a normalized repo-relative path, or None when invalid."""

    text = str(raw_path or "").strip()
    if not text:
        return None

    candidate = Path(text)
    if candidate.is_absolute():
        return None

    try:
        resolved = (PROJECT_ROOT / candidate).resolve()
        relative = resolved.relative_to(PROJECT_ROOT.resolve())
    except Exception:
        return None
    return relative.as_posix()


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered


def get_editable_targets(config: dict | None = None) -> list[str]:
    """Resolve the configured editable files or directories."""

    agent_cfg = (config or {}).get("agent", {})
    raw_targets = agent_cfg.get("modifiable_paths")
    if raw_targets is None:
        raw_targets = agent_cfg.get("modifiable_files", MODIFIABLE_FILES)

    normalized: list[str] = []
    for target in _normalize_text_list(raw_targets):
        normalized_target = _normalize_repo_relative_path(target)
        if normalized_target:
            normalized.append(normalized_target)
    return _dedupe_preserve_order(normalized)


def get_forbidden_paths(config: dict | None = None) -> list[str]:
    """Resolve the configured forbidden files or directories."""

    agent_cfg = (config or {}).get("agent", {})
    raw_paths = agent_cfg.get("forbidden_paths", DEFAULT_FORBIDDEN_PATHS)
    normalized: list[str] = []
    for target in _normalize_text_list(raw_paths):
        normalized_target = _normalize_repo_relative_path(target)
        if normalized_target:
            normalized.append(normalized_target)
    return _dedupe_preserve_order(normalized)


def _path_is_within_target(path: str, target: str) -> bool:
    return path == target or path.startswith(target + "/")


def is_forbidden_edit_path(path: str, forbidden_paths: list[str]) -> bool:
    normalized_path = _normalize_repo_relative_path(path)
    if not normalized_path:
        return True
    return any(_path_is_within_target(normalized_path, blocked) for blocked in forbidden_paths)


def is_allowed_edit_path(
    path: str,
    editable_targets: list[str],
    forbidden_paths: list[str] | None = None,
) -> bool:
    """Return True when a path is inside the configured editable surface."""

    normalized_path = _normalize_repo_relative_path(path)
    if not normalized_path:
        return False

    if forbidden_paths and is_forbidden_edit_path(normalized_path, forbidden_paths):
        return False

    for target in editable_targets:
        absolute_target = PROJECT_ROOT / target
        if absolute_target.is_dir():
            if _path_is_within_target(normalized_path, target):
                return True
        elif normalized_path == target:
            return True
    return False


def list_editable_repo_files(
    editable_targets: list[str],
    forbidden_paths: list[str] | None = None,
) -> list[str]:
    """List editable files beneath the configured editable files/directories."""

    collected: list[str] = []
    for target in editable_targets:
        absolute_target = PROJECT_ROOT / target
        if absolute_target.is_dir():
            for path in sorted(absolute_target.rglob("*")):
                if not path.is_file():
                    continue
                relative = path.relative_to(PROJECT_ROOT).as_posix()
                if forbidden_paths and is_forbidden_edit_path(relative, forbidden_paths):
                    continue
                collected.append(relative)
        elif absolute_target.is_file():
            if forbidden_paths and is_forbidden_edit_path(target, forbidden_paths):
                continue
            collected.append(target)
    return _dedupe_preserve_order(collected)


def summarize_changed_paths(paths: list[str]) -> str:
    """Return a concise label for one or more changed files."""

    if not paths:
        return ""
    if len(paths) == 1:
        return paths[0]
    return f"{paths[0]} (+{len(paths) - 1} files)"


def _normalize_text_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, list):
        items: list[str] = []
        for item in value:
            text = str(item).strip()
            if text:
                items.append(text)
        return items
    return [str(value).strip()]


def _sanitize_idea_id(raw_value: str, fallback: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_.-]+", "-", (raw_value or "").strip()).strip("-").lower()
    return cleaned or fallback


def load_structured_ideas(
    config: dict | None = None,
    override_path: str | None = None,
) -> list[dict]:
    """Load an ordered idea backlog from YAML/JSON."""

    agent_cfg = (config or {}).get("agent", {})
    ideas_file = override_path or agent_cfg.get("ideas_file")
    if not ideas_file:
        return []

    ideas_path = resolve_repo_path(str(ideas_file))
    try:
        raw = yaml.safe_load(ideas_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return []
    except Exception as exc:
        log_to_file(f"WARNING ideas load failed: {ideas_path} ({exc})")
        return []

    if isinstance(raw, dict):
        raw_items = raw.get("ideas", [])
    elif isinstance(raw, list):
        raw_items = raw
    else:
        return []

    ideas: list[dict] = []
    for index, item in enumerate(raw_items, start=1):
        if not isinstance(item, dict):
            continue
        if item.get("enabled", True) is False:
            continue

        title = str(item.get("title") or item.get("description") or "").strip()
        fallback_id = f"idea-{index}"
        idea_id = _sanitize_idea_id(str(item.get("id") or title), fallback_id)
        target_file = str(item.get("target_file") or item.get("file") or "").strip()
        priority = item.get("priority", index)
        try:
            priority_value = int(priority)
        except Exception:
            priority_value = index

        ideas.append(
            {
                "id": idea_id,
                "title": title or idea_id,
                "target_file": target_file,
                "priority": priority_value,
                "rationale": str(item.get("rationale") or "").strip(),
                "instructions": _normalize_text_list(item.get("instructions") or item.get("steps")),
                "tags": _normalize_text_list(item.get("tags")),
            }
        )

    ideas.sort(key=lambda idea: (idea["priority"], idea["id"]))
    return ideas


def load_autoresearch_config() -> dict:
    """Load the active autoresearch config."""

    try:
        with ACTIVE_AUTORESEARCH_CONFIG.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Results ledger
# ---------------------------------------------------------------------------

def init_results() -> None:
    ensure_results_file(get_results_tsv_path())


def log_result(commit: str, val_loss: float, status: str, file_changed: str, description: str) -> None:
    now = datetime.now()
    clean_description = description.replace("\t", " ").replace("\n", " ").strip()
    if file_changed and file_changed != "baseline":
        clean_description = f"[{file_changed}] {clean_description}"

    has_metric = status != "crash" and val_loss > 0.0
    append_result_row(
        get_results_tsv_path(),
        {
            "timestamp": now.strftime("%Y%m%d-%H%M%S"),
            "run_id": make_run_id(clean_description or status, now=now),
            "commit": commit,
            "metric_name": "best_val_loss" if has_metric else "",
            "metric_value": f"{val_loss:.6f}" if has_metric else "",
            "baseline_before": "",
            "training_seconds": "",
            "status": status,
            "description": clean_description,
        },
    )
    metric_str = f"{val_loss:.6f}" if has_metric else "n/a"
    log_to_file(f"RESULT: {status} | val_loss={metric_str} | {file_changed} | {description}")


def _parse_logged_metric(raw_value: str) -> float:
    raw_value = (raw_value or "").strip()
    if raw_value in ("", "0", "0.0", "0.000000"):
        return 999.0
    try:
        value = float(raw_value)
    except ValueError:
        return 999.0
    return value if value > 0.0 else 999.0


def _decode_history_description(description: str) -> tuple[str, str]:
    description = (description or "").strip()
    match = re.match(r"^\[(.+?)\]\s+(.*)$", description)
    if not match:
        return "", description
    return match.group(1), match.group(2)


def get_results_history() -> list[dict]:
    results_tsv = get_results_tsv_path()
    if not results_tsv.exists():
        return []
    rows = []
    with results_tsv.open("r", encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        column_index = {name: idx for idx, name in enumerate(header)}
        is_new_format = (
            "metric_value" in column_index
            and "status" in column_index
            and "description" in column_index
        )
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if not parts or all(not part for part in parts):
                continue

            if is_new_format:
                def get_col(name: str, default: str = "") -> str:
                    idx = column_index.get(name)
                    if idx is None or idx >= len(parts):
                        return default
                    return parts[idx]

                file_changed, description = _decode_history_description(get_col("description"))
                rows.append({
                    "commit": get_col("commit", "-------") or "-------",
                    "val_loss": _parse_logged_metric(get_col("metric_value")),
                    "status": get_col("status", "crash") or "crash",
                    "file": file_changed,
                    "description": description,
                })
                continue

            if len(parts) >= 5:
                rows.append({
                    "commit": parts[0],
                    "val_loss": _parse_logged_metric(parts[1]),
                    "status": parts[2],
                    "file": parts[3],
                    "description": parts[4],
                })
    return rows


def _coerce_metric_value(value) -> float | None:
    try:
        metric_value = float(value)
    except (TypeError, ValueError):
        return None
    return metric_value if metric_value > 0.0 else None


def get_promoted_baseline_metadata() -> dict | None:
    baseline_dir = get_workspace_dir() / "baseline"
    metadata = _read_json_if_exists(baseline_dir / "metadata.json")
    if metadata:
        metric_value = _coerce_metric_value(metadata.get("metric_value"))
        if metric_value is not None:
            enriched = dict(metadata)
            enriched["metric_value"] = metric_value
            return enriched

    summary = _read_json_if_exists(baseline_dir / "summary.json")
    if summary:
        metric_name = str(load_autoresearch_config().get("research", {}).get("metric_name", "best_val_loss"))
        metric_value = _coerce_metric_value(summary.get(metric_name))
        if metric_value is not None:
            return {
                "metric_name": metric_name,
                "metric_value": metric_value,
                "description": "baseline summary fallback",
            }

    return None


def get_current_reference_loss(history: list[dict] | None = None) -> float:
    baseline_metadata = get_promoted_baseline_metadata()
    if baseline_metadata:
        metric_value = _coerce_metric_value(baseline_metadata.get("metric_value"))
        if metric_value is not None:
            return metric_value

    history = history if history is not None else get_results_history()
    valid_losses = [r["val_loss"] for r in history if r.get("val_loss", 999.0) < 999.0]
    return min(valid_losses) if valid_losses else 999.0


def get_runs_dir() -> Path:
    """Resolve the active autoresearch runs directory."""

    return get_workspace_dir() / "runs"


def _read_json_if_exists(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return None


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)


def _first_attr(obj, *names, default=None):
    for name in names:
        if hasattr(obj, name):
            value = getattr(obj, name)
            if value is not None:
                return value
    return default


def _parse_datetime_value(value) -> datetime | None:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _safe_model_slug(model_name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", (model_name or "").strip()) or "gemini"


def get_gemini_cache_metadata_path(model_name: str) -> Path:
    return get_workspace_dir() / ".gemini_cache" / f"{_safe_model_slug(model_name)}.json"


def ensure_gemini_context_cache(
    static_context: str,
    model: str,
    ttl: str = "3600s",
    min_tokens: int = 2048,
) -> str | None:
    """Create or reuse a Gemini explicit cache for repeated static prompt context."""

    if not static_context.strip():
        return None

    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        return None

    try:
        from google import genai
        from google.genai import types
    except ImportError:
        log_to_file("WARNING Gemini cache disabled: google-genai not installed")
        return None

    client = genai.Client(api_key=api_key)
    meta_path = get_gemini_cache_metadata_path(model)
    content_hash = hashlib.sha256(static_context.encode("utf-8")).hexdigest()
    cached_meta = _read_json_if_exists(meta_path) or {}
    now = datetime.now(timezone.utc)

    if (
        cached_meta.get("model") == model
        and cached_meta.get("content_hash") == content_hash
        and cached_meta.get("cache_name")
    ):
        expire_time = _parse_datetime_value(cached_meta.get("expire_time"))
        if expire_time is None or expire_time > now:
            try:
                cache = client.caches.get(name=str(cached_meta["cache_name"]))
                cache_name = str(_first_attr(cache, "name", default="") or "")
                cache_expire_time = _parse_datetime_value(
                    _first_attr(cache, "expire_time", "expireTime")
                )
                if cache_name:
                    _write_json(
                        meta_path,
                        {
                            "cache_name": cache_name,
                            "model": model,
                            "content_hash": content_hash,
                            "expire_time": cache_expire_time.isoformat() if cache_expire_time else None,
                            "updated_at": now.isoformat(),
                            "token_count": cached_meta.get("token_count"),
                        },
                    )
                    log_to_file(f"Gemini cache: reusing {cache_name}")
                    return cache_name
            except Exception as exc:
                log_to_file(f"WARNING Gemini cache reuse failed: {exc}")

    token_count = None
    try:
        token_response = client.models.count_tokens(model=model, contents=static_context)
        token_count = int(_first_attr(token_response, "total_tokens", "totalTokens", default=0) or 0)
        if token_count and token_count < min_tokens:
            log_to_file(
                f"Gemini cache skipped: static context too small "
                f"({token_count} tokens < {min_tokens})"
            )
            return None
    except Exception as exc:
        log_to_file(f"WARNING Gemini cache token count failed: {exc}")

    try:
        cache = client.caches.create(
            model=model,
            config=types.CreateCachedContentConfig(
                display_name=f"dte-agent-{_safe_model_slug(model)}",
                contents=static_context,
                ttl=ttl,
            ),
        )
    except Exception as exc:
        log_to_file(f"WARNING Gemini cache create failed: {exc}")
        return None

    cache_name = str(_first_attr(cache, "name", default="") or "")
    if not cache_name:
        return None

    cache_expire_time = _parse_datetime_value(_first_attr(cache, "expire_time", "expireTime"))
    _write_json(
        meta_path,
        {
            "cache_name": cache_name,
            "model": model,
            "content_hash": content_hash,
            "expire_time": cache_expire_time.isoformat() if cache_expire_time else None,
            "updated_at": now.isoformat(),
            "token_count": token_count,
        },
    )
    log_to_file(
        f"Gemini cache: created {cache_name}"
        + (f" ({token_count} tokens)" if token_count else "")
    )
    return cache_name


def _is_gemini_cache_miss_error(error: Exception | str) -> bool:
    """Return True when Gemini failed because the cached content ID is invalid."""

    text = str(error).strip().lower()
    if not text:
        return False
    return (
        "cachedcontent not found" in text
        or "cached_content not found" in text
        or "cached content not found" in text
    )


def _call_gemini_with_cache_retry(
    call_fn,
    prompt: str,
    *,
    kwargs: dict,
    refresh_cache,
) -> str | None:
    """Retry once when Gemini rejects an expired or missing cached-content ID."""

    try:
        return call_fn(prompt, **kwargs)
    except FatalAPIError as exc:
        cached_content = kwargs.get("cached_content")
        if not cached_content or not _is_gemini_cache_miss_error(exc):
            raise

        log_to_file(f"Gemini cache miss for {cached_content}; refreshing and retrying once")
        refreshed_cache = refresh_cache()
        retry_kwargs = dict(kwargs)
        if refreshed_cache:
            retry_kwargs["cached_content"] = refreshed_cache
        else:
            retry_kwargs.pop("cached_content", None)
            log_to_file("WARNING Gemini cache refresh failed; retrying without cached_content")

        return call_fn(prompt, **retry_kwargs)


def _tail_log_lines(path: Path, limit: int = 25) -> list[str]:
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            return list(deque(handle, maxlen=limit))
    except Exception:
        return []


def _parse_loss_dict_from_line(line: str, prefix: str) -> dict[str, float] | None:
    """Parse a printed Python dict of losses from a training log line."""

    if prefix not in line:
        return None
    try:
        raw_dict = line.split(prefix, 1)[1].strip()
        parsed = ast.literal_eval(raw_dict)
    except (SyntaxError, ValueError):
        return None
    if not isinstance(parsed, dict):
        return None

    numeric_losses: dict[str, float] = {}
    for key, value in parsed.items():
        try:
            numeric_losses[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return numeric_losses or None


def _extract_component_losses(log_path: Path) -> tuple[dict[str, float] | None, dict[str, float] | None]:
    """Extract the latest printed train/val component-loss dicts from a train log."""

    lines = _tail_log_lines(log_path, limit=250)
    train_losses: dict[str, float] | None = None
    val_losses: dict[str, float] | None = None

    for line in lines:
        parsed_train = _parse_loss_dict_from_line(line, "Train losses:")
        if parsed_train is not None:
            train_losses = parsed_train

        parsed_val = _parse_loss_dict_from_line(line, "Val losses:")
        if parsed_val is not None:
            val_losses = parsed_val

    return train_losses, val_losses


def _format_component_losses(label: str, losses: dict[str, float] | None) -> str:
    """Format a compact subset of loss components for prompt context."""

    if not losses:
        return ""

    keys = ("total", "trajectory", "reconstruction", "kl", "physics", "mass_balance", "energy_balance")
    parts = []
    for key in keys:
        if key in losses:
            parts.append(f"{key}={losses[key]:.4f}")
    if not parts:
        return ""
    return f"{label}: " + ", ".join(parts)


def _format_named_metrics(label: str, metrics: dict[str, float] | None) -> str:
    """Format named metrics like rmse_per_state for prompt context."""

    if not isinstance(metrics, dict) or not metrics:
        return ""

    parts = []
    for name, value in metrics.items():
        try:
            parts.append(f"{name}={float(value):.4f}")
        except (TypeError, ValueError):
            continue
    if not parts:
        return ""
    return f"{label}: " + ", ".join(parts)


def _extract_eval_metric_text(eval_summary: dict | None) -> str:
    """Extract compact per-state eval metrics from evaluation_summary.json."""

    if not isinstance(eval_summary, dict):
        return ""

    model_metrics = eval_summary.get("model_metrics")
    if not isinstance(model_metrics, dict):
        return ""

    rmse_text = _format_named_metrics("eval_rmse", model_metrics.get("rmse_per_state"))
    nrmse_text = _format_named_metrics("eval_nrmse", model_metrics.get("nrmse_per_state"))
    sample_count = eval_summary.get("sample_count")
    detail_bits = []
    if isinstance(sample_count, int):
        detail_bits.append(f"samples={sample_count}")
    if eval_summary.get("predict_mode"):
        detail_bits.append(f"mode={eval_summary['predict_mode']}")
    metric_bits = [bit for bit in (rmse_text, nrmse_text) if bit]
    if detail_bits:
        metric_bits.append(", ".join(detail_bits))
    return " | ".join(metric_bits)


def _summarize_crash_tail(log_path: Path) -> str:
    """Extract a compact, high-signal crash summary from a run log."""

    lines = [line.strip() for line in _tail_log_lines(log_path, limit=40) if line.strip()]
    if not lines:
        return "no log tail available"

    interesting = [
        line for line in lines
        if any(
            token in line
            for token in (
                "Traceback",
                "Error",
                "Exception",
                "failure_reason",
                "AttributeError",
                "TypeError",
                "ValueError",
                "OSError",
            )
        )
    ]
    if interesting:
        return " | ".join(interesting[-3:])[:400]
    return " | ".join(lines[-3:])[:400]


def build_recent_run_context(limit: int = 8) -> str:
    """Summarize recent run artifacts for the LLM prompt."""

    runs_dir = get_runs_dir()
    if not runs_dir.exists():
        return ""

    run_dirs = sorted(
        [path for path in runs_dir.iterdir() if path.is_dir()],
        key=lambda path: path.stat().st_mtime,
    )
    if not run_dirs:
        return ""

    lines: list[str] = []
    for run_dir in run_dirs[-limit:]:
        result = _read_json_if_exists(run_dir / "result.json") or {}
        summary = _read_json_if_exists(run_dir / "summary.json") or {}
        eval_summary = _read_json_if_exists(run_dir / "eval_det" / "evaluation_summary.json")
        train_log_path = run_dir / "train.log"

        status = str(result.get("status", "unknown"))
        description = str(result.get("description", run_dir.name))
        metric_value = result.get("metric_value")
        metric_str = "n/a"
        if isinstance(metric_value, (int, float)):
            metric_str = f"{float(metric_value):.6f}"

        line = f"- {status.upper()} {metric_str} {description}"

        if summary:
            detail_parts = []
            epochs = summary.get("epochs_completed")
            if epochs is not None:
                detail_parts.append(f"epochs={epochs}")
            final_train_loss = summary.get("final_train_loss")
            if isinstance(final_train_loss, (int, float)):
                detail_parts.append(f"final_train={float(final_train_loss):.4f}")
            final_val_loss = summary.get("final_val_loss")
            if isinstance(final_val_loss, (int, float)):
                detail_parts.append(f"final_val={float(final_val_loss):.4f}")
            if summary.get("timed_out"):
                detail_parts.append("timed_out=true")
            if summary.get("non_finite_detected"):
                detail_parts.append("non_finite=true")
            failure_reason = summary.get("failure_reason")
            if failure_reason:
                detail_parts.append(f"failure={str(failure_reason)[:160]}")
            if detail_parts:
                line += " | " + ", ".join(detail_parts)
        train_losses, val_losses = _extract_component_losses(train_log_path)
        train_loss_text = _format_component_losses("train", train_losses)
        val_loss_text = _format_component_losses("val", val_losses)
        component_bits = [bit for bit in (train_loss_text, val_loss_text) if bit]
        if component_bits:
            line += " | " + " ; ".join(component_bits)

        eval_text = _extract_eval_metric_text(eval_summary)
        if eval_text:
            line += " | " + eval_text

        if not summary and status == "crash":
            line += f" | crash_tail={_summarize_crash_tail(train_log_path)}"

        lines.append(line[:700])

    baseline_metadata = get_promoted_baseline_metadata()
    baseline_block = ""
    if baseline_metadata:
        baseline_metric = baseline_metadata.get("metric_value")
        baseline_desc = baseline_metadata.get("description", "")
        if isinstance(baseline_metric, (int, float)):
            baseline_block = (
                "## Current promoted baseline\n"
                f"- {float(baseline_metric):.6f} {str(baseline_desc)[:220]}\n"
            )
        baseline_eval = _read_json_if_exists(
            get_workspace_dir() / "baseline" / "eval_det" / "evaluation_summary.json"
        )
        baseline_eval_text = _extract_eval_metric_text(baseline_eval)
        if baseline_eval_text:
            baseline_block += f"  {baseline_eval_text}\n"

    return baseline_block + "## Recent run artifact insights\n" + "\n".join(lines)


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------

def git(*args: str) -> tuple[str, int]:
    result = subprocess.run(
        ["git"] + list(args),
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip(), result.returncode


def git_current_branch() -> str:
    out, _ = git("branch", "--show-current")
    return out.strip()


def git_short_sha() -> str:
    out, _ = git("rev-parse", "--short", "HEAD")
    return out.strip() or "unknown"


def git_head_sha() -> str:
    out, _ = git("rev-parse", "HEAD")
    return out.strip()


def git_is_ancestor(ancestor: str, descendant: str = "HEAD") -> bool:
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, descendant],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def git_commit(message: str, files: list[str]) -> str:
    head_before = git_head_sha()
    for f in files:
        git("add", f)
    result = subprocess.run(
        ["git", "commit", "-m", message, "--"] + list(files),
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    head_after = git_head_sha()
    if result.returncode != 0 or head_after == head_before:
        detail = (result.stderr or result.stdout or "unknown git commit failure").strip()
        raise RuntimeError(detail)
    return head_after[:7]


def git_revert_file(filepath: str) -> None:
    """Restore a single file to the current HEAD commit."""
    git("restore", "--source=HEAD", "--staged", "--worktree", "--", filepath)


def git_reset_last_commit() -> None:
    """Undo the last commit but keep working tree changes."""
    git("reset", "--mixed", "HEAD~1")


def git_reset_commit(commit_sha: str) -> None:
    """Undo a specific HEAD commit while keeping working tree changes."""
    git("reset", "--mixed", f"{commit_sha}^")


def git_revert_commit(commit_sha: str) -> None:
    """Create a revert commit for an older experiment commit."""
    result = subprocess.run(
        ["git", "revert", "--no-edit", commit_sha],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        subprocess.run(
            ["git", "revert", "--abort"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
        )
        detail = (result.stderr or result.stdout or "unknown git revert failure").strip()
        raise RuntimeError(detail)


def git_discard_last_experiment(filepath: str) -> None:
    """Drop the last experiment commit and restore the touched file."""

    git_reset_last_commit()
    git_revert_file(filepath)


def git_file_is_dirty(filepath: str) -> bool:
    out, _ = git("status", "--porcelain", "--", filepath)
    return bool(out.strip())


def write_repo_file(filepath: str, source: str) -> None:
    target = PROJECT_ROOT / filepath
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(source, encoding="utf-8")


def restore_repo_file(filepath: str, original_source: str | None) -> None:
    """Restore a file to its pre-experiment state."""

    target = PROJECT_ROOT / filepath
    if original_source is None:
        if target.exists():
            target.unlink()
    else:
        write_repo_file(filepath, original_source)
    git("restore", "--staged", "--", filepath)


def write_repo_files_atomically(
    modified_sources: dict[str, str],
    original_sources: dict[str, str | None],
) -> None:
    """Write multiple files and restore already-written ones on failure."""

    written_paths: list[str] = []
    try:
        for filepath, source in modified_sources.items():
            write_repo_file(filepath, source)
            written_paths.append(filepath)
    except Exception:
        for filepath in reversed(written_paths):
            restore_repo_file(filepath, original_sources.get(filepath))
        raise


def rollback_experiment_change(
    filepath: str | None,
    original_source: str | None,
    *,
    committed: bool,
    commit_sha: str | None = None,
) -> None:
    """Restore the touched file to its pre-experiment contents."""

    if not filepath or original_source is None:
        return

    rollback_experiment_changes(
        {filepath: original_source},
        committed=committed,
        commit_sha=commit_sha,
    )


def rollback_experiment_changes(
    original_sources: dict[str, str | None],
    *,
    committed: bool,
    commit_sha: str | None = None,
) -> None:
    """Restore one or more files to their pre-experiment contents."""

    if not original_sources:
        return

    if committed and commit_sha:
        head_sha = git_head_sha()
        if head_sha == commit_sha:
            git_reset_commit(commit_sha)
        elif git_is_ancestor(commit_sha, head_sha):
            try:
                git_revert_commit(commit_sha)
            except RuntimeError as revert_err:
                log_to_file(f"ROLLBACK REVERT FAILED {commit_sha[:7]}: {revert_err}")
        else:
            log_to_file(
                f"ROLLBACK SKIPPED COMMIT RESET {commit_sha[:7]}: commit is no longer reachable from HEAD"
            )
    elif committed:
        git_reset_last_commit()
    for filepath, original_source in original_sources.items():
        restore_repo_file(filepath, original_source)


def git_push() -> None:
    out, rc = git("push", "origin", "HEAD")
    if rc != 0:
        git("push", "--set-upstream", "origin", "HEAD")
    log_to_file("git push: ok")


def setup_branch(tag: str, resume: bool) -> str:
    """Create or resume an autoresearch branch. Returns branch name."""
    branch = f"autoresearch/{tag}"
    existing, _ = git("branch", "--list", branch)
    if existing.strip():
        current, _ = git("branch", "--show-current")
        if current.strip() != branch:
            git("checkout", branch)
        log_to_file(f"Resumed branch: {branch}")
    else:
        git("checkout", "-b", branch)
        log_to_file(f"Created branch: {branch}")
    return branch


# ---------------------------------------------------------------------------
# Fatal API error
# ---------------------------------------------------------------------------

class FatalAPIError(Exception):
    pass


# ---------------------------------------------------------------------------
# LLM providers
# ---------------------------------------------------------------------------

DEFAULT_CLAUDE_MODEL = "claude-sonnet-4-6"
DEFAULT_GEMINI_MODEL = "gemini-3.1-pro-preview"
DEFAULT_DEEPSEEK_MODEL = "deepseek-reasoner"
DEFAULT_GROK_MODEL = "grok-3"
DEFAULT_DEEPSEEK_MAX_TOKENS = 64000
GEMINI_CACHE_MIN_TOKENS = 2048
DEFAULT_EXECUTION_MODE = "single_file"
DEFAULT_FORBIDDEN_PATHS = [
    "scripts/autoresearch.py",
    "dte/autoresearch",
    "scripts/agent.py",
    "auto_research.md",
    "program.md",
]
PATCH_RESPONSE_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "idea_id": {
            "type": "string",
            "description": "Optional structured idea identifier when following an ideas backlog.",
        },
        "file": {
            "type": "string",
            "description": "Repo-relative path to the single file to modify.",
        },
        "description": {
            "type": "string",
            "description": "Short one-line description of the proposed change.",
        },
        "changes": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "properties": {
                    "old": {
                        "type": "string",
                        "description": (
                            "Exact multi-line SEARCH block copied verbatim from the current file. "
                            "Include enough unchanged surrounding context so it matches exactly once."
                        ),
                    },
                    "new": {
                        "type": "string",
                        "description": (
                            "REPLACE block for that exact SEARCH block. Keep indentation and "
                            "unchanged code intact outside the edited region."
                        ),
                    },
                },
                "required": ["old", "new"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["file", "description", "changes"],
    "additionalProperties": False,
}
MULTI_FILE_SELECTION_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "idea_id": {
            "type": "string",
            "description": "Optional structured idea identifier when following an ideas backlog.",
        },
        "description": {
            "type": "string",
            "description": "Short one-line description of the proposed change.",
        },
        "files": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "string",
                "description": "Repo-relative path to an existing file that should be opened for editing.",
            },
        },
    },
    "required": ["description", "files"],
    "additionalProperties": False,
}
MULTI_FILE_PATCH_RESPONSE_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "idea_id": {
            "type": "string",
            "description": "Optional structured idea identifier when following an ideas backlog.",
        },
        "description": {
            "type": "string",
            "description": "Short one-line description of the proposed change.",
        },
        "files": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Repo-relative path to modify or create.",
                    },
                    "operation": {
                        "type": "string",
                        "description": "Optional file operation. Use modify for existing files or create for new files.",
                    },
                    "changes": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "old": {"type": "string"},
                                "new": {"type": "string"},
                            },
                            "required": ["old", "new"],
                            "additionalProperties": False,
                        },
                    },
                    "contents": {
                        "type": "string",
                        "description": "Full file contents when operation=create.",
                    },
                },
                "required": ["path"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["description", "files"],
    "additionalProperties": False,
}


def _normalize_gemini_thinking_level(model: str, thinking_level: str | None) -> str | None:
    if thinking_level is None:
        return None

    normalized = str(thinking_level).strip().lower()
    if not normalized:
        return None

    model_name = (model or "").lower()
    if "pro" in model_name:
        if normalized in {"low", "medium", "high"}:
            return normalized
        fallback = "low" if normalized == "minimal" else None
        if fallback:
            log_to_file(
                f"WARNING Gemini: thinking_level={normalized} is not supported for {model}; "
                f"using {fallback} instead"
            )
        else:
            log_to_file(
                f"WARNING Gemini: unsupported thinking_level={normalized} for {model}; omitting it"
            )
        return fallback

    if "flash" in model_name:
        if normalized in {"minimal", "low", "medium", "high"}:
            return normalized
        log_to_file(
            f"WARNING Gemini: unsupported thinking_level={normalized} for {model}; omitting it"
        )
        return None

    return normalized


def call_gemini(
    prompt: str,
    temperature: float | None = None,
    thinking_level: str | None = None,
    model: str = DEFAULT_GEMINI_MODEL,
    response_schema: dict | None = None,
    cached_content: str | None = None,
) -> str | None:
    """Call Google Gemini via the Google Gen AI SDK, with legacy fallback."""
    try:
        api_key = os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            log_to_file("ERROR Gemini: GEMINI_API_KEY not set")
            return None

        try:
            from google import genai
            from google.genai import types

            config_kwargs = {
                "max_output_tokens": 8192,
                "response_mime_type": "application/json",
            }
            if temperature is not None:
                config_kwargs["temperature"] = temperature
            if response_schema:
                config_kwargs["response_json_schema"] = response_schema
            if cached_content:
                config_kwargs["cached_content"] = cached_content
            if thinking_level:
                try:
                    normalized_level = _normalize_gemini_thinking_level(model, thinking_level)
                    if normalized_level:
                        config_kwargs["thinking_config"] = types.ThinkingConfig(
                            thinking_level=normalized_level
                        )
                except Exception:
                    log_to_file(
                        f"WARNING Gemini: could not configure thinking_level={thinking_level}"
                    )
            client = genai.Client(api_key=api_key)
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(**config_kwargs),
            )
            return response.text
        except ImportError:
            import google.generativeai as legacy_genai

            legacy_genai.configure(api_key=api_key)
            generation_config: dict = {
                "max_output_tokens": 8192,
                "response_mime_type": "application/json",
            }
            if temperature is not None:
                generation_config["temperature"] = temperature
            if response_schema:
                generation_config["response_schema"] = response_schema
            if cached_content:
                log_to_file("WARNING Gemini: cached_content is not supported by legacy SDK fallback")
            if thinking_level:
                try:
                    normalized_level = _normalize_gemini_thinking_level(model, thinking_level)
                    if normalized_level:
                        generation_config["thinking_config"] = {
                            "thinking_level": normalized_level,
                        }
                except Exception:
                    log_to_file(
                        f"WARNING Gemini: could not configure thinking_level={thinking_level}"
                    )
            model = legacy_genai.GenerativeModel(
                model_name=model,
                generation_config=generation_config,
            )
            response = model.generate_content(prompt)
            return response.text
    except ImportError:
        log_to_file(
            "ERROR Gemini: no Gemini SDK installed. Run: pip install google-genai"
        )
        return None
    except Exception as e:
        err = str(e)
        log_to_file(f"ERROR Gemini: {e}")
        if any(x in err.lower() for x in ("api_key_invalid", "quota", "billing", "permission")):
            raise FatalAPIError(err)
        return None


def call_claude(
    prompt: str,
    temperature: float | None = None,
    thinking_level: str | None = None,
    model: str = DEFAULT_CLAUDE_MODEL,
) -> str | None:
    try:
        import anthropic
        client = anthropic.Anthropic(timeout=180.0)
        kwargs: dict = {
            "model": model,
            "max_tokens": 4096,
            "messages": [{"role": "user", "content": prompt}],
        }
        if temperature is not None:
            kwargs["temperature"] = temperature
        response = client.messages.create(**kwargs)
        return response.content[0].text
    except ImportError:
        return None
    except Exception as e:
        err = str(e)
        log_to_file(f"ERROR Claude: {e}")
        if any(x in err.lower() for x in ("credit balance", "authentication", "billing")):
            raise FatalAPIError(err)
        return None


def call_claude_opus(
    prompt: str,
    temperature: float | None = None,
    thinking_level: str | None = None,
) -> str | None:
    try:
        import anthropic
        client = anthropic.Anthropic(timeout=600.0)
        text_parts: list[str] = []
        with client.messages.stream(
            model="claude-opus-4-6",
            max_tokens=36000,
            thinking={"type": "enabled", "budget_tokens": 32000},
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            for event in stream:
                if hasattr(event, "type") and event.type == "content_block_delta":
                    if hasattr(event, "delta") and event.delta.type == "text_delta":
                        text_parts.append(event.delta.text)
        return "".join(text_parts) if text_parts else None
    except ImportError:
        return None
    except Exception as e:
        err = str(e)
        log_to_file(f"ERROR Claude Opus: {e}")
        if any(x in err.lower() for x in ("credit balance", "authentication", "billing")):
            raise FatalAPIError(err)
        return None


def call_openai(
    prompt: str,
    temperature: float | None = None,
    thinking_level: str | None = None,
    model: str = "o3",
) -> str | None:
    try:
        from openai import OpenAI
        client = OpenAI(timeout=600.0)
        kwargs: dict = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
        }
        if model.startswith("o"):
            kwargs["max_completion_tokens"] = 32000
        else:
            kwargs["max_completion_tokens"] = 4096
            if temperature is not None:
                kwargs["temperature"] = temperature
        response = client.chat.completions.create(**kwargs)
        return response.choices[0].message.content
    except ImportError:
        return None
    except Exception as e:
        err = str(e)
        log_to_file(f"ERROR OpenAI ({model}): {e}")
        if any(x in err.lower() for x in ("insufficient_quota", "billing", "authentication")):
            raise FatalAPIError(err)
        return None


def call_deepseek(
    prompt: str,
    temperature: float | None = None,
    thinking_level: str | None = None,
    model: str = DEFAULT_DEEPSEEK_MODEL,
    max_tokens: int | None = None,
) -> str | None:
    try:
        from openai import OpenAI

        api_key = os.environ.get("DEEPSEEK_API_KEY", "")
        if not api_key:
            log_to_file("ERROR DeepSeek: DEEPSEEK_API_KEY not set")
            return None

        client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com",
            timeout=600.0,
        )
        kwargs: dict = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            # DeepSeek documents a 64K max_tokens ceiling for reasoning mode,
            # and this budget includes the reasoning content.
            "max_tokens": int(max_tokens or DEFAULT_DEEPSEEK_MAX_TOKENS),
        }
        # DeepSeek documents that sampling controls like temperature do not apply
        # to deepseek-reasoner, so we keep the config value but do not send it.
        if temperature is not None and model != DEFAULT_DEEPSEEK_MODEL:
            kwargs["temperature"] = temperature
        response = client.chat.completions.create(**kwargs)
        return response.choices[0].message.content
    except ImportError:
        return None
    except Exception as e:
        err = str(e)
        log_to_file(f"ERROR DeepSeek ({model}): {e}")
        if any(x in err.lower() for x in ("insufficient_balance", "authentication", "billing")):
            raise FatalAPIError(err)
        return None


def call_grok(
    prompt: str,
    temperature: float | None = None,
    thinking_level: str | None = None,
    model: str = DEFAULT_GROK_MODEL,
) -> str | None:
    try:
        from openai import OpenAI
        client = OpenAI(
            api_key=os.environ.get("XAI_API_KEY", ""),
            base_url="https://api.x.ai/v1",
            timeout=300.0,
        )
        kwargs: dict = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 4096,
        }
        if temperature is not None:
            kwargs["temperature"] = temperature
        response = client.chat.completions.create(**kwargs)
        return response.choices[0].message.content
    except ImportError:
        return None
    except Exception as e:
        err = str(e)
        log_to_file(f"ERROR Grok: {e}")
        if any(x in err.lower() for x in ("insufficient_quota", "billing", "authentication")):
            raise FatalAPIError(err)
        return None


def call_local(
    prompt: str,
    temperature: float | None = None,
    thinking_level: str | None = None,
) -> str | None:
    try:
        import requests
        resp = requests.post(
            "http://127.0.0.1:1234/v1/chat/completions",
            json={
                "model": "deepseek/deepseek-r1-0528-qwen3-8b",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 4096,
                "temperature": temperature if temperature is not None else 0.7,
            },
            timeout=120,
        )
        return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        log_to_file(f"ERROR LM Studio: {e}")
        return None


def _build_llm_backend(provider_name: str, provider_model: str):
    """Build a normalized invoker and display name for a provider/model pair."""

    if provider_name == "local":
        return call_local, "LM Studio (local)"
    if provider_name == "opus":
        return call_claude_opus, "Claude Opus 4.6"
    if provider_name == "claude":
        model_name = provider_model or DEFAULT_CLAUDE_MODEL
        return (
            lambda prompt, temperature=None, thinking_level=None: call_claude(
                prompt,
                temperature=temperature,
                thinking_level=thinking_level,
                model=model_name,
            ),
            f"Claude {model_name}",
        )
    if provider_name == "openai":
        model_name = provider_model or "o3"
        return (
            lambda prompt, temperature=None, thinking_level=None: call_openai(
                prompt,
                temperature=temperature,
                thinking_level=thinking_level,
                model=model_name,
            ),
            f"OpenAI {model_name}",
        )
    if provider_name == "grok":
        model_name = provider_model or DEFAULT_GROK_MODEL
        return (
            lambda prompt, temperature=None, thinking_level=None: call_grok(
                prompt,
                temperature=temperature,
                thinking_level=thinking_level,
                model=model_name,
            ),
            f"xAI {model_name}",
        )
    if provider_name == "deepseek":
        model_name = provider_model or DEFAULT_DEEPSEEK_MODEL
        return (
            lambda prompt, temperature=None, thinking_level=None, max_tokens=None: call_deepseek(
                prompt,
                temperature=temperature,
                thinking_level=thinking_level,
                model=model_name,
                max_tokens=max_tokens,
            ),
            f"DeepSeek {model_name}",
        )

    model_name = provider_model or DEFAULT_GEMINI_MODEL
    return (
        lambda prompt, temperature=None, thinking_level=None, response_schema=None, cached_content=None: call_gemini(
            prompt,
            temperature=temperature,
            thinking_level=thinking_level,
            model=model_name,
            response_schema=response_schema,
            cached_content=cached_content,
        ),
        f"Gemini {model_name}",
    )


def _infer_provider_from_model(model_name: str) -> str:
    """Infer the provider from a configured default model name."""

    name = (model_name or "").strip().lower()
    if not name:
        return "deepseek"
    if name == "local":
        return "local"
    if name.startswith("deepseek"):
        return "deepseek"
    if name.startswith("gemini"):
        return "gemini"
    if name.startswith("claude-opus"):
        return "opus"
    if name.startswith("claude"):
        return "claude"
    if name.startswith("grok"):
        return "grok"
    if name.startswith(("gpt-", "o")):
        return "openai"
    return "gemini"


def _default_model_for_provider(provider_name: str, configured_model: str | None = None) -> str:
    if configured_model:
        return configured_model
    if provider_name == "local":
        return "local"
    if provider_name == "deepseek":
        return DEFAULT_DEEPSEEK_MODEL
    if provider_name == "claude":
        return DEFAULT_CLAUDE_MODEL
    if provider_name == "opus":
        return "claude-opus-4-6"
    if provider_name == "grok":
        return DEFAULT_GROK_MODEL
    if provider_name == "openai":
        return "o3"
    return DEFAULT_GEMINI_MODEL


def _provider_supports_temperature(provider: str, model_name: str) -> bool:
    """Return whether the selected provider/model accepts temperature."""

    if provider == "local":
        return True
    if provider == "deepseek" and model_name == DEFAULT_DEEPSEEK_MODEL:
        return False
    if provider == "openai" and model_name.startswith("o"):
        return False
    if provider == "opus":
        return False
    return True


def _provider_supports_thinking_level(provider: str) -> bool:
    """Return whether a provider exposes a configurable thinking level here."""

    return provider == "gemini"


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

MODIFIABLE_FILES = [
    "configs/training_default.yaml",
    "scripts/train.py",
    "dte/models/encoder.py",
    "dte/models/decoder.py",
    "dte/models/latent_sde.py",
    "dte/models/digital_twin.py",
    "dte/training/trainer.py",
    "dte/training/losses.py",
]

RESEARCH_PRIORITIES = """
## Research priorities:
- Do NOT spend experiments primarily on routine hyperparameter optimization. Assume Optuna or a separate sweep handles LR, batch size, clip, loss weights, latent_dim, hidden_dim, layer counts, seq_len, and similar scalar tuning.
- Small hyperparameter changes are allowed only when they support a larger architectural, mathematical, or training-logic idea.
- Prefer substantial code, logic, and mathematics changes in Python source files over YAML/config edits.
- Focus on ideas like better drift/diffusion parameterisations, more stable latent stochasticity, improved rollout training logic, stronger physics-informed inductive biases, smarter residual/skip pathways, or novel consistency/objective formulations.
- Favor out-of-the-box but maintainable changes that alter the training architecture or learning dynamics, not tiny coefficient nudges.
- Bold changes are encouraged: adding/removing full helper functions, classes, pathways, or training mechanisms is good when the idea is coherent and testable in one file.
- If you touch scripts/train.py, do it for algorithmic or training-loop logic reasons, not to expose or tweak another scalar hyperparameter.
- Minimal support plumbing for a new algorithmic idea is acceptable. Pure config-only tuning should be a fallback, not the main search strategy.
"""

PROMISING_DIRECTIONS = """
## Promising code-level directions:
- Reparameterise diffusion or variance pathways to improve stability without collapsing uncertainty.
- Add principled residual, gating, or skip structures that help long-horizon rollout fidelity.
- Improve how one-step and rollout objectives interact so optimization better matches downstream trajectory quality.
- Introduce architecture-level regularisers or consistency losses tied to latent dynamics rather than static scalar weights alone.
- Improve numerical robustness in latent propagation, decoder coupling, or teacher-forcing transitions.
- Add mathematically motivated structure that helps generalisation across systems while preserving the generic SystemSpec design.
"""

JAX_PITFALLS = """
## JAX/Equinox pitfalls to avoid:
- Do NOT use Python float() casts inside JIT-compiled functions (ConcretizationTypeError).
- Do NOT change the tree structure of Equinox modules mid-training (shape errors on load).
- YAML changes must keep all required keys present; missing keys cause KeyError on load.
- In configs, numeric values like 3e-4 must stay as floats, not strings.
- Do NOT use eqx.field(static=True) for JAX arrays — only for non-array metadata.
- Use jnp.array(indices) not plain Python lists when indexing with Array.at[].set().
- HDF5 datasets must use key "time" (not "t") for the time axis.
- LossComputer and Trainer expect a PhysicsLoss instance and state_names list; do NOT hardcode CSTR physics.
- Do NOT modify scripts/autoresearch.py, dte/autoresearch/workflow.py, or program.md.
- The system is chosen via --system_config (SystemSpec + ProcessSimulator registry), not hardcoded.
"""

ARCHITECTURE_GUARDRAILS = """
## Architecture guardrails:
- Preserve the generic SystemSpec / ProcessSimulator / PhysicsLoss architecture.
- In dte/models and dte/training, keep changes system-agnostic: no branches or string literals for specific systems.
- Do NOT hardcode config-like numeric values in the generic core: decoder bounds, normalization constants, default physical states, or fixed dimensions.
- If a numeric bound, scale, or constraint matters, put it in configs or thread it through SystemSpec/config instead.
- Generic algorithmic changes that apply across systems are encouraged, including ordinary numeric tuning that is not a baked-in system constraint.
"""

_GENERIC_CORE_PREFIXES = (
    "dte/models/",
    "dte/training/",
)
_CONFIG_LIKE_NAMES = (
    "state_center",
    "state_scale",
    "control_center",
    "control_scale",
    "disturbance_center",
    "disturbance_scale",
    "decoder_constraints",
    "default_initial_state",
    "default_nominal_disturbance",
    "nominal_disturbance",
)
_DIMENSION_NAMES = (
    "state_dim",
    "control_dim",
    "disturbance_dim",
    "param_dim",
)
_SYSTEM_NAME_LITERAL_RE = re.compile(r"""["'](?:cstr|heat_exchanger)["']""")
_NUMERIC_LITERAL_RE = re.compile(r"(?<![\w.])\d+(?:\.\d+)?(?:e[+-]?\d+)?(?![\w.])", re.IGNORECASE)
_ALLOWED_GENERIC_LITERALS = {"0", "1", "0.0", "1.0"}
_CONFIG_LIKE_ASSIGNMENT_RE = re.compile(
    rf"""\b(?:{"|".join(_CONFIG_LIKE_NAMES)})\s*=\s*(?:jnp\.)?(?:array\s*\(|zeros\s*\(|ones\s*\(|full\s*\(|\[)"""
)
_DIMENSION_ASSIGNMENT_RE = re.compile(
    rf"""\b(?:{"|".join(_DIMENSION_NAMES)})\s*=\s*\d+"""
)
_DECODER_CONSTRAINT_LITERAL_RE = re.compile(
    r"""(?:"type"\s*:\s*"softplus"|"""
    r""""type"\s*:\s*"sigmoid_range"|"""
    r"""\b(?:low|high|bias)\s*=\s*[-+]?\d|"""
    r""""(?:low|high|bias)"\s*:\s*[-+]?\d)"""
)


def _iter_added_lines(original_source: str, modified_source: str):
    """Yield added lines from a proposed patch."""

    for line in difflib.ndiff(original_source.splitlines(), modified_source.splitlines()):
        if line.startswith("+ "):
            yield line[2:]


def validate_architecture_guardrails(
    filepath: str,
    original_source: str,
    modified_source: str,
) -> str | None:
    """Reject proposals that reintroduce hardcoded system assumptions into dte/."""

    if not filepath.startswith(_GENERIC_CORE_PREFIXES):
        return None

    for line in _iter_added_lines(original_source, modified_source):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        if _SYSTEM_NAME_LITERAL_RE.search(line):
            return (
                "Architecture guardrail: keep dte/ system-agnostic; "
                "do not introduce system-specific names or branches."
            )

        if (
            _CONFIG_LIKE_ASSIGNMENT_RE.search(line)
            or _DIMENSION_ASSIGNMENT_RE.search(line)
            or _DECODER_CONSTRAINT_LITERAL_RE.search(line)
        ):
            literals = [
                match.group(0)
                for match in _NUMERIC_LITERAL_RE.finditer(line)
                if match.group(0).lower() not in _ALLOWED_GENERIC_LITERALS
            ]
            if literals:
                return (
                    "Architecture guardrail: avoid hardcoded config-like numeric "
                    "constraints in dte/models or dte/training; move bounds, "
                    "normalization values, defaults, and fixed dims to "
                    "config/SystemSpec instead."
                )

    return None


def _categorize(desc: str) -> str:
    d = desc.lower()
    if any(w in d for w in ("latent_dim", "hidden_dim", "layer", "depth", "width", "encoder", "decoder", "diffusion", "drift")):
        return "architecture"
    if any(w in d for w in ("lr", "learning rate", "warmup", "schedule", "cosine", "peak_lr")):
        return "lr_schedule"
    if any(w in d for w in ("kl", "reconstruction", "trajectory", "mass", "energy", "loss weight", "physics")):
        return "loss_weights"
    if any(w in d for w in ("batch", "seq_len", "stride", "val_split")):
        return "data/batch"
    if any(w in d for w in ("clip", "weight decay", "regulariz", "dropout")):
        return "regularization"
    if any(w in d for w in ("solver", "heun", "euler", "dt_ratio", "sde")):
        return "sde_solver"
    if any(w in d for w in ("optax", "adam", "optimizer", "beta")):
        return "optimizer"
    return "other"


_IDEA_MARKER_RE = re.compile(r"\[idea:([a-zA-Z0-9_.-]+)\]")


def _extract_idea_id_from_description(description: str) -> str:
    match = _IDEA_MARKER_RE.search(description or "")
    return match.group(1).strip().lower() if match else ""


def annotate_description_with_idea(description: str, idea_id: str | None) -> str:
    clean_description = (description or "").strip()
    clean_idea_id = (idea_id or "").strip().lower()
    if not clean_idea_id:
        return clean_description
    if _extract_idea_id_from_description(clean_description) == clean_idea_id:
        return clean_description
    return f"[idea:{clean_idea_id}] {clean_description}"


def _row_matches_structured_idea(row: dict, idea: dict) -> bool:
    target_id = str(idea.get("id", "")).strip().lower()
    target_title = str(idea.get("title", "")).strip().lower()
    target_file = str(idea.get("target_file", "")).strip()
    description = str(row.get("description", ""))
    if target_id and _extract_idea_id_from_description(description) == target_id:
        return True
    if target_title and target_title in description.lower():
        if not target_file or row.get("file", "") == target_file:
            return True
    return False


def _history_has_attempted_idea(
    history: list[dict],
    idea: dict,
    *,
    resolved_only: bool = False,
) -> bool:
    for row in history:
        if resolved_only and row.get("status") not in ("keep", "discard"):
            continue
        if _row_matches_structured_idea(row, idea):
            return True
    return False


def get_eligible_structured_ideas(
    ideas: list[dict],
    editable_targets: list[str],
    forced_file: str | None,
    execution_mode: str = DEFAULT_EXECUTION_MODE,
    forbidden_paths: list[str] | None = None,
) -> list[dict]:
    eligible: list[dict] = []
    for idea in ideas:
        idea_copy = dict(idea)
        target_file = str(idea_copy.get("target_file", "")).strip()
        if forced_file and target_file and target_file != forced_file:
            continue
        if forced_file and not target_file:
            idea_copy["target_file"] = forced_file
            target_file = forced_file
        if target_file:
            if execution_mode == "multi_file":
                if not is_allowed_edit_path(target_file, editable_targets, forbidden_paths):
                    continue
            elif target_file not in editable_targets:
                continue
        elif forced_file and execution_mode == "multi_file":
            if not is_allowed_edit_path(forced_file, editable_targets, forbidden_paths):
                continue
        elif forced_file and execution_mode != "multi_file" and forced_file not in editable_targets:
            continue
        eligible.append(idea_copy)
    return eligible


def count_resolved_structured_ideas(history: list[dict], eligible_ideas: list[dict]) -> int:
    return sum(
        1
        for idea in eligible_ideas
        if _history_has_attempted_idea(history, idea, resolved_only=True)
    )


def select_next_structured_idea(
    ideas: list[dict],
    history: list[dict],
    editable_targets: list[str],
    forced_file: str | None,
    execution_mode: str = DEFAULT_EXECUTION_MODE,
    forbidden_paths: list[str] | None = None,
) -> dict | None:
    """Pick the next pending idea from the backlog."""

    candidates: list[dict] = []
    for idea in get_eligible_structured_ideas(
        ideas,
        editable_targets,
        forced_file,
        execution_mode=execution_mode,
        forbidden_paths=forbidden_paths,
    ):
        if _history_has_attempted_idea(history, idea, resolved_only=True):
            continue
        candidates.append(idea)
    return candidates[0] if candidates else None


def build_structured_ideas_section(
    ideas: list[dict],
    selected_idea: dict | None,
    execution_mode: str = DEFAULT_EXECUTION_MODE,
) -> str:
    if not ideas:
        return ""

    lines = ["## Structured idea backlog:"]
    for index, idea in enumerate(ideas[:8], start=1):
        idea_bits = [f"{index}. {idea['id']}"]
        target_file = str(idea.get("target_file", "")).strip()
        if target_file:
            idea_bits.append(f"file={target_file}")
        if selected_idea and idea.get("id") == selected_idea.get("id"):
            idea_bits.append("STATUS=NEXT")
        lines.append("  " + " | ".join(idea_bits))
        lines.append(f"     title: {idea['title']}")
        rationale = str(idea.get("rationale", "")).strip()
        if rationale:
            lines.append(f"     rationale: {rationale}")
        instructions = idea.get("instructions", []) or []
        for instruction in instructions[:3]:
            lines.append(f"     - {instruction}")

    if selected_idea:
        lines.append("")
        lines.append("## Required next idea")
        lines.append(f"- idea_id: {selected_idea['id']}")
        lines.append(f"- title: {selected_idea['title']}")
        target_file = str(selected_idea.get("target_file", "")).strip()
        if target_file:
            lines.append(f"- target_file: {target_file}")
        rationale = str(selected_idea.get("rationale", "")).strip()
        if rationale:
            lines.append(f"- rationale: {rationale}")
        instructions = selected_idea.get("instructions", []) or []
        if instructions:
            lines.append("- implementation guidance:")
            for instruction in instructions:
                lines.append(f"  - {instruction}")
        lines.append(
            "- You must attempt this exact queued idea now."
        )
        lines.append(
            "- Use exactly this idea_id in the response. Do not invent a different backlog item or a new idea_id."
        )
        if execution_mode == "multi_file":
            lines.append(
                "- Keep the implementation anchored on this queued idea. Additional files are allowed when needed."
            )
        else:
            lines.append(
                "- Stay on this target file and keep the implementation aligned with the queued title/rationale."
            )

    return "\n".join(lines) + "\n"


def enforce_selected_idea_on_proposal(
    proposal: dict,
    selected_idea: dict | None,
    execution_mode: str = DEFAULT_EXECUTION_MODE,
) -> dict:
    if not selected_idea:
        return proposal

    coerced = dict(proposal)
    coerced["idea_id"] = selected_idea["id"]
    target_file = str(selected_idea.get("target_file", "")).strip()
    if target_file and execution_mode != "multi_file":
        coerced["file"] = target_file
    elif target_file and execution_mode == "multi_file":
        files = _normalize_text_list(coerced.get("files"))
        if target_file not in files:
            coerced["files"] = [target_file] + files
    return coerced


def effective_run_target(
    requested_max_runs: int,
    history: list[dict],
    structured_ideas: list[dict],
    editable_targets: list[str],
    forced_file: str | None,
    execution_mode: str = DEFAULT_EXECUTION_MODE,
    forbidden_paths: list[str] | None = None,
) -> int:
    eligible_ideas = get_eligible_structured_ideas(
        structured_ideas,
        editable_targets,
        forced_file,
        execution_mode=execution_mode,
        forbidden_paths=forbidden_paths,
    )
    if eligible_ideas:
        return len(eligible_ideas)
    return requested_max_runs


def _count_recent_failures(history: list[dict]) -> int:
    count = 0
    for r in reversed(history):
        if r["status"] == "keep":
            break
        count += 1
    return min(count, 10)


def build_static_prompt_context(
    modifiable_files: list[str],
    agent_context: str = "",
) -> str:
    files_list = "\n".join(f"  - {f}" for f in modifiable_files)
    context_section = ""
    if agent_context:
        context_section = f"\n## Repo context\n{agent_context}\n"

    return f"""You are an autonomous ML researcher. Your goal: minimise best_val_loss for a \
physics-informed latent Neural SDE (digital twin) trained on process system data (CSTR, heat exchanger, or other registered systems).

## Constraints
- You MUST only modify ONE of these files per experiment:
{files_list}
- Do NOT modify the experiment harness: scripts/autoresearch.py, dte/autoresearch/*, program.md
- One idea per experiment. Keep changes minimal and surgical.
- Do NOT focus on routine hyperparameter optimization or config sweeps. Scalar training/config tweaks are allowed only when they are clearly in service of a larger architectural, mathematical, or training-logic change.
- Preserve the generic architecture. In dte/models and dte/training, do not add system-specific branches or bake config-like numbers into code; generic numeric algorithmic tweaks are fine, but bounds/scales/defaults/dims should live in config/SystemSpec.
- Available packages: jax, equinox, diffrax, optax, jaxtyping, numpy, yaml, h5py (no new installs).
{JAX_PITFALLS}{ARCHITECTURE_GUARDRAILS}{RESEARCH_PRIORITIES}{PROMISING_DIRECTIONS}{context_section}
## Output contract
- Respond with ONLY one JSON object.
- Treat each entry in `changes` as an exact SEARCH/REPLACE block:
  - `old` is the SEARCH block copied verbatim from the current file.
  - `new` is the REPLACE block.
- Use multi-line `old` blocks with enough unchanged surrounding context to match exactly once.
- Preserve indentation and unchanged code outside the edited region.
"""


def build_multi_file_static_prompt_context(
    editable_targets: list[str],
    editable_file_inventory: list[str],
    forbidden_paths: list[str],
    agent_context: str = "",
) -> str:
    targets_list = "\n".join(f"  - {path}" for path in editable_targets)
    forbidden_list = "\n".join(f"  - {path}" for path in forbidden_paths)
    inventory_list = "\n".join(f"  - {path}" for path in editable_file_inventory)
    context_section = ""
    if agent_context:
        context_section = f"\n## Repo context\n{agent_context}\n"

    return f"""You are an autonomous ML researcher. Your goal: minimise best_val_loss for a \
physics-informed latent Neural SDE (digital twin) trained on process system data (CSTR, heat exchanger, or other registered systems).

## Constraints
- You may modify MULTIPLE files per experiment when needed to implement one coherent idea fully.
- Editable files must stay within these repo roots/files:
{targets_list}
- You must NEVER modify these locked files/directories:
{forbidden_list}
- One idea per experiment. Full implementations are allowed when the idea genuinely needs multiple coordinated edits.
- Preserve the generic architecture. In dte/models and dte/training, do not add system-specific branches or bake config-like numbers into code; generic numeric algorithmic tweaks are fine, but bounds/scales/defaults/dims should live in config/SystemSpec.
- Prefer modifying existing files. You may create new files inside the editable roots when necessary.
- Available packages: jax, equinox, diffrax, optax, jaxtyping, numpy, yaml, h5py (no new installs).
{JAX_PITFALLS}{ARCHITECTURE_GUARDRAILS}{RESEARCH_PRIORITIES}{PROMISING_DIRECTIONS}{context_section}
## Editable file inventory
{inventory_list}
"""


def build_dynamic_prompt(
    file_path: str,
    file_source: str,
    history: list[dict],
    best_loss: float,
    recent_run_context: str = "",
    ideas: list[dict] | None = None,
    selected_idea: dict | None = None,
) -> str:
    history_section = ""
    near_misses_str = ""

    if history:
        lines = []
        categories: dict[str, dict] = {}
        near_misses: list[tuple[float, str, str]] = []

        for r in history:
            status = r["status"]
            loss = r.get("val_loss", 999.0)
            desc = r.get("description", "")
            changed_file = r.get("file", "?")
            cat = _categorize(desc)

            if status == "crash":
                lines.append(f"  CRASH  [{changed_file}] {desc}")
            else:
                gap = loss - best_loss
                marker = " <-- BEST" if loss == best_loss and status == "keep" else ""
                lines.append(f"  {loss:.4f} (+{gap:.4f})  [{changed_file}] {desc}{marker}")
                if status == "discard" and loss < 999 and gap < best_loss * 0.05:
                    near_misses.append((gap, changed_file, desc))

            if cat not in categories:
                categories[cat] = {"count": 0, "best_gap": 999.0}
            categories[cat]["count"] += 1
            if status == "discard" and loss < 999:
                categories[cat]["best_gap"] = min(categories[cat]["best_gap"], loss - best_loss)

        cat_summary = "\n## Category summary (attempts / closest gap to baseline):\n"
        for cat, stats in sorted(categories.items(), key=lambda x: -x[1]["count"]):
            gap_str = f"+{stats['best_gap']:.4f}" if stats["best_gap"] < 999 else "n/a"
            cat_summary += f"  {cat}: {stats['count']} tried, closest gap {gap_str}\n"

        if near_misses:
            near_misses.sort(key=lambda x: x[0])
            near_misses_str = "\n## Near misses (within 5% of best - consider combining or tweaking):\n"
            for gap, fpath, desc in near_misses[:5]:
                near_misses_str += f"  +{gap:.4f}  [{fpath}] {desc}\n"

        history_section = cat_summary + near_misses_str
        history_section += f"\n## All {len(history)} experiments:\n"
        history_section += "\n".join(lines) + "\n"

    fail_streak = _count_recent_failures(history) if history else 0
    streak_str = ""
    if fail_streak >= 5:
        streak_str = f"""
## WARNING: {fail_streak} consecutive failures. Change strategy completely.
Look at category summary - avoid exhausted categories. Try a different file or approach.
"""
    elif fail_streak >= 3:
        streak_str = f"\n## CAUTION: {fail_streak} consecutive failures. Try simpler change.\n"

    recent_runs_section = ""
    if recent_run_context:
        recent_runs_section = f"\n{recent_run_context}\n"

    ideas_section = ""
    if ideas:
        ideas_section = "\n" + build_structured_ideas_section(
            ideas,
            selected_idea,
            execution_mode="single_file",
        )

    return f"""## Current best_val_loss: {best_loss:.6f}
{streak_str}{history_section}{recent_runs_section}{ideas_section}
## Currently showing: {file_path}
```
{file_source}
```

## Your task
Propose ONE modification to lower best_val_loss. Do NOT repeat or closely variant anything already tried.
Think creatively about what hasn't been attempted yet.

Respond with ONLY a JSON object (no markdown fences, no explanation):
{{
  "idea_id": "optional structured idea identifier when following a queued idea",
  "file": "repo-relative path to the file you want to modify (must be from the allowed list)",
  "description": "short description of the change (no tabs)",
  "changes": [
    {{
      "old": "exact SEARCH block copied from the current file",
      "new": "replacement REPLACE block"
    }}
  ]
}}

Each change is a find-and-replace. "old" MUST appear exactly once in the file.
Keep changes minimal. One idea at a time."""


def build_prompt(
    file_path: str,
    file_source: str,
    history: list[dict],
    best_loss: float,
    modifiable_files: list[str],
    agent_context: str = "",
    recent_run_context: str = "",
    ideas: list[dict] | None = None,
    selected_idea: dict | None = None,
) -> str:
    return build_static_prompt_context(
        modifiable_files,
        agent_context=agent_context,
    ) + build_dynamic_prompt(
        file_path,
        file_source,
        history,
        best_loss,
        recent_run_context=recent_run_context,
        ideas=ideas,
        selected_idea=selected_idea,
    )


def _build_history_summary(history: list[dict], best_loss: float) -> str:
    if not history:
        return ""

    lines = ["## Recent experiment history:"]
    for row in history[-12:]:
        status = str(row.get("status", "crash"))
        description = str(row.get("description", "")).strip()
        changed_file = str(row.get("file", "")).strip() or "?"
        if status == "crash":
            lines.append(f"- CRASH [{changed_file}] {description}")
            continue
        loss = float(row.get("val_loss", 999.0) or 999.0)
        if loss >= 999.0:
            lines.append(f"- {status.upper()} [{changed_file}] {description}")
            continue
        gap = loss - best_loss
        lines.append(f"- {status.upper()} {loss:.6f} ({gap:+.6f}) [{changed_file}] {description}")
    return "\n".join(lines) + "\n"


def build_multi_file_selection_prompt(
    history: list[dict],
    best_loss: float,
    recent_run_context: str = "",
    ideas: list[dict] | None = None,
    selected_idea: dict | None = None,
) -> str:
    history_section = _build_history_summary(history, best_loss)
    ideas_section = ""
    if ideas:
        ideas_section = "\n" + build_structured_ideas_section(
            ideas,
            selected_idea,
            execution_mode="multi_file",
        )
    recent_runs_section = f"\n{recent_run_context}\n" if recent_run_context else ""

    return f"""## Current best_val_loss: {best_loss:.6f}
{history_section}{recent_runs_section}{ideas_section}
## Your task
Choose the existing repository files you need to inspect or modify to implement one coherent experiment fully.

Respond with ONLY one JSON object:
{{
  "idea_id": "optional structured idea identifier when following a queued idea",
  "description": "short description of the full experiment",
  "files": [
    "repo-relative path to an existing file you need opened"
  ]
}}

Requirements:
- Every file path must already exist in the editable inventory.
- Choose all files you genuinely need; there is no single-file restriction in this mode.
- Do not include locked files.
- Keep the experiment aligned with the queued idea when one is provided."""


def build_multi_file_patch_prompt(
    description: str,
    file_sources: dict[str, str],
    editable_targets: list[str],
    recent_run_context: str = "",
    selected_idea: dict | None = None,
) -> str:
    files_section = "\n\n".join(
        f"## File: {path}\n```\n{source}\n```"
        for path, source in file_sources.items()
    )
    idea_section = ""
    if selected_idea:
        title = str(selected_idea.get("title", "")).strip()
        rationale = str(selected_idea.get("rationale", "")).strip()
        instructions = selected_idea.get("instructions", []) or []
        idea_lines = ["## Selected idea"]
        idea_lines.append(f"- idea_id: {selected_idea['id']}")
        if title:
            idea_lines.append(f"- title: {title}")
        if rationale:
            idea_lines.append(f"- rationale: {rationale}")
        if instructions:
            idea_lines.append("- implementation guidance:")
            for instruction in instructions:
                idea_lines.append(f"  - {instruction}")
        idea_section = "\n" + "\n".join(idea_lines) + "\n"

    recent_runs_section = f"\n{recent_run_context}\n" if recent_run_context else ""
    editable_list = "\n".join(f"  - {path}" for path in editable_targets)

    return f"""## Experiment description
{description}
{recent_runs_section}{idea_section}
## Editable roots/files
{editable_list}

## Opened files
{files_section}

## Your task
Implement the experiment fully across these files. You may also CREATE new files inside the editable roots when necessary.

Respond with ONLY one JSON object:
{{
  "idea_id": "optional structured idea identifier when following a queued idea",
  "description": "short description of the final experiment",
  "files": [
    {{
      "path": "repo-relative path",
      "operation": "modify",
      "changes": [
        {{
          "old": "exact SEARCH block copied from the current file",
          "new": "replacement REPLACE block"
        }}
      ]
    }},
    {{
      "path": "repo-relative path for a new file",
      "operation": "create",
      "contents": "full contents of the new file"
    }}
  ]
}}

Requirements:
- For modify operations, every SEARCH block must appear exactly once in the provided file text.
- For create operations, provide the full file contents in `contents`.
- Modify only files that are shown above. Create operations may target new paths inside the editable roots.
- Do not target locked files."""


# ---------------------------------------------------------------------------
# Response parsing & patch application
# ---------------------------------------------------------------------------

MAX_CONSECUTIVE_PARSE_FAILURES = 3
LLM_REQUEST_TIMEOUT_SECONDS = 180


def _strip_markdown_fences(text: str) -> str:
    raw = text.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\s*```$", "", raw)
    return raw.strip()


def _extract_balanced_json_object(text: str) -> str | None:
    start = None
    depth = 0
    in_string = False
    escape = False

    for idx, char in enumerate(text):
        if escape:
            escape = False
            continue
        if char == "\\" and in_string:
            escape = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            if depth == 0:
                start = idx
            depth += 1
        elif char == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start is not None:
                return text[start:idx + 1]
    return None


def parse_response(text: str) -> dict | None:
    if not text:
        return None

    candidates: list[str] = []
    raw = _strip_markdown_fences(text)
    if raw:
        candidates.append(raw)
    balanced = _extract_balanced_json_object(raw)
    if balanced and balanced not in candidates:
        candidates.append(balanced)
    if raw != text:
        balanced_original = _extract_balanced_json_object(text)
        if balanced_original and balanced_original not in candidates:
            candidates.append(balanced_original)

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def repair_json_response_with_schema(
    text: str | None,
    schema_prompt: str,
    call_llm,
    response_schema: dict,
    fail_streak: int = 0,
) -> tuple[dict | None, str | None, bool]:
    """Ask the LLM to repair malformed JSON into a known schema."""

    if not text:
        return None, None, False

    repair_prompt = f"""The following response was supposed to be a single JSON object but was malformed.
Return ONLY one valid JSON object.

{schema_prompt}

Malformed response:
{text}
"""
    repaired, timed_out = invoke_llm_with_timeout(
        call_llm,
        repair_prompt,
        fail_streak=max(fail_streak, 1),
        phase="apply",
        response_schema=response_schema,
    )
    return parse_response(repaired), repaired, timed_out


def repair_response(
    text: str | None,
    call_llm,
    fail_streak: int = 0,
) -> tuple[dict | None, str | None, bool]:
    """Ask the LLM to repair malformed JSON and parse the repaired response."""

    if not text:
        return None, None, False

    repair_prompt = f"""The following response was supposed to be a single JSON object but was malformed.
Return ONLY one valid JSON object with this schema:
{{
  "file": "repo-relative path",
  "description": "short description",
  "changes": [
    {{
      "old": "exact SEARCH block copied from the current file",
      "new": "replacement REPLACE block"
    }}
  ]
}}

Do not add markdown fences or commentary.
Use multi-line SEARCH blocks with enough unchanged surrounding context to match exactly once.

Malformed response:
{text}
"""
    repaired, timed_out = invoke_llm_with_timeout(
        call_llm,
        repair_prompt,
        fail_streak=max(fail_streak, 1),
        phase="apply",
        response_schema=PATCH_RESPONSE_JSON_SCHEMA,
    )
    return parse_response(repaired), repaired, timed_out


def repair_multi_file_selection_response(
    text: str | None,
    call_llm,
    fail_streak: int = 0,
) -> tuple[dict | None, str | None, bool]:
    schema_prompt = """Use this schema:
{
  "idea_id": "optional structured idea identifier",
  "description": "short description",
  "files": [
    "repo-relative path to an existing file you need opened"
  ]
}

Do not add markdown fences or commentary."""
    return repair_json_response_with_schema(
        text,
        schema_prompt,
        call_llm,
        MULTI_FILE_SELECTION_JSON_SCHEMA,
        fail_streak=fail_streak,
    )


def repair_apply_failure(
    proposal: dict,
    filepath: str,
    file_source: str,
    apply_error: str,
    call_llm,
    fail_streak: int = 0,
) -> tuple[dict | None, str | None, bool]:
    """Ask the LLM to repair a JSON patch whose `old` strings did not match."""

    prior_json = json.dumps(proposal, indent=2, ensure_ascii=True)
    repair_prompt = f"""The following JSON patch could not be applied to the current file.

Failure:
{apply_error}

Return ONLY one valid JSON object with this schema:
{{
  "file": "{filepath}",
  "description": "short description",
  "changes": [
    {{
      "old": "exact SEARCH block copied from the current file",
      "new": "replacement REPLACE block"
    }}
  ]
}}

Requirements:
- Keep "file" exactly "{filepath}".
- Preserve the same intent if possible, but prioritize a patch that applies cleanly.
- Every "old" string MUST appear exactly once in the current file text below.
- Use multi-line SEARCH blocks with enough unchanged surrounding context to match exactly once.
- Keep the patch minimal.
- Do not add markdown fences or commentary.

Current file:
{file_source}

Previous failed proposal:
{prior_json}
"""
    repaired, timed_out = invoke_llm_with_timeout(
        call_llm,
        repair_prompt,
        fail_streak=max(fail_streak, 1),
        phase="apply",
        response_schema=PATCH_RESPONSE_JSON_SCHEMA,
    )
    return parse_response(repaired), repaired, timed_out


def repair_multi_file_apply_failure(
    proposal: dict,
    file_sources: dict[str, str],
    apply_error: str,
    editable_targets: list[str],
    call_llm,
    fail_streak: int = 0,
) -> tuple[dict | None, str | None, bool]:
    """Ask the LLM to repair a multi-file patch after an apply or validation failure."""

    prior_json = json.dumps(proposal, indent=2, ensure_ascii=True)
    files_section = "\n\n".join(
        f"## File: {path}\n```\n{source}\n```"
        for path, source in file_sources.items()
    )
    editable_list = "\n".join(f"  - {path}" for path in editable_targets)
    schema_prompt = f"""Return ONLY one valid JSON object with this schema:
{{
  "idea_id": "optional structured idea identifier",
  "description": "short description",
  "files": [
    {{
      "path": "repo-relative path",
      "operation": "modify",
      "changes": [
        {{
          "old": "exact SEARCH block copied from the current file",
          "new": "replacement REPLACE block"
        }}
      ]
    }},
    {{
      "path": "repo-relative path for a new file",
      "operation": "create",
      "contents": "full contents of the new file"
    }}
  ]
}}

Requirements:
- Only modify files shown below.
- Create operations may add new files only within these editable roots/files:
{editable_list}
- Every SEARCH block must appear exactly once in its current file text.
- Keep the same intent if possible, but prioritize a patch that applies cleanly.
- Do not add markdown fences or commentary.

Failure:
{apply_error}

Current opened files:
{files_section}

Previous failed proposal:
{prior_json}"""
    return repair_json_response_with_schema(
        prior_json,
        schema_prompt,
        call_llm,
        MULTI_FILE_PATCH_RESPONSE_JSON_SCHEMA,
        fail_streak=fail_streak,
    )


def invoke_llm_with_timeout(
    call_llm,
    prompt: str,
    *,
    fail_streak: int = 0,
    timeout_seconds: int = LLM_REQUEST_TIMEOUT_SECONDS,
    phase: str = "reason",
    response_schema: dict | None = None,
    cached_content: str | None = None,
) -> tuple[str | None, bool]:
    """Run an LLM call in a daemon thread and enforce a hard timeout."""

    llm_result: list[str | None] = [None]
    llm_fatal: list[FatalAPIError | None] = [None]

    def _llm_thread() -> None:
        try:
            llm_result[0] = call_llm(
                prompt,
                fail_streak=fail_streak,
                phase=phase,
                response_schema=response_schema,
                cached_content=cached_content,
            )
        except FatalAPIError as e:
            llm_fatal[0] = e

    llm_t = threading.Thread(target=_llm_thread, daemon=True)
    llm_t.start()
    llm_t.join(timeout_seconds)

    if llm_t.is_alive():
        return None, True

    if llm_fatal[0]:
        raise llm_fatal[0]

    return llm_result[0], False


def apply_changes(source: str, changes: list[dict]) -> tuple[str | None, str | None]:
    modified = source
    for change in changes:
        old = change.get("old", "")
        new = change.get("new", "")
        if not old:
            continue
        count = modified.count(old)
        if count == 0:
            error = f"string not found: {old[:160]!r}"
            log_to_file(f"APPLY FAIL: {error}")
            return None, error
        if count > 1:
            error = f"ambiguous match ({count}x): {old[:160]!r}"
            log_to_file(f"APPLY FAIL: {error}")
            return None, error
        modified = modified.replace(old, new, 1)
    return modified, None


def validate_file(filepath: str, source: str, original_source: str | None = None) -> str | None:
    """Return error string or None if valid."""
    if filepath.endswith(".py"):
        try:
            ast.parse(source)
        except SyntaxError as e:
            return f"SyntaxError: {e}"
    elif filepath.endswith((".yaml", ".yml")):
        try:
            yaml.safe_load(source)
        except yaml.YAMLError as e:
            return f"YAMLError: {e}"
    if original_source is not None:
        guardrail_error = validate_architecture_guardrails(filepath, original_source, source)
        if guardrail_error:
            return guardrail_error
    return None


def validate_multi_file_selection(
    proposal: dict,
    editable_targets: list[str],
    forbidden_paths: list[str],
) -> tuple[list[str] | None, str | None]:
    """Validate and normalize a multi-file selection proposal."""

    raw_files = proposal.get("files")
    if not isinstance(raw_files, list) or not raw_files:
        return None, "selection is missing a non-empty files list"

    selected_files: list[str] = []
    for raw_path in raw_files:
        normalized = _normalize_repo_relative_path(str(raw_path))
        if not normalized:
            return None, f"invalid repo-relative path: {raw_path!r}"
        if not is_allowed_edit_path(normalized, editable_targets, forbidden_paths):
            return None, f"disallowed file in selection: {normalized}"
        absolute_path = PROJECT_ROOT / normalized
        if not absolute_path.exists() or not absolute_path.is_file():
            return None, f"selection path does not exist: {normalized}"
        selected_files.append(normalized)

    selected_files = _dedupe_preserve_order(selected_files)
    return selected_files, None


def _normalize_multi_file_entries(proposal: dict) -> list[dict] | None:
    raw_files = proposal.get("files")
    if not isinstance(raw_files, list) or not raw_files:
        return None
    entries: list[dict] = []
    for item in raw_files:
        if not isinstance(item, dict):
            return None
        entries.append(dict(item))
    return entries


def prepare_multi_file_patch(
    proposal: dict,
    opened_file_sources: dict[str, str],
    editable_targets: list[str],
    forbidden_paths: list[str],
) -> tuple[list[str] | None, dict[str, str | None] | None, dict[str, str] | None, str | None]:
    """Validate and materialize a multi-file patch proposal."""

    entries = _normalize_multi_file_entries(proposal)
    if entries is None:
        return None, None, None, "proposal is missing a non-empty files list"

    changed_paths: list[str] = []
    original_sources: dict[str, str | None] = {}
    modified_sources: dict[str, str] = {}
    seen_paths: set[str] = set()

    for entry in entries:
        raw_path = entry.get("path", "")
        normalized_path = _normalize_repo_relative_path(str(raw_path))
        if not normalized_path:
            return None, None, None, f"invalid repo-relative path: {raw_path!r}"
        if normalized_path in seen_paths:
            return None, None, None, f"duplicate file entry: {normalized_path}"
        if not is_allowed_edit_path(normalized_path, editable_targets, forbidden_paths):
            return None, None, None, f"disallowed file in patch: {normalized_path}"

        operation = str(entry.get("operation", "")).strip().lower()
        if not operation:
            operation = "create" if "contents" in entry and "changes" not in entry else "modify"

        absolute_path = PROJECT_ROOT / normalized_path
        if operation == "modify":
            if normalized_path not in opened_file_sources:
                return None, None, None, f"modify target was not opened first: {normalized_path}"
            original = opened_file_sources[normalized_path]
            changes = entry.get("changes")
            if not isinstance(changes, list) or not changes:
                return None, None, None, f"modify entry is missing changes: {normalized_path}"
            modified, apply_error = apply_changes(original, changes)
            if modified is None:
                return None, None, None, f"{normalized_path}: {apply_error or 'apply failed'}"
        elif operation == "create":
            if absolute_path.exists():
                return None, None, None, f"create target already exists: {normalized_path}"
            contents = entry.get("contents")
            if not isinstance(contents, str):
                return None, None, None, f"create entry is missing contents: {normalized_path}"
            original = None
            modified = contents
        else:
            return None, None, None, f"unsupported operation {operation!r} for {normalized_path}"

        validate_err = validate_file(normalized_path, modified, original_source=original)
        if validate_err:
            return None, None, None, f"{normalized_path}: {validate_err}"

        seen_paths.add(normalized_path)
        changed_paths.append(normalized_path)
        original_sources[normalized_path] = original
        modified_sources[normalized_path] = modified

    return changed_paths, original_sources, modified_sources, None


def get_eval_settings(config: dict | None = None) -> dict:
    """Resolve lightweight evaluation settings from autoresearch config."""

    agent_cfg = (config or {}).get("agent", {})
    return {
        "eval_on_keep": bool(agent_cfg.get("eval_on_keep", True)),
        "eval_every_n_runs": max(0, int(agent_cfg.get("eval_every_n_runs", 0) or 0)),
        "n_trajectories": max(1, int(agent_cfg.get("eval_n_trajectories", 8) or 8)),
        "predict_mode": str(agent_cfg.get("eval_predict_mode", "deterministic")),
        "plot_count": max(0, int(agent_cfg.get("eval_plot_count", 0) or 0)),
        "skip_ensemble": bool(agent_cfg.get("eval_skip_ensemble", True)),
        "timeout_seconds": max(60, int(agent_cfg.get("eval_timeout_seconds", 900) or 900)),
    }


def should_run_lightweight_eval(result: dict | None, run_number: int, eval_settings: dict) -> bool:
    """Return True when a sidecar evaluation should be run for a completed experiment."""

    if not result or result.get("status") not in ("keep", "discard"):
        return False

    if eval_settings.get("eval_on_keep", True) and result.get("status") == "keep":
        return True

    every_n = int(eval_settings.get("eval_every_n_runs", 0) or 0)
    return every_n > 0 and run_number % every_n == 0


def _resolve_run_dir_from_result(result: dict) -> Path | None:
    """Resolve the run directory from a result payload."""

    summary_path = result.get("summary_path")
    if summary_path:
        return Path(summary_path).resolve().parent

    artifacts_dir = result.get("artifacts_dir")
    if artifacts_dir:
        return Path(artifacts_dir).resolve().parent

    run_id = result.get("run_id")
    if run_id:
        return get_runs_dir() / str(run_id)

    return None


def _resolve_eval_model_path(run_dir: Path, result: dict) -> Path | None:
    """Pick the most useful checkpoint to evaluate from a run directory."""

    candidates: list[Path] = []
    artifacts_dir = result.get("artifacts_dir")
    if artifacts_dir:
        artifacts_path = Path(str(artifacts_dir)).resolve()
        candidates.extend(
            [
                artifacts_path / "best_model.eqx",
                artifacts_path / "final_model.eqx",
            ]
        )
    candidates.extend(
        [
            run_dir / "artifacts" / "best_model.eqx",
            run_dir / "artifacts" / "final_model.eqx",
            run_dir / "best_model.eqx",
            run_dir / "final_model.eqx",
        ]
    )
    for path in candidates:
        if path.exists():
            return path
    return None


def maybe_run_lightweight_eval(
    result: dict | None,
    run_number: int,
    train_cfg: dict,
    eval_settings: dict,
    on_line: callable | None = None,
) -> Path | None:
    """Run a compact evaluation pass when configured to do so."""

    if not should_run_lightweight_eval(result, run_number, eval_settings):
        return None

    run_dir = _resolve_run_dir_from_result(result or {})
    if run_dir is None:
        log_to_file("EVAL SKIP: could not resolve run directory")
        return None

    model_path = _resolve_eval_model_path(run_dir, result or {})
    if model_path is None:
        log_to_file(f"EVAL SKIP: no checkpoint found in {run_dir}")
        return None

    predict_mode = str(eval_settings.get("predict_mode", "deterministic"))
    output_dir = run_dir / ("eval_det" if predict_mode == "deterministic" else "eval_stoch")
    summary_path = output_dir / "evaluation_summary.json"

    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "evaluate.py"),
        "--model_path",
        str(model_path),
        "--predict_mode",
        predict_mode,
        "--n_trajectories",
        str(eval_settings.get("n_trajectories", 8)),
        "--plot_count",
        str(eval_settings.get("plot_count", 0)),
        "--output_dir",
        str(output_dir),
        "--summary_path",
        str(summary_path),
    ]
    if eval_settings.get("skip_ensemble", True):
        command.append("--skip_ensemble")

    data_dir = train_cfg.get("data_dir")
    if data_dir:
        command.extend(["--data_dir", str(resolve_repo_path(str(data_dir)))])

    try:
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=int(eval_settings.get("timeout_seconds", 900)),
            check=False,
        )
    except subprocess.TimeoutExpired:
        log_to_file(f"EVAL TIMEOUT: run={run_dir.name}")
        if on_line:
            on_line("Lightweight eval timed out.")
        return None
    except Exception as exc:
        log_to_file(f"EVAL ERROR: run={run_dir.name} ({exc})")
        if on_line:
            on_line(f"Lightweight eval failed: {str(exc)[:100]}")
        return None

    output_lines = [
        line.strip()
        for line in (completed.stdout or "").splitlines()
        if line.strip()
    ]
    if completed.returncode != 0:
        tail = " | ".join(output_lines[-3:]) if output_lines else (completed.stderr or "").strip()
        log_to_file(f"EVAL FAIL: run={run_dir.name} rc={completed.returncode} {tail[:300]}")
        if on_line:
            on_line(f"Lightweight eval failed: {tail[:100] or 'unknown error'}")
        return None

    if on_line:
        on_line(
            "Lightweight eval saved "
            f"{summary_path.relative_to(PROJECT_ROOT)}"
        )

    if result and result.get("baseline_promoted") and output_dir.exists():
        baseline_eval_dir = get_workspace_dir() / "baseline" / output_dir.name
        if baseline_eval_dir.exists():
            shutil.rmtree(baseline_eval_dir)
        shutil.copytree(output_dir, baseline_eval_dir)

    return summary_path if summary_path.exists() else None


# ---------------------------------------------------------------------------
# Experiment runner – calls scripts/autoresearch.py as subprocess
# ---------------------------------------------------------------------------

DEFAULT_TRAIN_TIMEOUT_SECONDS = 3600


def get_train_timeout_seconds() -> int:
    """Align the agent-side timeout with the autoresearch config."""

    try:
        with ACTIVE_AUTORESEARCH_CONFIG.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}
        research = config.get("research", {})
        minutes = float(research.get("time_budget_minutes", 30))
        buffer_minutes = float(research.get("hard_timeout_buffer_minutes", 5))
        return max(60, int((minutes + buffer_minutes + 1.0) * 60))
    except Exception:
        return DEFAULT_TRAIN_TIMEOUT_SECONDS


def run_experiment(
    description: str,
    on_line: callable | None = None,
) -> dict | None:
    """Run one autoresearch experiment and return result dict or None on crash."""
    cmd = [
        sys.executable,
        str(AUTORESEARCH_SCRIPT),
        "--config", str(ACTIVE_AUTORESEARCH_CONFIG),
        "--description", description,
    ]

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    train_timeout_seconds = get_train_timeout_seconds()

    # Determine where result.json will land
    # autoresearch.py writes it to workspace_dir/runs/<run_id>/result.json
    # We read the workspace_dir from config.
    workspace_dir = get_workspace_dir()
    runs_dir = workspace_dir / "runs"
    existing_result_files = set(runs_dir.glob("*/result.json")) if runs_dir.exists() else set()

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=PROJECT_ROOT,
            env=env,
            bufsize=0,
        )

        t_start = time.time()
        buf = b""
        all_lines: list[str] = []

        while True:
            if time.time() - t_start > train_timeout_seconds:
                proc.kill()
                log_to_file(f"TIMEOUT: experiment killed after {train_timeout_seconds}s")
                return {"error": f"timeout ({train_timeout_seconds}s)"}

            # GPU thermal abort
            if _nvml_available:
                try:
                    temp = nvmlDeviceGetTemperature(_nvml_handle, NVML_TEMPERATURE_GPU)
                    if temp >= GPU_TEMP_ABORT:
                        proc.kill()
                        log_to_file(f"SAFETY: killed – GPU {temp}C >= {GPU_TEMP_ABORT}C")
                        return {"error": f"GPU overheat ({temp}C)"}
                except Exception:
                    pass

            chunk = proc.stdout.read(1)
            if not chunk:
                break

            if chunk in (b"\r", b"\n"):
                if buf:
                    line = buf.decode("utf-8", errors="replace").strip()
                    if line:
                        all_lines.append(line)
                        if on_line:
                            on_line(line)
                    buf = b""
            else:
                buf += chunk

        if buf:
            line = buf.decode("utf-8", errors="replace").strip()
            if line:
                all_lines.append(line)
                if on_line:
                    on_line(line)

        proc.wait(timeout=30)

    except subprocess.TimeoutExpired:
        proc.kill()
        return {"error": "timeout"}
    except Exception as e:
        return {"error": str(e)}

    # Find the most-recently-written result.json under runs/
    if runs_dir.exists():
        result_files = [
            path for path in runs_dir.glob("*/result.json")
            if path not in existing_result_files
        ]
        result_files.sort(key=lambda p: p.stat().st_mtime)
        if result_files:
            try:
                with result_files[-1].open("r") as f:
                    result = json.load(f)
                return result
            except Exception:
                pass

    # Fallback: scan stdout for metric
    for line in reversed(all_lines):
        if "best_val_loss:" in line:
            try:
                val = float(line.split("best_val_loss:")[-1].strip())
                return {"metric_value": val, "status": "keep", "description": description}
            except ValueError:
                pass

    if proc.returncode not in (0, None):
        errors = [l for l in all_lines if "Error" in l or "Traceback" in l]
        msg = "\n".join(errors[-10:]) if errors else "\n".join(all_lines[-10:])
        return {"error": msg[:500]}

    return None


def _unique_baseline_archive_dir() -> Path:
    archive_root = get_workspace_dir() / ".baseline_archive"
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    candidate = archive_root / timestamp
    suffix = 1
    while candidate.exists():
        suffix += 1
        candidate = archive_root / f"{timestamp}-{suffix}"
    return candidate


def archive_promoted_baseline() -> Path | None:
    baseline_dir = get_workspace_dir() / "baseline"
    if not baseline_dir.exists():
        return None

    archive_dir = _unique_baseline_archive_dir()
    archive_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(baseline_dir), str(archive_dir))
    log_to_file(f"Archived promoted baseline to {_display_path(archive_dir)}")
    return archive_dir


def restore_archived_baseline(archive_dir: Path | None) -> None:
    if not archive_dir or not archive_dir.exists():
        return

    baseline_dir = get_workspace_dir() / "baseline"
    if baseline_dir.exists():
        shutil.rmtree(baseline_dir)
    shutil.move(str(archive_dir), str(baseline_dir))
    log_to_file(f"Restored promoted baseline from {_display_path(archive_dir)}")


def discard_archived_baseline(archive_dir: Path | None) -> None:
    if not archive_dir or not archive_dir.exists():
        return
    shutil.rmtree(archive_dir)


def run_rebaseline_experiment(on_line: callable | None = None) -> dict | None:
    archived_baseline = archive_promoted_baseline()
    refresh_note = "Refreshing promoted baseline against the current branch state..."
    log_to_file(refresh_note)
    if on_line:
        on_line(refresh_note)

    result = run_experiment("baseline", on_line=on_line)
    promoted = bool(result and result.get("baseline_promoted"))

    if promoted:
        discard_archived_baseline(archived_baseline)
        return result

    restore_archived_baseline(archived_baseline)
    restore_note = (
        "Baseline refresh failed; restored previous promoted baseline."
        if archived_baseline
        else "Baseline refresh failed."
    )
    log_to_file(restore_note)
    if on_line:
        on_line(restore_note)
    return result


# ---------------------------------------------------------------------------
# Rich TUI dashboard
# ---------------------------------------------------------------------------

SPARK_CHARS = list(" .:-=+*#@")

PHASE_STYLES = {
    "THINKING":  ("THINKING",  "bold magenta"),
    "TRAINING":  ("TRAINING",  "bold yellow"),
    "EVALUATING": ("EVALUATING", "bold cyan"),
    "APPLYING":  ("APPLYING",  "bold blue"),
    "BASELINE":  ("BASELINE",  "bold yellow"),
    "KEEP":      ("IMPROVED",  "bold green"),
    "DISCARD":   ("DISCARDED", "bold red"),
    "CRASH":     ("CRASHED",   "bold red"),
    "COOLING":   ("COOLING",   "bold cyan"),
    "DONE":      ("COMPLETE",  "bold green"),
    "STARTING":  ("STARTING",  "dim"),
}


def sparkline(values: list[float], width: int = 40) -> str:
    if not values:
        return ""
    recent = list(values)[-width:]
    lo, hi = min(recent), max(recent)
    rng = hi - lo if hi > lo else 1.0
    return "".join(SPARK_CHARS[min(int((v - lo) / rng * 8), 8)] for v in recent)


def bar(pct: float, width: int = 20) -> str:
    pct = max(0.0, min(100.0, float(pct)))
    filled = int(width * pct / 100)
    return "\u2588" * filled + "\u2591" * (width - filled)


def build_dashboard(state: dict) -> Layout:
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="body", ratio=1, minimum_size=16),
        Layout(name="footer", size=5),
    )
    layout["body"].split_row(
        Layout(name="left", ratio=3, minimum_size=50),
        Layout(name="right", ratio=2, minimum_size=36),
    )
    layout["left"].split_column(
        Layout(name="training", size=7, minimum_size=7),
        Layout(name="experiments", ratio=1, minimum_size=9),
    )
    layout["right"].split_column(
        Layout(name="gpu", size=8, minimum_size=8),
        Layout(name="activity", ratio=1, minimum_size=8),
    )

    phase = state.get("phase", "STARTING")
    exp_num = state.get("experiment_num", 0)
    max_runs = state.get("max_runs", 0)
    best_loss = state.get("best_loss", 999.0)
    branch = state.get("branch", "")
    elapsed = state.get("total_elapsed", 0.0)
    history: list[dict] = state.get("history", [])
    kept = [r for r in history if r["status"] == "keep"]
    llm_name = state.get("llm_name", "LLM")
    config_path = state.get("config_path", "")
    workspace_dir = state.get("workspace_dir", "")
    train_epochs = state.get("train_epochs")
    time_budget_minutes = state.get("time_budget_minutes")
    val_every = state.get("val_every")
    eh, rem = divmod(int(elapsed), 3600)
    em, es = divmod(rem, 60)

    # ---- Header ----
    phase_label, phase_style = PHASE_STYLES.get(phase, (phase, "dim"))
    h = Text()
    h.append(" digital-twin autoresearch ", style="bold white on blue")
    h.append("  ")
    h.append(f" {phase_label} ", style=phase_style)
    h.append(f"  Exp {exp_num}/{max_runs}  ")
    if best_loss < 999:
        h.append(f"Best {best_loss:.6f}", style="bold green")
    h.append(f"  {len(history)} runs  ")
    h.append(f"{len(kept)} kept", style="green")
    if len(kept) >= 2:
        first_keep = next((r["val_loss"] for r in history if r["status"] == "keep"), 999)
        if first_keep < 999 and best_loss < first_keep:
            imp = (first_keep - best_loss) / first_keep * 100
            h.append(f"  -{imp:.1f}%", style="bold green")
    h.append(f"  {eh}:{em:02d}:{es:02d}")
    h.append(f"  {branch}", style="dim")
    layout["header"].update(Panel(h, border_style="blue"))

    # ---- Training / Status panel ----
    t = Text()
    idea = state.get("current_idea", "")
    if idea:
        t.append(f" {idea[:100]}\n\n", style="italic white")

    loss_history: list[float] = list(state.get("loss_history", []))
    metrics = state.get("metrics")

    if phase == "THINKING":
        elapsed_phase = state.get("phase_elapsed", 0.0)
        dots = "." * (int(elapsed_phase) % 4 + 1)
        t.append(f" Querying {llm_name}{dots}\n", style="italic magenta")
        if best_loss < 999:
            t.append(f" Target: < {best_loss:.6f}\n", style="dim green")
        streak = _count_recent_failures(history)
        if streak > 0:
            sty = "bold red" if streak >= 5 else "yellow"
            t.append(f" Fail streak: {streak}\n", style=sty)
    elif phase == "COOLING":
        ce = state.get("phase_elapsed", 0.0)
        pct = min(ce / 30 * 100, 100)
        t.append(f" Cooling {bar(pct)} {int(ce)}s\n", style="cyan")
    elif phase in ("TRAINING", "BASELINE"):
        if loss_history:
            t.append(f" {sparkline(loss_history)}\n", style="bright_yellow")
            t.append(f" Latest loss: {loss_history[-1]:.6f}\n")
        else:
            t.append(f" Training in progress...\n", style="yellow")
    elif phase == "EVALUATING":
        t.append(" Running lightweight deterministic eval...\n", style="cyan")
    elif phase == "APPLYING":
        t.append(f" Applying patch and committing...\n", style="bright_blue")
    elif phase == "KEEP":
        t.append(f" IMPROVED — change kept!\n", style="bold green")
    elif phase == "DISCARD":
        t.append(f" No improvement — reverted.\n", style="bold yellow")
    elif phase == "CRASH":
        t.append(f" Experiment crashed — reverted.\n", style="bold red")
    if not t.plain.strip():
        t.append(" Waiting for the next state update...\n", style="dim")

    training_title = "Training" if phase in ("TRAINING", "BASELINE") else phase.title()
    border_color = {
        "TRAINING": "bright_green", "BASELINE": "bright_yellow", "THINKING": "bright_magenta",
        "KEEP": "green", "DISCARD": "yellow", "CRASH": "red", "COOLING": "cyan",
        "EVALUATING": "cyan",
    }.get(phase, "dim")
    layout["training"].update(Panel(t, title=f"[bold]{training_title}[/]", border_style=border_color))

    # ---- Experiment log ----
    table = Table(expand=True, show_lines=False, show_header=True,
                  header_style="bold dim", box=None, padding=(0, 1))
    table.add_column("#", style="dim", width=4, justify="right")
    table.add_column("SHA", width=8, style="dim")
    table.add_column("St", width=5)
    table.add_column("val_loss", width=10, justify="right")
    table.add_column("File", width=20, no_wrap=True)
    table.add_column("Description", ratio=1, no_wrap=True)

    for idx, r in enumerate(history[-14:], 1):
        row_num = max(0, len(history) - 14) + idx
        status_map = {"keep": ("KEEP", "bold green"), "discard": ("SKIP", "yellow"), "crash": ("FAIL", "red")}
        icon, sty = status_map.get(r["status"], (r["status"][:4], "white"))
        loss_str = f"{r['val_loss']:.4f}" if r["val_loss"] < 999 else "—"
        is_best = r["val_loss"] == best_loss and r["status"] == "keep" and best_loss < 999
        loss_style = "bold bright_green" if is_best else ("white" if r["val_loss"] < 999 else "dim")
        commit = r["commit"][:7] if r["commit"] != "-------" else "—"
        file_short = r.get("file", "")[-20:]
        table.add_row(
            str(row_num), commit, Text(icon, style=sty),
            Text(loss_str, style=loss_style),
            Text(file_short, style="dim"),
            Text(r["description"][:50], style="white" if r["status"] == "keep" else "dim"),
        )

    if not history:
        table.add_row("", "", Text("—", style="dim"), "—", "—", Text("Waiting for baseline...", style="dim"))

    layout["experiments"].update(Panel(table, title="[bold]Experiments[/]", border_style="bright_blue"))

    # ---- GPU panel ----
    gpu = state.get("gpu", {})
    temp = gpu.get("temp")
    vram_used = gpu.get("vram_used_mb", 0)
    vram_total = gpu.get("vram_total_mb", VRAM_TOTAL_MB)
    util = gpu.get("gpu_util", 0)

    g = Text()
    g.append(f" {GPU_NAME}\n\n", style="dim")

    if temp is not None:
        temp_style = "bold red" if temp >= GPU_TEMP_ABORT else ("yellow" if temp >= 70 else "green")
        g.append(f" Temp  {temp:3d}C  {bar(min(temp, 100), 15)}\n", style=temp_style)
    else:
        g.append(f" Temp  N/A (CPU mode)\n", style="dim")

    if vram_total > 0:
        vram_pct = vram_used / vram_total * 100
        vram_style = "bold red" if vram_used >= VRAM_LIMIT_MB else "green"
        g.append(f" VRAM  {vram_used:5.0f}/{vram_total:.0f}MB  {bar(vram_pct, 10)}\n", style=vram_style)
    else:
        g.append(" VRAM  N/A\n", style="dim")

    g.append(f" Load  {util:3d}%  {bar(util, 15)}\n", style="bright_blue")
    g.append(f"\n Runs {len(history):3d}  Kept {len(kept):3d}  Fail {len([r for r in history if r['status']=='crash']):3d}\n")
    if not g.plain.strip():
        g.append(" Hardware stats unavailable\n", style="dim")

    layout["gpu"].update(Panel(g, title="[bold]Hardware[/]", border_style="bright_blue"))

    # ---- Activity log ----
    log_lines: deque[str] = state.get("log_lines", deque())
    log_text = Text()
    for line in list(log_lines)[-20:]:
        log_text.append(f"{line}\n", style="dim")
    if not log_lines:
        log_text.append(" Waiting...\n", style="dim")
    layout["activity"].update(Panel(log_text, title="[dim]Activity[/]", border_style="dim"))

    # ---- Footer ----
    footer = Text()
    footer.append(" Config: ", style="bold")
    footer.append(f"{config_path or 'N/A'}", style="cyan")
    footer.append("   Workspace: ", style="bold")
    footer.append(f"{workspace_dir or 'N/A'}", style="cyan")
    footer.append("   LLM: ", style="bold")
    footer.append(llm_name, style="magenta")
    footer.append("\n")
    footer.append(" Train: ", style="bold")
    footer.append(
        f"{train_epochs if train_epochs is not None else 'N/A'} epochs",
        style="green",
    )
    footer.append("   Budget: ", style="bold")
    if time_budget_minutes is not None:
        footer.append(f"{float(time_budget_minutes):.0f} min", style="green")
    else:
        footer.append("N/A", style="dim")
    footer.append("   Val every: ", style="bold")
    footer.append(f"{val_every if val_every is not None else 'N/A'}", style="green")
    footer.append("   Results: ", style="bold")
    footer.append(str(len(history)), style="yellow")
    footer.append("   Kept: ", style="bold")
    footer.append(str(len(kept)), style="green")
    layout["footer"].update(Panel(footer, title="[dim]Run Context[/]", border_style="dim"))

    return layout


# ---------------------------------------------------------------------------
# Main loop helpers
# ---------------------------------------------------------------------------

def _choose_file(
    history: list[dict],
    modifiable_files: list[str],
    forced_file: str | None,
    preferred_file: str | None = None,
) -> str:
    """Pick the file to show in the prompt. Rotates through modifiable files."""
    if forced_file:
        return forced_file
    if preferred_file and preferred_file in modifiable_files:
        return preferred_file
    clean_files = [f for f in modifiable_files if not git_file_is_dirty(f)]
    candidates = clean_files or modifiable_files

    def _research_priority(path: str) -> int:
        if path.startswith(("dte/models/", "dte/training/")):
            return 0
        if path.endswith(".py"):
            return 1
        return 2

    # Prefer least-recently-tried files
    file_counts: dict[str, int] = {f: 0 for f in candidates}
    for r in history:
        fpath = r.get("file", "")
        if fpath in file_counts:
            file_counts[fpath] += 1
    return min(file_counts, key=lambda f: (_research_priority(f), file_counts[f], f))


# ---------------------------------------------------------------------------
# Text-mode fallback
# ---------------------------------------------------------------------------

def _run_text_mode(
    args,
    history: list[dict],
    best_loss: float,
    call_llm,
    editable_targets: list[str],
    max_runs: int,
    train_cfg: dict,
    eval_settings: dict,
    agent_context: str,
    recent_run_context: str,
    static_prompt: str,
    reason_cached_content: str | None,
    branch: str,
    structured_ideas: list[dict],
    execution_mode: str,
    forbidden_paths: list[str],
) -> None:
    """Simple text output mode when Rich is not desired."""
    prior_count = len(history)
    consecutive_parse_failures = 0
    eligible_structured_ideas = get_eligible_structured_ideas(
        structured_ideas,
        editable_targets,
        args.file,
        execution_mode=execution_mode,
        forbidden_paths=forbidden_paths,
    )
    strict_structured_ideas = bool(eligible_structured_ideas)
    display_max_runs = (
        len(eligible_structured_ideas)
        if strict_structured_ideas
        else max_runs
    )

    def update_text_state(
        exp_num: int,
        description: str,
        phase: str,
        *,
        file_path: str = "",
        current_idea: str = "",
        extra: dict | None = None,
    ) -> None:
        payload = {
            "max_runs": display_max_runs,
            "best_loss": None if best_loss >= 999.0 else best_loss,
            "llm_name": args.llm_name,
            "branch": branch,
        }
        if file_path:
            payload["file"] = file_path
        if current_idea:
            payload["current_idea"] = current_idea
        if extra:
            payload.update(extra)
        write_state(exp_num, description, phase, payload)

    try:
        idea_iteration = 0
        while True:
            history = get_results_history()
            resolved_idea_count = count_resolved_structured_ideas(history, eligible_structured_ideas)
            if strict_structured_ideas:
                if resolved_idea_count >= len(eligible_structured_ideas):
                    print("All queued structured ideas have resolved keep/discard results.")
                    break
                exp_num = resolved_idea_count + 1
            else:
                if idea_iteration >= max(0, max_runs - prior_count):
                    break
                exp_num = prior_count + idea_iteration + 1
                idea_iteration += 1

            proposed_file = ""
            original: str | None = None
            changed_paths: list[str] = []
            original_sources: dict[str, str | None] = {}
            experiment_committed = False
            experiment_commit_sha: str | None = None

            try:
                print(f"\n[Exp {exp_num}/{display_max_runs}] Asking {args.llm_name}...")
                update_text_state(exp_num, "querying LLM", "THINKING")
                recent_run_context = build_recent_run_context()

                selected_idea = select_next_structured_idea(
                    structured_ideas,
                    history,
                    editable_targets,
                    args.file,
                    execution_mode=execution_mode,
                    forbidden_paths=forbidden_paths,
                )
                if strict_structured_ideas and not selected_idea:
                    print("  No pending structured ideas remain. Stopping.")
                    break

                fail_streak = _count_recent_failures(history)
                if execution_mode == "multi_file":
                    selection_prompt = build_multi_file_selection_prompt(
                        history,
                        best_loss,
                        recent_run_context=recent_run_context,
                        ideas=structured_ideas,
                        selected_idea=selected_idea,
                    )
                    prompt = selection_prompt if reason_cached_content else static_prompt + selection_prompt
                    response_schema = MULTI_FILE_SELECTION_JSON_SCHEMA
                else:
                    target_file = _choose_file(
                        history,
                        editable_targets,
                        args.file,
                        preferred_file=selected_idea.get("target_file") if selected_idea else None,
                    )
                    try:
                        file_source = (PROJECT_ROOT / target_file).read_text(encoding="utf-8")
                    except FileNotFoundError:
                        print(f"  File not found: {target_file}")
                        continue
                    dynamic_prompt = build_dynamic_prompt(
                        target_file,
                        file_source,
                        history,
                        best_loss,
                        recent_run_context=recent_run_context,
                        ideas=structured_ideas,
                        selected_idea=selected_idea,
                    )
                    prompt = dynamic_prompt if reason_cached_content else static_prompt + dynamic_prompt
                    response_schema = PATCH_RESPONSE_JSON_SCHEMA

                try:
                    response, timed_out = invoke_llm_with_timeout(
                        call_llm,
                        prompt,
                        fail_streak=fail_streak,
                        phase="reason",
                        response_schema=response_schema,
                        cached_content=reason_cached_content,
                    )
                except FatalAPIError as e:
                    print(f"Fatal API error: {e}")
                    break
                if timed_out:
                    print(f"  LLM request timed out after {LLM_REQUEST_TIMEOUT_SECONDS}s. Skipping.")
                    history = get_results_history()
                    continue

                proposal = parse_response(response)
                if not proposal:
                    try:
                        if execution_mode == "multi_file":
                            proposal, repaired, repair_timed_out = repair_multi_file_selection_response(
                                response,
                                call_llm,
                                fail_streak=fail_streak,
                            )
                        else:
                            proposal, repaired, repair_timed_out = repair_response(
                                response,
                                call_llm,
                                fail_streak=fail_streak,
                            )
                    except FatalAPIError as e:
                        print(f"Fatal API error: {e}")
                        break
                    if repair_timed_out:
                        print(f"  JSON repair request timed out after {LLM_REQUEST_TIMEOUT_SECONDS}s. Skipping.")
                        history = get_results_history()
                        continue
                    if proposal:
                        print("  Repaired malformed LLM response.")

                if not proposal:
                    print("  LLM gave unparseable response. Skipping.")
                    consecutive_parse_failures += 1
                    if (
                        consecutive_parse_failures >= MAX_CONSECUTIVE_PARSE_FAILURES
                        and not strict_structured_ideas
                    ):
                        print("  Too many consecutive malformed LLM responses. Stopping cleanly.")
                        break
                    continue
                consecutive_parse_failures = 0

                proposal = enforce_selected_idea_on_proposal(
                    proposal,
                    selected_idea,
                    execution_mode=execution_mode,
                )

                if execution_mode == "multi_file":
                    selected_paths, selection_error = validate_multi_file_selection(
                        proposal,
                        editable_targets,
                        forbidden_paths,
                    )
                    if selection_error or not selected_paths:
                        print(f"  Invalid multi-file selection: {selection_error or 'no files selected'}. Skipping.")
                        history = get_results_history()
                        continue
                    dirty_paths = [path for path in selected_paths if git_file_is_dirty(path)]
                    if dirty_paths:
                        print(
                            "  Skipping selection because these files are dirty: "
                            + ", ".join(dirty_paths)
                        )
                        history = get_results_history()
                        continue

                    description = annotate_description_with_idea(
                        proposal.get("description", "unknown"),
                        proposal.get("idea_id"),
                    )
                    opened_file_sources = {
                        path: (PROJECT_ROOT / path).read_text(encoding="utf-8")
                        for path in selected_paths
                    }
                    selection_label = summarize_changed_paths(selected_paths)
                    current_idea = f"[{selection_label}] {description}"
                    print(f"  Idea: {description}")
                    print(f"  Files: {selection_label}")
                    update_text_state(
                        exp_num,
                        description,
                        "APPLYING",
                        file_path=selection_label,
                        current_idea=current_idea,
                    )

                    patch_prompt = build_multi_file_patch_prompt(
                        description,
                        opened_file_sources,
                        editable_targets,
                        recent_run_context=recent_run_context,
                        selected_idea=selected_idea,
                    )
                    patch_input = patch_prompt if reason_cached_content else static_prompt + patch_prompt
                    try:
                        patch_response, patch_timed_out = invoke_llm_with_timeout(
                            call_llm,
                            patch_input,
                            fail_streak=fail_streak,
                            phase="reason",
                            response_schema=MULTI_FILE_PATCH_RESPONSE_JSON_SCHEMA,
                            cached_content=reason_cached_content,
                        )
                    except FatalAPIError as e:
                        print(f"Fatal API error: {e}")
                        break
                    if patch_timed_out:
                        print(f"  Patch generation timed out after {LLM_REQUEST_TIMEOUT_SECONDS}s. Skipping.")
                        history = get_results_history()
                        continue

                    patch_proposal = parse_response(patch_response)
                    if not patch_proposal:
                        try:
                            patch_proposal, repaired_patch, repair_timed_out = repair_json_response_with_schema(
                                patch_response,
                                """Use this schema:
{
  "idea_id": "optional structured idea identifier",
  "description": "short description",
  "files": [
    {
      "path": "repo-relative path",
      "operation": "modify",
      "changes": [
        {
          "old": "exact SEARCH block copied from the current file",
          "new": "replacement REPLACE block"
        }
      ]
    },
    {
      "path": "repo-relative path for a new file",
      "operation": "create",
      "contents": "full contents of the new file"
    }
  ]
}

Do not add markdown fences or commentary.""",
                                call_llm,
                                MULTI_FILE_PATCH_RESPONSE_JSON_SCHEMA,
                                fail_streak=fail_streak,
                            )
                        except FatalAPIError as e:
                            print(f"Fatal API error: {e}")
                            break
                        if repair_timed_out:
                            print(f"  Patch JSON repair timed out after {LLM_REQUEST_TIMEOUT_SECONDS}s. Skipping.")
                            history = get_results_history()
                            continue
                        if patch_proposal:
                            print("  Repaired malformed multi-file patch response.")

                    if not patch_proposal:
                        print("  LLM gave unparseable multi-file patch. Skipping.")
                        history = get_results_history()
                        continue

                    patch_proposal = enforce_selected_idea_on_proposal(
                        patch_proposal,
                        selected_idea,
                        execution_mode=execution_mode,
                    )
                    description = annotate_description_with_idea(
                        patch_proposal.get("description", description),
                        patch_proposal.get("idea_id"),
                    )
                    changed_paths, original_sources, modified_sources, apply_error = prepare_multi_file_patch(
                        patch_proposal,
                        opened_file_sources,
                        editable_targets,
                        forbidden_paths,
                    )
                    if apply_error or not changed_paths or original_sources is None or modified_sources is None:
                        print("  Could not apply multi-file patch. Trying one repair...")
                        try:
                            repaired_patch, repaired_text, repair_timed_out = repair_multi_file_apply_failure(
                                patch_proposal,
                                opened_file_sources,
                                apply_error or "unknown multi-file apply failure",
                                editable_targets,
                                call_llm,
                                fail_streak=fail_streak,
                            )
                        except FatalAPIError as e:
                            print(f"Fatal API error: {e}")
                            break
                        if repair_timed_out:
                            print(f"  Patch repair request timed out after {LLM_REQUEST_TIMEOUT_SECONDS}s. Skipping.")
                            log_result(
                                "-------",
                                0.0,
                                "crash",
                                selection_label,
                                f"APPLY FAIL: {description} [repair timeout]",
                            )
                            history = get_results_history()
                            continue
                        if repaired_patch:
                            repaired_patch = enforce_selected_idea_on_proposal(
                                repaired_patch,
                                selected_idea,
                                execution_mode=execution_mode,
                            )
                            repaired_paths, repaired_originals, repaired_modified, repaired_error = prepare_multi_file_patch(
                                repaired_patch,
                                opened_file_sources,
                                editable_targets,
                                forbidden_paths,
                            )
                            if repaired_error is None and repaired_paths and repaired_originals is not None and repaired_modified is not None:
                                patch_proposal = repaired_patch
                                changed_paths = repaired_paths
                                original_sources = repaired_originals
                                modified_sources = repaired_modified
                                description = annotate_description_with_idea(
                                    patch_proposal.get("description", description),
                                    patch_proposal.get("idea_id"),
                                )
                                print("  Recovered multi-file apply failure with repair prompt.")
                            else:
                                apply_error = repaired_error or apply_error
                        if apply_error or not changed_paths or not original_sources:
                            print(f"  Could not apply multi-file patch after repair. Skipping.")
                            log_result(
                                "-------",
                                0.0,
                                "crash",
                                selection_label,
                                f"APPLY FAIL: {description} [{apply_error or 'repair failed'}]",
                            )
                            history = get_results_history()
                            continue

                    proposed_file = changed_paths[0]
                    current_idea = f"[{summarize_changed_paths(changed_paths)}] {description}"
                    print(f"  Final files: {summarize_changed_paths(changed_paths)}")
                    update_text_state(
                        exp_num,
                        description,
                        "APPLYING",
                        file_path=summarize_changed_paths(changed_paths),
                        current_idea=current_idea,
                    )
                    try:
                        write_repo_files_atomically(modified_sources, original_sources)
                    except Exception as write_err:
                        print(f"  WRITE FAIL: {str(write_err)[:120]}")
                        log_result(
                            "-------",
                            0.0,
                            "crash",
                            summarize_changed_paths(changed_paths),
                            f"WRITE FAIL: {description} [{str(write_err)[:120]}]",
                        )
                        history = get_results_history()
                        continue
                else:
                    if "changes" not in proposal:
                        print("  LLM gave unparseable response. Skipping.")
                        history = get_results_history()
                        continue

                    proposed_file = proposal.get("file", target_file)
                    if proposed_file not in editable_targets:
                        print(f"  LLM chose disallowed file {proposed_file}. Skipping.")
                        continue
                    if git_file_is_dirty(proposed_file):
                        print(f"  Skipping dirty file {proposed_file} to avoid clobbering existing changes.")
                        continue

                    description = annotate_description_with_idea(
                        proposal.get("description", "unknown"),
                        proposal.get("idea_id"),
                    )
                    current_idea = f"[{proposed_file}] {description}"
                    print(f"  Idea: {description}")
                    print(f"  File: {proposed_file}")
                    update_text_state(
                        exp_num,
                        description,
                        "APPLYING",
                        file_path=proposed_file,
                        current_idea=current_idea,
                    )

                    try:
                        original = (PROJECT_ROOT / proposed_file).read_text(encoding="utf-8")
                    except FileNotFoundError:
                        print(f"  File not found: {proposed_file}")
                        continue

                    modified, apply_error = apply_changes(original, proposal["changes"])
                    if modified is None:
                        print("  Could not apply patch. Trying one repair...")
                        try:
                            repaired_proposal, repaired_text, repair_timed_out = repair_apply_failure(
                                proposal,
                                proposed_file,
                                original,
                                apply_error or "unknown apply failure",
                                call_llm,
                                fail_streak=fail_streak,
                            )
                        except FatalAPIError as e:
                            print(f"Fatal API error: {e}")
                            break

                        if repair_timed_out:
                            print(f"  Patch repair request timed out after {LLM_REQUEST_TIMEOUT_SECONDS}s. Skipping.")
                            log_result(
                                "-------",
                                0.0,
                                "crash",
                                proposed_file,
                                f"APPLY FAIL: {description} [repair timeout]",
                            )
                            history = get_results_history()
                            continue

                        if repaired_proposal and "changes" in repaired_proposal:
                            repaired_proposal = enforce_selected_idea_on_proposal(
                                repaired_proposal,
                                selected_idea,
                                execution_mode=execution_mode,
                            )
                            repaired_file = repaired_proposal.get("file", proposed_file)
                            if repaired_file == proposed_file:
                                repaired_modified, repaired_error = apply_changes(
                                    original,
                                    repaired_proposal["changes"],
                                )
                                if repaired_modified is not None:
                                    proposal = repaired_proposal
                                    description = annotate_description_with_idea(
                                        proposal.get("description", description),
                                        proposal.get("idea_id"),
                                    )
                                    current_idea = f"[{proposed_file}] {description}"
                                    modified = repaired_modified
                                    print("  Recovered apply-fail with repair prompt.")
                                    update_text_state(
                                        exp_num,
                                        description,
                                        "APPLYING",
                                        file_path=proposed_file,
                                        current_idea=current_idea,
                                    )
                                else:
                                    apply_error = repaired_error or apply_error
                            else:
                                apply_error = f"repair changed target file to {repaired_file}"

                        if modified is None:
                            print("  Could not apply patch after repair. Skipping.")
                            failure_detail = apply_error or "repair failed"
                            log_result(
                                "-------",
                                0.0,
                                "crash",
                                proposed_file,
                                f"APPLY FAIL: {description} [{failure_detail}]",
                            )
                            history = get_results_history()
                            continue

                    err = validate_file(proposed_file, modified, original_source=original)
                    if err:
                        print(f"  Validation failed: {err}. Skipping.")
                        log_result("-------", 0.0, "crash", proposed_file, f"VALIDATE: {description} [{err}]")
                        history = get_results_history()
                        continue

                    changed_paths = [proposed_file]
                    original_sources = {proposed_file: original}
                    modified_sources = {proposed_file: modified}
                    write_repo_file(proposed_file, modified)

                file_label = summarize_changed_paths(changed_paths)
                try:
                    sha = git_commit(description, changed_paths)
                    experiment_committed = True
                    experiment_commit_sha = git_head_sha()
                except RuntimeError as commit_err:
                    rollback_experiment_changes(
                        original_sources,
                        committed=False,
                        commit_sha=experiment_commit_sha,
                    )
                    err_msg = str(commit_err)[:120]
                    print(f"  COMMIT FAIL: {err_msg}")
                    log_result("-------", 0.0, "crash", file_label, f"COMMIT FAIL: {description} [{err_msg}]")
                    history = get_results_history()
                    continue

                print(f"  Committed: {sha}")

                print(f"  Running experiment...")
                update_text_state(
                    exp_num,
                    description,
                    "TRAINING",
                    file_path=file_label,
                    current_idea=current_idea,
                    extra={"commit": sha},
                )
                result = run_experiment(f"[{file_label}] {description}")

                if result and result.get("status") in ("keep", "discard"):
                    val_loss = float(result.get("metric_value", 999.0) or 999.0)
                    if should_run_lightweight_eval(result, exp_num, eval_settings):
                        print("  Running lightweight eval...")
                        maybe_run_lightweight_eval(
                            result,
                            exp_num,
                            train_cfg,
                            eval_settings,
                            on_line=lambda line: print(f"  {line}"),
                        )
                    if result.get("status") == "keep":
                        best_loss = val_loss
                        print(f"  KEEP: val_loss={val_loss:.6f} (NEW BEST)")
                        update_text_state(
                            exp_num,
                            description,
                            "KEEP",
                            file_path=file_label,
                            current_idea=current_idea,
                            extra={"commit": sha, "metric_value": val_loss},
                        )
                        git_push()
                    else:
                        print(f"  DISCARD: val_loss={val_loss:.6f} (best={best_loss:.6f})")
                        update_text_state(
                            exp_num,
                            description,
                            "DISCARD",
                            file_path=file_label,
                            current_idea=current_idea,
                            extra={"commit": sha, "metric_value": val_loss},
                        )
                        rollback_experiment_changes(
                            original_sources,
                            committed=experiment_committed,
                            commit_sha=experiment_commit_sha,
                        )
                elif result and result.get("status") == "crash":
                    print("  CRASH: experiment failed")
                    update_text_state(
                        exp_num,
                        description,
                        "CRASH",
                        file_path=file_label,
                        current_idea=current_idea,
                        extra={"commit": sha},
                    )
                    rollback_experiment_changes(
                        original_sources,
                        committed=experiment_committed,
                        commit_sha=experiment_commit_sha,
                    )
                elif result and "error" in result:
                    err_msg = str(result["error"])[:80]
                    print(f"  CRASH: {err_msg}")
                    log_result(sha, 0.0, "crash", file_label, f"{description} [{err_msg}]")
                    update_text_state(
                        exp_num,
                        description,
                        "CRASH",
                        file_path=file_label,
                        current_idea=current_idea,
                        extra={"commit": sha, "error": err_msg},
                    )
                    rollback_experiment_changes(
                        original_sources,
                        committed=experiment_committed,
                        commit_sha=experiment_commit_sha,
                    )
                else:
                    print("  CRASH: no result")
                    log_result(sha, 0.0, "crash", file_label, f"{description} [no output]")
                    update_text_state(
                        exp_num,
                        description,
                        "CRASH",
                        file_path=file_label,
                        current_idea=current_idea,
                        extra={"commit": sha, "error": "no result"},
                    )
                    rollback_experiment_changes(
                        original_sources,
                        committed=experiment_committed,
                        commit_sha=experiment_commit_sha,
                    )

                history = get_results_history()
                valid = [r["val_loss"] for r in history if r["val_loss"] < 999]
                if valid:
                    best_loss = min(valid)
                recent_run_context = build_recent_run_context()

            except Exception as loop_err:
                print(f"  Exception: {str(loop_err)[:120]}")
                log_to_file(f"TEXT LOOP EXCEPTION: {loop_err}")
                try:
                    rollback_experiment_changes(
                        original_sources,
                        committed=experiment_committed,
                        commit_sha=experiment_commit_sha,
                    )
                except Exception:
                    pass
                update_text_state(
                    exp_num,
                    str(loop_err)[:120],
                    "CRASH",
                    file_path=summarize_changed_paths(changed_paths) or proposed_file,
                    extra={"error": str(loop_err)[:120]},
                )
                history = get_results_history()
    finally:
        clear_state()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Autonomous research agent for Digital Twin Engine")
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_AUTORESEARCH_CONFIG.relative_to(PROJECT_ROOT)),
        help="Path to autoresearch config",
    )
    parser.add_argument("--max-runs",    type=int, default=100)
    parser.add_argument("--resume",      action="store_true", help="Resume existing branch")
    parser.add_argument(
        "--rebaseline",
        action="store_true",
        help="Run a fresh baseline against the current branch and promote it before new experiments",
    )
    parser.add_argument("--tag",         type=str,  default=None, help="Branch tag (default: date)")
    parser.add_argument("--no-dashboard", action="store_true", help="Text-only output")
    parser.add_argument("--file",        type=str,  default=None, help="Restrict to one modifiable file")
    parser.add_argument("--ideas-file",  type=str,  default=None, help="Structured YAML/JSON ideas backlog to prioritize")
    # LLM provider flags (default comes from autoresearch config)
    parser.add_argument("--gemini",  type=str, nargs="?", const=DEFAULT_GEMINI_MODEL, default=None,
                        help=f"Use Google Gemini model (default: {DEFAULT_GEMINI_MODEL})")
    parser.add_argument("--deepseek", type=str, nargs="?", const=DEFAULT_DEEPSEEK_MODEL, default=None,
                        help=f"Use DeepSeek model (default: {DEFAULT_DEEPSEEK_MODEL})")
    parser.add_argument("--claude",  action="store_true", help="Use Claude Sonnet 4.6")
    parser.add_argument("--opus",    action="store_true", help="Use Claude Opus 4.6 (32k thinking)")
    parser.add_argument("--sonnet4", action="store_true", help="Use Claude Sonnet 4 (legacy)")
    parser.add_argument("--openai",  type=str, nargs="?", const="o3", default=None,
                        help="Use OpenAI model (default: o3). Options: gpt-4.1, gpt-5.1, o3")
    parser.add_argument("--grok",    action="store_true", help="Use xAI Grok 3")
    parser.add_argument("--local",   action="store_true", help="Use local LM Studio")
    parser.add_argument("--temperature", type=float, default=None,
                        help="Override sampling temperature when the selected model supports it")
    parser.add_argument("--thinking-level", type=str, default=None,
                        help="Provider-specific thinking level (currently used for Gemini when set)")
    parser.add_argument("--apply-llm", type=str, default=None,
                        help="Separate model for JSON repair / patch repair calls (defaults to primary model)")
    parser.add_argument("--apply-temperature", type=float, default=None,
                        help="Override sampling temperature for repair/apply LLM calls")
    parser.add_argument("--apply-thinking-level", type=str, default=None,
                        help="Provider-specific thinking level for repair/apply calls")
    parser.add_argument(
        "--deepseek-max-tokens",
        type=int,
        default=None,
        help=f"Override DeepSeek max_tokens (default config: {DEFAULT_DEEPSEEK_MAX_TOKENS})",
    )
    args = parser.parse_args()
    active_config_path = set_autoresearch_config(args.config)

    ar_cfg = load_autoresearch_config()
    agent_cfg = ar_cfg.get("agent", {})
    execution_mode = get_execution_mode(ar_cfg)
    editable_targets = get_editable_targets(ar_cfg)
    forbidden_paths = get_forbidden_paths(ar_cfg)
    editable_file_inventory = list_editable_repo_files(editable_targets, forbidden_paths)
    research_cfg = ar_cfg.get("research", {})
    train_cfg = ar_cfg.get("train", {})
    eval_settings = get_eval_settings(ar_cfg)
    agent_context = load_agent_context(ar_cfg)
    structured_ideas = load_structured_ideas(ar_cfg, override_path=args.ideas_file)
    recent_run_context = build_recent_run_context()

    default_model = str(agent_cfg.get("default_llm", DEFAULT_GEMINI_MODEL))
    if args.local:
        provider_name = "local"
        provider_model = "local"
    elif args.opus:
        provider_name = "opus"
        provider_model = "claude-opus-4-6"
    elif args.sonnet4:
        provider_name = "claude"
        provider_model = "claude-sonnet-4-20250514"
    elif args.claude:
        provider_name = "claude"
        provider_model = DEFAULT_CLAUDE_MODEL
    elif args.openai:
        provider_name = "openai"
        provider_model = args.openai
    elif args.grok:
        provider_name = "grok"
        provider_model = DEFAULT_GROK_MODEL
    elif args.gemini:
        provider_name = "gemini"
        provider_model = args.gemini
    elif args.deepseek:
        provider_name = "deepseek"
        provider_model = args.deepseek
    else:
        provider_name = _infer_provider_from_model(default_model)
        provider_model = default_model

    provider_model = _default_model_for_provider(provider_name, provider_model)
    _call_llm_fn, llm_name = _build_llm_backend(provider_name, provider_model)

    apply_model_raw = args.apply_llm or agent_cfg.get("apply_llm") or provider_model
    apply_provider_name = _infer_provider_from_model(str(apply_model_raw))
    apply_provider_model = _default_model_for_provider(apply_provider_name, str(apply_model_raw))
    if apply_provider_name == provider_name and apply_provider_model == provider_model:
        _apply_llm_fn = _call_llm_fn
        apply_llm_name = llm_name
    else:
        _apply_llm_fn, apply_llm_name = _build_llm_backend(
            apply_provider_name,
            apply_provider_model,
        )

    args.llm_name = llm_name if apply_llm_name == llm_name else f"{llm_name} | repair {apply_llm_name}"

    adaptive_temperature_enabled = bool(agent_cfg.get("adaptive_temperature", True))
    adaptive_temperature_cap = float(agent_cfg.get("max_fail_streak_temp", 0.5))
    configured_temperature = args.temperature
    if configured_temperature is None and "temperature" in agent_cfg:
        configured_temperature = agent_cfg.get("temperature")
    if configured_temperature is not None:
        configured_temperature = float(configured_temperature)

    configured_apply_temperature = args.apply_temperature
    if configured_apply_temperature is None and "apply_temperature" in agent_cfg:
        configured_apply_temperature = agent_cfg.get("apply_temperature")
    if configured_apply_temperature is not None:
        configured_apply_temperature = float(configured_apply_temperature)

    deepseek_temperature = args.temperature
    if deepseek_temperature is None:
        deepseek_temperature = agent_cfg.get("deepseek_temperature", 0.0)
    if deepseek_temperature is not None:
        deepseek_temperature = float(deepseek_temperature)

    deepseek_max_tokens = args.deepseek_max_tokens
    if deepseek_max_tokens is None:
        deepseek_max_tokens = agent_cfg.get("deepseek_max_tokens", DEFAULT_DEEPSEEK_MAX_TOKENS)
    deepseek_max_tokens = max(1, min(int(deepseek_max_tokens), DEFAULT_DEEPSEEK_MAX_TOKENS))

    configured_thinking_level = args.thinking_level or agent_cfg.get("thinking_level")
    if configured_thinking_level is not None:
        configured_thinking_level = str(configured_thinking_level)

    configured_apply_thinking_level = args.apply_thinking_level or agent_cfg.get("apply_thinking_level")
    if configured_apply_thinking_level is None and apply_provider_name == "gemini":
        configured_apply_thinking_level = "low"
    if configured_apply_thinking_level is not None:
        configured_apply_thinking_level = str(configured_apply_thinking_level)

    supports_temperature = _provider_supports_temperature(provider_name, provider_model)
    supports_thinking_level = _provider_supports_thinking_level(provider_name)
    apply_supports_temperature = _provider_supports_temperature(
        apply_provider_name,
        apply_provider_model,
    )
    apply_supports_thinking_level = _provider_supports_thinking_level(apply_provider_name)

    def call_llm(
        prompt: str,
        fail_streak: int = 0,
        phase: str = "reason",
        response_schema: dict | None = None,
        cached_content: str | None = None,
    ) -> str | None:
        """Invoke the selected model with phase-aware routing and sampling controls."""

        is_apply_phase = phase == "apply"
        active_provider_name = apply_provider_name if is_apply_phase else provider_name
        active_provider_model = apply_provider_model if is_apply_phase else provider_model
        active_call_fn = _apply_llm_fn if is_apply_phase else _call_llm_fn
        active_supports_temperature = (
            apply_supports_temperature if is_apply_phase else supports_temperature
        )
        active_supports_thinking_level = (
            apply_supports_thinking_level if is_apply_phase else supports_thinking_level
        )

        if is_apply_phase:
            temperature = configured_apply_temperature
            if temperature is None and active_supports_temperature:
                temperature = 0.0
            thinking_level = (
                configured_apply_thinking_level if active_supports_thinking_level else None
            )
        else:
            temperature = configured_temperature
            if (
                temperature is None
                and active_provider_name == "deepseek"
                and active_provider_model == DEFAULT_DEEPSEEK_MODEL
            ):
                temperature = deepseek_temperature
            if temperature is None and adaptive_temperature_enabled and active_supports_temperature:
                if fail_streak >= 10:
                    temperature = adaptive_temperature_cap
                elif fail_streak >= 5:
                    temperature = min(0.3, adaptive_temperature_cap)
            thinking_level = configured_thinking_level if active_supports_thinking_level else None

        if active_provider_name == "deepseek":
            if active_supports_temperature:
                return active_call_fn(
                    prompt,
                    temperature=temperature,
                    thinking_level=thinking_level,
                    max_tokens=deepseek_max_tokens,
                )
            return active_call_fn(
                prompt,
                thinking_level=thinking_level,
                max_tokens=deepseek_max_tokens,
            )

        if active_provider_name == "gemini":
            kwargs: dict = {}
            if active_supports_temperature and temperature is not None:
                kwargs["temperature"] = temperature
            if thinking_level is not None:
                kwargs["thinking_level"] = thinking_level
            if response_schema is not None:
                kwargs["response_schema"] = response_schema
            if cached_content:
                kwargs["cached_content"] = cached_content

            def refresh_cache() -> str | None:
                nonlocal reason_cached_content
                if not gemini_context_cache_enabled or active_provider_model != provider_model:
                    return None
                reason_cached_content = ensure_gemini_context_cache(
                    static_prompt,
                    provider_model,
                    ttl=gemini_context_cache_ttl,
                    min_tokens=gemini_context_cache_min_tokens,
                )
                return reason_cached_content

            return _call_gemini_with_cache_retry(
                active_call_fn,
                prompt,
                kwargs=kwargs,
                refresh_cache=refresh_cache,
            )

        if active_supports_temperature:
            return active_call_fn(prompt, temperature=temperature, thinking_level=thinking_level)
        return active_call_fn(prompt, thinking_level=thinking_level)

    if args.file:
        normalized_forced_file = _normalize_repo_relative_path(args.file)
        if not normalized_forced_file:
            print(f"Warning: {args.file} is not a valid repo-relative path, proceeding anyway.")
            normalized_forced_file = args.file
        if execution_mode == "multi_file":
            if not is_allowed_edit_path(normalized_forced_file, editable_targets, forbidden_paths):
                print(f"Warning: {args.file} not under editable targets, proceeding anyway.")
            editable_targets_active = [normalized_forced_file]
        else:
            if normalized_forced_file not in editable_targets:
                print(f"Warning: {args.file} not in modifiable_files list, proceeding anyway.")
            editable_targets_active = [normalized_forced_file]
    else:
        editable_targets_active = editable_targets

    effective_max_runs = effective_run_target(
        args.max_runs,
        history=[],
        structured_ideas=structured_ideas,
        editable_targets=editable_targets_active,
        forced_file=args.file,
        execution_mode=execution_mode,
        forbidden_paths=forbidden_paths,
    )

    if execution_mode == "multi_file":
        editable_inventory_active = list_editable_repo_files(editable_targets_active, forbidden_paths)
        static_prompt = build_multi_file_static_prompt_context(
            editable_targets_active,
            editable_inventory_active,
            forbidden_paths,
            agent_context=agent_context,
        )
    else:
        editable_inventory_active = list_editable_repo_files(editable_targets_active, forbidden_paths)
        static_prompt = build_static_prompt_context(
            editable_targets_active,
            agent_context=agent_context,
        )
    if execution_mode == "multi_file" and not args.no_dashboard:
        print("Multi-file moonshot mode currently uses text-only output; enabling --no-dashboard.")
        args.no_dashboard = True
    gemini_context_cache_enabled = bool(agent_cfg.get("gemini_context_cache", False))
    gemini_context_cache_ttl = str(agent_cfg.get("gemini_context_cache_ttl", "3600s"))
    gemini_context_cache_min_tokens = max(
        0,
        int(agent_cfg.get("gemini_context_cache_min_tokens", GEMINI_CACHE_MIN_TOKENS) or 0),
    )
    reason_cached_content = None
    if gemini_context_cache_enabled and provider_name == "gemini":
        reason_cached_content = ensure_gemini_context_cache(
            static_prompt,
            provider_model,
            ttl=gemini_context_cache_ttl,
            min_tokens=gemini_context_cache_min_tokens,
        )

    # Git branch
    tag = args.tag or datetime.now().strftime("%b%d").lower()
    branch = setup_branch(tag, args.resume)

    init_results()

    # Check previous crash
    prev = read_state()
    if prev:
        if other_agent_process_running():
            log_to_file(f"Previous crash: exp={prev.get('experiment_num')} phase={prev.get('phase')}")
            print(f"Previous crash detected: exp={prev.get('experiment_num')}, phase={prev.get('phase')}")
        else:
            log_to_file("Ignoring stale agent_state.json from an earlier run")
        clear_state()

    # Load history
    history = get_results_history()
    prior_count = len(history)
    best_loss = get_current_reference_loss(history)

    log_to_file(
        f"Agent started: llm={llm_name} apply_llm={apply_llm_name} "
        f"cache={'on' if reason_cached_content else 'off'} max_runs={effective_max_runs} "
        f"branch={branch} prior={prior_count} mode={execution_mode} config={active_config_path}"
    )

    if args.no_dashboard:
        if args.rebaseline or not history:
            baseline_label = "Running fresh baseline experiment..." if args.rebaseline and history else "Running baseline experiment..."
            print(baseline_label)
            write_state(
                prior_count + 1,
                "baseline",
                "TRAINING",
                {
                    "max_runs": effective_max_runs,
                    "best_loss": None if best_loss >= 999.0 else best_loss,
                    "llm_name": args.llm_name,
                    "branch": branch,
                    "current_idea": "baseline",
                },
            )
            result = run_rebaseline_experiment() if args.rebaseline else run_experiment("baseline")
            if result and result.get("status") in ("keep", "discard"):
                val_loss = float(result.get("metric_value", 999.0) or 999.0)
                if should_run_lightweight_eval(result, prior_count + 1, eval_settings):
                    print("Running lightweight eval for baseline...")
                    maybe_run_lightweight_eval(
                        result,
                        prior_count + 1,
                        train_cfg,
                        eval_settings,
                        on_line=lambda line: print(f"  {line}"),
                    )
                history = get_results_history()
                best_loss = get_current_reference_loss(history)
                print(f"Baseline: val_loss={val_loss:.6f}")
            elif result and result.get("status") == "crash":
                print("Baseline failed inside autoresearch harness.")
                clear_state()
                return
            history = get_results_history()
            recent_run_context = build_recent_run_context()

        _run_text_mode(
            args,
            history,
            best_loss,
            call_llm,
            editable_targets_active,
            effective_max_runs,
            train_cfg,
            eval_settings,
            agent_context,
            recent_run_context,
            static_prompt,
            reason_cached_content,
            branch,
            structured_ideas,
            execution_mode,
            forbidden_paths,
        )
        return

    # ---- Dashboard mode ----
    console = Console()
    console.clear()

    state: dict = {
        "phase": "STARTING",
        "experiment_num": prior_count,
        "max_runs": effective_max_runs,
        "best_loss": best_loss,
        "llm_name": args.llm_name,
        "branch": branch,
        "current_idea": "",
        "history": history,
        "loss_history": deque(maxlen=200),
        "metrics": None,
        "gpu": get_gpu_stats(),
        "log_lines": deque(maxlen=60),
        "total_elapsed": 0.0,
        "phase_start": time.time(),
        "phase_elapsed": 0.0,
        "config_path": str(active_config_path.relative_to(PROJECT_ROOT)),
        "workspace_dir": str(get_workspace_dir().relative_to(PROJECT_ROOT)),
        "train_epochs": train_cfg.get("n_epochs"),
        "time_budget_minutes": research_cfg.get("time_budget_minutes"),
        "val_every": train_cfg.get("val_every"),
    }

    t_start = time.time()
    _last_gpu_poll = [0.0]

    def add_log(msg: str) -> None:
        state["log_lines"].append(f"  {msg}")
        log_to_file(msg)

    def update_gpu() -> None:
        now = time.time()
        if now - _last_gpu_poll[0] >= 3.0:
            state["gpu"] = get_gpu_stats()
            _last_gpu_poll[0] = now

    def set_phase(name: str) -> None:
        state["phase"] = name
        state["phase_start"] = time.time()
        state["phase_elapsed"] = 0.0
        if name not in ("TRAINING", "BASELINE"):
            state["metrics"] = None

    def on_training_line(line: str) -> None:
        # Capture any loss numbers printed during training
        if "train_loss" in line.lower() or "val_loss" in line.lower():
            # Try to parse a float after the colon
            match = re.search(r"(?:train|val)_loss[=:\s]+([0-9.eE+\-]+)", line, re.I)
            if match:
                try:
                    state["loss_history"].append(float(match.group(1)))
                except ValueError:
                    pass
        if line.strip():
            state["log_lines"].append(f"  {line[:110]}")

    eligible_structured_ideas = get_eligible_structured_ideas(
        structured_ideas,
        editable_targets_active,
        args.file,
        execution_mode=execution_mode,
        forbidden_paths=forbidden_paths,
    )
    strict_structured_ideas = bool(eligible_structured_ideas)

    with Live(build_dashboard(state), console=console, refresh_per_second=4) as live:

        def refresh() -> None:
            try:
                state["total_elapsed"] = time.time() - t_start
                state["phase_elapsed"] = time.time() - state["phase_start"]
                update_gpu()
                live.update(build_dashboard(state))
            except Exception as e:
                log_to_file(f"Dashboard error: {e}")

        # ---- Baseline ----
        if args.rebaseline or not history:
            set_phase("BASELINE")
            state["current_idea"] = (
                "Refreshing promoted baseline from current branch"
                if args.rebaseline and history
                else "Establishing baseline (no code changes)"
            )
            add_log(
                "Running fresh baseline experiment..."
                if args.rebaseline and history
                else "Running baseline experiment..."
            )
            refresh()

            if not wait_for_cool_gpu():
                add_log("GPU too hot. Aborting.")
                set_phase("DONE")
                refresh()
                time.sleep(3)
                return

            baseline_holder: list[dict | None] = [None]

            def _baseline_thread() -> None:
                baseline_holder[0] = (
                    run_rebaseline_experiment(on_line=on_training_line)
                    if args.rebaseline
                    else run_experiment("baseline", on_line=on_training_line)
                )

            bt = threading.Thread(target=_baseline_thread, daemon=True)
            bt.start()
            while bt.is_alive():
                refresh()
                time.sleep(0.5)
            bt.join()

            result = baseline_holder[0]
            refresh()

            if result and result.get("status") in ("keep", "discard"):
                val_loss = float(result.get("metric_value", 999.0) or 999.0)
                if should_run_lightweight_eval(result, prior_count + 1, eval_settings):
                    set_phase("EVALUATING")
                    add_log("Running lightweight eval for baseline...")
                    refresh()
                    maybe_run_lightweight_eval(
                        result,
                        prior_count + 1,
                        train_cfg,
                        eval_settings,
                        on_line=add_log,
                    )
                history = get_results_history()
                best_loss = get_current_reference_loss(history)
                state["best_loss"] = best_loss
                add_log(f"Baseline: val_loss={val_loss:.6f}")
            elif result and result.get("status") == "crash":
                add_log("Baseline failed inside autoresearch harness.")
                set_phase("DONE")
                refresh()
                time.sleep(5)
                return
            elif result and "error" in result:
                add_log(f"Baseline failed: {result['error'][:80]}")
                set_phase("DONE")
                refresh()
                time.sleep(5)
                return
            else:
                add_log("Baseline produced no result.")
                set_phase("DONE")
                refresh()
                time.sleep(5)
                return

            history = get_results_history()
            state["history"] = history
            recent_run_context = build_recent_run_context()
            refresh()

        best_loss = get_current_reference_loss(history)
        state["best_loss"] = best_loss
        add_log(f"Ready. Best val_loss={best_loss:.6f} | {len(history)} prior experiments")
        refresh()

        # ---- Main loop ----
        consecutive_parse_failures = 0
        non_idea_iteration = 0

        while True:
            history = get_results_history()
            state["history"] = history
            if strict_structured_ideas:
                resolved_idea_count = count_resolved_structured_ideas(history, eligible_structured_ideas)
                if resolved_idea_count >= len(eligible_structured_ideas):
                    add_log("All queued structured ideas have resolved keep/discard results.")
                    break
                exp_num = resolved_idea_count + 1
            else:
                if non_idea_iteration >= max(0, args.max_runs - prior_count):
                    break
                exp_num = prior_count + non_idea_iteration + 1
                non_idea_iteration += 1

            proposed_file = ""
            original: str | None = None
            experiment_committed = False
            experiment_commit_sha: str | None = None
            try:
                state["experiment_num"] = exp_num
                state["metrics"] = None

                # ---- THINK ----
                set_phase("THINKING")
                state["current_idea"] = ""
                add_log(f"Exp {exp_num}/{effective_max_runs}: querying {llm_name}...")
                write_state(exp_num, "querying LLM", "THINKING")
                refresh()

                selected_idea = select_next_structured_idea(
                    structured_ideas,
                    history,
                    editable_targets_active,
                    args.file,
                    execution_mode=execution_mode,
                    forbidden_paths=forbidden_paths,
                )
                if strict_structured_ideas and not selected_idea:
                    add_log("No pending structured ideas remain. Stopping.")
                    break
                target_file = _choose_file(
                    history,
                    editable_targets_active,
                    args.file,
                    preferred_file=selected_idea.get("target_file") if selected_idea else None,
                )
                try:
                    file_source = (PROJECT_ROOT / target_file).read_text(encoding="utf-8")
                except FileNotFoundError:
                    add_log(f"File not found: {target_file}")
                    continue

                fail_streak = _count_recent_failures(history)
                recent_run_context = build_recent_run_context()
                if fail_streak >= 5:
                    add_log(f"Streak {fail_streak} failures — strategy change needed")
                elif fail_streak >= 3:
                    add_log(f"Streak {fail_streak} failures — trying simpler approach")

                dynamic_prompt = build_dynamic_prompt(
                    target_file,
                    file_source,
                    history,
                    best_loss,
                    recent_run_context=recent_run_context,
                    ideas=structured_ideas,
                    selected_idea=selected_idea,
                )
                prompt = dynamic_prompt if reason_cached_content else static_prompt + dynamic_prompt

                # Cool GPU while LLM thinks
                if not wait_for_cool_gpu():
                    add_log("GPU too hot. Stopping.")
                    break

                try:
                    llm_response, llm_timed_out = invoke_llm_with_timeout(
                        call_llm,
                        prompt,
                        fail_streak=fail_streak,
                        phase="reason",
                        response_schema=PATCH_RESPONSE_JSON_SCHEMA,
                        cached_content=reason_cached_content,
                    )
                except FatalAPIError as e:
                    add_log(f"Fatal API error: {e}")
                    break

                if llm_timed_out:
                    add_log(f"LLM request timed out after {LLM_REQUEST_TIMEOUT_SECONDS}s. Skipping.")
                    history = get_results_history()
                    state["history"] = history
                    refresh()
                    continue

                proposal = parse_response(llm_response)
                if not proposal:
                    add_log("Malformed LLM response. Attempting JSON repair...")
                    try:
                        proposal, repaired, repair_timed_out = repair_response(
                            llm_response,
                            call_llm,
                            fail_streak=fail_streak,
                        )
                    except FatalAPIError as e:
                        add_log(f"Fatal API error: {e}")
                        break
                    if repair_timed_out:
                        add_log(f"JSON repair request timed out after {LLM_REQUEST_TIMEOUT_SECONDS}s. Skipping.")
                        history = get_results_history()
                        state["history"] = history
                        refresh()
                        continue
                    if proposal:
                        add_log("Recovered malformed LLM response.")

                if not proposal or "changes" not in proposal:
                    add_log("LLM gave unparseable response. Skipping.")
                    log_to_file(f"RAW: {(llm_response or '')[:500]}")
                    consecutive_parse_failures += 1
                    if (
                        consecutive_parse_failures >= MAX_CONSECUTIVE_PARSE_FAILURES
                        and not strict_structured_ideas
                    ):
                        add_log("Too many consecutive malformed LLM responses. Stopping cleanly.")
                        clear_state()
                        set_phase("DONE")
                        refresh()
                        break
                    history = get_results_history()
                    state["history"] = history
                    refresh()
                    continue
                consecutive_parse_failures = 0

                proposal = enforce_selected_idea_on_proposal(
                    proposal,
                    selected_idea,
                    execution_mode=execution_mode,
                )

                proposed_file = proposal.get("file", target_file)
                description = annotate_description_with_idea(
                    proposal.get("description", "unknown").replace("\t", " "),
                    proposal.get("idea_id"),
                )
                state["current_idea"] = f"[{proposed_file}] {description}"
                add_log(f"Idea: {description}")
                add_log(f"File: {proposed_file}")

                if proposed_file not in editable_targets_active:
                    add_log(f"Disallowed file: {proposed_file}. Skipping.")
                    history = get_results_history()
                    state["history"] = history
                    continue
                if git_file_is_dirty(proposed_file):
                    add_log(f"Skipping dirty file {proposed_file} to avoid clobbering existing changes.")
                    history = get_results_history()
                    state["history"] = history
                    refresh()
                    continue

                # ---- APPLY ----
                set_phase("APPLYING")
                refresh()

                try:
                    original = (PROJECT_ROOT / proposed_file).read_text(encoding="utf-8")
                except FileNotFoundError:
                    add_log(f"File not found: {proposed_file}")
                    continue

                modified, apply_error = apply_changes(original, proposal["changes"])
                if modified is None:
                    add_log("Could not apply patch. Trying one repair...")
                    try:
                        repaired_proposal, repaired_text, repair_timed_out = repair_apply_failure(
                            proposal,
                            proposed_file,
                            original,
                            apply_error or "unknown apply failure",
                            call_llm,
                            fail_streak=fail_streak,
                        )
                    except FatalAPIError as e:
                        add_log(f"Fatal API error: {e}")
                        break

                    if repair_timed_out:
                        add_log(f"Patch repair timed out after {LLM_REQUEST_TIMEOUT_SECONDS}s.")
                        log_result(
                            "-------",
                            0.0,
                            "crash",
                            proposed_file,
                            f"APPLY FAIL: {description} [repair timeout]",
                        )
                        history = get_results_history()
                        state["history"] = history
                        refresh()
                        continue

                    if repaired_proposal and "changes" in repaired_proposal:
                        repaired_proposal = enforce_selected_idea_on_proposal(
                            repaired_proposal,
                            selected_idea,
                            execution_mode=execution_mode,
                        )
                        repaired_file = repaired_proposal.get("file", proposed_file)
                        if repaired_file == proposed_file:
                            repaired_modified, repaired_error = apply_changes(
                                original,
                                repaired_proposal["changes"],
                            )
                            if repaired_modified is not None:
                                proposal = repaired_proposal
                                description = annotate_description_with_idea(
                                    proposal.get("description", description).replace("\t", " "),
                                    proposal.get("idea_id"),
                                )
                                state["current_idea"] = f"[{proposed_file}] {description}"
                                modified = repaired_modified
                                add_log("Recovered apply-fail with repair prompt.")
                            else:
                                apply_error = repaired_error or apply_error
                        else:
                            apply_error = f"repair changed target file to {repaired_file}"

                    if modified is None:
                        add_log("Could not apply patch after repair. Skipping.")
                        failure_detail = apply_error or "repair failed"
                        log_result(
                            "-------",
                            0.0,
                            "crash",
                            proposed_file,
                            f"APPLY FAIL: {description} [{failure_detail}]",
                        )
                        history = get_results_history()
                        state["history"] = history
                        refresh()
                        continue

                validate_err = validate_file(proposed_file, modified, original_source=original)
                if validate_err:
                    add_log(f"Validation failed: {validate_err}")
                    log_result("-------", 0.0, "crash", proposed_file, f"VALIDATE: {description} [{validate_err}]")
                    history = get_results_history()
                    state["history"] = history
                    refresh()
                    continue

                write_repo_file(proposed_file, modified)
                try:
                    sha = git_commit(description, [proposed_file])
                    experiment_committed = True
                    experiment_commit_sha = git_head_sha()
                except RuntimeError as commit_err:
                    rollback_experiment_change(
                        proposed_file,
                        original,
                        committed=False,
                        commit_sha=experiment_commit_sha,
                    )
                    err_msg = str(commit_err)[:120]
                    log_result(
                        "-------",
                        0.0,
                        "crash",
                        proposed_file,
                        f"COMMIT FAIL: {description} [{err_msg}]",
                    )
                    set_phase("CRASH")
                    add_log(f"COMMIT FAIL: {err_msg}")
                    history = get_results_history()
                    state["history"] = history
                    clear_state()
                    refresh()
                    continue
                add_log(f"Committed: {sha}")

                # ---- TRAIN ----
                set_phase("TRAINING")
                state["loss_history"] = deque(maxlen=200)
                write_state(exp_num, description, "TRAINING")
                refresh()

                t0 = time.time()
                result_holder: list[dict | None] = [None]

                def _train_thread() -> None:
                    result_holder[0] = run_experiment(
                        f"[{proposed_file}] {description}",
                        on_line=on_training_line,
                    )

                train_t = threading.Thread(target=_train_thread, daemon=True)
                train_t.start()
                while train_t.is_alive():
                    refresh()
                    time.sleep(0.5)
                train_t.join()

                result = result_holder[0]
                elapsed_exp = time.time() - t0

                # ---- Keep / Discard ----
                if result and result.get("status") in ("keep", "discard"):
                    val_loss = float(result.get("metric_value", 999.0) or 999.0)
                    improved = result.get("status") == "keep"
                    if should_run_lightweight_eval(result, exp_num, eval_settings):
                        set_phase("EVALUATING")
                        add_log("Running lightweight eval...")
                        refresh()
                        maybe_run_lightweight_eval(
                            result,
                            exp_num,
                            train_cfg,
                            eval_settings,
                            on_line=add_log,
                        )

                    if improved:
                        best_loss = val_loss
                        state["best_loss"] = best_loss
                        set_phase("KEEP")
                        add_log(f"KEEP: val_loss={val_loss:.6f} NEW BEST!")
                        git_push()
                    else:
                        set_phase("DISCARD")
                        add_log(f"DISCARD: val_loss={val_loss:.6f} (best={best_loss:.6f})")
                        rollback_experiment_change(
                            proposed_file,
                            original,
                            committed=experiment_committed,
                            commit_sha=experiment_commit_sha,
                        )

                elif result and result.get("status") == "crash":
                    set_phase("CRASH")
                    add_log("CRASH: experiment failed inside autoresearch harness")
                    rollback_experiment_change(
                        proposed_file,
                        original,
                        committed=experiment_committed,
                        commit_sha=experiment_commit_sha,
                    )
                elif result and "error" in result:
                    err_msg = str(result["error"])[:80]
                    log_result(sha, 0.0, "crash", proposed_file, f"{description} [{err_msg}]")
                    set_phase("CRASH")
                    add_log(f"CRASH: {err_msg}")
                    rollback_experiment_change(
                        proposed_file,
                        original,
                        committed=experiment_committed,
                        commit_sha=experiment_commit_sha,
                    )
                else:
                    log_result(sha, 0.0, "crash", proposed_file, f"{description} [no output]")
                    set_phase("CRASH")
                    add_log("CRASH: no output from experiment")
                    rollback_experiment_change(
                        proposed_file,
                        original,
                        committed=experiment_committed,
                        commit_sha=experiment_commit_sha,
                    )

                history = get_results_history()
                state["history"] = history
                recent_run_context = build_recent_run_context()
                clear_state()
                add_log(f"Elapsed: {elapsed_exp:.0f}s")
                refresh()

                # Brief pause between experiments
                if i < remaining - 1:
                    for _ in range(6):
                        refresh()
                        time.sleep(1)

            except Exception as loop_err:
                log_to_file(f"LOOP EXCEPTION: {loop_err}")
                add_log(f"Exception: {str(loop_err)[:80]}")
                try:
                    rollback_experiment_change(
                        proposed_file,
                        original,
                        committed=experiment_committed,
                        commit_sha=experiment_commit_sha,
                    )
                except Exception:
                    pass
                try:
                    clear_state()
                except Exception:
                    pass
                refresh()

        set_phase("DONE")
        refresh()
        time.sleep(3)

    # Summary
    console.print()
    history = get_results_history()
    kept = [r for r in history if r["status"] == "keep"]
    console.print(f"[bold]Total experiments:[/] {len(history)}")
    console.print(f"[bold]Kept:[/] {len(kept)}")
    console.print(f"[bold]Best val_loss:[/] {best_loss:.6f}")
    console.print(f"[bold]Branch:[/] {branch}")
    console.print(f"[bold]Results:[/] {get_results_tsv_path()}")
    console.print(f"[bold]Log:[/] {LOG_FILE}")


if __name__ == "__main__":
    main()
