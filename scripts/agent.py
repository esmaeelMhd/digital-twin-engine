"""Autonomous research agent for Digital Twin Engine.

Loops indefinitely: asks an LLM to propose a single-file code change,
applies it, runs the autoresearch harness (scripts/autoresearch.py),
keeps improvements, reverts failures, and shows a Rich TUI dashboard.

Usage:
    python scripts/agent.py                    # Provider from autoresearch config
    python scripts/agent.py --config configs/autoresearch_stage1.yaml
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
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from collections import deque
from datetime import datetime
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
LOG_FILE = PROJECT_ROOT / "agent.log"
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
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(f"[{ts}] {msg}\n")
        f.flush()
        os.fsync(f.fileno())


def write_state(exp_num: int, description: str, phase: str, extra: dict | None = None) -> None:
    payload = {
        "timestamp": datetime.now().isoformat(),
        "experiment_num": exp_num,
        "description": description,
        "phase": phase,
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

    baseline_metadata = _read_json_if_exists(get_workspace_dir() / "baseline" / "metadata.json")
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


def git_commit(message: str, files: list[str]) -> str:
    for f in files:
        git("add", f)
    subprocess.run(
        ["git", "commit", "-m", message],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    return git_short_sha()


def git_revert_file(filepath: str) -> None:
    """Restore a single file to the current HEAD commit."""
    git("restore", "--source=HEAD", "--staged", "--worktree", "--", filepath)


def git_reset_last_commit() -> None:
    """Undo the last commit but keep working tree changes."""
    git("reset", "--mixed", "HEAD~1")


def git_discard_last_experiment(filepath: str) -> None:
    """Drop the last experiment commit and restore the touched file."""

    git_reset_last_commit()
    git_revert_file(filepath)


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


def call_gemini(
    prompt: str,
    temperature: float | None = None,
    thinking_level: str | None = None,
    model: str = DEFAULT_GEMINI_MODEL,
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
            if thinking_level:
                try:
                    config_kwargs["thinking_config"] = types.ThinkingConfig(
                        thinking_level=str(thinking_level).lower()
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
            "max_tokens": 8192,
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

KNOWN_GOOD = """
## Proven techniques for JAX Neural SDE digital twins:
- Increasing trajectory loss weight (10.0 -> 20.0) for better long-horizon predictions
- Reducing KL weight during early training (0.001 -> 0.0001 or 0.0)
- Cosine annealing with warm restarts instead of linear warmup decay
- Gradient clipping reduction (1.0 -> 0.5) for more stable SDE training
- Increasing latent_dim (16 -> 32) for more expressive latent space
- Adding weight decay (e.g., 1e-4) to encoder/decoder linear layers
- Using EulerHeun instead of Heun for faster SDE steps (lower dt_ratio)
- Adjusting diffusion scale initialisation (e.g., log-scale init at -1 instead of 0)
- Reducing physics loss weights (0.1 -> 0.01) early in training to avoid gradient dominance
- Larger batch_size for JAX compiled code (64 -> 128) to improve throughput
- Curriculum training: start with shorter seq_len and grow it (curriculum_seq_len_start / _end in config)
- Teacher forcing: high tf_weight early, anneal to 0 for free-rollout robustness (teacher_forcing_weight)
- Stochastic SDE path (use_stochastic_training: true) vs deterministic mean trajectory
- Increasing seq_len to expose the model to longer dependencies
These are NOT guaranteed to work but are worth trying if not yet attempted.
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


def _count_recent_failures(history: list[dict]) -> int:
    count = 0
    for r in reversed(history):
        if r["status"] == "keep":
            break
        count += 1
    return min(count, 10)


def build_prompt(
    file_path: str,
    file_source: str,
    history: list[dict],
    best_loss: float,
    modifiable_files: list[str],
    agent_context: str = "",
    recent_run_context: str = "",
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
            near_misses_str = "\n## Near misses (within 5% of best — consider combining or tweaking):\n"
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
Look at category summary — avoid exhausted categories. Try a different file or approach.
"""
    elif fail_streak >= 3:
        streak_str = f"\n## CAUTION: {fail_streak} consecutive failures. Try simpler change.\n"

    files_list = "\n".join(f"  - {f}" for f in modifiable_files)
    context_section = ""
    if agent_context:
        context_section = f"\n## Repo context\n{agent_context}\n"
    recent_runs_section = ""
    if recent_run_context:
        recent_runs_section = f"\n{recent_run_context}\n"

    return f"""You are an autonomous ML researcher. Your goal: minimise best_val_loss for a \
physics-informed latent Neural SDE (digital twin) trained on process system data (CSTR, heat exchanger, or other registered systems).

## Constraints
- You MUST only modify ONE of these files per experiment:
{files_list}
- Do NOT modify the experiment harness: scripts/autoresearch.py, dte/autoresearch/*, program.md
- One idea per experiment. Keep changes minimal and surgical.
- Preserve the generic architecture. In dte/models and dte/training, do not add system-specific branches or bake config-like numbers into code; generic numeric algorithmic tweaks are fine, but bounds/scales/defaults/dims should live in config/SystemSpec.
- Available packages: jax, equinox, diffrax, optax, jaxtyping, numpy, yaml, h5py (no new installs).
{JAX_PITFALLS}{ARCHITECTURE_GUARDRAILS}
## Current best_val_loss: {best_loss:.6f}
{streak_str}{history_section}{KNOWN_GOOD}{context_section}{recent_runs_section}
## Currently showing: {file_path}
```
{file_source}
```

## Your task
Propose ONE modification to lower best_val_loss. Do NOT repeat or closely variant anything already tried.
Think creatively about what hasn't been attempted yet.

Respond with ONLY a JSON object (no markdown fences, no explanation):
{{
  "file": "repo-relative path to the file you want to modify (must be from the allowed list)",
  "description": "short description of the change (no tabs)",
  "changes": [
    {{
      "old": "exact string to find in that file",
      "new": "replacement string"
    }}
  ]
}}

Each change is a find-and-replace. "old" MUST appear exactly once in the file.
Keep changes minimal. One idea at a time."""


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
      "old": "exact string to find",
      "new": "replacement string"
    }}
  ]
}}

Do not add markdown fences or commentary.

Malformed response:
{text}
"""
    repaired, timed_out = invoke_llm_with_timeout(
        call_llm,
        repair_prompt,
        fail_streak=max(fail_streak, 1),
    )
    return parse_response(repaired), repaired, timed_out


def invoke_llm_with_timeout(
    call_llm,
    prompt: str,
    *,
    fail_streak: int = 0,
    timeout_seconds: int = LLM_REQUEST_TIMEOUT_SECONDS,
) -> tuple[str | None, bool]:
    """Run an LLM call in a daemon thread and enforce a hard timeout."""

    llm_result: list[str | None] = [None]
    llm_fatal: list[FatalAPIError | None] = [None]

    def _llm_thread() -> None:
        try:
            llm_result[0] = call_llm(prompt, fail_streak=fail_streak)
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


def apply_changes(source: str, changes: list[dict]) -> str | None:
    modified = source
    for change in changes:
        old = change.get("old", "")
        new = change.get("new", "")
        if not old:
            continue
        count = modified.count(old)
        if count == 0:
            log_to_file(f"APPLY FAIL: string not found: {old[:80]!r}")
            return None
        if count > 1:
            log_to_file(f"APPLY FAIL: ambiguous match ({count}x): {old[:80]!r}")
            return None
        modified = modified.replace(old, new, 1)
    return modified


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
        Layout(name="body",   ratio=3),
        Layout(name="footer", ratio=2),
    )
    layout["body"].split_row(
        Layout(name="left",  ratio=3),
        Layout(name="right", ratio=2),
    )
    layout["left"].split_column(
        Layout(name="training",    ratio=2),
        Layout(name="experiments", ratio=3),
    )
    layout["right"].split_column(
        Layout(name="gpu",      ratio=2),
        Layout(name="activity", ratio=3),
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

def _choose_file(history: list[dict], modifiable_files: list[str], forced_file: str | None) -> str:
    """Pick the file to show in the prompt. Rotates through modifiable files."""
    if forced_file:
        return forced_file
    # Prefer least-recently-tried files
    file_counts: dict[str, int] = {f: 0 for f in modifiable_files}
    for r in history:
        fpath = r.get("file", "")
        if fpath in file_counts:
            file_counts[fpath] += 1
    return min(file_counts, key=lambda f: file_counts[f])


# ---------------------------------------------------------------------------
# Text-mode fallback
# ---------------------------------------------------------------------------

def _run_text_mode(
    args,
    history: list[dict],
    best_loss: float,
    call_llm,
    modifiable_files: list[str],
    max_runs: int,
    train_cfg: dict,
    eval_settings: dict,
    agent_context: str,
    recent_run_context: str,
) -> None:
    """Simple text output mode when Rich is not desired."""
    prior_count = len(history)
    consecutive_parse_failures = 0

    for i in range(max_runs - prior_count):
        exp_num = prior_count + i + 1
        print(f"\n[Exp {exp_num}/{max_runs}] Asking {args.llm_name}...")
        recent_run_context = build_recent_run_context()

        target_file = _choose_file(history, modifiable_files, args.file)
        try:
            file_source = (PROJECT_ROOT / target_file).read_text(encoding="utf-8")
        except FileNotFoundError:
            print(f"  File not found: {target_file}")
            continue

        fail_streak = _count_recent_failures(history)
        prompt = build_prompt(
            target_file,
            file_source,
            history,
            best_loss,
            modifiable_files,
            agent_context=agent_context,
            recent_run_context=recent_run_context,
        )

        try:
            response, timed_out = invoke_llm_with_timeout(
                call_llm,
                prompt,
                fail_streak=fail_streak,
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

        if not proposal or "changes" not in proposal:
            print("  LLM gave unparseable response. Skipping.")
            consecutive_parse_failures += 1
            if consecutive_parse_failures >= MAX_CONSECUTIVE_PARSE_FAILURES:
                print("  Too many consecutive malformed LLM responses. Stopping cleanly.")
                break
            continue
        consecutive_parse_failures = 0

        proposed_file = proposal.get("file", target_file)
        if proposed_file not in modifiable_files:
            print(f"  LLM chose disallowed file {proposed_file}. Skipping.")
            continue

        description = proposal.get("description", "unknown")
        print(f"  Idea: {description}")
        print(f"  File: {proposed_file}")

        try:
            original = (PROJECT_ROOT / proposed_file).read_text(encoding="utf-8")
        except FileNotFoundError:
            print(f"  File not found: {proposed_file}")
            continue

        modified = apply_changes(original, proposal["changes"])
        if modified is None:
            print("  Could not apply patch. Skipping.")
            log_result("-------", 0.0, "crash", proposed_file, f"APPLY FAIL: {description}")
            history = get_results_history()
            continue

        err = validate_file(proposed_file, modified, original_source=original)
        if err:
            print(f"  Validation failed: {err}. Skipping.")
            git_revert_file(proposed_file)
            log_result("-------", 0.0, "crash", proposed_file, f"VALIDATE: {description} [{err}]")
            history = get_results_history()
            continue

        (PROJECT_ROOT / proposed_file).write_text(modified, encoding="utf-8")
        sha = git_commit(description, [proposed_file])
        print(f"  Committed: {sha}")

        print(f"  Running experiment...")
        result = run_experiment(f"[{proposed_file}] {description}")

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
                git_push()
            else:
                print(f"  DISCARD: val_loss={val_loss:.6f} (best={best_loss:.6f})")
                git_discard_last_experiment(proposed_file)
        elif result and result.get("status") == "crash":
            print("  CRASH: experiment failed")
            git_discard_last_experiment(proposed_file)
        elif result and "error" in result:
            err_msg = str(result["error"])[:80]
            print(f"  CRASH: {err_msg}")
            log_result(sha, 0.0, "crash", proposed_file, f"{description} [{err_msg}]")
            git_discard_last_experiment(proposed_file)
        else:
            print("  CRASH: no result")
            log_result(sha, 0.0, "crash", proposed_file, f"{description} [no output]")
            git_discard_last_experiment(proposed_file)

        history = get_results_history()
        valid = [r["val_loss"] for r in history if r["val_loss"] < 999]
        if valid:
            best_loss = min(valid)
        recent_run_context = build_recent_run_context()


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
    parser.add_argument("--tag",         type=str,  default=None, help="Branch tag (default: date)")
    parser.add_argument("--no-dashboard", action="store_true", help="Text-only output")
    parser.add_argument("--file",        type=str,  default=None, help="Restrict to one modifiable file")
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
    args = parser.parse_args()
    active_config_path = set_autoresearch_config(args.config)

    ar_cfg = load_autoresearch_config()
    agent_cfg = ar_cfg.get("agent", {})
    modifiable_files = agent_cfg.get("modifiable_files", MODIFIABLE_FILES)
    research_cfg = ar_cfg.get("research", {})
    train_cfg = ar_cfg.get("train", {})
    eval_settings = get_eval_settings(ar_cfg)
    agent_context = load_agent_context(ar_cfg)
    recent_run_context = build_recent_run_context()

    default_model = str(agent_cfg.get("default_llm", DEFAULT_DEEPSEEK_MODEL))
    provider_name = ""
    provider_model = ""

    if args.local:
        provider_name = "local"
        provider_model = "local"
        _call_llm_fn = call_local
        llm_name = "LM Studio (local)"
    elif args.opus:
        provider_name = "opus"
        provider_model = "claude-opus-4-6"
        _call_llm_fn = call_claude_opus
        llm_name = "Claude Opus 4.6"
    elif args.sonnet4:
        provider_name = "claude"
        provider_model = "claude-sonnet-4-20250514"
        _call_llm_fn = lambda prompt, temperature=None, thinking_level=None: call_claude(
            prompt,
            temperature=temperature,
            thinking_level=thinking_level,
            model=provider_model,
        )
        llm_name = "Claude Sonnet 4"
    elif args.claude:
        provider_name = "claude"
        provider_model = DEFAULT_CLAUDE_MODEL
        _call_llm_fn = lambda prompt, temperature=None, thinking_level=None: call_claude(
            prompt,
            temperature=temperature,
            thinking_level=thinking_level,
            model=provider_model,
        )
        llm_name = "Claude Sonnet 4.6"
    elif args.openai:
        provider_name = "openai"
        provider_model = args.openai
        _call_llm_fn = lambda prompt, temperature=None, thinking_level=None: call_openai(
            prompt,
            temperature=temperature,
            thinking_level=thinking_level,
            model=provider_model,
        )
        llm_name = f"OpenAI {provider_model}"
    elif args.grok:
        provider_name = "grok"
        provider_model = DEFAULT_GROK_MODEL
        _call_llm_fn = lambda prompt, temperature=None, thinking_level=None: call_grok(
            prompt,
            temperature=temperature,
            thinking_level=thinking_level,
            model=provider_model,
        )
        llm_name = "xAI Grok 3"
    elif args.gemini:
        provider_name = "gemini"
        provider_model = args.gemini
        _call_llm_fn = lambda prompt, temperature=None, thinking_level=None: call_gemini(
            prompt,
            temperature=temperature,
            thinking_level=thinking_level,
            model=provider_model,
        )
        llm_name = f"Gemini {provider_model}"
    elif args.deepseek:
        provider_name = "deepseek"
        provider_model = args.deepseek
        _call_llm_fn = lambda prompt, temperature=None, thinking_level=None: call_deepseek(
            prompt,
            temperature=temperature,
            thinking_level=thinking_level,
            model=provider_model,
        )
        llm_name = f"DeepSeek {provider_model}"
    else:
        provider_name = _infer_provider_from_model(default_model)
        provider_model = default_model
        if provider_name == "local":
            _call_llm_fn = call_local
            llm_name = "LM Studio (local)"
        elif provider_name == "deepseek":
            _call_llm_fn = lambda prompt, temperature=None, thinking_level=None: call_deepseek(
                prompt,
                temperature=temperature,
                thinking_level=thinking_level,
                model=provider_model,
            )
            llm_name = f"DeepSeek {provider_model}"
        elif provider_name == "claude":
            _call_llm_fn = lambda prompt, temperature=None, thinking_level=None: call_claude(
                prompt,
                temperature=temperature,
                thinking_level=thinking_level,
                model=provider_model or DEFAULT_CLAUDE_MODEL,
            )
            llm_name = f"Claude {provider_model}"
        elif provider_name == "opus":
            _call_llm_fn = call_claude_opus
            llm_name = "Claude Opus 4.6"
        elif provider_name == "openai":
            _call_llm_fn = lambda prompt, temperature=None, thinking_level=None: call_openai(
                prompt,
                temperature=temperature,
                thinking_level=thinking_level,
                model=provider_model,
            )
            llm_name = f"OpenAI {provider_model}"
        elif provider_name == "grok":
            _call_llm_fn = lambda prompt, temperature=None, thinking_level=None: call_grok(
                prompt,
                temperature=temperature,
                thinking_level=thinking_level,
                model=provider_model or DEFAULT_GROK_MODEL,
            )
            llm_name = f"xAI {provider_model}"
        else:
            provider_name = "gemini"
            provider_model = provider_model or DEFAULT_GEMINI_MODEL
            _call_llm_fn = lambda prompt, temperature=None, thinking_level=None: call_gemini(
                prompt,
                temperature=temperature,
                thinking_level=thinking_level,
                model=provider_model,
            )
            llm_name = f"Gemini {provider_model}"

    args.llm_name = llm_name

    adaptive_temperature_enabled = bool(agent_cfg.get("adaptive_temperature", True))
    adaptive_temperature_cap = float(agent_cfg.get("max_fail_streak_temp", 0.5))
    configured_temperature = args.temperature
    if configured_temperature is None and "temperature" in agent_cfg:
        configured_temperature = agent_cfg.get("temperature")
    if configured_temperature is not None:
        configured_temperature = float(configured_temperature)

    deepseek_temperature = args.temperature
    if deepseek_temperature is None:
        deepseek_temperature = agent_cfg.get("deepseek_temperature", 0.0)
    if deepseek_temperature is not None:
        deepseek_temperature = float(deepseek_temperature)

    configured_thinking_level = args.thinking_level or agent_cfg.get("thinking_level")
    if configured_thinking_level is not None:
        configured_thinking_level = str(configured_thinking_level)

    supports_temperature = _provider_supports_temperature(provider_name, provider_model)
    supports_thinking_level = _provider_supports_thinking_level(provider_name)

    def call_llm(prompt: str, fail_streak: int = 0) -> str | None:
        """Invoke the selected model with config-aware temperature/thinking settings."""

        temperature = configured_temperature
        if temperature is None and provider_name == "deepseek" and provider_model == DEFAULT_DEEPSEEK_MODEL:
            temperature = deepseek_temperature
        if temperature is None and adaptive_temperature_enabled and supports_temperature:
            if fail_streak >= 10:
                temperature = adaptive_temperature_cap
            elif fail_streak >= 5:
                temperature = min(0.3, adaptive_temperature_cap)

        thinking_level = configured_thinking_level if supports_thinking_level else None
        if supports_temperature:
            return _call_llm_fn(prompt, temperature=temperature, thinking_level=thinking_level)
        return _call_llm_fn(prompt, thinking_level=thinking_level)

    if args.file:
        if args.file not in modifiable_files:
            print(f"Warning: {args.file} not in modifiable_files list, proceeding anyway.")
        modifiable_files_active = [args.file]
    else:
        modifiable_files_active = modifiable_files

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
    valid_losses = [r["val_loss"] for r in history if r["val_loss"] < 999]
    best_loss = min(valid_losses) if valid_losses else 999.0

    log_to_file(
        f"Agent started: llm={llm_name} max_runs={args.max_runs} "
        f"branch={branch} prior={prior_count} config={active_config_path}"
    )

    if args.no_dashboard:
        if not history:
            print("Running baseline experiment...")
            result = run_experiment("baseline")
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
                best_loss = val_loss
                print(f"Baseline: val_loss={val_loss:.6f}")
            elif result and result.get("status") == "crash":
                print("Baseline failed inside autoresearch harness.")
                return
            history = get_results_history()
            recent_run_context = build_recent_run_context()

        _run_text_mode(
            args,
            history,
            best_loss,
            call_llm,
            modifiable_files_active,
            args.max_runs,
            train_cfg,
            eval_settings,
            agent_context,
            recent_run_context,
        )
        return

    # ---- Dashboard mode ----
    console = Console()
    console.clear()

    state: dict = {
        "phase": "STARTING",
        "experiment_num": prior_count,
        "max_runs": args.max_runs,
        "best_loss": best_loss,
        "llm_name": llm_name,
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
        if not history:
            set_phase("BASELINE")
            state["current_idea"] = "Establishing baseline (no code changes)"
            add_log("Running baseline experiment...")
            refresh()

            if not wait_for_cool_gpu():
                add_log("GPU too hot. Aborting.")
                set_phase("DONE")
                refresh()
                time.sleep(3)
                return

            baseline_holder: list[dict | None] = [None]

            def _baseline_thread() -> None:
                baseline_holder[0] = run_experiment("baseline", on_line=on_training_line)

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
                best_loss = val_loss
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

        best_loss_valid = [r["val_loss"] for r in history if r["val_loss"] < 999]
        best_loss = min(best_loss_valid) if best_loss_valid else 999.0
        state["best_loss"] = best_loss
        add_log(f"Ready. Best val_loss={best_loss:.6f} | {len(history)} prior experiments")
        refresh()

        # ---- Main loop ----
        remaining = max(0, args.max_runs - len(history))
        consecutive_parse_failures = 0

        for i in range(remaining):
            try:
                exp_num = len(history) + 1
                state["experiment_num"] = exp_num
                state["metrics"] = None

                # ---- THINK ----
                set_phase("THINKING")
                state["current_idea"] = ""
                add_log(f"Exp {exp_num}/{args.max_runs}: querying {llm_name}...")
                write_state(exp_num, "querying LLM", "THINKING")
                refresh()

                target_file = _choose_file(history, modifiable_files_active, args.file)
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

                prompt = build_prompt(
                    target_file,
                    file_source,
                    history,
                    best_loss,
                    modifiable_files_active,
                    agent_context=agent_context,
                    recent_run_context=recent_run_context,
                )

                # Cool GPU while LLM thinks
                if not wait_for_cool_gpu():
                    add_log("GPU too hot. Stopping.")
                    break

                try:
                    llm_response, llm_timed_out = invoke_llm_with_timeout(
                        call_llm,
                        prompt,
                        fail_streak=fail_streak,
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
                    if consecutive_parse_failures >= MAX_CONSECUTIVE_PARSE_FAILURES:
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

                proposed_file = proposal.get("file", target_file)
                description = proposal.get("description", "unknown").replace("\t", " ")
                state["current_idea"] = f"[{proposed_file}] {description}"
                add_log(f"Idea: {description}")
                add_log(f"File: {proposed_file}")

                if proposed_file not in modifiable_files_active:
                    add_log(f"Disallowed file: {proposed_file}. Skipping.")
                    history = get_results_history()
                    state["history"] = history
                    continue

                # ---- APPLY ----
                set_phase("APPLYING")
                refresh()

                try:
                    original = (PROJECT_ROOT / proposed_file).read_text(encoding="utf-8")
                except FileNotFoundError:
                    add_log(f"File not found: {proposed_file}")
                    continue

                modified = apply_changes(original, proposal["changes"])
                if modified is None:
                    add_log("Could not apply patch. Skipping.")
                    log_result("-------", 0.0, "crash", proposed_file, f"APPLY FAIL: {description}")
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

                (PROJECT_ROOT / proposed_file).write_text(modified, encoding="utf-8")
                sha = git_commit(description, [proposed_file])
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
                        git_discard_last_experiment(proposed_file)

                elif result and result.get("status") == "crash":
                    set_phase("CRASH")
                    add_log("CRASH: experiment failed inside autoresearch harness")
                    git_discard_last_experiment(proposed_file)
                elif result and "error" in result:
                    err_msg = str(result["error"])[:80]
                    log_result(sha, 0.0, "crash", proposed_file, f"{description} [{err_msg}]")
                    set_phase("CRASH")
                    add_log(f"CRASH: {err_msg}")
                    git_discard_last_experiment(proposed_file)
                else:
                    log_result(sha, 0.0, "crash", proposed_file, f"{description} [no output]")
                    set_phase("CRASH")
                    add_log("CRASH: no output from experiment")
                    git_discard_last_experiment(proposed_file)

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
                    git_discard_last_experiment(proposed_file)
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
