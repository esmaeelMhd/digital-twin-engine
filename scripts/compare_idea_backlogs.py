"""Run one autoresearch branch per ideas file from the same main snapshot.

This is intended for comparing model-generated idea backlogs fairly:
each ideas file gets
1. the same base config,
2. the same starting `main` commit,
3. an isolated workspace directory,
4. its own branch on `origin`.

The script does not merge anything into `main`. It writes a comparison summary
under `outputs/idea_backlog_compare/<session_tag>/summary.{json,tsv}`.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shlex
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
AGENT_SCRIPT = PROJECT_ROOT / "scripts" / "agent.py"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "idea_backlog_compare"
CAMPAIGN_NAMES = {
    "latent",
    "trainer",
    "losses",
    "encoder",
    "decoder",
    "digital_twin",
}


class RunnerError(RuntimeError):
    """Raised when the comparison runner cannot proceed safely."""


@dataclass
class BacklogPlan:
    name: str
    ideas_file: str
    config_path: str
    workspace_dir: str
    branch: str
    tag: str


def _run_git(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        ["git", *args],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if check and completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "git command failed").strip()
        raise RunnerError(f"git {' '.join(args)} failed: {detail}")
    return completed


def _git_output(*args: str, check: bool = True) -> str:
    return _run_git(*args, check=check).stdout.strip()


def ensure_clean_worktree() -> None:
    status = _git_output("status", "--porcelain", "--untracked-files=no")
    if status:
        raise RunnerError(
            "working tree has tracked changes; commit or stash them before running backlog comparison"
        )


def ensure_no_other_agent_running() -> None:
    completed = subprocess.run(
        ["pgrep", "-af", "scripts/agent.py"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    current_pid = os.getpid()
    for line in completed.stdout.splitlines():
        parts = line.strip().split(maxsplit=1)
        if not parts:
            continue
        try:
            pid = int(parts[0])
        except ValueError:
            continue
        if pid == current_pid:
            continue
        raise RunnerError(f"another scripts/agent.py process is already running: {line.strip()}")


def checkout_and_sync_main() -> str:
    current = _git_output("branch", "--show-current")
    if current != "main":
        _run_git("checkout", "main")
    _run_git("pull", "--ff-only", "origin", "main")
    return _git_output("rev-parse", "main")


def branch_exists(branch: str) -> bool:
    return bool(_git_output("branch", "--list", branch))


def remote_branch_exists(branch: str) -> bool:
    completed = _run_git("ls-remote", "--exit-code", "--heads", "origin", branch, check=False)
    return completed.returncode == 0


def create_branch_from_base(branch: str, base_sha: str) -> None:
    if branch_exists(branch):
        raise RunnerError(f"branch already exists locally: {branch}")
    if remote_branch_exists(branch):
        raise RunnerError(f"branch already exists on origin: {branch}")
    _run_git("checkout", "-b", branch, base_sha)
    _run_git("push", "--set-upstream", "origin", branch)


def push_branch(branch: str) -> None:
    current = _git_output("branch", "--show-current")
    if current != branch:
        _run_git("checkout", branch)
    _run_git("push", "origin", branch)


def load_yaml(path: Path) -> dict:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}
    except Exception as exc:
        raise RunnerError(f"failed to load YAML {path}: {exc}") from exc


def write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def slugify_name(raw: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", raw).strip("-").lower()
    if not slug:
        raise RunnerError(f"cannot derive slug from {raw!r}")
    return slug


def default_backlog_name(ideas_path: Path) -> str:
    stem = ideas_path.stem
    prefix = "auto_research_"
    suffix = "_ideas"
    if stem.startswith(prefix) and stem.endswith(suffix):
        middle = stem[len(prefix):-len(suffix)]
        if middle:
            return middle
    return stem


def matching_context_path(ideas_path: Path) -> Path | None:
    stem = ideas_path.stem
    if stem.endswith("_ideas"):
        candidate = ideas_path.with_name(f"{stem[:-6]}.md")
        if candidate.exists():
            return candidate
    return None


def discover_model_ideas_files() -> list[Path]:
    paths = sorted((PROJECT_ROOT / "docs" / "autoresearch_ideas").glob("auto_research_*_ideas.yaml"))
    discovered: list[Path] = []
    for path in paths:
        name = default_backlog_name(path)
        if name in CAMPAIGN_NAMES:
            continue
        discovered.append(path)
    return discovered


def build_plan(
    *,
    base_config_path: Path,
    ideas_files: list[Path],
    session_tag: str,
) -> list[BacklogPlan]:
    base_config = load_yaml(base_config_path)
    plans: list[BacklogPlan] = []
    config_output_dir = OUTPUT_ROOT / session_tag / "configs"

    for ideas_path in ideas_files:
        backlog_name = slugify_name(default_backlog_name(ideas_path))
        tag = f"{session_tag}-{backlog_name}"
        branch = f"autoresearch/{tag}"
        workspace_dir = f"outputs/idea_backlog_compare/{session_tag}/{backlog_name}"
        config_payload = json.loads(json.dumps(base_config))
        config_payload.setdefault("research", {})["workspace_dir"] = workspace_dir
        agent_cfg = config_payload.setdefault("agent", {})
        agent_cfg["ideas_file"] = str(ideas_path.relative_to(PROJECT_ROOT))
        context_path = matching_context_path(ideas_path)
        if context_path is not None:
            agent_cfg["context_file"] = str(context_path.relative_to(PROJECT_ROOT))
        config_path = config_output_dir / f"{backlog_name}.yaml"
        write_yaml(config_path, config_payload)

        plans.append(
            BacklogPlan(
                name=backlog_name,
                ideas_file=str(ideas_path.relative_to(PROJECT_ROOT)),
                config_path=str(config_path.relative_to(PROJECT_ROOT)),
                workspace_dir=workspace_dir,
                branch=branch,
                tag=tag,
            )
        )
    return plans


def stream_command(command: list[str], *, log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(f"$ {' '.join(shlex.quote(part) for part in command)}\n\n")
        log_file.flush()

        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )

        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            log_file.write(line)
        process.wait()
        return process.returncode


def read_results_rows(results_path: Path) -> list[dict]:
    if not results_path.exists():
        return []
    with results_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return list(reader)


def parse_positive_float(raw_value: str | None) -> float | None:
    if raw_value is None:
        return None
    text = str(raw_value).strip()
    if text in ("", "0", "0.0", "0.000000"):
        return None
    try:
        value = float(text)
    except ValueError:
        return None
    return value if value > 0.0 else None


def summarize_workspace(workspace_dir: Path) -> dict:
    results_path = workspace_dir / "results.tsv"
    baseline_path = workspace_dir / "baseline" / "metadata.json"
    rows = read_results_rows(results_path)
    baseline_metadata = json.loads(baseline_path.read_text(encoding="utf-8")) if baseline_path.exists() else {}

    baseline_rows = [row for row in rows if row.get("description", "").strip() == "baseline"]
    initial_baseline = None
    if baseline_rows:
        initial_baseline = parse_positive_float(baseline_rows[0].get("metric_value"))

    keep_rows = [row for row in rows if row.get("status") == "keep"]
    non_baseline_keep_rows = [row for row in keep_rows if row.get("description", "").strip() != "baseline"]
    best_keep_row = None
    best_keep_metric = None
    for row in non_baseline_keep_rows:
        metric = parse_positive_float(row.get("metric_value"))
        if metric is None:
            continue
        if best_keep_metric is None or metric < best_keep_metric:
            best_keep_metric = metric
            best_keep_row = row

    final_best_metric = parse_positive_float(baseline_metadata.get("metric_value"))
    if final_best_metric is None:
        valid_metrics = [
            metric
            for metric in (parse_positive_float(row.get("metric_value")) for row in rows)
            if metric is not None
        ]
        final_best_metric = min(valid_metrics) if valid_metrics else None

    improvement = None
    improvement_pct = None
    if initial_baseline is not None and final_best_metric is not None:
        improvement = initial_baseline - final_best_metric
        if initial_baseline > 0.0:
            improvement_pct = (improvement / initial_baseline) * 100.0

    return {
        "workspace_dir": str(workspace_dir.relative_to(PROJECT_ROOT)),
        "results_path": str(results_path.relative_to(PROJECT_ROOT)) if results_path.exists() else "",
        "initial_baseline": initial_baseline,
        "final_best_metric": final_best_metric,
        "improvement": improvement,
        "improvement_pct": improvement_pct,
        "keep_count": len(non_baseline_keep_rows),
        "discard_count": sum(1 for row in rows if row.get("status") == "discard"),
        "crash_count": sum(1 for row in rows if row.get("status") == "crash"),
        "best_keep_commit": (best_keep_row or {}).get("commit", ""),
        "best_keep_description": (best_keep_row or {}).get("description", ""),
        "promoted_commit": str(baseline_metadata.get("commit", "")),
        "promoted_description": str(baseline_metadata.get("description", "")),
        "total_rows": len(rows),
    }


def write_summary_tsv(summary_rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not summary_rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(summary_rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(summary_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare multiple autoresearch ideas backlogs from the same main snapshot"
    )
    parser.add_argument(
        "--base-config",
        type=str,
        default="configs/autoresearch/autoresearch_latent_stage1.yaml",
        help="Base autoresearch config to clone for each backlog run",
    )
    parser.add_argument(
        "--ideas-files",
        type=str,
        nargs="*",
        default=None,
        help="Explicit ideas files to compare; if omitted, auto-detect model-specific files",
    )
    parser.add_argument(
        "--session-tag",
        type=str,
        default=datetime.now().strftime("ideas-compare-%Y%m%d-%H%M%S"),
        help="Session tag used in branch names and output folders",
    )
    parser.add_argument(
        "--agent-args",
        type=str,
        default="",
        help="Extra arguments appended verbatim to each scripts/agent.py invocation",
    )
    parser.add_argument(
        "--no-rebaseline",
        action="store_true",
        help="Skip --rebaseline when launching each backlog run",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned branches/configs and exit without running anything",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    base_config_path = PROJECT_ROOT / args.base_config
    if not base_config_path.exists():
        raise RunnerError(f"base config not found: {args.base_config}")

    if args.ideas_files:
        ideas_files = []
        for raw_path in args.ideas_files:
            path = Path(raw_path)
            if not path.is_absolute():
                path = PROJECT_ROOT / path
            if not path.exists():
                raise RunnerError(f"ideas file not found: {raw_path}")
            ideas_files.append(path)
    else:
        ideas_files = discover_model_ideas_files()
        if not ideas_files:
            raise RunnerError(
                "no model-specific ideas files found; pass them explicitly with --ideas-files"
            )

    plans = build_plan(
        base_config_path=base_config_path,
        ideas_files=ideas_files,
        session_tag=args.session_tag,
    )

    if args.dry_run:
        for plan in plans:
            print(json.dumps(asdict(plan), indent=2))
        return 0

    ensure_clean_worktree()
    ensure_no_other_agent_running()
    base_sha = checkout_and_sync_main()

    extra_agent_args = shlex.split(args.agent_args)
    session_dir = OUTPUT_ROOT / args.session_tag
    session_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, object] = {
        "session_tag": args.session_tag,
        "started_at": datetime.now().isoformat(),
        "base_branch": "main",
        "base_sha": base_sha,
        "base_config": str(base_config_path.relative_to(PROJECT_ROOT)),
        "plans": [asdict(plan) for plan in plans],
        "runs": [],
    }
    summary_json = session_dir / "summary.json"
    summary_tsv = session_dir / "summary.tsv"

    for index, plan in enumerate(plans, start=1):
        print(f"\n=== Backlog {index}/{len(plans)}: {plan.name} -> {plan.branch} ===\n")
        create_branch_from_base(plan.branch, base_sha)

        command = [
            sys.executable,
            str(AGENT_SCRIPT),
            "--config",
            plan.config_path,
            "--tag",
            plan.tag,
            "--no-dashboard",
        ] + extra_agent_args
        if not args.no_rebaseline:
            command.append("--rebaseline")

        log_path = session_dir / f"{index:02d}-{plan.name}.log"
        started_at = datetime.now().isoformat()
        returncode = stream_command(command, log_path=log_path)
        if returncode != 0:
            raise RunnerError(
                f"backlog run {plan.name} exited with code {returncode}; see {log_path}"
            )

        push_branch(plan.branch)
        workspace_summary = summarize_workspace(PROJECT_ROOT / plan.workspace_dir)
        result = {
            "name": plan.name,
            "ideas_file": plan.ideas_file,
            "config_path": plan.config_path,
            "branch": plan.branch,
            "started_at": started_at,
            "finished_at": datetime.now().isoformat(),
            "log_path": str(log_path.relative_to(PROJECT_ROOT)),
            **workspace_summary,
        }
        summary["runs"].append(result)
        ordered_rows = sorted(
            summary["runs"],
            key=lambda row: (
                float("inf") if row.get("final_best_metric") is None else row["final_best_metric"],
                0.0 if row.get("improvement") is None else -row["improvement"],
                row["name"],
            ),
        )
        summary["leaderboard"] = ordered_rows
        summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        write_summary_tsv(ordered_rows, summary_tsv)

        _run_git("checkout", "main")

    summary["finished_at"] = datetime.now().isoformat()
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nBacklog comparison complete. Summary saved to {summary_json}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RunnerError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
