"""Run focused autoresearch campaigns sequentially and merge winners into main."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
AGENT_SCRIPT = PROJECT_ROOT / "scripts" / "agent.py"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "campaign_runner"

RECOMMENDED_ORDER = [
    "latent",
    "trainer",
    "losses",
    "encoder",
    "decoder",
    "digital_twin",
]

DEFAULT_STAGE1_RUNS = {
    "latent": 16,
    "trainer": 8,
    "losses": 6,
    "encoder": 6,
    "decoder": 6,
    "digital_twin": 4,
}

DEFAULT_STAGE2_RUNS = {
    "latent": 4,
    "trainer": 4,
    "losses": 4,
    "encoder": 3,
    "decoder": 3,
    "digital_twin": 3,
}


class RunnerError(RuntimeError):
    """Raised when the overnight campaign runner cannot proceed safely."""


@dataclass
class CampaignPlan:
    name: str
    stage: str
    config_path: str
    max_runs: int
    tag: str
    branch: str


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
    status = _git_output("status", "--porcelain")
    if status:
        raise RunnerError(
            "working tree is not clean; commit or stash tracked/untracked changes before running "
            "overnight campaigns"
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


def checkout_main_and_sync() -> None:
    current = _git_output("branch", "--show-current")
    if current != "main":
        _run_git("checkout", "main")
    _run_git("pull", "--ff-only", "origin", "main")


def branch_exists(branch: str) -> bool:
    return bool(_git_output("branch", "--list", branch))


def branch_ahead_of_main(branch: str) -> bool:
    count = _git_output("rev-list", "--count", f"main..{branch}")
    try:
        return int(count or "0") > 0
    except ValueError:
        return False


def merge_campaign_branch(branch: str) -> bool:
    checkout_main_and_sync()
    if not branch_exists(branch):
        raise RunnerError(f"campaign branch does not exist: {branch}")
    if not branch_ahead_of_main(branch):
        return False
    try:
        _run_git("merge", "--no-edit", branch)
    except RunnerError as exc:
        _run_git("merge", "--abort", check=False)
        raise RunnerError(f"merge failed for {branch}: {exc}") from exc
    _run_git("push", "origin", "main")
    return True


def build_campaign_plan(
    *,
    stage: str,
    session_tag: str,
    campaigns: Iterable[str],
    max_runs_override: int | None = None,
) -> list[CampaignPlan]:
    runs_by_name = DEFAULT_STAGE1_RUNS if stage == "stage1" else DEFAULT_STAGE2_RUNS
    plans: list[CampaignPlan] = []
    for name in campaigns:
        config_path = f"configs/autoresearch_{name}_{stage}.yaml"
        branch_tag = f"{session_tag}-{name}-{stage}"
        plans.append(
            CampaignPlan(
                name=name,
                stage=stage,
                config_path=config_path,
                max_runs=max_runs_override if max_runs_override is not None else runs_by_name[name],
                tag=branch_tag,
                branch=f"autoresearch/{branch_tag}",
            )
        )
    return plans


def stream_command(
    command: list[str],
    *,
    log_path: Path,
    env: dict[str, str] | None = None,
) -> int:
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
            env=env,
        )

        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            log_file.write(line)
        process.wait()
        return process.returncode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run recommended autoresearch campaigns sequentially overnight"
    )
    parser.add_argument(
        "--stage",
        choices=("stage1", "stage2"),
        default="stage1",
        help="Campaign stage to run overnight (default: stage1)",
    )
    parser.add_argument(
        "--session-tag",
        type=str,
        default=datetime.now().strftime("overnight-%Y%m%d-%H%M%S"),
        help="Prefix used in campaign branch tags and output folder names",
    )
    parser.add_argument(
        "--campaigns",
        type=str,
        default=",".join(RECOMMENDED_ORDER),
        help="Comma-separated campaign names in run order",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=None,
        help="Override the recommended max-runs for every campaign",
    )
    parser.add_argument(
        "--agent-args",
        type=str,
        default="",
        help="Extra arguments appended verbatim to each scripts/agent.py invocation",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    campaign_names = [name.strip() for name in args.campaigns.split(",") if name.strip()]
    unknown = [name for name in campaign_names if name not in RECOMMENDED_ORDER]
    if unknown:
        raise RunnerError(f"unknown campaign names: {', '.join(unknown)}")

    ensure_clean_worktree()
    ensure_no_other_agent_running()
    checkout_main_and_sync()

    plans = build_campaign_plan(
        stage=args.stage,
        session_tag=args.session_tag,
        campaigns=campaign_names,
        max_runs_override=args.max_runs,
    )

    session_dir = OUTPUT_ROOT / args.session_tag
    session_dir.mkdir(parents=True, exist_ok=True)
    summary_path = session_dir / "summary.json"
    extra_agent_args = shlex.split(args.agent_args)

    summary: dict[str, object] = {
        "session_tag": args.session_tag,
        "stage": args.stage,
        "started_at": datetime.now().isoformat(),
        "plans": [asdict(plan) for plan in plans],
        "campaigns": [],
    }

    for index, plan in enumerate(plans, start=1):
        if branch_exists(plan.branch):
            raise RunnerError(
                f"branch already exists for planned campaign {plan.name}: {plan.branch}; "
                "use a new --session-tag"
            )

        checkout_main_and_sync()

        log_path = session_dir / f"{index:02d}-{plan.name}-{plan.stage}.log"
        command = [
            sys.executable,
            str(AGENT_SCRIPT),
            "--config",
            plan.config_path,
            "--tag",
            plan.tag,
            "--max-runs",
            str(plan.max_runs),
            "--no-dashboard",
        ] + extra_agent_args

        print(
            f"\n=== Campaign {index}/{len(plans)}: {plan.name} ({plan.stage}) "
            f"-> {plan.branch} ===\n"
        )

        started_at = datetime.now().isoformat()
        returncode = stream_command(
            command,
            log_path=log_path,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        if returncode != 0:
            raise RunnerError(
                f"campaign {plan.name} exited with code {returncode}; see {log_path}"
            )

        merged = merge_campaign_branch(plan.branch)
        result = {
            "campaign": plan.name,
            "stage": plan.stage,
            "branch": plan.branch,
            "config_path": plan.config_path,
            "max_runs": plan.max_runs,
            "log_path": str(log_path.relative_to(PROJECT_ROOT)),
            "started_at": started_at,
            "finished_at": datetime.now().isoformat(),
            "merged_into_main": merged,
            "main_head": _git_output("rev-parse", "HEAD"),
        }
        summary["campaigns"].append(result)
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    summary["finished_at"] = datetime.now().isoformat()
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nOvernight campaigns complete. Summary saved to {summary_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RunnerError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
