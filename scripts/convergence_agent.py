"""Canonical convergence runner for phase-closure workflows."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from dte.convergence import auto_close_phase, get_phase_spec, list_phase_ids, read_phase_status


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run or inspect a canonical convergence phase.",
    )
    parser.add_argument(
        "--phase",
        default="phase1_unit_foundation_v1",
        choices=list_phase_ids(),
        help="Convergence phase identifier.",
    )
    parser.add_argument(
        "--workspace_dir",
        default=None,
        help="Workspace directory for this phase run.",
    )
    parser.add_argument(
        "--jax_platform",
        default="gpu",
        choices=("cpu", "gpu"),
        help="Execution backend for the canonical phase runner.",
    )
    parser.add_argument(
        "--skip_generation",
        action="store_true",
        help="Reuse existing generated data when supported by the phase runner.",
    )
    parser.add_argument(
        "--transfer_targets",
        nargs="*",
        default=None,
        help="Optional target subset for canonical transfer evaluation.",
    )
    parser.add_argument(
        "--status_only",
        action="store_true",
        help="Print the current machine-readable phase status and exit.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Pass through to the canonical runner when supported.",
    )
    parser.add_argument(
        "--auto_close",
        action="store_true",
        help="Iteratively run bounded phase-closure attempts until accepted or attempts are exhausted.",
    )
    parser.add_argument(
        "--max_attempts",
        type=int,
        default=4,
        help="Maximum number of bounded closure attempts in --auto_close mode.",
    )
    return parser.parse_args()


def _status_payload(phase_id: str, workspace_dir: Path) -> dict[str, object]:
    spec = get_phase_spec(phase_id)
    status = read_phase_status(phase_id, workspace_dir=workspace_dir)
    return {
        "phase_id": spec.phase_id,
        "title": spec.title,
        "workspace_dir": str(workspace_dir),
        "summary_path": str(status.summary_path),
        "status": status.status,
        "accepted": status.accepted,
        "error": status.error,
        "gates": status.gates,
        "editable_targets": list(spec.editable_targets),
        "forbidden_paths": list(spec.forbidden_paths),
    }


def _print_status(phase_id: str, workspace_dir: Path) -> None:
    print(json.dumps(_status_payload(phase_id, workspace_dir), indent=2, sort_keys=True))


def main() -> int:
    args = _parse_args()
    spec = get_phase_spec(args.phase)
    workspace_dir = spec.resolve_workspace(args.workspace_dir)
    base_run_kwargs = {
        "skip_generation": args.skip_generation,
        "transfer_targets": args.transfer_targets,
    }

    if args.status_only:
        _print_status(spec.phase_id, workspace_dir)
        return 0

    if args.auto_close:
        payload = auto_close_phase(
            spec=spec,
            workspace_dir=workspace_dir,
            jax_platform=args.jax_platform,
            dry_run=args.dry_run,
            max_attempts=args.max_attempts,
            base_run_kwargs=base_run_kwargs,
        )
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0 if payload["accepted"] else 1

    current_status = read_phase_status(spec.phase_id, workspace_dir)
    if current_status.accepted:
        _print_status(spec.phase_id, workspace_dir)
        return 0

    command = spec.build_command(
        workspace_dir=workspace_dir,
        jax_platform=args.jax_platform,
        skip_generation=bool(base_run_kwargs["skip_generation"]),
        transfer_targets=args.transfer_targets,
        dry_run=args.dry_run,
    )
    result = subprocess.run(command, cwd=PROJECT_ROOT, check=False)

    _print_status(spec.phase_id, workspace_dir)

    if args.dry_run:
        return 0 if result.returncode == 0 else result.returncode
    if result.returncode != 0:
        return result.returncode
    final_status = read_phase_status(spec.phase_id, workspace_dir)
    return 0 if final_status.accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())
