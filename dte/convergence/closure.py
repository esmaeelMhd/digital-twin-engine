"""Bounded phase-closure loop on top of the convergence phase registry."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import yaml

from dte.convergence.workflow import PROJECT_ROOT, PhaseRunStatus, PhaseSpec, read_phase_status


@dataclass(frozen=True)
class FileSnapshot:
    path: Path
    existed: bool
    content: str | None


@dataclass(frozen=True)
class ClosureStrategy:
    strategy_id: str
    phase_id: str
    description: str
    touched_files: tuple[str, ...]
    run_overrides: dict[str, Any]
    applies: Callable[[PhaseRunStatus], bool]
    mutator: Callable[[Path], None]


def _relative_path(path: Path) -> str:
    return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))


def _path_matches_target(path: str, target: str) -> bool:
    if target.endswith("/"):
        return path == target[:-1] or path.startswith(target)
    return path == target


def _is_allowed_path(spec: PhaseSpec, path: Path) -> bool:
    rel = _relative_path(path)
    editable = any(_path_matches_target(rel, target) for target in spec.editable_targets)
    forbidden = any(_path_matches_target(rel, target) for target in spec.forbidden_paths)
    return editable and not forbidden


def _snapshot_files(paths: list[Path]) -> list[FileSnapshot]:
    snapshots: list[FileSnapshot] = []
    for path in paths:
        if path.exists():
            snapshots.append(
                FileSnapshot(
                    path=path,
                    existed=True,
                    content=path.read_text(encoding="utf-8"),
                )
            )
        else:
            snapshots.append(FileSnapshot(path=path, existed=False, content=None))
    return snapshots


def restore_snapshots(snapshots: list[FileSnapshot]) -> None:
    for snapshot in snapshots:
        if snapshot.existed:
            snapshot.path.parent.mkdir(parents=True, exist_ok=True)
            snapshot.path.write_text(snapshot.content or "", encoding="utf-8")
        elif snapshot.path.exists():
            snapshot.path.unlink()


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def _set_nested(mapping: dict[str, Any], keypath: str, value: Any) -> None:
    cursor: dict[str, Any] = mapping
    parts = keypath.split(".")
    for key in parts[:-1]:
        next_item = cursor.get(key)
        if not isinstance(next_item, dict):
            next_item = {}
            cursor[key] = next_item
        cursor = next_item
    cursor[parts[-1]] = value


def _yaml_update_mutator(relpath: str, updates: dict[str, Any]) -> Callable[[Path], None]:
    def _mutate(repo_root: Path) -> None:
        path = repo_root / relpath
        payload = _load_yaml(path)
        updated = copy.deepcopy(payload)
        for keypath, value in updates.items():
            _set_nested(updated, keypath, value)
        _write_yaml(path, updated)

    return _mutate


def _always(_status: PhaseRunStatus) -> bool:
    return True


def _failed_with(substr: str) -> Callable[[PhaseRunStatus], bool]:
    needle = substr.lower()

    def _predicate(status: PhaseRunStatus) -> bool:
        return status.status == "failed" and needle in (status.error or "").lower()

    return _predicate


def _gate_false(gate_name: str) -> Callable[[PhaseRunStatus], bool]:
    def _predicate(status: PhaseRunStatus) -> bool:
        return status.status == "not_accepted" and not bool(status.gates.get(gate_name, False))

    return _predicate


def _phase1_strategies() -> tuple[ClosureStrategy, ...]:
    return (
        ClosureStrategy(
            strategy_id="phase1_batch_size_16",
            phase_id="phase1_unit_foundation_v1",
            description="Reduce universal Phase 1 batch size to relieve GPU memory pressure.",
            touched_files=("configs/training_universal_phase1_regime.yaml",),
            run_overrides={"skip_generation": True},
            applies=_failed_with("out of memory"),
            mutator=_yaml_update_mutator(
                "configs/training_universal_phase1_regime.yaml",
                {"training.batch_size": 16},
            ),
        ),
        ClosureStrategy(
            strategy_id="phase1_solver_max_steps_16384",
            phase_id="phase1_unit_foundation_v1",
            description="Increase latent solver step budget for stiff universal dynamics.",
            touched_files=("configs/training_universal_phase1_regime.yaml",),
            run_overrides={"skip_generation": True},
            applies=_failed_with("maximum number of solver steps"),
            mutator=_yaml_update_mutator(
                "configs/training_universal_phase1_regime.yaml",
                {"model.latent_solver.max_steps": 16384},
            ),
        ),
        ClosureStrategy(
            strategy_id="phase1_solver_relaxed_tolerances",
            phase_id="phase1_unit_foundation_v1",
            description="Relax stiff latent solver tolerances for rollout gate closure.",
            touched_files=("configs/training_universal_phase1_regime.yaml",),
            run_overrides={"skip_generation": True},
            applies=_gate_false("rollout_stability_on_held_out_variants"),
            mutator=_yaml_update_mutator(
                "configs/training_universal_phase1_regime.yaml",
                {
                    "model.latent_solver.rtol": 0.002,
                    "model.latent_solver.atol": 0.0002,
                },
            ),
        ),
        ClosureStrategy(
            strategy_id="phase1_solver_implicit_euler",
            phase_id="phase1_unit_foundation_v1",
            description="Switch the canonical latent solver to implicit Euler for stiff families.",
            touched_files=("configs/training_universal_phase1_regime.yaml",),
            run_overrides={"skip_generation": True},
            applies=_gate_false("rollout_stability_on_held_out_variants"),
            mutator=_yaml_update_mutator(
                "configs/training_universal_phase1_regime.yaml",
                {
                    "model.latent_solver.method": "implicit_euler",
                    "model.latent_solver.max_steps": 16384,
                    "model.latent_solver.rtol": 0.002,
                    "model.latent_solver.atol": 0.0002,
                },
            ),
        ),
    )


_STRATEGIES: dict[str, tuple[ClosureStrategy, ...]] = {
    "phase1_unit_foundation_v1": _phase1_strategies(),
}


def status_score(status: PhaseRunStatus) -> tuple[int, int, int]:
    rank = {
        "missing": 0,
        "failed": 1,
        "dry_run": 2,
        "running": 3,
        "not_accepted": 4,
        "accepted": 5,
    }.get(status.status, -1)
    return (
        rank,
        sum(1 for value in status.gates.values() if value),
        1 if status.accepted else 0,
    )


def get_phase_strategies(phase_id: str) -> tuple[ClosureStrategy, ...]:
    return _STRATEGIES.get(phase_id, ())


def choose_strategy(
    *,
    spec: PhaseSpec,
    status: PhaseRunStatus,
    attempted_strategy_ids: set[str],
) -> ClosureStrategy | None:
    for strategy in get_phase_strategies(spec.phase_id):
        if strategy.strategy_id in attempted_strategy_ids:
            continue
        if strategy.applies(status):
            return strategy
    return None


def apply_strategy(spec: PhaseSpec, strategy: ClosureStrategy) -> list[FileSnapshot]:
    touched_paths = [PROJECT_ROOT / relpath for relpath in strategy.touched_files]
    for path in touched_paths:
        if not _is_allowed_path(spec, path):
            raise ValueError(f"Strategy '{strategy.strategy_id}' touches disallowed path '{_relative_path(path)}'.")
    snapshots = _snapshot_files(touched_paths)
    strategy.mutator(PROJECT_ROOT)
    return snapshots


def run_phase_once(
    *,
    spec: PhaseSpec,
    workspace_dir: Path,
    jax_platform: str,
    dry_run: bool,
    base_run_kwargs: dict[str, Any] | None = None,
    run_overrides: dict[str, Any] | None = None,
) -> tuple[list[str], int, PhaseRunStatus]:
    merged = dict(base_run_kwargs or {})
    merged.update(run_overrides or {})
    command = spec.build_command(
        workspace_dir=workspace_dir,
        jax_platform=jax_platform,
        skip_generation=bool(merged.get("skip_generation", False)),
        skip_training=bool(merged.get("skip_training", False)),
        skip_evaluation=bool(merged.get("skip_evaluation", False)),
        skip_transfer=bool(merged.get("skip_transfer", False)),
        skip_control=bool(merged.get("skip_control", False)),
        transfer_targets=merged.get("transfer_targets"),
        dry_run=dry_run,
    )
    process = subprocess.Popen(
        command,
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=False,
    )
    assert process.stdout is not None
    while True:
        chunk = process.stdout.read(8192)
        if not chunk:
            break
        sys.stderr.buffer.write(chunk)
        sys.stderr.buffer.flush()
    returncode = process.wait()
    status = read_phase_status(spec.phase_id, workspace_dir=workspace_dir)
    return command, returncode, status


def _state_path(workspace_dir: Path) -> Path:
    return workspace_dir / ".convergence_agent" / "state.json"


def load_state(workspace_dir: Path) -> dict[str, Any]:
    path = _state_path(workspace_dir)
    if not path.exists():
        return {"attempts": []}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_state(workspace_dir: Path, state: dict[str, Any]) -> None:
    path = _state_path(workspace_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(state, handle, indent=2, sort_keys=True)


def auto_close_phase(
    *,
    spec: PhaseSpec,
    workspace_dir: Path,
    jax_platform: str,
    dry_run: bool,
    max_attempts: int,
    base_run_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    current_status = read_phase_status(spec.phase_id, workspace_dir=workspace_dir)
    state = load_state(workspace_dir)
    attempts: list[dict[str, Any]] = list(state.get("attempts", []))

    if current_status.status == "running":
        return {
            "phase_id": spec.phase_id,
            "workspace_dir": str(workspace_dir),
            "status": current_status.status,
            "accepted": current_status.accepted,
            "error": current_status.error,
            "gates": current_status.gates,
            "attempts": attempts,
            "message": "workspace already marked running",
        }

    if current_status.accepted:
        return {
            "phase_id": spec.phase_id,
            "workspace_dir": str(workspace_dir),
            "status": current_status.status,
            "accepted": True,
            "error": current_status.error,
            "gates": current_status.gates,
            "attempts": attempts,
        }

    attempted_strategy_ids = {
        attempt["strategy_id"]
        for attempt in attempts
        if attempt.get("strategy_id") not in {None, "__baseline__"} and attempt.get("kept", False)
    } | {
        attempt["strategy_id"]
        for attempt in attempts
        if attempt.get("strategy_id") not in {None, "__baseline__"} and not attempt.get("improved", False)
    }

    while len(attempts) < max_attempts:
        status_before = read_phase_status(spec.phase_id, workspace_dir=workspace_dir)
        attempt_index = len(attempts) + 1
        snapshots: list[FileSnapshot] = []
        strategy_id = "__baseline__"
        description = "run canonical phase without a patch"
        run_overrides: dict[str, Any] = {}

        if status_before.status != "missing":
            strategy = choose_strategy(
                spec=spec,
                status=status_before,
                attempted_strategy_ids=attempted_strategy_ids,
            )
            if strategy is None:
                break
            strategy_id = strategy.strategy_id
            description = strategy.description
            run_overrides = dict(strategy.run_overrides)
            snapshots = apply_strategy(spec, strategy)

        started_at = time.time()
        command, returncode, status_after = run_phase_once(
            spec=spec,
            workspace_dir=workspace_dir,
            jax_platform=jax_platform,
            dry_run=dry_run,
            base_run_kwargs=base_run_kwargs,
            run_overrides=run_overrides,
        )

        improved = status_score(status_after) > status_score(status_before)
        kept = strategy_id == "__baseline__" or improved
        if not kept and snapshots:
            restore_snapshots(snapshots)
            status_restored = read_phase_status(spec.phase_id, workspace_dir=workspace_dir)
        else:
            status_restored = status_after

        attempt_record = {
            "attempt_index": attempt_index,
            "strategy_id": strategy_id,
            "description": description,
            "started_at": started_at,
            "duration_seconds": time.time() - started_at,
            "command": command,
            "returncode": returncode,
            "status_before": status_before.status,
            "accepted_before": status_before.accepted,
            "gates_before": status_before.gates,
            "error_before": status_before.error,
            "status_after": status_after.status,
            "accepted_after": status_after.accepted,
            "gates_after": status_after.gates,
            "error_after": status_after.error,
            "improved": improved,
            "kept": kept,
            "restored": bool(snapshots) and not kept,
        }
        attempts.append(attempt_record)
        state["attempts"] = attempts
        save_state(workspace_dir, state)

        if kept and strategy_id not in {None, "__baseline__"}:
            attempted_strategy_ids.add(strategy_id)

        if status_restored.accepted:
            return {
                "phase_id": spec.phase_id,
                "workspace_dir": str(workspace_dir),
                "status": status_restored.status,
                "accepted": True,
                "error": status_restored.error,
                "gates": status_restored.gates,
                "attempts": attempts,
            }

        if strategy_id == "__baseline__" and returncode != 0 and not dry_run:
            continue

    final_status = read_phase_status(spec.phase_id, workspace_dir=workspace_dir)
    return {
        "phase_id": spec.phase_id,
        "workspace_dir": str(workspace_dir),
        "status": final_status.status,
        "accepted": final_status.accepted,
        "error": final_status.error,
        "gates": final_status.gates,
        "attempts": attempts,
    }
