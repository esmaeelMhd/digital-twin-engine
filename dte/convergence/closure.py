"""Bounded phase-closure loop on top of the convergence phase registry."""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import yaml

from dte.convergence.workflow import PROJECT_ROOT, PhaseRunStatus, PhaseSpec, read_phase_status
from dte.utils.runtime import runtime_env_defaults


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


@dataclass(frozen=True)
class ProbeRunResult:
    succeeded: bool
    config_path: Path
    run_dir: Path
    log_dir: Path
    train_summary_path: Path
    eval_summary_path: Path
    error: str | None
    train_aggregate_metric: float | None
    eval_aggregate_metric: float | None
    rollout_rmse_max: float | None
    duration_seconds: float


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


def _noop_mutator(_repo_root: Path) -> None:
    """Apply no file edits; useful for scope-only rerun strategies."""


def _always(_status: PhaseRunStatus) -> bool:
    return True


def _resolve_jax_platform_env(platform: str) -> str:
    if platform == "gpu":
        return "cuda,cpu"
    return platform


def _failed_with(substr: str) -> Callable[[PhaseRunStatus], bool]:
    needle = substr.lower()

    def _predicate(status: PhaseRunStatus) -> bool:
        return status.status == "failed" and needle in (status.error or "").lower()

    return _predicate


def _gate_false(gate_name: str) -> Callable[[PhaseRunStatus], bool]:
    def _predicate(status: PhaseRunStatus) -> bool:
        return status.status == "not_accepted" and not bool(status.gates.get(gate_name, False))

    return _predicate


def _read_text_tail(path: Path, max_chars: int = 16000) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def _phase1_enrich_failed_status(status: PhaseRunStatus, workspace_dir: Path) -> PhaseRunStatus:
    if status.phase_id != "phase1_unit_foundation_v1" or status.status != "failed":
        return status

    error_text = (status.error or "").lower()
    if "out of memory" in error_text or "maximum number of solver steps" in error_text:
        return status

    logs_dir = workspace_dir / "logs"
    candidate_logs = (
        logs_dir / "train_unit_foundation.log",
        logs_dir / "evaluate_unit_foundation.log",
        logs_dir / "transfer_source_pretrain.log",
        logs_dir / "control_gate.log",
    )
    tail = "\n".join(_read_text_tail(path) for path in candidate_logs if path.exists()).lower()
    if not tail:
        return status

    inferred_error = status.error
    if "out of memory" in tail or "resource_exhausted" in tail:
        inferred_error = "resource exhausted: out of memory"
    elif "maximum number of solver steps" in tail:
        inferred_error = "maximum number of solver steps was reached"
    elif "pure_callback failed to find a local cpu device" in tail:
        inferred_error = "pure_callback failed to find a local cpu device"

    if inferred_error == status.error:
        return status

    return PhaseRunStatus(
        phase_id=status.phase_id,
        summary_path=status.summary_path,
        status=status.status,
        accepted=status.accepted,
        error=inferred_error,
        gates=status.gates,
    )


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
            strategy_id="phase1_rollout_only_rerun",
            phase_id="phase1_unit_foundation_v1",
            description="Rerun only the canonical rollout gate path for shared-checkpoint closure.",
            touched_files=(),
            run_overrides={
                "skip_generation": True,
                "skip_transfer": True,
                "skip_control": True,
            },
            applies=_gate_false("rollout_stability_on_held_out_variants"),
            mutator=_noop_mutator,
        ),
        ClosureStrategy(
            strategy_id="phase1_transfer_two_tank_only_rerun",
            phase_id="phase1_unit_foundation_v1",
            description="Rerun only the remaining hydraulic transfer target through the canonical benchmark.",
            touched_files=(),
            run_overrides={
                "skip_generation": True,
                "skip_training": True,
                "skip_evaluation": True,
                "skip_control": True,
                "transfer_targets": ["two_tank_high_throughput"],
            },
            applies=_gate_false("transfer_beats_scratch_on_targets"),
            mutator=_noop_mutator,
        ),
    )


_STRATEGIES: dict[str, tuple[ClosureStrategy, ...]] = {
    "phase1_unit_foundation_v1": _phase1_strategies(),
}


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _run_logged_command(
    *,
    command: list[str],
    log_path: Path,
    env_updates: dict[str, str],
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(runtime_env_defaults())
    env.update(env_updates)
    env["PYTHONUNBUFFERED"] = "1"
    process = subprocess.Popen(
        command,
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=False,
        env=env,
    )
    assert process.stdout is not None
    with log_path.open("wb") as handle:
        while True:
            chunk = process.stdout.read(8192)
            if not chunk:
                break
            sys.stderr.buffer.write(chunk)
            sys.stderr.buffer.flush()
            handle.write(chunk)
    return process.wait()


def _phase1_probe_root(workspace_dir: Path, strategy_id: str, attempt_index: int) -> Path:
    return workspace_dir / ".convergence_agent" / "probes" / f"{attempt_index:02d}_{strategy_id}"


def _setdefault_nested(mapping: dict[str, Any], keypath: str, value: Any) -> None:
    cursor: dict[str, Any] = mapping
    parts = keypath.split(".")
    for key in parts[:-1]:
        next_item = cursor.get(key)
        if not isinstance(next_item, dict):
            next_item = {}
            cursor[key] = next_item
        cursor = next_item
    cursor.setdefault(parts[-1], value)


def _build_phase1_probe_config_payload(config: dict[str, Any]) -> dict[str, Any]:
    probe = copy.deepcopy(config)
    training_cfg = probe.setdefault("training", {})
    training_cfg["n_epochs"] = min(int(training_cfg.get("n_epochs", 2)), 2)
    training_cfg["max_batches_per_epoch"] = min(int(training_cfg.get("max_batches_per_epoch", 8)), 8)

    checkpointing = probe.setdefault("checkpointing", {})
    checkpointing["val_every"] = 1
    checkpointing["save_every"] = 1
    checkpointing["max_val_batches"] = min(int(checkpointing.get("max_val_batches", 2)), 2)

    evaluation = probe.setdefault("evaluation", {})
    evaluation["per_system_batches"] = min(int(evaluation.get("per_system_batches", 2)), 2)
    evaluation["forecast_batches"] = min(int(evaluation.get("forecast_batches", 1)), 1)
    evaluation["rollout_batches"] = min(int(evaluation.get("rollout_batches", 1)), 1)
    evaluation["rollout_samples"] = min(int(evaluation.get("rollout_samples", 2)), 2)
    evaluation["uncertainty_batches"] = 0
    evaluation["uncertainty_samples"] = 0
    evaluation["sensitivity_batches"] = 0
    return probe


def _infer_phase1_probe_error(
    *,
    train_error: str,
    probe_config: dict[str, Any],
    jax_platform: str,
) -> str:
    lower = train_error.lower()
    if "out of memory" in lower or "resource_exhausted" in lower:
        return "resource exhausted: out of memory"
    if "maximum number of solver steps" in lower:
        return "maximum number of solver steps was reached"
    if "pure_callback failed to find a local cpu device" in lower:
        return "pure_callback failed to find a local cpu device"

    training_cfg = probe_config.get("training", {}) if isinstance(probe_config, dict) else {}
    batch_size = int(training_cfg.get("batch_size", 0) or 0)
    if (
        jax_platform == "gpu"
        and batch_size >= 32
        and "universal digital twin training" in lower
        and "universal training:" in lower
        and "traceback" not in lower
    ):
        return "resource exhausted: out of memory (inferred)"

    return "probe train failed"


def _extract_rollout_rmse_max(eval_summary: dict[str, Any]) -> float | None:
    rollout_metrics = eval_summary.get("rollout_metrics", {})
    if not isinstance(rollout_metrics, dict) or not rollout_metrics:
        return None
    values: list[float] = []
    for metrics in rollout_metrics.values():
        if not isinstance(metrics, dict):
            continue
        value = metrics.get("rmse")
        if isinstance(value, (float, int)):
            values.append(float(value))
    if not values:
        return None
    return max(values)


def _run_phase1_probe(
    *,
    workspace_dir: Path,
    attempt_index: int,
    strategy_id: str,
    jax_platform: str,
    seed: int = 42,
) -> ProbeRunResult:
    started_at = time.time()
    probe_root = _phase1_probe_root(workspace_dir, strategy_id, attempt_index)
    probe_root.mkdir(parents=True, exist_ok=True)
    run_dir = probe_root / "run"
    eval_dir = probe_root / "eval"
    logs_dir = probe_root / "logs"
    config_path = probe_root / "config.yaml"
    train_summary_path = run_dir / "summary.json"
    eval_summary_path = eval_dir / "summary.json"

    base_config = _load_yaml(PROJECT_ROOT / "configs" / "training_universal_phase1_regime.yaml")
    probe_config = _build_phase1_probe_config_payload(base_config)
    _write_yaml(config_path, probe_config)

    env_updates = {
        "JAX_PLATFORMS": _resolve_jax_platform_env(jax_platform),
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
    }

    train_command = [
        sys.executable,
        "scripts/train_universal.py",
        "--config",
        str(config_path),
        "--output_dir",
        str(run_dir),
        "--seed",
        str(seed),
    ]
    train_returncode = _run_logged_command(
        command=train_command,
        log_path=logs_dir / "train_probe.log",
        env_updates=env_updates,
    )
    if train_returncode != 0 or not train_summary_path.exists() or not (run_dir / "best_model.eqx").exists():
        train_error = _read_text_tail(logs_dir / "train_probe.log", max_chars=8000)
        error = _infer_phase1_probe_error(
            train_error=train_error,
            probe_config=probe_config,
            jax_platform=jax_platform,
        )
        return ProbeRunResult(
            succeeded=False,
            config_path=config_path,
            run_dir=run_dir,
            log_dir=logs_dir,
            train_summary_path=train_summary_path,
            eval_summary_path=eval_summary_path,
            error=error,
            train_aggregate_metric=None,
            eval_aggregate_metric=None,
            rollout_rmse_max=None,
            duration_seconds=time.time() - started_at,
        )

    train_summary = _read_json(train_summary_path)
    eval_command = [
        sys.executable,
        "scripts/evaluate_universal.py",
        "--model_path",
        str(run_dir / "best_model.eqx"),
        "--config",
        str(run_dir / "config.yaml"),
        "--output_dir",
        str(eval_dir),
        "--seed",
        str(seed),
    ]
    eval_returncode = _run_logged_command(
        command=eval_command,
        log_path=logs_dir / "evaluate_probe.log",
        env_updates=env_updates,
    )
    if eval_returncode != 0 or not eval_summary_path.exists():
        eval_error = _read_text_tail(logs_dir / "evaluate_probe.log", max_chars=8000)
        return ProbeRunResult(
            succeeded=False,
            config_path=config_path,
            run_dir=run_dir,
            log_dir=logs_dir,
            train_summary_path=train_summary_path,
            eval_summary_path=eval_summary_path,
            error=eval_error or "probe evaluation failed",
            train_aggregate_metric=train_summary.get("aggregate_metric_value"),
            eval_aggregate_metric=None,
            rollout_rmse_max=None,
            duration_seconds=time.time() - started_at,
        )

    eval_summary = _read_json(eval_summary_path)
    return ProbeRunResult(
        succeeded=True,
        config_path=config_path,
        run_dir=run_dir,
        log_dir=logs_dir,
        train_summary_path=train_summary_path,
        eval_summary_path=eval_summary_path,
        error=None,
        train_aggregate_metric=train_summary.get("aggregate_metric_value"),
        eval_aggregate_metric=eval_summary.get("aggregate_metric_value"),
        rollout_rmse_max=_extract_rollout_rmse_max(eval_summary),
        duration_seconds=time.time() - started_at,
    )


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
        env={
            **os.environ,
            **runtime_env_defaults(),
        },
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
    status = _phase1_enrich_failed_status(status, workspace_dir)
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


def _workspace_phase_id(workspace_dir: Path) -> str | None:
    summary_path = workspace_dir / "summary.json"
    if not summary_path.exists():
        return None
    try:
        payload = _read_json(summary_path)
    except Exception:
        return None
    acceptance = payload.get("acceptance")
    if isinstance(acceptance, dict):
        phase = acceptance.get("phase")
        if isinstance(phase, str) and phase:
            return phase
    return None


def find_prior_kept_strategy_id(spec: PhaseSpec, workspace_dir: Path) -> str | None:
    parent = workspace_dir.parent
    if not parent.exists():
        return None
    candidates = sorted(
        (item for item in parent.iterdir() if item.is_dir() and item != workspace_dir),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for candidate in candidates:
        if _workspace_phase_id(candidate) != spec.phase_id:
            continue
        state_path = _state_path(candidate)
        if not state_path.exists():
            continue
        try:
            state = load_state(candidate)
        except Exception:
            continue
        attempts = state.get("attempts", [])
        if not isinstance(attempts, list):
            continue
        for attempt in reversed(attempts):
            if not isinstance(attempt, dict):
                continue
            strategy_id = attempt.get("strategy_id")
            if not isinstance(strategy_id, str) or strategy_id in {"", "__baseline__"}:
                continue
            if bool(attempt.get("kept", False)):
                return strategy_id
    return None


def _strategy_by_id(spec: PhaseSpec, strategy_id: str) -> ClosureStrategy | None:
    for strategy in get_phase_strategies(spec.phase_id):
        if strategy.strategy_id == strategy_id:
            return strategy
    return None


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
    current_status = _phase1_enrich_failed_status(current_status, workspace_dir)
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
    bootstrap_strategy_id = (
        find_prior_kept_strategy_id(spec, workspace_dir)
        if current_status.status == "missing"
        else None
    )

    while len(attempts) < max_attempts:
        status_before = current_status
        attempt_index = len(attempts) + 1
        snapshots: list[FileSnapshot] = []
        strategy_id = "__baseline__"
        description = "run canonical phase without a patch"
        run_overrides: dict[str, Any] = {}
        probe: ProbeRunResult | None = None

        if status_before.status == "missing" and bootstrap_strategy_id is not None:
            strategy = _strategy_by_id(spec, bootstrap_strategy_id)
            if strategy is not None and strategy.strategy_id not in attempted_strategy_ids:
                strategy_id = strategy.strategy_id
                description = (
                    f"{strategy.description} "
                    f"[bootstrapped from prior kept strategy]"
                )
                run_overrides = dict(strategy.run_overrides)
                snapshots = apply_strategy(spec, strategy)
                bootstrap_strategy_id = None
            else:
                bootstrap_strategy_id = None
        elif status_before.status != "missing":
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
        if dry_run:
            command, returncode, status_after = run_phase_once(
                spec=spec,
                workspace_dir=workspace_dir,
                jax_platform=jax_platform,
                dry_run=True,
                base_run_kwargs=base_run_kwargs,
                run_overrides=run_overrides,
            )
        else:
            probe = _run_phase1_probe(
                workspace_dir=workspace_dir,
                attempt_index=attempt_index,
                strategy_id=strategy_id,
                jax_platform=jax_platform,
            )
            if probe.succeeded:
                command, returncode, status_after = run_phase_once(
                    spec=spec,
                    workspace_dir=workspace_dir,
                    jax_platform=jax_platform,
                    dry_run=False,
                    base_run_kwargs=base_run_kwargs,
                    run_overrides=run_overrides,
                )
            else:
                command = [
                    sys.executable,
                    "scripts/train_universal.py",
                    "--config",
                    str(probe.config_path),
                    "--output_dir",
                    str(probe.run_dir),
                    "--seed",
                    "42",
                ]
                returncode = 1
                status_after = PhaseRunStatus(
                    phase_id=spec.phase_id,
                    summary_path=spec.summary_path_for_workspace(workspace_dir),
                    status="failed",
                    accepted=False,
                    error=probe.error,
                    gates=status_before.gates,
                )

        improved = status_score(status_after) > status_score(status_before)
        kept = strategy_id == "__baseline__" or improved
        if not kept and snapshots:
            restore_snapshots(snapshots)
            status_restored = read_phase_status(spec.phase_id, workspace_dir=workspace_dir)
            status_restored = _phase1_enrich_failed_status(status_restored, workspace_dir)
        else:
            status_restored = status_after
        current_status = status_restored

        attempt_record = {
            "attempt_index": attempt_index,
            "strategy_id": strategy_id,
            "description": description,
            "started_at": started_at,
            "duration_seconds": time.time() - started_at,
            "command": command,
            "returncode": returncode,
            "probe": None
            if probe is None
            else {
                "succeeded": probe.succeeded,
                "config_path": str(probe.config_path),
                "run_dir": str(probe.run_dir),
                "log_dir": str(probe.log_dir),
                "train_summary_path": str(probe.train_summary_path),
                "eval_summary_path": str(probe.eval_summary_path),
                "error": probe.error,
                "train_aggregate_metric": probe.train_aggregate_metric,
                "eval_aggregate_metric": probe.eval_aggregate_metric,
                "rollout_rmse_max": probe.rollout_rmse_max,
                "duration_seconds": probe.duration_seconds,
            },
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

        if strategy_id not in {None, "__baseline__"} and (kept or not improved):
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

    final_status = current_status
    return {
        "phase_id": spec.phase_id,
        "workspace_dir": str(workspace_dir),
        "status": final_status.status,
        "accepted": final_status.accepted,
        "error": final_status.error,
        "gates": final_status.gates,
        "attempts": attempts,
    }
