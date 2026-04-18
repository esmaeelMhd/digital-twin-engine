"""Tests for canonical convergence-phase registry and CLI behavior."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from dte.convergence.closure import (
    _phase1_enrich_failed_status,
    apply_strategy,
    auto_close_phase,
    choose_strategy,
    restore_snapshots,
    status_score,
)
from dte.convergence import get_phase_spec, list_phase_ids, read_phase_status


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_list_phase_ids_contains_phase1() -> None:
    assert list_phase_ids() == ["phase1_unit_foundation_v1"]


def test_phase1_status_is_missing_when_summary_does_not_exist(tmp_path: Path) -> None:
    status = read_phase_status("phase1_unit_foundation_v1", workspace_dir=tmp_path / "missing")
    assert status.status == "missing"
    assert status.accepted is False
    assert status.error is None
    assert status.gates == {}


def test_phase1_status_maps_ok_and_accepted(tmp_path: Path) -> None:
    workspace_dir = tmp_path / "accepted"
    workspace_dir.mkdir(parents=True)
    (workspace_dir / "summary.json").write_text(
        json.dumps(
            {
                "status": "ok",
                "acceptance": {
                    "accepted": True,
                    "gates": {
                        "shared_checkpoint_trains_reproducibly": True,
                        "transfer_beats_scratch_on_targets": True,
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    status = read_phase_status("phase1_unit_foundation_v1", workspace_dir=workspace_dir)
    assert status.status == "accepted"
    assert status.accepted is True
    assert status.gates == {
        "shared_checkpoint_trains_reproducibly": True,
        "transfer_beats_scratch_on_targets": True,
    }


def test_phase1_status_maps_failed_with_error(tmp_path: Path) -> None:
    workspace_dir = tmp_path / "failed"
    workspace_dir.mkdir(parents=True)
    (workspace_dir / "summary.json").write_text(
        json.dumps(
            {
                "status": "failed",
                "error": "step failed",
                "acceptance": {
                    "accepted": False,
                    "gates": {
                        "shared_checkpoint_trains_reproducibly": False,
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    status = read_phase_status("phase1_unit_foundation_v1", workspace_dir=workspace_dir)
    assert status.status == "failed"
    assert status.accepted is False
    assert status.error == "step failed"
    assert status.gates == {"shared_checkpoint_trains_reproducibly": False}


def test_phase1_build_command_supports_transfer_targets_and_skip_generation(tmp_path: Path) -> None:
    spec = get_phase_spec("phase1_unit_foundation_v1")
    command = spec.build_command(
        workspace_dir=tmp_path / "phase1",
        jax_platform="gpu",
        skip_generation=True,
        skip_training=True,
        skip_evaluation=True,
        skip_transfer=True,
        skip_control=True,
        transfer_targets=["cstr_fast_kinetics", "two_tank_high_throughput"],
        dry_run=True,
    )

    assert command == [
        "python",
        "scripts/run_unit_foundation_baseline.py",
        "--workspace_dir",
        str(tmp_path / "phase1"),
        "--jax_platform",
        "gpu",
        "--skip_generation",
        "--skip_training",
        "--skip_evaluation",
        "--skip_transfer",
        "--skip_control",
        "--transfer_targets",
        "cstr_fast_kinetics",
        "two_tank_high_throughput",
        "--dry_run",
    ]


def test_convergence_agent_status_only_prints_machine_readable_status(tmp_path: Path) -> None:
    workspace_dir = tmp_path / "status_only"
    workspace_dir.mkdir(parents=True)
    (workspace_dir / "summary.json").write_text(
        json.dumps(
            {
                "status": "ok",
                "acceptance": {
                    "accepted": False,
                    "gates": {
                        "control_gate_completed": True,
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/convergence_agent.py",
            "--phase",
            "phase1_unit_foundation_v1",
            "--workspace_dir",
            str(workspace_dir),
            "--status_only",
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout)
    assert payload["phase_id"] == "phase1_unit_foundation_v1"
    assert payload["workspace_dir"] == str(workspace_dir)
    assert payload["status"] == "not_accepted"
    assert payload["accepted"] is False
    assert payload["gates"] == {"control_gate_completed": True}


def test_convergence_agent_short_circuits_when_phase_is_already_accepted(tmp_path: Path) -> None:
    workspace_dir = tmp_path / "accepted"
    workspace_dir.mkdir(parents=True)
    summary_path = workspace_dir / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "status": "ok",
                "acceptance": {
                    "accepted": True,
                    "gates": {
                        "control_gate_completed": True,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    before = summary_path.read_text(encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/convergence_agent.py",
            "--phase",
            "phase1_unit_foundation_v1",
            "--workspace_dir",
            str(workspace_dir),
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout)
    assert payload["status"] == "accepted"
    assert payload["accepted"] is True
    assert summary_path.read_text(encoding="utf-8") == before


def test_status_score_orders_phase_progress(tmp_path: Path) -> None:
    missing = read_phase_status("phase1_unit_foundation_v1", workspace_dir=tmp_path / "missing")

    failed_dir = tmp_path / "failed"
    failed_dir.mkdir()
    (failed_dir / "summary.json").write_text(
        json.dumps({"status": "failed", "acceptance": {"accepted": False, "gates": {}}}),
        encoding="utf-8",
    )
    failed = read_phase_status("phase1_unit_foundation_v1", workspace_dir=failed_dir)

    accepted_dir = tmp_path / "accepted_score"
    accepted_dir.mkdir()
    (accepted_dir / "summary.json").write_text(
        json.dumps({"status": "ok", "acceptance": {"accepted": True, "gates": {"a": True}}}),
        encoding="utf-8",
    )
    accepted = read_phase_status("phase1_unit_foundation_v1", workspace_dir=accepted_dir)

    assert status_score(failed) > status_score(missing)
    assert status_score(accepted) > status_score(failed)


def test_choose_strategy_prefers_max_steps_fix_for_matching_failure(tmp_path: Path) -> None:
    workspace_dir = tmp_path / "failed_max_steps"
    workspace_dir.mkdir()
    (workspace_dir / "summary.json").write_text(
        json.dumps(
            {
                "status": "failed",
                "error": "The maximum number of solver steps was reached.",
                "acceptance": {"accepted": False, "gates": {}},
            }
        ),
        encoding="utf-8",
    )
    spec = get_phase_spec("phase1_unit_foundation_v1")
    status = read_phase_status(spec.phase_id, workspace_dir=workspace_dir)
    strategy = choose_strategy(spec=spec, status=status, attempted_strategy_ids=set())

    assert strategy is not None
    assert strategy.strategy_id == "phase1_solver_max_steps_16384"


def test_phase1_enrich_failed_status_reads_workspace_logs_for_generic_wrapper_error(tmp_path: Path) -> None:
    workspace_dir = tmp_path / "phase1_failed"
    logs_dir = workspace_dir / "logs"
    logs_dir.mkdir(parents=True)
    (workspace_dir / "summary.json").write_text(
        json.dumps(
            {
                "status": "failed",
                "error": "step 'train_unit_foundation' failed with exit code 1",
                "acceptance": {"accepted": False, "gates": {}},
            }
        ),
        encoding="utf-8",
    )
    (logs_dir / "train_unit_foundation.log").write_text(
        "jax.errors.JaxRuntimeError: RESOURCE_EXHAUSTED: Out of memory while trying to allocate 5.51GiB.\n",
        encoding="utf-8",
    )

    status = read_phase_status("phase1_unit_foundation_v1", workspace_dir=workspace_dir)
    enriched = _phase1_enrich_failed_status(status, workspace_dir)

    assert enriched.status == "failed"
    assert enriched.error == "resource exhausted: out of memory"


def test_choose_strategy_uses_enriched_workspace_error_for_oom_failures(tmp_path: Path) -> None:
    workspace_dir = tmp_path / "phase1_failed_oom"
    logs_dir = workspace_dir / "logs"
    logs_dir.mkdir(parents=True)
    (workspace_dir / "summary.json").write_text(
        json.dumps(
            {
                "status": "failed",
                "error": "step 'train_unit_foundation' failed with exit code 1",
                "acceptance": {"accepted": False, "gates": {}},
            }
        ),
        encoding="utf-8",
    )
    (logs_dir / "train_unit_foundation.log").write_text(
        "jax.errors.JaxRuntimeError: RESOURCE_EXHAUSTED: Out of memory while trying to allocate 5.51GiB.\n",
        encoding="utf-8",
    )

    spec = get_phase_spec("phase1_unit_foundation_v1")
    raw_status = read_phase_status(spec.phase_id, workspace_dir=workspace_dir)
    status = _phase1_enrich_failed_status(raw_status, workspace_dir)
    strategy = choose_strategy(spec=spec, status=status, attempted_strategy_ids=set())

    assert strategy is not None
    assert strategy.strategy_id == "phase1_batch_size_16"


def test_apply_strategy_and_restore_round_trips_phase1_yaml() -> None:
    spec = get_phase_spec("phase1_unit_foundation_v1")
    status = read_phase_status(spec.phase_id, workspace_dir=PROJECT_ROOT / "outputs" / "does_not_exist")
    strategy = choose_strategy(
        spec=spec,
        status=type(status)(
            phase_id=status.phase_id,
            summary_path=status.summary_path,
            status="not_accepted",
            accepted=False,
            error=None,
            gates={"rollout_stability_on_held_out_variants": False},
        ),
        attempted_strategy_ids=set(),
    )
    assert strategy is not None

    target_path = PROJECT_ROOT / "configs" / "training_universal_phase1_regime.yaml"
    before = target_path.read_text(encoding="utf-8")
    snapshots = apply_strategy(spec, strategy)
    mutated = target_path.read_text(encoding="utf-8")
    assert mutated != before
    restore_snapshots(snapshots)
    assert target_path.read_text(encoding="utf-8") == before


def test_auto_close_phase_runs_dry_run_baseline_and_records_attempt(tmp_path: Path) -> None:
    spec = get_phase_spec("phase1_unit_foundation_v1")
    workspace_dir = tmp_path / "auto_close"

    payload = auto_close_phase(
        spec=spec,
        workspace_dir=workspace_dir,
        jax_platform="cpu",
        dry_run=True,
        max_attempts=1,
        base_run_kwargs={"skip_generation": True},
    )

    assert payload["accepted"] is False
    assert payload["attempts"]
    assert payload["attempts"][0]["strategy_id"] == "__baseline__"
    assert payload["attempts"][0]["status_after"] == "dry_run"
    state_path = workspace_dir / ".convergence_agent" / "state.json"
    assert state_path.exists()


def test_convergence_agent_auto_close_dry_run_returns_nonzero_until_phase_is_accepted(tmp_path: Path) -> None:
    workspace_dir = tmp_path / "cli_auto_close"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/convergence_agent.py",
            "--phase",
            "phase1_unit_foundation_v1",
            "--workspace_dir",
            str(workspace_dir),
            "--auto_close",
            "--dry_run",
            "--max_attempts",
            "1",
            "--skip_generation",
        ],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["accepted"] is False
    assert payload["attempts"][0]["strategy_id"] == "__baseline__"
