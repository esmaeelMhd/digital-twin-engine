"""Tests for canonical convergence-phase registry and CLI behavior."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from dte.convergence.closure import (
    _build_phase1_probe_config_payload,
    _infer_phase1_probe_error,
    _phase1_enrich_failed_status,
    apply_strategy,
    auto_close_phase,
    choose_strategy,
    find_prior_kept_strategy_id,
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


def test_build_phase1_probe_config_payload_caps_runtime_without_overwriting_user_shape() -> None:
    config = {
        "training": {
            "batch_size": 64,
            "n_epochs": 10,
            "max_batches_per_epoch": 64,
        },
        "checkpointing": {
            "val_every": 4,
            "save_every": 5,
            "max_val_batches": 4,
        },
        "evaluation": {
            "per_system_batches": 4,
            "forecast_batches": 2,
            "rollout_batches": 2,
            "rollout_samples": 4,
            "uncertainty_batches": 2,
            "uncertainty_samples": 8,
            "sensitivity_batches": 2,
        },
        "model": {
            "latent_solver": {
                "method": "heun",
            }
        },
    }

    probe = _build_phase1_probe_config_payload(config)

    assert probe["training"]["batch_size"] == 64
    assert probe["training"]["n_epochs"] == 2
    assert probe["training"]["max_batches_per_epoch"] == 8
    assert probe["checkpointing"]["val_every"] == 1
    assert probe["checkpointing"]["save_every"] == 1
    assert probe["checkpointing"]["max_val_batches"] == 2
    assert probe["evaluation"]["per_system_batches"] == 2
    assert probe["evaluation"]["forecast_batches"] == 1
    assert probe["evaluation"]["rollout_batches"] == 1
    assert probe["evaluation"]["rollout_samples"] == 2
    assert probe["evaluation"]["uncertainty_batches"] == 0
    assert probe["evaluation"]["uncertainty_samples"] == 0
    assert probe["evaluation"]["sensitivity_batches"] == 0
    assert probe["model"]["latent_solver"]["method"] == "heun"
    assert config["training"]["n_epochs"] == 10
    assert config["checkpointing"]["val_every"] == 4


def test_infer_phase1_probe_error_marks_generic_gpu_startup_failure_as_inferred_oom() -> None:
    probe_config = {
        "training": {
            "batch_size": 64,
        }
    }
    train_error = """
============================================================
UNIVERSAL DIGITAL TWIN TRAINING
============================================================
Universal Training:   0%|          | 0/8 [00:00<?, ?it/s]
"""

    error = _infer_phase1_probe_error(
        train_error=train_error,
        probe_config=probe_config,
        jax_platform="gpu",
    )

    assert error == "resource exhausted: out of memory (inferred)"


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


def test_find_prior_kept_strategy_id_reads_latest_matching_phase_workspace(tmp_path: Path) -> None:
    spec = get_phase_spec("phase1_unit_foundation_v1")
    prior_workspace = tmp_path / "unit_foundation_phase1_run_prev"
    prior_workspace.mkdir(parents=True)
    (prior_workspace / "summary.json").write_text(
        json.dumps(
            {
                "status": "ok",
                "acceptance": {
                    "phase": "phase1_unit_foundation_v1",
                    "accepted": False,
                    "gates": {},
                },
            }
        ),
        encoding="utf-8",
    )
    state_dir = prior_workspace / ".convergence_agent"
    state_dir.mkdir(parents=True)
    (state_dir / "state.json").write_text(
        json.dumps(
            {
                "attempts": [
                    {"strategy_id": "__baseline__", "kept": True, "improved": True},
                    {"strategy_id": "phase1_batch_size_16", "kept": True, "improved": True},
                    {"strategy_id": "phase1_solver_relaxed_tolerances", "kept": False, "improved": False},
                ]
            }
        ),
        encoding="utf-8",
    )

    current_workspace = tmp_path / "unit_foundation_phase1_run_next"
    current_workspace.mkdir(parents=True)

    strategy_id = find_prior_kept_strategy_id(spec, current_workspace)

    assert strategy_id == "phase1_batch_size_16"


def test_auto_close_phase_carries_failed_probe_status_forward_instead_of_repeating_missing_baseline(
    tmp_path: Path,
    monkeypatch,
) -> None:
    spec = get_phase_spec("phase1_unit_foundation_v1")
    workspace_dir = tmp_path / "carry_forward"

    probe_calls = {"count": 0}

    class _FakeProbe:
        def __init__(self, config_path: Path, run_dir: Path, log_dir: Path, train_summary_path: Path, eval_summary_path: Path):
            self.succeeded = False
            self.config_path = config_path
            self.run_dir = run_dir
            self.log_dir = log_dir
            self.train_summary_path = train_summary_path
            self.eval_summary_path = eval_summary_path
            self.error = "resource exhausted: out of memory"
            self.train_aggregate_metric = None
            self.eval_aggregate_metric = None
            self.rollout_rmse_max = None
            self.duration_seconds = 0.01

    def _fake_probe(*, workspace_dir: Path, attempt_index: int, strategy_id: str, jax_platform: str, seed: int = 42):
        probe_calls["count"] += 1
        probe_root = workspace_dir / ".convergence_agent" / "probes" / f"{attempt_index:02d}_{strategy_id}"
        return _FakeProbe(
            config_path=probe_root / "config.yaml",
            run_dir=probe_root / "run",
            log_dir=probe_root / "logs",
            train_summary_path=probe_root / "run" / "summary.json",
            eval_summary_path=probe_root / "eval" / "summary.json",
        )

    monkeypatch.setattr("dte.convergence.closure._run_phase1_probe", _fake_probe)

    payload = auto_close_phase(
        spec=spec,
        workspace_dir=workspace_dir,
        jax_platform="gpu",
        dry_run=False,
        max_attempts=2,
        base_run_kwargs={"skip_generation": True},
    )

    assert probe_calls["count"] == 2
    assert len(payload["attempts"]) == 2
    assert payload["attempts"][0]["strategy_id"] == "__baseline__"
    assert payload["attempts"][0]["error_after"] == "resource exhausted: out of memory"
    assert payload["attempts"][1]["strategy_id"] == "phase1_batch_size_16"
    assert payload["status"] in {"failed", "missing"}


def test_auto_close_phase_bootstraps_from_prior_kept_strategy_on_fresh_workspace(
    tmp_path: Path,
    monkeypatch,
) -> None:
    spec = get_phase_spec("phase1_unit_foundation_v1")
    prior_workspace = tmp_path / "unit_foundation_phase1_run_prev"
    prior_workspace.mkdir(parents=True)
    (prior_workspace / "summary.json").write_text(
        json.dumps(
            {
                "status": "ok",
                "acceptance": {
                    "phase": "phase1_unit_foundation_v1",
                    "accepted": False,
                    "gates": {},
                },
            }
        ),
        encoding="utf-8",
    )
    state_dir = prior_workspace / ".convergence_agent"
    state_dir.mkdir(parents=True)
    (state_dir / "state.json").write_text(
        json.dumps(
            {
                "attempts": [
                    {"strategy_id": "phase1_batch_size_16", "kept": True, "improved": True},
                ]
            }
        ),
        encoding="utf-8",
    )

    workspace_dir = tmp_path / "unit_foundation_phase1_run_next"
    probe_calls: list[str] = []

    class _FakeProbe:
        def __init__(self, strategy_id: str):
            probe_root = workspace_dir / ".convergence_agent" / "probes" / strategy_id
            self.succeeded = False
            self.config_path = probe_root / "config.yaml"
            self.run_dir = probe_root / "run"
            self.log_dir = probe_root / "logs"
            self.train_summary_path = probe_root / "run" / "summary.json"
            self.eval_summary_path = probe_root / "eval" / "summary.json"
            self.error = "resource exhausted: out of memory"
            self.train_aggregate_metric = None
            self.eval_aggregate_metric = None
            self.rollout_rmse_max = None
            self.duration_seconds = 0.01

    def _fake_probe(*, workspace_dir: Path, attempt_index: int, strategy_id: str, jax_platform: str, seed: int = 42):
        probe_calls.append(strategy_id)
        return _FakeProbe(strategy_id)

    monkeypatch.setattr("dte.convergence.closure._run_phase1_probe", _fake_probe)

    payload = auto_close_phase(
        spec=spec,
        workspace_dir=workspace_dir,
        jax_platform="gpu",
        dry_run=False,
        max_attempts=1,
        base_run_kwargs={"skip_generation": True},
    )

    assert probe_calls == ["phase1_batch_size_16"]
    assert payload["attempts"][0]["strategy_id"] == "phase1_batch_size_16"
    assert "[bootstrapped from prior kept strategy]" in payload["attempts"][0]["description"]


def test_auto_close_phase_does_not_repeat_non_improving_strategy_in_same_run(
    tmp_path: Path,
    monkeypatch,
) -> None:
    spec = get_phase_spec("phase1_unit_foundation_v1")
    workspace_dir = tmp_path / "no_repeat"

    probe_calls: list[str] = []

    class _FakeProbe:
        def __init__(
            self,
            *,
            strategy_id: str,
            succeeded: bool,
            error: str | None = None,
            train_aggregate_metric: float | None = 0.1,
            eval_aggregate_metric: float | None = 0.1,
            rollout_rmse_max: float | None = 50.0,
        ):
            probe_root = workspace_dir / ".convergence_agent" / "probes" / strategy_id
            self.succeeded = succeeded
            self.config_path = probe_root / "config.yaml"
            self.run_dir = probe_root / "run"
            self.log_dir = probe_root / "logs"
            self.train_summary_path = probe_root / "run" / "summary.json"
            self.eval_summary_path = probe_root / "eval" / "summary.json"
            self.error = error
            self.train_aggregate_metric = train_aggregate_metric
            self.eval_aggregate_metric = eval_aggregate_metric
            self.rollout_rmse_max = rollout_rmse_max
            self.duration_seconds = 0.01

    def _fake_probe(*, workspace_dir: Path, attempt_index: int, strategy_id: str, jax_platform: str, seed: int = 42):
        probe_calls.append(strategy_id)
        if strategy_id == "__baseline__":
            return _FakeProbe(strategy_id=strategy_id, succeeded=False, error="resource exhausted: out of memory")
        return _FakeProbe(strategy_id=strategy_id, succeeded=True)

    statuses = [
        # attempt 1 baseline failure
        ("failed", False, "resource exhausted: out of memory", {}),
        # attempt 2 improves to not_accepted
        (
            "not_accepted",
            False,
            None,
            {
                "shared_checkpoint_trains_reproducibly": True,
                "control_response_fidelity_is_measured": True,
                "control_gate_completed": True,
                "rollout_stability_on_held_out_variants": False,
                "transfer_beats_scratch_on_targets": False,
            },
        ),
        # attempt 3 stays not_accepted, so strategy should not repeat
        (
            "not_accepted",
            False,
            None,
            {
                "shared_checkpoint_trains_reproducibly": True,
                "control_response_fidelity_is_measured": True,
                "control_gate_completed": True,
                "rollout_stability_on_held_out_variants": False,
                "transfer_beats_scratch_on_targets": False,
            },
        ),
        # attempt 4 also not_accepted, but should come from the next strategy
        (
            "not_accepted",
            False,
            None,
            {
                "shared_checkpoint_trains_reproducibly": True,
                "control_response_fidelity_is_measured": True,
                "control_gate_completed": True,
                "rollout_stability_on_held_out_variants": False,
                "transfer_beats_scratch_on_targets": False,
            },
        ),
    ]

    def _fake_run_phase_once(*, spec, workspace_dir, jax_platform, dry_run, base_run_kwargs=None, run_overrides=None):
        status_name, accepted, error, gates = statuses.pop(0)
        from dte.convergence.workflow import PhaseRunStatus

        return (
            ["python", "scripts/run_unit_foundation_baseline.py"],
            0,
            PhaseRunStatus(
                phase_id=spec.phase_id,
                summary_path=spec.summary_path_for_workspace(workspace_dir),
                status=status_name,
                accepted=accepted,
                error=error,
                gates=gates,
            ),
        )

    monkeypatch.setattr("dte.convergence.closure._run_phase1_probe", _fake_probe)
    monkeypatch.setattr("dte.convergence.closure.run_phase_once", _fake_run_phase_once)
    monkeypatch.setattr("dte.convergence.closure.apply_strategy", lambda spec, strategy: [])

    payload = auto_close_phase(
        spec=spec,
        workspace_dir=workspace_dir,
        jax_platform="gpu",
        dry_run=False,
        max_attempts=4,
        base_run_kwargs={"skip_generation": True},
    )

    strategy_ids = [attempt["strategy_id"] for attempt in payload["attempts"]]
    assert strategy_ids[:2] == [
        "__baseline__",
        "phase1_batch_size_16",
    ]
    assert probe_calls[:2] == strategy_ids[:2]


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
