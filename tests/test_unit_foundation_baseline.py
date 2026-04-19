"""Tests for the canonical unit-foundation baseline runner."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from scripts.run_unit_foundation_baseline import (
    _build_transfer_calibration_policy,
    _build_transfer_source_config,
    _build_target_only_universal_config,
    _resolve_jax_platform_env,
    _build_transfer_warm_start_config,
    _load_universal_sources,
    _select_best_transfer_restart,
    load_yaml,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_unit_foundation_baseline_dry_run_writes_summary(tmp_path: Path):
    workspace_dir = tmp_path / "baseline"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_unit_foundation_baseline.py",
            "--workspace_dir",
            str(workspace_dir),
            "--dry_run",
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    summary_path = workspace_dir / "summary.json"
    assert summary_path.exists()

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["status"] == "dry_run"
    assert [step["name"] for step in summary["steps"]] == [
        "generate_regime_corpus",
        "train_unit_foundation",
        "evaluate_unit_foundation",
        "transfer_benchmark",
        "control_gate",
    ]
    assert summary["acceptance"]["phase"] == "phase1_unit_foundation_v1"


def test_build_transfer_warm_start_config_uses_few_shot_schedule():
    target_config = {
        "training": {
            "batch_size": 64,
            "max_batches_per_epoch": 64,
            "n_epochs": 4,
        },
        "optimizer": {
            "peak_lr": 5e-4,
            "end_lr": 1e-5,
            "warmup_steps": 50,
            "total_steps": 640,
            "gradient_clip": 1.0,
        },
    }

    warm_start_config = _build_transfer_warm_start_config(
        target_config,
        n_train_samples=64 * 22,
    )

    assert target_config["optimizer"]["peak_lr"] == 5e-4
    assert target_config["optimizer"]["warmup_steps"] == 50
    assert target_config["optimizer"]["total_steps"] == 640
    assert warm_start_config["optimizer"]["peak_lr"] == 2e-4
    assert warm_start_config["optimizer"]["end_lr"] == 1e-5
    assert warm_start_config["optimizer"]["warmup_steps"] == 8
    assert warm_start_config["optimizer"]["total_steps"] == 88


def test_resolve_jax_platform_env_maps_gpu_to_cuda():
    assert _resolve_jax_platform_env("gpu") == "cuda,cpu"
    assert _resolve_jax_platform_env("cpu") == "cpu"


def test_build_transfer_warm_start_config_supports_conservative_variant():
    target_config = {
        "training": {
            "batch_size": 64,
            "max_batches_per_epoch": 64,
            "n_epochs": 4,
        },
        "optimizer": {
            "peak_lr": 5e-4,
            "end_lr": 1e-5,
            "warmup_steps": 50,
            "total_steps": 640,
            "gradient_clip": 1.0,
        },
    }

    warm_start_config = _build_transfer_warm_start_config(
        target_config,
        n_train_samples=64 * 22,
        optimizer_variant="conservative",
    )

    assert warm_start_config["training"]["n_epochs"] == 2
    assert warm_start_config["optimizer"]["peak_lr"] == 1e-4
    assert warm_start_config["optimizer"]["warmup_steps"] == 4
    assert warm_start_config["optimizer"]["total_steps"] == 44


def test_build_transfer_source_config_filters_missing_targeted_loss_systems():
    training_config = {
        "training": {"n_epochs": 10},
        "checkpointing": {"val_every": 2, "save_every": 2},
        "data": {
            "systems": [
                {"name": "cstr"},
                {"name": "cstr_fast_kinetics"},
                {"name": "separator"},
            ]
        },
        "system_specific_losses": {
            "role_derivative_terms": [
                {
                    "name": "reactor_species_dynamics",
                    "systems": ["cstr", "cstr_fast_kinetics"],
                    "state_role": "concentration",
                    "weight": 0.35,
                },
                {
                    "name": "separator_cut_dynamics",
                    "systems": ["separator"],
                    "state_role": "concentration",
                    "weight": 0.35,
                },
            ]
        },
    }

    source_config = _build_transfer_source_config(
        training_config,
        transfer_targets=["cstr_fast_kinetics"],
        source_epochs=6,
    )

    assert source_config["training"]["n_epochs"] == 6
    assert [item["name"] for item in source_config["data"]["systems"]] == ["cstr", "separator"]
    terms = source_config["system_specific_losses"]["role_derivative_terms"]
    assert terms[0]["name"] == "reactor_species_dynamics"
    assert terms[0]["systems"] == ["cstr"]
    assert terms[1]["name"] == "separator_cut_dynamics"
    assert terms[1]["systems"] == ["separator"]


def test_build_target_only_universal_config_filters_missing_targeted_loss_systems():
    training_config = {
        "training": {"n_epochs": 10},
        "checkpointing": {"val_every": 2, "save_every": 2},
        "data": {
            "systems": [
                {"name": "cstr"},
                {"name": "cstr_fast_kinetics"},
                {"name": "separator"},
            ]
        },
        "evaluation": {},
        "system_specific_losses": {
            "role_derivative_terms": [
                {
                    "name": "reactor_species_dynamics",
                    "systems": ["cstr", "cstr_fast_kinetics"],
                    "state_role": "concentration",
                    "weight": 0.35,
                },
                {
                    "name": "separator_cut_dynamics",
                    "systems": ["separator"],
                    "state_role": "concentration",
                    "weight": 0.35,
                },
            ]
        },
    }

    target_config = _build_target_only_universal_config(
        training_config,
        target_name="cstr_fast_kinetics",
        n_epochs=4,
    )

    assert [item["name"] for item in target_config["data"]["systems"]] == ["cstr_fast_kinetics"]
    terms = target_config["system_specific_losses"]["role_derivative_terms"]
    assert len(terms) == 1
    assert terms[0]["name"] == "reactor_species_dynamics"
    assert terms[0]["systems"] == ["cstr_fast_kinetics"]


def test_build_transfer_calibration_policy_uses_family_specific_initializer_for_reactor_and_hydraulic_targets():
    full_config = load_yaml(PROJECT_ROOT / "configs" / "training_universal_phase1_regime.yaml")
    source_config = _build_transfer_source_config(
        full_config,
        transfer_targets=["cstr_fast_kinetics", "heat_exchanger_high_ua", "two_tank_high_throughput"],
        source_epochs=6,
    )
    source_sources = _load_universal_sources(source_config)

    cstr_target = _build_target_only_universal_config(
        full_config,
        target_name="cstr_fast_kinetics",
        n_epochs=4,
    )
    cstr_policy = _build_transfer_calibration_policy(cstr_target, source_sources)
    assert cstr_policy == {
        "name": "reactor_fresh_dynamics_full",
        "optimizer_variant": "default",
        "trainable_mode": "full",
        "tune_normalization": True,
        "tune_physics_params": False,
        "active_param_indices": [],
        "restart_seed_offsets": [0],
        "selection_metric": "rollout_rmse",
        "init_kwargs": {
            "copy_drift_backbone": False,
            "copy_cde_backbone": False,
            "copy_drift_adapter": False,
        },
    }

    two_tank_target = _build_target_only_universal_config(
        full_config,
        target_name="two_tank_high_throughput",
        n_epochs=4,
    )
    two_tank_policy = _build_transfer_calibration_policy(two_tank_target, source_sources)
    assert two_tank_policy["name"] == "hydraulic_policy_set"
    assert two_tank_policy["selection_metric"] == "rollout_rmse"
    assert [candidate["name"] for candidate in two_tank_policy["candidate_policies"]] == [
        "hydraulic_fresh_cde_adapters_norm",
        "hydraulic_fresh_cde_full_norm",
    ]
    assert two_tank_policy["candidate_policies"][0]["trainable_mode"] == "adapters"
    assert two_tank_policy["candidate_policies"][1]["trainable_mode"] == "full"
    assert two_tank_policy["candidate_policies"][0]["init_kwargs"] == {
        "copy_drift_backbone": True,
        "copy_cde_backbone": False,
        "copy_drift_adapter": True,
    }
    assert two_tank_policy["candidate_policies"][1]["init_kwargs"] == {
        "copy_drift_backbone": True,
        "copy_cde_backbone": False,
        "copy_drift_adapter": True,
    }


def test_build_transfer_calibration_policy_keeps_full_warm_start_for_heat_exchanger_target():
    full_config = load_yaml(PROJECT_ROOT / "configs" / "training_universal_phase1_regime.yaml")
    source_config = _build_transfer_source_config(
        full_config,
        transfer_targets=["cstr_fast_kinetics", "heat_exchanger_high_ua", "two_tank_high_throughput"],
        source_epochs=6,
    )
    source_sources = _load_universal_sources(source_config)

    heat_exchanger_target = _build_target_only_universal_config(
        full_config,
        target_name="heat_exchanger_high_ua",
        n_epochs=4,
    )
    heat_exchanger_policy = _build_transfer_calibration_policy(
        heat_exchanger_target,
        source_sources,
    )
    assert heat_exchanger_policy == {
        "name": "full_warm_start",
        "optimizer_variant": "default",
        "trainable_mode": "full",
        "tune_normalization": True,
        "tune_physics_params": False,
        "active_param_indices": [],
        "restart_seed_offsets": [0],
        "selection_metric": "rollout_rmse",
        "init_kwargs": {
            "copy_drift_backbone": True,
            "copy_cde_backbone": True,
            "copy_drift_adapter": True,
        },
    }


def test_select_best_transfer_restart_prefers_lower_rollout_rmse_then_loss():
    restarts = [
        {
            "target": "two_tank_high_throughput",
            "restart_index": 0,
            "train_summary": {
                "per_system_val_losses": {
                    "two_tank_high_throughput": {"total": 0.0048}
                }
            },
            "rollout_metrics": {"rmse": 0.31},
        },
        {
            "target": "two_tank_high_throughput",
            "restart_index": 1,
            "train_summary": {
                "per_system_val_losses": {
                    "two_tank_high_throughput": {"total": 0.0051}
                }
            },
            "rollout_metrics": {"rmse": 0.24},
        },
    ]

    selected = _select_best_transfer_restart(restarts, selection_metric="rollout_rmse")

    assert selected["restart_index"] == 1
