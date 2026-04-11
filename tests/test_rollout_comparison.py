"""Tests for rollout-stability comparison config generation."""

from pathlib import Path

import yaml

from scripts.compare_rollout_stability import build_training_config, compare_system_metrics


def test_old_style_comparison_config_disables_physics_losses(tmp_path: Path):
    base_config_path = tmp_path / "base.yaml"
    base_config_path.write_text(
        yaml.safe_dump(
            {
                "model": {
                    "simulator_prior": {"enabled": True},
                    "learned_solver": {"enabled": True},
                    "self_correcting_policy": {"enabled": True},
                    "neural_cde": {"enabled": True},
                },
                "training": {},
                "optimizer": {},
                "checkpointing": {},
                "loss_weights": {
                    "reconstruction": 1.0,
                    "trajectory": 1.0,
                    "one_step": 1.0,
                    "energy": 0.1,
                    "energy_balance": 0.2,
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    config = build_training_config(
        system_name="heat_exchanger",
        variant="old_style",
        base_config_path=base_config_path,
        n_epochs=2,
        max_batches_per_epoch=2,
        max_val_batches=1,
        disable_neural_cde_in_old=True,
    )

    assert config["model"]["simulator_prior"]["enabled"] is False
    assert config["model"]["learned_solver"]["enabled"] is False
    assert config["model"]["self_correcting_policy"]["enabled"] is False
    assert config["model"]["neural_cde"]["enabled"] is False
    assert config["loss_weights"]["energy"] == 0.0
    assert config["loss_weights"]["energy_balance"] == 0.0
    assert config["physics_loss_weights"]["energy"] == 0.0


def test_tail_error_and_physics_improvement_can_satisfy_stability_pass():
    comparison = compare_system_metrics(
        system_name="heat_exchanger",
        old_metrics={
            "non_finite_detected": False,
            "mse_fullseq": 100.0,
            "mse_laststep": 120.0,
            "mse_10step": 110.0,
            "mse_1step": 50.0,
            "energy_violation_max": 20.0,
            "mass_violation_max": 0.0,
        },
        new_metrics={
            "non_finite_detected": False,
            "mse_fullseq": 108.0,
            "mse_laststep": 90.0,
            "mse_10step": 118.0,
            "mse_1step": 55.0,
            "energy_violation_max": 10.0,
            "mass_violation_max": 0.0,
        },
        min_improvement_ratio=0.9,
        stability_tolerance_ratio=1.2,
    )

    assert comparison["pass"] is True
    assert "physics violation improved" in comparison["reason"]
