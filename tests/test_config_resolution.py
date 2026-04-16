"""Tests for explicit single-system training config resolution."""

from dte.training.shared.config_resolution import resolve_single_system_training_config


def test_strict_config_mode_keeps_loaded_values():
    config = {
        "model": {"initial_diffusion_scale": 0.1},
        "optimizer": {"peak_lr": 1e-3, "gradient_clip": 1.0},
        "loss_weights": {"kl": 1e-4},
    }

    resolved, applied = resolve_single_system_training_config(config, mode="strict")

    assert resolved == config
    assert applied == []


def test_legacy_safe_config_mode_applies_historic_train_overrides():
    config = {
        "model": {"initial_diffusion_scale": 0.1, "disturbance_dim": 3},
        "optimizer": {"peak_lr": 1e-3, "gradient_clip": 1.0},
        "loss_weights": {"kl": 1e-4},
    }

    resolved, applied = resolve_single_system_training_config(config, mode="legacy_safe")

    assert resolved["model"]["initial_diffusion_scale"] == 0.0001
    assert resolved["model"]["disturbance_dim"] == 2
    assert resolved["optimizer"]["peak_lr"] == 5.0e-4
    assert resolved["optimizer"]["gradient_clip"] == 0.5
    assert resolved["loss_weights"]["kl"] == 0.0
    assert len(applied) == 5
