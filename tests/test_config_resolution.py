"""Tests for canonical single-system training config resolution."""

from dte.training.shared.config_resolution import resolve_single_system_training_config


def test_resolution_keeps_loaded_values():
    config = {
        "model": {"initial_diffusion_scale": 0.1},
        "optimizer": {"peak_lr": 1e-3, "gradient_clip": 1.0},
        "loss_weights": {"kl": 1e-4},
    }

    resolved = resolve_single_system_training_config(config)

    assert resolved == config


def test_resolution_returns_independent_copy():
    config = {
        "model": {"initial_diffusion_scale": 0.1, "disturbance_dim": 3},
        "optimizer": {"peak_lr": 1e-3, "gradient_clip": 1.0},
        "loss_weights": {"kl": 1e-4},
    }

    resolved = resolve_single_system_training_config(config)
    resolved["model"]["initial_diffusion_scale"] = 0.0001

    assert config["model"]["initial_diffusion_scale"] == 0.1
