"""Tests for dataset splitting behavior."""

import jax.numpy as jnp

from dte.data.dataset import TrajectoryDataset


def test_split_uses_trajectory_axis_and_keeps_validation_nonempty():
    """Validation data should be split on trajectories, not subsequences."""

    data = {
        "states": jnp.zeros((5, 60, 4)),
        "controls": jnp.zeros((5, 60, 2)),
        "disturbances": jnp.zeros((5, 60, 2)),
        "params": jnp.zeros((5, 6)),
        "time": jnp.tile(jnp.linspace(0.0, 5.9, 60), (5, 1)),
        "normalization": {
            "state_mean": jnp.zeros(4),
            "state_std": jnp.ones(4),
            "control_mean": jnp.zeros(2),
            "control_std": jnp.ones(2),
            "disturbance_mean": jnp.zeros(2),
            "disturbance_std": jnp.ones(2),
            "param_mean": jnp.zeros(6),
            "param_std": jnp.ones(6),
        },
    }

    dataset = TrajectoryDataset(data, seq_len=20, stride=10)
    train_dataset, val_dataset = dataset.split(val_fraction=0.2)

    assert train_dataset.data["states"].shape[0] == 4
    assert val_dataset.data["states"].shape[0] == 1
    assert train_dataset.n_samples > 0
    assert val_dataset.n_samples > 0
