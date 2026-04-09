"""Tests for dataset splitting and sampling behavior."""

import jax
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


def test_sample_batch_seq_len_override_does_not_truncate_params():
    data = {
        "states": jnp.zeros((3, 20, 4)),
        "controls": jnp.zeros((3, 20, 2)),
        "disturbances": jnp.zeros((3, 20, 2)),
        "params": jnp.arange(18, dtype=jnp.float32).reshape(3, 6),
        "time": jnp.tile(jnp.linspace(0.0, 1.9, 20), (3, 1)),
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

    dataset = TrajectoryDataset(data, seq_len=10, stride=5)
    batch = dataset.sample_batch(jax.random.PRNGKey(0), batch_size=2, seq_len=2)

    assert batch["states"].shape == (2, 2, 4)
    assert batch["controls"].shape == (2, 2, 2)
    assert batch["disturbances"].shape == (2, 2, 2)
    assert batch["t"].shape == (2, 2)
    assert batch["params"].shape == (2, 6)
