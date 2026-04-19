"""Tests for dataset splitting and sampling behavior."""

import jax
import jax.numpy as jnp

from dte.data.datasets.unit_dataset import TrajectoryDataset


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


def test_sample_batch_falls_back_to_replacement_for_tiny_datasets():
    data = {
        "states": jnp.zeros((2, 8, 2)),
        "controls": jnp.zeros((2, 8, 1)),
        "disturbances": jnp.zeros((2, 8, 1)),
        "params": jnp.array([[1.0], [2.0]], dtype=jnp.float32),
        "time": jnp.tile(jnp.linspace(0.0, 0.7, 8), (2, 1)),
        "normalization": {
            "state_mean": jnp.zeros(2),
            "state_std": jnp.ones(2),
            "control_mean": jnp.zeros(1),
            "control_std": jnp.ones(1),
            "disturbance_mean": jnp.zeros(1),
            "disturbance_std": jnp.ones(1),
            "param_mean": jnp.zeros(1),
            "param_std": jnp.ones(1),
        },
    }

    dataset = TrajectoryDataset(data, seq_len=8, stride=8)
    batch = dataset.sample_batch(jax.random.PRNGKey(0), batch_size=5)

    assert dataset.n_samples == 2
    assert batch["states"].shape == (5, 8, 2)
    assert batch["params"].shape == (5, 1)
    assert jnp.all(jnp.isin(batch["params"].reshape(-1), jnp.array([1.0, 2.0], dtype=jnp.float32)))


def test_missing_param_normalization_stats_are_backfilled():
    data = {
        "states": jnp.zeros((2, 12, 3)),
        "controls": jnp.zeros((2, 12, 1)),
        "disturbances": jnp.zeros((2, 12, 2)),
        "params": jnp.array([[1.0, 2.0], [3.0, 4.0]], dtype=jnp.float32),
        "time": jnp.tile(jnp.linspace(0.0, 1.1, 12), (2, 1)),
        "normalization": {
            "state_mean": jnp.zeros(3),
            "state_std": jnp.ones(3),
            "control_mean": jnp.zeros(1),
            "control_std": jnp.ones(1),
            "disturbance_mean": jnp.zeros(2),
            "disturbance_std": jnp.ones(2),
        },
    }

    dataset = TrajectoryDataset(data, seq_len=6, stride=3)
    stats = dataset.get_normalization_stats()

    assert "param_mean" in stats
    assert "param_std" in stats
    assert jnp.allclose(stats["param_mean"], jnp.array([2.0, 3.0], dtype=jnp.float32))
    assert jnp.all(stats["param_std"] > 0.0)
    assert dataset.state_dim == 3
    assert dataset.control_dim == 1
    assert dataset.disturbance_dim == 2
    assert dataset.param_dim == 2


def test_dataset_clamps_seq_len_to_available_trajectory_length():
    data = {
        "states": jnp.zeros((2, 8, 2)),
        "controls": jnp.zeros((2, 8, 1)),
        "disturbances": jnp.zeros((2, 8, 1)),
        "params": jnp.ones((2, 3)),
        "time": jnp.tile(jnp.linspace(0.0, 0.7, 8), (2, 1)),
        "normalization": {
            "state_mean": jnp.zeros(2),
            "state_std": jnp.ones(2),
            "control_mean": jnp.zeros(1),
            "control_std": jnp.ones(1),
            "disturbance_mean": jnp.zeros(1),
            "disturbance_std": jnp.ones(1),
            "param_mean": jnp.zeros(3),
            "param_std": jnp.ones(3),
        },
    }

    dataset = TrajectoryDataset(data, seq_len=16, stride=4)

    assert dataset.seq_len == 8
    assert dataset.n_samples == 2
    sample = dataset[0]
    assert sample["states"].shape == (8, 2)
