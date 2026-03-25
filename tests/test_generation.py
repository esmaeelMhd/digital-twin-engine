"""Tests for data generation retry behavior."""

import jax
import jax.numpy as jnp
import pytest

from dte.data.generation import DataGenerator
from dte.simulators.cstr import CSTRParams, CSTRSimulator


def test_generate_dataset_retries_until_requested_count(monkeypatch):
    """The generator should keep retrying until it reaches the requested count."""

    simulator = CSTRSimulator(CSTRParams())
    config = {
        "operating_ranges": {
            "F_in": [40.0, 60.0],
            "Tc_in": [290.0, 310.0],
            "Ca_in": [0.8, 1.2],
            "T_in": [315.0, 325.0],
        },
        "simulation": {"dt": 0.1},
    }
    generator = DataGenerator(simulator, config)

    attempts = {"count": 0}

    def fake_generate_single_trajectory(
        _key,
        n_steps=5,
        params=None,
        simulation_mode="dataset",
        profiling=None,
    ):
        attempts["count"] += 1
        if profiling is not None:
            profiling["simulation_mode"] = simulation_mode

        if attempts["count"] <= 4:
            states = jnp.full((n_steps, 4), jnp.nan)
        else:
            states = jnp.ones((n_steps, 4))

        return {
            "time": jnp.linspace(0.0, 1.0, n_steps),
            "states": states,
            "controls": jnp.ones((n_steps, 2)),
            "disturbances": jnp.ones((n_steps, 2)),
            "params": jnp.ones((6,)),
        }

    monkeypatch.setattr(generator, "generate_single_trajectory", fake_generate_single_trajectory)

    dataset = generator.generate_dataset(
        jax.random.PRNGKey(0), n_trajectories=3, n_steps=5
    )

    assert dataset["states"].shape == (3, 5, 4)
    assert dataset["controls"].shape == (3, 5, 2)
    assert dataset["disturbances"].shape == (3, 5, 2)
    assert dataset["params"].shape == (3, 6)
    assert attempts["count"] == 7
    assert generator.last_profile["successful_trajectories"] == 3
    assert generator.last_profile["invalid_trajectories"] == 4
    assert generator.last_profile["simulation_mode"] == "dataset"


def test_generate_dataset_batched_dataset_mode_matches_single_mode():
    """Batched dataset mode should preserve shapes and deterministic outputs."""
    simulator = CSTRSimulator(CSTRParams())
    config = {
        "operating_ranges": {
            "F_in": [40.0, 60.0],
            "Tc_in": [290.0, 310.0],
            "Ca_in": [0.8, 1.2],
            "T_in": [315.0, 325.0],
        },
        "simulation": {"dt": 0.1},
    }
    generator = DataGenerator(simulator, config)
    key = jax.random.PRNGKey(123)

    single_dataset = generator.generate_dataset(
        key,
        n_trajectories=2,
        n_steps=8,
        simulation_mode="dataset",
        batch_size=1,
    )
    batched_dataset = generator.generate_dataset(
        key,
        n_trajectories=2,
        n_steps=8,
        simulation_mode="dataset",
        batch_size=2,
    )

    assert batched_dataset["states"].shape == single_dataset["states"].shape
    assert batched_dataset["controls"].shape == single_dataset["controls"].shape
    assert batched_dataset["disturbances"].shape == single_dataset["disturbances"].shape
    assert batched_dataset["params"].shape == single_dataset["params"].shape
    assert jnp.allclose(batched_dataset["states"], single_dataset["states"], atol=1e-4, rtol=1e-4)
    assert generator.last_profile["batch_size"] == 2
    assert (
        generator.last_profile["steady_state_fast_successes"]
        + generator.last_profile["steady_state_fallbacks"]
        >= 2
    )


def test_batched_signal_generation_matches_single_path():
    """Batched signal helpers should reproduce single-path trajectories exactly."""
    simulator = CSTRSimulator(CSTRParams())
    config = {
        "operating_ranges": {
            "F_in": [40.0, 60.0],
            "Tc_in": [290.0, 310.0],
            "Ca_in": [0.8, 1.2],
            "T_in": [315.0, 325.0],
        },
        "simulation": {"dt": 0.1},
    }
    generator = DataGenerator(simulator, config)
    keys = jax.random.split(jax.random.PRNGKey(7), 3)

    batched = generator._generate_signal_batch(
        keys,
        n_steps=12,
        dt=0.1,
        min_val=40.0,
        max_val=60.0,
        signal_type="mixed",
        switch_prob=0.05,
        n_changes=5,
    )
    single = jnp.stack(
        [
            generator._generate_signal(
                key,
                n_steps=12,
                dt=0.1,
                min_val=40.0,
                max_val=60.0,
                signal_type="mixed",
                switch_prob=0.05,
                n_changes=5,
            )
            for key in keys
        ]
    )

    assert jnp.allclose(batched, single, atol=1e-6, rtol=1e-6)


def test_generate_dataset_to_hdf5_round_trips(tmp_path):
    """Streaming HDF5 generation should produce a loadable dataset with normalization stats."""
    simulator = CSTRSimulator(CSTRParams())
    config = {
        "operating_ranges": {
            "F_in": [40.0, 60.0],
            "Tc_in": [290.0, 310.0],
            "Ca_in": [0.8, 1.2],
            "T_in": [315.0, 325.0],
        },
        "simulation": {"dt": 0.1},
    }
    generator = DataGenerator(simulator, config)
    output_path = tmp_path / "train_data.h5"

    dataset_summary = generator.generate_dataset_to_hdf5(
        jax.random.PRNGKey(0),
        str(output_path),
        n_trajectories=2,
        n_steps=8,
        simulation_mode="dataset",
        batch_size=2,
    )

    reloaded = DataGenerator.load_dataset(str(output_path))

    assert output_path.exists()
    assert dataset_summary["states_shape"] == (2, 8, 4)
    assert dataset_summary["controls_shape"] == (2, 8, 2)
    assert reloaded["states"].shape == (2, 8, 4)
    assert "state_mean" in reloaded["normalization"]
    assert "state_mean" in dataset_summary["normalization"]
    assert generator.last_profile["successful_trajectories"] == 2
    assert generator.last_profile["attempts"] >= 2
    assert (
        generator.last_profile["steady_state_fast_successes"]
        + generator.last_profile["steady_state_fallbacks"]
        == generator.last_profile["attempts"]
    )


def test_recommend_batch_size_uses_backend_specific_defaults():
    """Batch-size recommendations should reflect backend class."""
    assert DataGenerator.recommend_batch_size("cpu") == 4
    assert DataGenerator.recommend_batch_size("gpu") == 8
    assert DataGenerator.recommend_batch_size("tpu") == 16
