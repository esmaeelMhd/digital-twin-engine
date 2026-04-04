"""Tests for data generation retry behavior."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dte.data.generation import load_dataset
from dte.data.generation_generic import GenericDataGenerator
from dte.simulators.cstr import CSTRParams, CSTRSimulator


def test_generate_dataset_retries_until_requested_count(monkeypatch, tmp_path):
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
    config["data_generation"] = {
        "control_signals": {
            "F_in": {"type": "mixed", "switch_prob": 0.05, "n_changes": 5},
            "Tc_in": {"type": "mixed", "switch_prob": 0.05, "n_changes": 5},
        },
        "disturbance_signals": {
            "Ca_in": {"type": "prbs", "switch_prob": 0.01},
            "T_in": {"type": "prbs", "switch_prob": 0.01},
        },
    }
    generator = GenericDataGenerator(simulator, config, simulator.spec)

    attempts = {"count": 0}

    def fake_generate_trajectory(_key, n_steps=5, dt=0.1, simulation_mode="dataset"):
        attempts["count"] += 1
        if attempts["count"] <= 4:
            return None
        states = jnp.ones((n_steps, 4))

        return {
            "t": jnp.linspace(0.0, dt * n_steps, n_steps),
            "states": states,
            "controls": jnp.ones((n_steps, 2)),
            "disturbances": jnp.ones((n_steps, 2)),
            "params": jnp.ones((6,)),
            "measurement_noise_seconds": 0.0,
            "validation_seconds": 0.0,
        }

    monkeypatch.setattr(generator, "_generate_trajectory", fake_generate_trajectory)

    output_path = tmp_path / "train_data.h5"

    dataset = generator.generate_dataset_to_hdf5(
        jax.random.PRNGKey(0), str(output_path), n_trajectories=3, n_steps=5, batch_size=1
    )
    reloaded = load_dataset(str(output_path))

    assert dataset["states_shape"] == (3, 5, 4)
    assert dataset["controls_shape"] == (3, 5, 2)
    assert dataset["disturbances_shape"] == (3, 5, 2)
    assert dataset["params_shape"] == (3, 6)
    assert reloaded["states"].shape == (3, 5, 4)
    assert attempts["count"] == 7
    assert generator.last_profile["attempts"] == 7
    assert generator.last_profile["invalid_trajectories"] == 4
    assert generator.last_profile["batch_size"] == 1


def test_generate_dataset_batched_dataset_mode_returns_expected_shapes():
    """Batched dataset mode should return shape-compatible outputs."""
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
    config["data_generation"] = {
        "control_signals": {
            "F_in": {"type": "mixed", "switch_prob": 0.05, "n_changes": 5},
            "Tc_in": {"type": "mixed", "switch_prob": 0.05, "n_changes": 5},
        },
        "disturbance_signals": {
            "Ca_in": {"type": "prbs", "switch_prob": 0.01},
            "T_in": {"type": "prbs", "switch_prob": 0.01},
        },
    }
    generator = GenericDataGenerator(simulator, config, simulator.spec)
    batched_batch = generator._generate_trajectories_batched(
        jax.random.split(jax.random.PRNGKey(123), 2),
        n_steps=8,
        dt=0.1,
        simulation_mode="dataset",
    )

    assert batched_batch["states"].shape[1:] == (8, 4)
    assert batched_batch["controls"].shape[1:] == (8, 2)
    assert batched_batch["disturbances"].shape[1:] == (8, 2)
    assert batched_batch["params"].shape[1] == 6
    assert batched_batch["time"].shape[1] == 8
    assert batched_batch["states"].shape[0] <= 2
    assert np.all(np.isfinite(np.asarray(batched_batch["states"])))


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
    config["data_generation"] = {
        "control_signals": {
            "F_in": {"type": "mixed", "switch_prob": 0.05, "n_changes": 5},
            "Tc_in": {"type": "mixed", "switch_prob": 0.05, "n_changes": 5},
        },
        "disturbance_signals": {
            "Ca_in": {"type": "prbs", "switch_prob": 0.01},
            "T_in": {"type": "prbs", "switch_prob": 0.01},
        },
    }
    generator = GenericDataGenerator(simulator, config, simulator.spec)
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
    config["data_generation"] = {
        "control_signals": {
            "F_in": {"type": "mixed", "switch_prob": 0.05, "n_changes": 5},
            "Tc_in": {"type": "mixed", "switch_prob": 0.05, "n_changes": 5},
        },
        "disturbance_signals": {
            "Ca_in": {"type": "prbs", "switch_prob": 0.01},
            "T_in": {"type": "prbs", "switch_prob": 0.01},
        },
    }
    generator = GenericDataGenerator(simulator, config, simulator.spec)
    output_path = tmp_path / "train_data.h5"

    dataset_summary = generator.generate_dataset_to_hdf5(
        jax.random.PRNGKey(0),
        str(output_path),
        n_trajectories=2,
        n_steps=8,
        simulation_mode="dataset",
        batch_size=2,
    )

    reloaded = load_dataset(str(output_path))

    assert output_path.exists()
    assert dataset_summary["states_shape"] == (2, 8, 4)
    assert dataset_summary["controls_shape"] == (2, 8, 2)
    assert reloaded["states"].shape == (2, 8, 4)
    assert "state_mean" in reloaded["normalization"]
    assert "state_mean" in dataset_summary["normalization"]
    assert dataset_summary["states_shape"] == (2, 8, 4)
    assert generator.last_profile["attempts"] >= 2
    assert generator.last_profile["invalid_trajectories"] >= 0


def test_recommend_batch_size_uses_backend_specific_defaults():
    """Batch-size recommendations should reflect backend class."""
    generator = GenericDataGenerator(CSTRSimulator(CSTRParams()), {"simulation": {"dt": 0.1}})
    assert generator.recommend_batch_size("cpu") == 4
    assert generator.recommend_batch_size("gpu") == 8
    assert generator.recommend_batch_size("tpu") == 16


def test_generic_generator_uses_cstr_signal_policies_and_param_formatting():
    """The shared generic generator should cover the CSTR fast path directly."""
    simulator = CSTRSimulator(CSTRParams())
    config = {
        "operating_ranges": {
            "F_in": [40.0, 60.0],
            "Tc_in": [290.0, 310.0],
            "Ca_in": [0.8, 1.2],
            "T_in": [315.0, 325.0],
        },
        "simulation": {"dt": 0.1},
        "data_generation": {
            "control_signals": {
                "F_in": {"type": "mixed", "switch_prob": 0.05, "n_changes": 5},
                "Tc_in": {"type": "mixed", "switch_prob": 0.05, "n_changes": 5},
            },
            "disturbance_signals": {
                "Ca_in": {"type": "prbs", "switch_prob": 0.01},
                "T_in": {"type": "prbs", "switch_prob": 0.01},
            },
        },
    }
    generator = GenericDataGenerator(simulator, config, simulator.spec)
    keys = jax.random.split(jax.random.PRNGKey(7), 3)

    batched_controls = generator._generate_control_trajectories(keys, n_steps=12, dt=0.1)
    single_controls = jnp.stack(
        [generator._generate_control_trajectory(key, n_steps=12, dt=0.1) for key in keys]
    )
    batched_disturbances = generator._generate_disturbance_trajectories(keys, n_steps=12, dt=0.1)
    single_disturbances = jnp.stack(
        [generator._generate_disturbance_trajectory(key, n_steps=12, dt=0.1) for key in keys]
    )

    assert batched_controls.shape == (3, 12, 2)
    assert batched_disturbances.shape == (3, 12, 2)
    assert jnp.allclose(batched_controls, single_controls, atol=1e-6, rtol=1e-6)
    assert jnp.allclose(batched_disturbances, single_disturbances, atol=1e-6, rtol=1e-6)

    stored_params = simulator.format_data_generation_params_batch(
        simulator.sample_data_generation_params_batch(keys)
    )
    assert stored_params.shape == (3, 6)
