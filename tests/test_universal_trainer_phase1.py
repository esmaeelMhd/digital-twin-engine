"""Focused tests for Phase 1 universal-trainer extensions."""

import jax
import jax.numpy as jnp

from dte.data.datasets.universal_unit_dataset import UniversalSystemMetadata
from dte.models.universal_digital_twin import UniversalDigitalTwin
from dte.training.universal_trainer import UniversalTrainer


class _DummyDataset:
    def __init__(self, metadata):
        self.metadata = metadata
        self.n_samples = 8
        self.system_names = list(metadata.system_names)
        self.system_ids = {name: idx for idx, name in enumerate(self.system_names)}
        self.n_systems = len(self.system_names)


def _build_metadata() -> UniversalSystemMetadata:
    return UniversalSystemMetadata(
        system_names=("cstr",),
        state_center=jnp.zeros((1, 4), dtype=jnp.float32),
        state_scale=jnp.ones((1, 4), dtype=jnp.float32),
        control_center=jnp.zeros((1, 2), dtype=jnp.float32),
        control_scale=jnp.ones((1, 2), dtype=jnp.float32),
        disturbance_center=jnp.zeros((1, 2), dtype=jnp.float32),
        disturbance_scale=jnp.ones((1, 2), dtype=jnp.float32),
        param_scale=jnp.ones((1, 6), dtype=jnp.float32),
        state_mask=jnp.ones((1, 4), dtype=jnp.float32),
        control_mask=jnp.ones((1, 2), dtype=jnp.float32),
        disturbance_mask=jnp.ones((1, 2), dtype=jnp.float32),
        param_mask=jnp.ones((1, 6), dtype=jnp.float32),
        state_dim=jnp.asarray([4], dtype=jnp.int32),
        control_dim=jnp.asarray([2], dtype=jnp.int32),
        disturbance_dim=jnp.asarray([2], dtype=jnp.int32),
        param_dim=jnp.asarray([6], dtype=jnp.int32),
        system_descriptor=jnp.zeros((1, 25), dtype=jnp.float32),
        state_group_kind_names=("concentration", "temperature"),
        state_group_mask=jnp.asarray(
            [[[1, 1, 0, 0], [0, 0, 1, 1]]],
            dtype=jnp.float32,
        ),
        state_group_active=jnp.asarray([[1, 1]], dtype=jnp.float32),
        state_group_kind_id=jnp.asarray([[0, 1]], dtype=jnp.int32),
        state_role_names=("concentration", "temperature"),
        state_role_id=jnp.asarray([[0, 0, 1, 1]], dtype=jnp.int32),
        state_lower_bound=jnp.asarray([[0.0, 0.0, 250.0, 250.0]], dtype=jnp.float32),
        state_upper_bound=jnp.asarray([[jnp.inf, jnp.inf, 400.0, 400.0]], dtype=jnp.float32),
    )


def _build_storage_metadata() -> UniversalSystemMetadata:
    return UniversalSystemMetadata(
        system_names=("storage_tank",),
        state_center=jnp.zeros((1, 3), dtype=jnp.float32),
        state_scale=jnp.ones((1, 3), dtype=jnp.float32),
        control_center=jnp.zeros((1, 1), dtype=jnp.float32),
        control_scale=jnp.ones((1, 1), dtype=jnp.float32),
        disturbance_center=jnp.zeros((1, 1), dtype=jnp.float32),
        disturbance_scale=jnp.ones((1, 1), dtype=jnp.float32),
        param_scale=jnp.ones((1, 2), dtype=jnp.float32),
        state_mask=jnp.ones((1, 3), dtype=jnp.float32),
        control_mask=jnp.ones((1, 1), dtype=jnp.float32),
        disturbance_mask=jnp.ones((1, 1), dtype=jnp.float32),
        param_mask=jnp.ones((1, 2), dtype=jnp.float32),
        state_dim=jnp.asarray([3], dtype=jnp.int32),
        control_dim=jnp.asarray([1], dtype=jnp.int32),
        disturbance_dim=jnp.asarray([1], dtype=jnp.int32),
        param_dim=jnp.asarray([2], dtype=jnp.int32),
        system_descriptor=jnp.zeros((1, 17), dtype=jnp.float32),
        state_group_kind_names=("inventory", "concentration", "thermal"),
        state_group_mask=jnp.asarray(
            [[[1, 0, 0], [0, 1, 0], [0, 0, 1]]],
            dtype=jnp.float32,
        ),
        state_group_active=jnp.asarray([[1, 1, 1]], dtype=jnp.float32),
        state_group_kind_id=jnp.asarray([[0, 1, 2]], dtype=jnp.int32),
        state_role_names=("inventory", "concentration", "temperature"),
        state_role_id=jnp.asarray([[0, 1, 2]], dtype=jnp.int32),
        state_lower_bound=jnp.asarray([[0.0, 0.0, 250.0]], dtype=jnp.float32),
        state_upper_bound=jnp.asarray([[10.0, 1.0, 400.0]], dtype=jnp.float32),
    )


def _build_config() -> dict:
    return {
        "model": {
            "latent_dim": 16,
            "shared_hidden_dim": 32,
            "system_embedding_dim": 8,
            "state_group_token_dim": 12,
            "state_group_kind_dim": 6,
            "state_group_encoder_layers": 2,
            "state_group_coupling_layers": 2,
            "encoder_layers": 2,
            "decoder_layers": 2,
            "drift_layers": 2,
            "use_system_spec_embedding": True,
            "use_variational_encoder": True,
            "neural_cde": {"enabled": True, "hidden_dim": 16, "n_layers": 2},
        }
    }


def test_universal_trainer_exposes_multi_horizon_loss_terms():
    metadata = _build_metadata()
    config = _build_config()
    config["optimizer"] = {
        "peak_lr": 1e-3,
        "end_lr": 1e-4,
        "warmup_steps": 1,
        "total_steps": 4,
        "gradient_clip": 1.0,
    }
    config["training"] = {"batch_size": 2, "seq_len": 5, "stride": 1}
    config["loss_weights"] = {
        "reconstruction": 1.0,
        "trajectory": 1.0,
        "one_step": 0.5,
        "k_step": 0.5,
        "kl": 1e-4,
        "state_bounds": 0.0,
        "positivity": 0.0,
    }
    config["multi_horizon"] = {"k_steps": [2, 3]}

    model = UniversalDigitalTwin.from_config(config, metadata, jax.random.PRNGKey(0))
    dataset = _DummyDataset(metadata)
    trainer = UniversalTrainer(model, config, dataset, dataset)

    batch = {
        "states": jnp.asarray(
            [
                [
                    [0.2, 0.1, 300.0, 290.0],
                    [0.22, 0.11, 301.0, 291.0],
                    [0.24, 0.12, 302.0, 292.0],
                    [0.25, 0.13, 303.0, 293.0],
                    [0.26, 0.14, 304.0, 294.0],
                ],
                [
                    [0.3, 0.2, 310.0, 295.0],
                    [0.31, 0.21, 311.0, 296.0],
                    [0.32, 0.22, 312.0, 297.0],
                    [0.33, 0.23, 313.0, 298.0],
                    [0.34, 0.24, 314.0, 299.0],
                ],
            ],
            dtype=jnp.float32,
        ),
        "controls": jnp.ones((2, 5, 2), dtype=jnp.float32),
        "disturbances": jnp.ones((2, 5, 2), dtype=jnp.float32),
        "params": jnp.ones((2, 6), dtype=jnp.float32),
        "t": jnp.tile(jnp.linspace(0.0, 0.4, 5, dtype=jnp.float32), (2, 1)),
        "state_mask": jnp.ones((2, 4), dtype=bool),
        "control_mask": jnp.ones((2, 2), dtype=bool),
        "disturbance_mask": jnp.ones((2, 2), dtype=bool),
        "param_mask": jnp.ones((2, 6), dtype=bool),
        "time_mask": jnp.ones((2, 5), dtype=bool),
        "system_id": jnp.zeros((2,), dtype=jnp.int32),
    }

    total_loss, loss_dict = trainer.compute_loss(model, batch, jax.random.PRNGKey(1))

    assert jnp.isfinite(total_loss)
    assert "k_step" in loss_dict
    assert "k_step_2" in loss_dict
    assert "k_step_3" in loss_dict


def test_universal_trainer_exposes_targeted_storage_dynamics_loss_term():
    metadata = _build_storage_metadata()
    config = _build_config()
    config["optimizer"] = {
        "peak_lr": 1e-3,
        "end_lr": 1e-4,
        "warmup_steps": 1,
        "total_steps": 4,
        "gradient_clip": 1.0,
    }
    config["training"] = {"batch_size": 2, "seq_len": 5, "stride": 1}
    config["loss_weights"] = {
        "reconstruction": 1.0,
        "trajectory": 1.0,
        "one_step": 0.5,
        "k_step": 0.0,
        "kl": 1e-4,
        "state_bounds": 0.0,
        "positivity": 0.0,
    }
    config["system_specific_losses"] = {
        "role_derivative_terms": [
            {
                "name": "storage_inventory_dynamics",
                "systems": ["storage_tank"],
                "state_role": "inventory",
                "weight": 0.5,
            }
        ]
    }

    model = UniversalDigitalTwin.from_config(config, metadata, jax.random.PRNGKey(2))
    dataset = _DummyDataset(metadata)
    trainer = UniversalTrainer(model, config, dataset, dataset)

    batch = {
        "states": jnp.asarray(
            [
                [
                    [1.0, 0.4, 320.0],
                    [1.1, 0.41, 321.0],
                    [1.25, 0.43, 322.0],
                    [1.35, 0.44, 323.0],
                    [1.5, 0.46, 324.0],
                ],
                [
                    [0.9, 0.5, 318.0],
                    [1.0, 0.49, 319.0],
                    [1.1, 0.48, 320.0],
                    [1.2, 0.47, 321.0],
                    [1.3, 0.46, 322.0],
                ],
            ],
            dtype=jnp.float32,
        ),
        "controls": jnp.ones((2, 5, 1), dtype=jnp.float32),
        "disturbances": jnp.ones((2, 5, 1), dtype=jnp.float32),
        "params": jnp.ones((2, 2), dtype=jnp.float32),
        "t": jnp.tile(jnp.linspace(0.0, 0.4, 5, dtype=jnp.float32), (2, 1)),
        "state_mask": jnp.ones((2, 3), dtype=bool),
        "control_mask": jnp.ones((2, 1), dtype=bool),
        "disturbance_mask": jnp.ones((2, 1), dtype=bool),
        "param_mask": jnp.ones((2, 2), dtype=bool),
        "time_mask": jnp.ones((2, 5), dtype=bool),
        "system_id": jnp.zeros((2,), dtype=jnp.int32),
    }

    trainer_without_term = UniversalTrainer(
        model,
        {key: value for key, value in config.items() if key != "system_specific_losses"},
        dataset,
        dataset,
    )
    total_without_term, _ = trainer_without_term.compute_loss(
        model,
        batch,
        jax.random.PRNGKey(3),
    )
    total_with_term, loss_dict = trainer.compute_loss(model, batch, jax.random.PRNGKey(3))

    assert "storage_inventory_dynamics" in loss_dict
    assert jnp.isfinite(loss_dict["storage_inventory_dynamics"])
    assert loss_dict["storage_inventory_dynamics"] >= 0.0
    assert total_with_term > total_without_term
