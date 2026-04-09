"""Tests for the grouped universal digital twin path."""

import jax
import jax.numpy as jnp

from dte.data.multi_system_dataset import UniversalSystemMetadata
from dte.models.universal_digital_twin import UniversalDigitalTwin


def _build_metadata() -> UniversalSystemMetadata:
    return UniversalSystemMetadata(
        system_names=("cstr", "heat_exchanger"),
        state_center=jnp.zeros((2, 4), dtype=jnp.float32),
        state_scale=jnp.ones((2, 4), dtype=jnp.float32),
        control_center=jnp.zeros((2, 2), dtype=jnp.float32),
        control_scale=jnp.ones((2, 2), dtype=jnp.float32),
        disturbance_center=jnp.zeros((2, 2), dtype=jnp.float32),
        disturbance_scale=jnp.ones((2, 2), dtype=jnp.float32),
        param_scale=jnp.ones((2, 6), dtype=jnp.float32),
        state_mask=jnp.asarray([[1, 1, 1, 1], [1, 1, 0, 0]], dtype=jnp.float32),
        control_mask=jnp.ones((2, 2), dtype=jnp.float32),
        disturbance_mask=jnp.ones((2, 2), dtype=jnp.float32),
        param_mask=jnp.asarray([[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 0]], dtype=jnp.float32),
        state_dim=jnp.asarray([4, 2], dtype=jnp.int32),
        control_dim=jnp.asarray([2, 2], dtype=jnp.int32),
        disturbance_dim=jnp.asarray([2, 2], dtype=jnp.int32),
        param_dim=jnp.asarray([6, 5], dtype=jnp.int32),
        system_descriptor=jnp.zeros((2, 25), dtype=jnp.float32),
        state_group_kind_names=("concentration", "thermal", "inventory"),
        state_group_mask=jnp.asarray(
            [
                [[1, 1, 0, 0], [0, 0, 1, 1]],
                [[1, 1, 0, 0], [0, 0, 0, 0]],
            ],
            dtype=jnp.float32,
        ),
        state_group_active=jnp.asarray([[1, 1], [1, 0]], dtype=jnp.float32),
        state_group_kind_id=jnp.asarray([[0, 1], [1, 0]], dtype=jnp.int32),
        state_role_names=("concentration", "temperature", "inventory"),
        state_role_id=jnp.asarray(
            [
                [0, 0, 1, 1],
                [1, 1, 2, 2],
            ],
            dtype=jnp.int32,
        ),
        state_lower_bound=jnp.asarray(
            [
                [0.0, 0.0, 250.0, 250.0],
                [250.0, 250.0, -jnp.inf, -jnp.inf],
            ],
            dtype=jnp.float32,
        ),
        state_upper_bound=jnp.asarray(
            [
                [jnp.inf, jnp.inf, 400.0, 400.0],
                [450.0, 450.0, jnp.inf, jnp.inf],
            ],
            dtype=jnp.float32,
        ),
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


def test_universal_model_grouped_encode_decode_shapes():
    metadata = _build_metadata()
    model = UniversalDigitalTwin.from_config(_build_config(), metadata, jax.random.PRNGKey(0))

    state_norm = jnp.array([0.5, -0.2, 0.1, 0.7], dtype=jnp.float32)
    params_scaled = jnp.ones((6,), dtype=jnp.float32)
    control_norm = jnp.array([0.2, -0.4], dtype=jnp.float32)
    state_mask = jnp.array([1.0, 1.0, 1.0, 1.0], dtype=jnp.float32)
    control_mask = jnp.array([1.0, 1.0], dtype=jnp.float32)
    param_mask = jnp.ones((6,), dtype=jnp.float32)

    z, z_mean, z_logvar = model.encode(
        state_norm,
        params_scaled,
        control_norm,
        state_mask,
        control_mask,
        param_mask,
        jnp.asarray(0, dtype=jnp.int32),
        jax.random.PRNGKey(1),
    )
    decoded = model.decode(
        z,
        params_scaled,
        control_norm,
        state_mask,
        control_mask,
        param_mask,
        jnp.asarray(0, dtype=jnp.int32),
    )

    assert z.shape == (16,)
    assert z_mean.shape == (16,)
    assert z_logvar.shape == (16,)
    assert decoded.shape == (4,)


def test_universal_model_respects_inactive_state_dimensions():
    metadata = _build_metadata()
    model = UniversalDigitalTwin.from_config(_build_config(), metadata, jax.random.PRNGKey(2))

    state_norm = jnp.array([0.3, -0.1, 0.0, 0.0], dtype=jnp.float32)
    params_scaled = jnp.array([1.0, 0.5, -0.2, 0.1, 0.0, 0.0], dtype=jnp.float32)
    control_norm = jnp.array([0.2, 0.4], dtype=jnp.float32)
    state_mask = jnp.array([1.0, 1.0, 0.0, 0.0], dtype=jnp.float32)
    control_mask = jnp.array([1.0, 1.0], dtype=jnp.float32)
    param_mask = jnp.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.0], dtype=jnp.float32)
    system_id = jnp.asarray(1, dtype=jnp.int32)

    z, _, _ = model.encode(
        state_norm,
        params_scaled,
        control_norm,
        state_mask,
        control_mask,
        param_mask,
        system_id,
        None,
    )
    decoded = model.decode(
        z,
        params_scaled,
        control_norm,
        state_mask,
        control_mask,
        param_mask,
        system_id,
    )

    assert decoded.shape == (4,)
    assert jnp.allclose(decoded[2:], 0.0)
