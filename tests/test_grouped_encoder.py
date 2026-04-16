"""Tests for the optional grouped encoder path in the single-system model."""

import jax
import jax.numpy as jnp
import yaml

from dte.models.unit.digital_twin import DigitalTwin
from dte.models.unit.grouped_encoder import GroupedStateEncoder
from dte.simulators.registry import get_system_spec


def _load_training_config() -> dict:
    with open("configs/training_default.yaml", "r") as f:
        config = yaml.safe_load(f)
    config["model"]["grouped_encoder"] = {
        "enabled": True,
        "group_token_dim": 48,
        "group_kind_dim": 8,
        "group_encoder_layers": 2,
        "group_mixer_layers": 2,
    }
    config["model"]["channel_conditioning"] = {"enabled": True}
    config["model"]["law_conditioning"] = {"enabled": True}
    return config


def test_grouped_encoder_can_drive_digital_twin_predict():
    with open("configs/cstr_default.yaml", "r") as f:
        system_spec = get_system_spec(yaml.safe_load(f))

    model = DigitalTwin.from_config(
        _load_training_config(),
        jax.random.PRNGKey(0),
        system_spec=system_spec,
    )

    assert isinstance(model.encoder, GroupedStateEncoder)

    n_steps = 10
    initial_state = jnp.array([0.5, 0.25, 340.0, 300.0], dtype=jnp.float32)
    controls = jnp.ones((n_steps, 2), dtype=jnp.float32) * jnp.array([50.0, 300.0], dtype=jnp.float32)
    disturbances = jnp.ones((n_steps, 2), dtype=jnp.float32) * jnp.array([1.0, 320.0], dtype=jnp.float32)
    params = jnp.ones((6,), dtype=jnp.float32)
    ts = jnp.linspace(0.0, 1.0, n_steps)

    result = model.predict(initial_state, controls, disturbances, params, ts, jax.random.PRNGKey(1))

    assert result["states"].shape == (n_steps, 4)
    assert result["latent"].shape[0] == n_steps
