"""Tests for OnlineAdapter freezing and part-specific fine-tuning."""

import jax
import jax.numpy as jnp
import numpy as np
import yaml

from dte.models.unit.digital_twin import DigitalTwin
from dte.simulators.registry import get_system_spec
from dte.training.online import OnlineAdapter, OnlineAdapterConfig


def _load_cstr_spec():
    with open("configs/cstr_default.yaml", "r", encoding="utf-8") as handle:
        return get_system_spec(yaml.safe_load(handle))


def _tiny_training_config() -> dict:
    with open("configs/training_default.yaml", "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["model"].update(
        {
            "latent_dim": 8,
            "hidden_dim": 16,
            "n_layers": 1,
            "drift_layers": 1,
            "diffusion_layers": 1,
            "diffusion_hidden_dim": 8,
        }
    )
    return config


def _fill_buffer(adapter: OnlineAdapter, spec, n: int):
    state = np.asarray(spec.default_initial_state, dtype=np.float32)
    control = np.array(
        [0.5 * sum(spec.control_ranges[name]) for name in spec.control_names],
        dtype=np.float32,
    )
    disturbance = np.asarray(spec.default_nominal_disturbance, dtype=np.float32)
    for i in range(n):
        adapter._buffer.push(
            states=state + 0.01 * i,
            controls=control,
            disturbances=disturbance,
            t=0.1 * i,
        )


def test_online_adapter_freezes_normalization_tables():
    spec = _load_cstr_spec()
    model = DigitalTwin.from_config(
        _tiny_training_config(), jax.random.PRNGKey(0), system_spec=spec
    )
    cfg = OnlineAdapterConfig(
        window_size=16,
        seq_len=4,
        n_finetune_steps=1,
        learning_rate=1e-3,
        finetune_every=10_000,
    )
    adapter = OnlineAdapter(model, spec, cfg, key=jax.random.PRNGKey(1))
    before_center = jnp.array(adapter.model.encoder.state_center)
    before_scale = jnp.array(adapter.model.encoder.state_scale)
    before_nom = jnp.array(adapter.model.latent_sde.nominal_disturbance)

    _fill_buffer(adapter, spec, cfg.seq_len + 2)
    adapter._finetune()

    assert jnp.allclose(adapter.model.encoder.state_center, before_center)
    assert jnp.allclose(adapter.model.encoder.state_scale, before_scale)
    assert jnp.allclose(adapter.model.latent_sde.nominal_disturbance, before_nom)


def test_online_adapter_encoder_only_leaves_decoder_frozen():
    spec = _load_cstr_spec()
    model = DigitalTwin.from_config(
        _tiny_training_config(), jax.random.PRNGKey(0), system_spec=spec
    )
    cfg = OnlineAdapterConfig(
        window_size=16,
        seq_len=4,
        n_finetune_steps=1,
        learning_rate=1e-3,
        finetune_every=10_000,
        finetune_encoder_only=True,
    )
    adapter = OnlineAdapter(model, spec, cfg, key=jax.random.PRNGKey(1))
    before_dec = jnp.array(adapter.model.decoder.output_layer.weight)
    before_enc = jnp.array(adapter.model.encoder.mean_layer.weight)

    _fill_buffer(adapter, spec, cfg.seq_len + 2)
    adapter._finetune()

    assert jnp.allclose(adapter.model.decoder.output_layer.weight, before_dec)
    assert not jnp.allclose(adapter.model.encoder.mean_layer.weight, before_enc)
