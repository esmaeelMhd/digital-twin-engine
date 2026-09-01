"""Tests for latent rollout through DigitalTwin.rollout_latent."""

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
import yaml

from dte.models.unit.digital_twin import DigitalTwin
from dte.simulators.registry import get_system_spec


def _load_config():
    with open("configs/training_default.yaml", "r") as handle:
        return yaml.safe_load(handle)


def _load_system_spec():
    with open("configs/cstr_default.yaml", "r") as handle:
        return get_system_spec(yaml.safe_load(handle))


def _build_model(key):
    return DigitalTwin.from_config(_load_config(), key, system_spec=_load_system_spec())


def _rollout_inputs(model, key, n_steps=20):
    ts = jnp.linspace(0.0, 2.0, n_steps)
    z0 = jax.random.normal(key, shape=(model.latent_sde.latent_dim,))
    controls = jax.random.uniform(key, shape=(n_steps, 2))
    disturbances = jax.random.uniform(key, shape=(n_steps, 2))
    params = jnp.ones(6)
    return ts, z0, controls, disturbances, params


def test_rollout_latent_shape():
    key = jax.random.PRNGKey(0)
    model = _build_model(key)
    ts, z0, controls, disturbances, params = _rollout_inputs(model, key)

    z_traj = model.rollout_latent(
        ts,
        z0,
        controls,
        params,
        disturbances=disturbances,
        stochastic=False,
    )

    assert z_traj.shape == (ts.shape[0], model.latent_sde.latent_dim)


def test_rollout_latent_deterministic():
    key = jax.random.PRNGKey(1)
    model = _build_model(key)
    ts, z0, controls, disturbances, params = _rollout_inputs(model, key)

    z_traj1 = model.rollout_latent(
        ts, z0, controls, params, disturbances=disturbances, stochastic=False
    )
    z_traj2 = model.rollout_latent(
        ts, z0, controls, params, disturbances=disturbances, stochastic=False
    )

    assert jnp.allclose(z_traj1, z_traj2, atol=1e-5)


def test_rollout_latent_stochastic_key_sensitivity():
    key = jax.random.PRNGKey(2)
    key_init, key1, key2 = jax.random.split(key, 3)
    model = _build_model(key_init)
    ts, z0, controls, disturbances, params = _rollout_inputs(model, key)

    z_traj1 = model.rollout_latent(
        ts,
        z0,
        controls,
        params,
        disturbances=disturbances,
        key=key1,
        stochastic=True,
    )
    z_traj2 = model.rollout_latent(
        ts,
        z0,
        controls,
        params,
        disturbances=disturbances,
        key=key2,
        stochastic=True,
    )

    diff = jnp.abs(z_traj1 - z_traj2).max()
    assert diff > 1e-4, f"Stochastic rollouts should differ with different keys, max diff: {diff}"


def test_rollout_latent_gradients_flow():
    key = jax.random.PRNGKey(3)
    model = _build_model(key)
    ts, z0, controls, disturbances, params = _rollout_inputs(model, key, n_steps=12)

    def loss_fn(current_model):
        z_traj = current_model.rollout_latent(
            ts, z0, controls, params, disturbances=disturbances, stochastic=False
        )
        return jnp.sum(z_traj**2)

    _loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
    grad_leaves = jax.tree.leaves(eqx.filter(grads, eqx.is_array))
    assert len(grad_leaves) > 0
    assert sum(jnp.abs(grad).sum() > 1e-6 for grad in grad_leaves) > 0


def test_rollout_latent_jit():
    key = jax.random.PRNGKey(4)
    model = _build_model(key)
    ts, z0, controls, disturbances, params = _rollout_inputs(model, key, n_steps=16)

    @eqx.filter_jit
    def solve(current_model, ts_, z0_, controls_, params_, disturbances_):
        return current_model.rollout_latent(
            ts_,
            z0_,
            controls_,
            params_,
            disturbances=disturbances_,
            stochastic=False,
        )

    z_traj1 = solve(model, ts, z0, controls, params, disturbances)
    z_traj2 = solve(model, ts, z0, controls, params, disturbances)
    assert jnp.allclose(z_traj1, z_traj2, atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
