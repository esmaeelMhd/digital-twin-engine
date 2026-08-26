"""Tests for Latent SDE module."""

import jax
import jax.numpy as jnp
import pytest

from dte.models.unit.latent_sde import LatentSDE


def test_forward_pass_shape():
    """Test 1: LatentSDE forward pass produces correct output shape."""
    key = jax.random.PRNGKey(0)
    key_init, key_solve = jax.random.split(key)
    
    latent_sde = LatentSDE(
        latent_dim=16,
        control_dim=2,
        param_dim=6,
        hidden_dim=128,
        drift_layers=3,
        diffusion_layers=2,
        key=key_init,
    )
    
    # Create test inputs
    n_steps = 50
    ts = jnp.linspace(0.0, 5.0, n_steps)
    z0 = jax.random.normal(key, shape=(16,))
    controls = jax.random.uniform(key, shape=(n_steps, 2))
    params = jnp.ones(6)
    
    # Solve SDE
    z_traj = latent_sde(ts, z0, controls, params, key_solve)
    
    assert z_traj.shape == (n_steps, 16), f"Expected shape ({n_steps}, 16), got {z_traj.shape}"


def test_sample_trajectories_shape():
    """Test 2: sample_trajectories with n_samples=5 produces correct shape."""
    key = jax.random.PRNGKey(1)
    key_init, key_solve = jax.random.split(key)
    
    latent_sde = LatentSDE(
        latent_dim=16,
        control_dim=2,
        param_dim=6,
        key=key_init,
    )
    
    # Create test inputs
    n_steps = 50
    n_samples = 5
    ts = jnp.linspace(0.0, 5.0, n_steps)
    z0 = jax.random.normal(key, shape=(16,))
    controls = jax.random.uniform(key, shape=(n_steps, 2))
    params = jnp.ones(6)
    
    # Sample multiple trajectories
    z_trajs = latent_sde.sample_trajectories(
        ts, z0, controls, params, key_solve, n_samples=n_samples
    )
    
    expected_shape = (n_samples, n_steps, 16)
    assert z_trajs.shape == expected_shape, f"Expected shape {expected_shape}, got {z_trajs.shape}"


def test_mean_trajectory_deterministic():
    """Test 3: mean_trajectory is deterministic."""
    key = jax.random.PRNGKey(2)
    key_init, key1, key2 = jax.random.split(key, 3)
    
    latent_sde = LatentSDE(
        latent_dim=16,
        control_dim=2,
        param_dim=6,
        key=key_init,
    )
    
    # Create test inputs
    n_steps = 50
    ts = jnp.linspace(0.0, 5.0, n_steps)
    z0 = jax.random.normal(key, shape=(16,))
    controls = jax.random.uniform(key, shape=(n_steps, 2))
    params = jnp.ones(6)
    
    # Run twice
    z_traj1 = latent_sde.mean_trajectory(ts, z0, controls, params)
    z_traj2 = latent_sde.mean_trajectory(ts, z0, controls, params)
    
    # Should be identical
    assert jnp.allclose(z_traj1, z_traj2, atol=1e-5), "Mean trajectory should be deterministic"


def test_sample_trajectories_stochastic():
    """Test 4: sample_trajectories is stochastic with different keys."""
    key = jax.random.PRNGKey(3)
    key_init, key1, key2 = jax.random.split(key, 3)
    
    latent_sde = LatentSDE(
        latent_dim=16,
        control_dim=2,
        param_dim=6,
        initial_diffusion_scale=0.5,  # Higher noise to ensure difference
        key=key_init,
    )
    
    # Create test inputs
    n_steps = 50
    ts = jnp.linspace(0.0, 5.0, n_steps)
    z0 = jax.random.normal(key, shape=(16,))
    controls = jax.random.uniform(key, shape=(n_steps, 2))
    params = jnp.ones(6)
    
    # Run with different keys
    z_traj1 = latent_sde(ts, z0, controls, params, key1)
    z_traj2 = latent_sde(ts, z0, controls, params, key2)
    
    # Should be different (with high probability)
    diff = jnp.abs(z_traj1 - z_traj2).max()
    assert diff > 1e-4, f"Trajectories should differ with different keys, max diff: {diff}"


def test_gradients_flow():
    """Test 5: Gradients flow through the SDE solve."""
    key = jax.random.PRNGKey(4)
    key_init, key_solve = jax.random.split(key)
    
    latent_sde = LatentSDE(
        latent_dim=16,
        control_dim=2,
        param_dim=6,
        hidden_dim=64,  # Smaller for faster test
        key=key_init,
    )
    
    # Create test inputs
    n_steps = 20  # Fewer steps for faster test
    ts = jnp.linspace(0.0, 2.0, n_steps)
    z0 = jax.random.normal(key, shape=(16,))
    controls = jax.random.uniform(key, shape=(n_steps, 2))
    params = jnp.ones(6)
    
    # Define loss function
    def loss_fn(model):
        z_traj = model.mean_trajectory(ts, z0, controls, params)
        return jnp.sum(z_traj**2)
    
    # Compute gradients
    loss, grads = eqx.filter_value_and_grad(loss_fn)(latent_sde)
    
    # Check that gradients exist and are non-zero
    grad_leaves = jax.tree.leaves(eqx.filter(grads, eqx.is_array))
    assert len(grad_leaves) > 0, "Should have gradients"
    
    non_zero_grads = sum(jnp.abs(g).sum() > 1e-6 for g in grad_leaves)
    assert non_zero_grads > 0, "At least some gradients should be non-zero"


def test_jit_compilation():
    """Test 6: JIT compilation works."""
    key = jax.random.PRNGKey(5)
    key_init, key_solve = jax.random.split(key)
    
    latent_sde = LatentSDE(
        latent_dim=16,
        control_dim=2,
        param_dim=6,
        key=key_init,
    )
    
    # JIT the forward pass
    @jax.jit
    def solve_jit(ts, z0, controls, params, key):
        return latent_sde.mean_trajectory(ts, z0, controls, params)
    
    # Create test inputs
    n_steps = 30
    ts = jnp.linspace(0.0, 3.0, n_steps)
    z0 = jax.random.normal(key, shape=(16,))
    controls = jax.random.uniform(key, shape=(n_steps, 2))
    params = jnp.ones(6)
    
    # Run once to compile
    z_traj1 = solve_jit(ts, z0, controls, params, key_solve)
    
    # Run again
    z_traj2 = solve_jit(ts, z0, controls, params, key_solve)
    
    # Should be identical
    assert jnp.allclose(z_traj1, z_traj2, atol=1e-5)


# Need to import equinox for gradient test
import equinox as eqx


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
