"""Tests for the full Digital Twin model."""

import jax
import jax.numpy as jnp
import equinox as eqx
import pytest
import yaml
import tempfile
import os

from dte.models.unit.digital_twin import DigitalTwin
from dte.simulators.registry import get_system_spec


def load_config():
    """Load training configuration."""
    with open("configs/training_default.yaml", "r") as f:
        return yaml.safe_load(f)


def load_system_spec():
    """Load the default CSTR system spec used by integration tests."""
    with open("configs/cstr_default.yaml", "r") as f:
        return get_system_spec(yaml.safe_load(f))


def build_model(config, key):
    """Construct a model with the default system spec."""
    return DigitalTwin.from_config(config, key, system_spec=load_system_spec())


def test_from_config():
    """Test 1: DigitalTwin.from_config creates model without error."""
    config = load_config()
    key = jax.random.PRNGKey(0)
    
    model = build_model(config, key)
    
    # Check that all components exist
    assert model.encoder is not None
    assert model.decoder is not None
    assert model.latent_sde is not None
    
    # Check parameter counts
    param_counts = model.get_parameter_count()
    print(f"\nParameter counts: {param_counts}")
    assert param_counts["total"] > 0


def test_predict_shapes():
    """Test 2: predict() returns correct shapes."""
    config = load_config()
    latent_dim = config["model"]["latent_dim"]
    key = jax.random.PRNGKey(1)
    
    model = build_model(config, key)
    
    # Create test inputs
    n_steps = 50
    initial_state = jnp.array([0.5, 0.5, 350.0, 300.0])
    controls = jax.random.uniform(key, shape=(n_steps, 2))
    disturbances = jax.random.uniform(key, shape=(n_steps, 2))
    params = jnp.ones(6)
    ts = jnp.linspace(0.0, 5.0, n_steps)
    
    # Predict
    result = model.predict(initial_state, controls, disturbances, params, ts, key)
    
    # Check shapes
    assert result["states"].shape == (n_steps, 4)
    assert result["latent"].shape == (n_steps, latent_dim)
    assert result["z_mean"].shape == (latent_dim,)
    assert result["z_logvar"].shape == (latent_dim,)


def test_predict_ensemble_shapes():
    """Test 3: predict_ensemble() returns correct shapes."""
    config = load_config()
    key = jax.random.PRNGKey(2)
    
    model = build_model(config, key)
    
    # Create test inputs
    n_steps = 50
    n_samples = 10
    initial_state = jnp.array([0.5, 0.5, 350.0, 300.0])
    controls = jax.random.uniform(key, shape=(n_steps, 2))
    disturbances = jax.random.uniform(key, shape=(n_steps, 2))
    params = jnp.ones(6)
    ts = jnp.linspace(0.0, 5.0, n_steps)
    
    # Predict ensemble
    result = model.predict_ensemble(
        initial_state, controls, disturbances, params, ts, key, n_samples=n_samples
    )
    
    # Check shapes
    assert result["states_mean"].shape == (n_steps, 4)
    assert result["states_std"].shape == (n_steps, 4)
    assert result["states_samples"].shape == (n_samples, n_steps, 4)
    
    # Check that std is positive
    assert jnp.all(result["states_std"] >= 0)


def test_save_load():
    """Test 4: save and load roundtrip preserves predictions."""
    config = load_config()
    system_spec = load_system_spec()
    key = jax.random.PRNGKey(3)
    
    # Create model
    model = DigitalTwin.from_config(config, key, system_spec=system_spec)
    
    # Create test inputs
    n_steps = 30
    initial_state = jnp.array([0.5, 0.5, 350.0, 300.0])
    controls = jax.random.uniform(key, shape=(n_steps, 2))
    disturbances = jax.random.uniform(key, shape=(n_steps, 2))
    params = jnp.ones(6)
    ts = jnp.linspace(0.0, 3.0, n_steps)
    
    # Get prediction before save
    key_pred1, key_pred2 = jax.random.split(key)
    result1 = model.predict(initial_state, controls, disturbances, params, ts, key_pred1)
    
    # Save model
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "test_model.eqx")
        model.save(model_path)
        
        # Load model
        loaded_model = DigitalTwin.load(model_path, config, system_spec=system_spec)
        
        # Get prediction after load (use same key for deterministic comparison)
        result2 = loaded_model.predict(initial_state, controls, disturbances, params, ts, key_pred1)
        
        # Results should be identical
        assert jnp.allclose(result1["states"], result2["states"], atol=1e-5)
        assert jnp.allclose(result1["latent"], result2["latent"], atol=1e-5)


def test_parameter_count():
    """Test 5: Total parameter count is reasonable."""
    config = load_config()
    key = jax.random.PRNGKey(4)
    
    model = build_model(config, key)
    param_counts = model.get_parameter_count()
    
    total = param_counts["total"]
    
    # Check reasonable range (200K-1M based on plan, but we have ~200K)
    assert 100_000 <= total <= 1_000_000, f"Total params {total} outside expected range"
    
    # Check that components have reasonable proportions
    assert param_counts["encoder"] > 0
    assert param_counts["decoder"] > 0
    assert param_counts["latent_sde"] > 0


def test_encode_decode_consistency():
    """Test that encode-decode maintains reasonable state ranges."""
    config = load_config()
    key = jax.random.PRNGKey(5)
    
    model = build_model(config, key)
    
    # Create test state in reasonable range
    state = jnp.array([0.5, 0.5, 350.0, 300.0])
    params = jnp.ones(6)
    control = jnp.array([50.0, 300.0])
    
    # Encode
    z, z_mean, z_logvar = model.encode(state, params, control, key)
    
    # Decode
    reconstructed_state = model.decode(z, params, control)
    
    # Check output constraints (from decoder)
    Ca, Cb, T, Tc = reconstructed_state
    assert Ca >= 0, "Ca must be non-negative"
    assert Cb >= 0, "Cb must be non-negative"
    assert 200.0 <= T <= 500.0, "T must be in valid range"
    assert 200.0 <= Tc <= 500.0, "Tc must be in valid range"


def test_jit_compatibility():
    """Test that the model is JIT-compatible."""
    config = load_config()
    key = jax.random.PRNGKey(6)
    
    model = build_model(config, key)
    
    # JIT the predict function
    @jax.jit
    def predict_jit(initial_state, controls, disturbances, params, ts, key):
        return model.predict(initial_state, controls, disturbances, params, ts, key)
    
    # Create test inputs
    n_steps = 20
    initial_state = jnp.array([0.5, 0.5, 350.0, 300.0])
    controls = jax.random.uniform(key, shape=(n_steps, 2))
    disturbances = jax.random.uniform(key, shape=(n_steps, 2))
    params = jnp.ones(6)
    ts = jnp.linspace(0.0, 2.0, n_steps)
    
    # Run once to compile
    result1 = predict_jit(initial_state, controls, disturbances, params, ts, key)
    
    # Run again - should be fast
    result2 = predict_jit(initial_state, controls, disturbances, params, ts, key)
    
    # Should be identical (same key)
    assert jnp.allclose(result1["states"], result2["states"], atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
