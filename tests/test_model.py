"""Tests for encoder and decoder models."""

import jax
import jax.numpy as jnp
import equinox as eqx
import pytest

from dte.models.encoder import Encoder
from dte.models.decoder import Decoder


def test_encoder_output_shapes():
    """Test 1: Encoder output shapes are correct."""
    key = jax.random.PRNGKey(0)
    key_init, key_encode = jax.random.split(key)
    
    encoder = Encoder(
        state_dim=4,
        param_dim=6,
        control_dim=2,
        latent_dim=16,
        hidden_dim=128,
        n_layers=3,
        key=key_init,
    )
    
    # Create dummy inputs
    state = jnp.array([0.5, 0.5, 350.0, 300.0])
    params = jnp.ones(6)
    control = jnp.array([50.0, 300.0])
    
    # Encode
    z_mean, z_logvar = encoder.encode(state, params, control)
    
    assert z_mean.shape == (16,), f"Expected shape (16,), got {z_mean.shape}"
    assert z_logvar.shape == (16,), f"Expected shape (16,), got {z_logvar.shape}"
    
    # Test batch encoding
    batch_size = 32
    states = jnp.tile(state[None, :], (batch_size, 1))
    params_batch = jnp.tile(params[None, :], (batch_size, 1))
    controls = jnp.tile(control[None, :], (batch_size, 1))
    
    encode_fn = jax.vmap(encoder.encode, in_axes=(0, 0, 0))
    z_means, z_logvars = encode_fn(states, params_batch, controls)
    
    assert z_means.shape == (batch_size, 16)
    assert z_logvars.shape == (batch_size, 16)


def test_decoder_output_shapes():
    """Test 2: Decoder output shapes are correct."""
    key = jax.random.PRNGKey(1)
    
    decoder = Decoder(
        latent_dim=16,
        param_dim=6,
        control_dim=2,
        state_dim=4,
        hidden_dim=128,
        n_layers=3,
        key=key,
    )
    
    # Create dummy inputs
    z = jax.random.normal(key, shape=(16,))
    params = jnp.ones(6)
    control = jnp.array([50.0, 300.0])
    
    # Decode
    state = decoder(z, params, control)
    
    assert state.shape == (4,), f"Expected shape (4,), got {state.shape}"
    
    # Test batch decoding
    batch_size = 32
    zs = jax.random.normal(key, shape=(batch_size, 16))
    params_batch = jnp.tile(params[None, :], (batch_size, 1))
    controls = jnp.tile(control[None, :], (batch_size, 1))
    
    decode_fn = jax.vmap(decoder, in_axes=(0, 0, 0))
    states = decode_fn(zs, params_batch, controls)
    
    assert states.shape == (batch_size, 4)


def test_decoder_constraints():
    """Test 3: Decoder outputs satisfy constraints."""
    key = jax.random.PRNGKey(2)
    constraints = [
        {"type": "softplus", "indices": [0, 1], "bias": 0.5},
        {"type": "sigmoid_range", "indices": [2, 3], "low": 250.0, "high": 400.0},
    ]
    
    decoder = Decoder(
        latent_dim=16,
        param_dim=6,
        control_dim=2,
        state_dim=4,
        hidden_dim=128,
        n_layers=3,
        constraints=constraints,
        key=key,
    )
    
    # Test with random latent vectors
    for i in range(100):
        z = jax.random.normal(jax.random.PRNGKey(i), shape=(16,)) * 3.0  # Large values
        params = jnp.ones(6)
        control = jnp.array([50.0, 300.0])
        
        state = decoder(z, params, control)
        
        Ca, Cb, T, Tc = state[0], state[1], state[2], state[3]
        
        # Check constraints
        assert Ca >= 0, f"Ca must be non-negative, got {Ca}"
        assert Cb >= 0, f"Cb must be non-negative, got {Cb}"
        assert 250.0 <= T <= 400.0, f"T must be in [250, 400], got {T}"
        assert 250.0 <= Tc <= 400.0, f"Tc must be in [250, 400], got {Tc}"


def test_encoder_decoder_roundtrip():
    """Test 4: Full encode->decode roundtrip has correct shapes."""
    key = jax.random.PRNGKey(3)
    key_enc, key_dec, key_sample = jax.random.split(key, 3)
    
    encoder = Encoder(
        state_dim=4,
        param_dim=6,
        control_dim=2,
        latent_dim=16,
        hidden_dim=128,
        n_layers=3,
        key=key_enc,
    )
    
    decoder = Decoder(
        latent_dim=16,
        param_dim=6,
        control_dim=2,
        state_dim=4,
        hidden_dim=128,
        n_layers=3,
        key=key_dec,
    )
    
    # Create input
    state = jnp.array([0.5, 0.5, 350.0, 300.0])
    params = jnp.ones(6)
    control = jnp.array([50.0, 300.0])
    
    # Encode
    z, z_mean, z_logvar = encoder(state, params, control, key_sample)
    
    # Decode
    state_recon = decoder(z, params, control)
    
    assert z.shape == (16,)
    assert state_recon.shape == (4,)


def test_jit_compatibility():
    """Test 5: Both modules are JIT-compatible."""
    key = jax.random.PRNGKey(4)
    key_enc, key_dec, key_sample = jax.random.split(key, 3)
    
    encoder = Encoder(
        state_dim=4,
        param_dim=6,
        control_dim=2,
        latent_dim=16,
        hidden_dim=128,
        n_layers=3,
        key=key_enc,
    )
    
    decoder = Decoder(
        latent_dim=16,
        param_dim=6,
        control_dim=2,
        state_dim=4,
        hidden_dim=128,
        n_layers=3,
        key=key_dec,
    )
    
    # JIT the functions
    @jax.jit
    def encode_jit(state, params, control, key):
        return encoder(state, params, control, key)
    
    @jax.jit
    def decode_jit(z, params, control):
        return decoder(z, params, control)
    
    # Test
    state = jnp.array([0.5, 0.5, 350.0, 300.0])
    params = jnp.ones(6)
    control = jnp.array([50.0, 300.0])
    
    z, z_mean, z_logvar = encode_jit(state, params, control, key_sample)
    state_recon = decode_jit(z, params, control)
    
    # Run again to verify JIT works
    z2, _, _ = encode_jit(state, params, control, key_sample)
    
    # With same key, should get same result
    assert jnp.allclose(z, z2)


def test_parameter_count():
    """Test 6: Parameter count is reasonable."""
    key = jax.random.PRNGKey(5)
    key_enc, key_dec = jax.random.split(key)
    
    encoder = Encoder(
        state_dim=4,
        param_dim=6,
        control_dim=2,
        latent_dim=16,
        hidden_dim=128,
        n_layers=3,
        key=key_enc,
    )
    
    decoder = Decoder(
        latent_dim=16,
        param_dim=6,
        control_dim=2,
        state_dim=4,
        hidden_dim=128,
        n_layers=3,
        key=key_dec,
    )
    
    # Count parameters
    def count_params(model):
        return sum(x.size for x in jax.tree.leaves(eqx.filter(model, eqx.is_array)))
    
    encoder_params = count_params(encoder)
    decoder_params = count_params(decoder)
    total_params = encoder_params + decoder_params
    
    print(f"\nEncoder parameters: {encoder_params:,}")
    print(f"Decoder parameters: {decoder_params:,}")
    print(f"Total parameters: {total_params:,}")
    
    # Check reasonable range (50K-500K) - model is lightweight!
    assert 50_000 <= total_params <= 500_000, f"Total params {total_params} outside expected range"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
