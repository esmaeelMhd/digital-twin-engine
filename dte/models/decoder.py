"""Decoder module for mapping latent states back to physical space."""

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray


class Decoder(eqx.Module):
    """
    Decodes latent state z back to physical state [Ca, Cb, T, Tc].
    Also conditioned on params and control for better reconstruction.
    """
    
    layers: list
    output_layer: eqx.nn.Linear
    
    def __init__(
        self,
        latent_dim: int = 16,
        param_dim: int = 6,
        control_dim: int = 2,
        state_dim: int = 4,
        hidden_dim: int = 128,
        n_layers: int = 3,
        *,
        key: PRNGKeyArray,
    ):
        """Initialize decoder.
        
        Args:
            latent_dim: Latent dimension
            param_dim: Parameter dimension
            control_dim: Control dimension
            state_dim: State dimension
            hidden_dim: Hidden layer dimension
            n_layers: Number of hidden layers
            key: PRNG key for initialization
        """
        keys = jax.random.split(key, n_layers + 1)
        
        # Input dimension
        input_dim = latent_dim + param_dim + control_dim
        
        # Hidden layers
        self.layers = []
        for i in range(n_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            self.layers.append(
                eqx.nn.Linear(in_dim, hidden_dim, key=keys[i])
            )
        
        # Output layer
        self.output_layer = eqx.nn.Linear(hidden_dim, state_dim, key=keys[-1])
    
    def __call__(
        self,
        z: Float[Array, "latent_dim"],
        params: Float[Array, "param_dim"],
        control: Float[Array, "control_dim"],
    ) -> Float[Array, "state_dim"]:
        """Decode latent vector to physical state.
        
        Args:
            z: Latent vector
            params: System parameters
            control: Control input
            
        Returns:
            Reconstructed physical state [Ca, Cb, T, Tc]
        """
        # Concatenate inputs
        x = jnp.concatenate([z, params * 0.01, control * 0.01])
        
        # Forward through hidden layers
        for layer in self.layers:
            x = layer(x)
            x = jax.nn.silu(x)
        
        # Output layer
        state_raw = self.output_layer(x)
        
        # Apply output constraints
        # Ca, Cb: must be non-negative (use softplus)
        Ca = jax.nn.softplus(state_raw[0])
        Cb = jax.nn.softplus(state_raw[1])
        
        # T, Tc: must be in reasonable range ~200-500K
        # Use 200 + 300*sigmoid to get range [200, 500]
        T = 200.0 + 300.0 * jax.nn.sigmoid(state_raw[2])
        Tc = 200.0 + 300.0 * jax.nn.sigmoid(state_raw[3])
        
        return jnp.array([Ca, Cb, T, Tc])
