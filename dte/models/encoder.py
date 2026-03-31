"""Encoder module for mapping physical states to latent space."""

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray


class Encoder(eqx.Module):
    """
    Encodes physical state [Ca, Cb, T, Tc] + system params + control
    into latent space z ∈ R^latent_dim.
    
    Outputs both mean and log-variance (VAE-style) for stochastic encoding.
    """
    
    layers: list
    mean_layer: eqx.nn.Linear
    logvar_layer: eqx.nn.Linear
    
    def __init__(
        self,
        state_dim: int = 4,
        param_dim: int = 6,
        control_dim: int = 2,
        latent_dim: int = 16,
        hidden_dim: int = 128,
        n_layers: int = 3,
        *,
        key: PRNGKeyArray,
    ):
        """Initialize encoder.
        
        Args:
            state_dim: State dimension
            param_dim: Parameter dimension
            control_dim: Control dimension
            latent_dim: Latent dimension
            hidden_dim: Hidden layer dimension
            n_layers: Number of hidden layers
            key: PRNG key for initialization
        """
        keys = jax.random.split(key, n_layers + 2)
        
        # Input dimension
        input_dim = state_dim + param_dim + control_dim
        
        # Hidden layers
        self.layers = []
        for i in range(n_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            self.layers.append(
                eqx.nn.Linear(in_dim, hidden_dim, key=keys[i])
            )
        
        # Output heads
        self.mean_layer = eqx.nn.Linear(hidden_dim, latent_dim, key=keys[-2])
        self.logvar_layer = eqx.nn.Linear(hidden_dim, latent_dim, key=keys[-1])
    
    def encode(
        self,
        state: Float[Array, "state_dim"],
        params: Float[Array, "param_dim"],
        control: Float[Array, "control_dim"],
    ) -> tuple[Float[Array, "latent_dim"], Float[Array, "latent_dim"]]:
        """Encode inputs to latent mean and log-variance.
        
        Args:
            state: Physical state
            params: System parameters
            control: Control input
            
        Returns:
            Tuple of (z_mean, z_logvar)
        """
        # Concatenate inputs
        log_params = jnp.sign(params) * jnp.log1p(jnp.abs(params))
        scaled_state = (state - jnp.array([1.0, 1.0, 320.0, 300.0])) * jnp.array([1.0, 1.0, 0.02, 0.02])
        scaled_control = (control - jnp.array([55.0, 300.0])) * jnp.array([0.02, 0.02])
        x = jnp.concatenate([scaled_state, log_params * 0.1, scaled_control])
        
        # Forward through hidden layers
        for layer in self.layers:
            x = layer(x)
            x = jax.nn.silu(x)
        
        # Output heads
        z_mean = jnp.tanh(self.mean_layer(x) * 0.05) * 5.0
        z_logvar = jax.nn.log_sigmoid(self.logvar_layer(x) - 2.0)
        
        return z_mean, z_logvar
    
    def sample(
        self,
        z_mean: Float[Array, "latent_dim"],
        z_logvar: Float[Array, "latent_dim"],
        key: PRNGKeyArray,
    ) -> Float[Array, "latent_dim"]:
        """Sample latent vector using reparameterization trick.
        
        Args:
            z_mean: Latent mean
            z_logvar: Latent log-variance
            key: PRNG key
            
        Returns:
            Sampled latent vector z
        """
        std = jnp.exp(0.5 * z_logvar)
        eps = jax.random.normal(key, shape=z_mean.shape)
        return z_mean + eps * std
    
    def __call__(
        self,
        state: Float[Array, "state_dim"],
        params: Float[Array, "param_dim"],
        control: Float[Array, "control_dim"],
        key: PRNGKeyArray,
    ) -> tuple[Float[Array, "latent_dim"], Float[Array, "latent_dim"], Float[Array, "latent_dim"]]:
        """Full forward pass with sampling.
        
        Args:
            state: Physical state
            params: System parameters
            control: Control input
            key: PRNG key for sampling
            
        Returns:
            Tuple of (z, z_mean, z_logvar)
        """
        z_mean, z_logvar = self.encode(state, params, control)
        z = self.sample(z_mean, z_logvar, key)
        return z, z_mean, z_logvar
