"""Encoder module for mapping physical states to latent space."""

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray


class Encoder(eqx.Module):
    """VAE-style encoder: physical state + params + control -> latent (mean, logvar).

    All input normalization is driven by arrays passed at construction time so
    that no system-specific magic numbers live in this file.
    """

    layers: list
    mean_layer: eqx.nn.Linear
    logvar_layer: eqx.nn.Linear

    # Normalization arrays stored as non-trainable leaf arrays.
    # We use regular (trainable) storage but freeze them by never including them
    # in optimizer updates.  They are JAX arrays, so they live on device.
    state_center: Float[Array, "state_dim"]
    state_scale: Float[Array, "state_dim"]
    control_center: Float[Array, "control_dim"]
    control_scale: Float[Array, "control_dim"]
    param_scale: float = eqx.field(static=True)

    def __init__(
        self,
        state_dim: int = 4,
        param_dim: int = 6,
        control_dim: int = 2,
        latent_dim: int = 16,
        hidden_dim: int = 128,
        n_layers: int = 3,
        # Normalization -- defaults match original CSTR values
        state_center: list | None = None,
        state_scale: list | None = None,
        control_center: list | None = None,
        control_scale: list | None = None,
        param_scale: float = 0.1,
        *,
        key: PRNGKeyArray,
    ):
        keys = jax.random.split(key, n_layers + 2)

        input_dim = state_dim + param_dim + control_dim

        self.layers = []
        for i in range(n_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            self.layers.append(eqx.nn.Linear(in_dim, hidden_dim, key=keys[i]))

        self.mean_layer = eqx.nn.Linear(hidden_dim, latent_dim, key=keys[-2])
        self.logvar_layer = eqx.nn.Linear(hidden_dim, latent_dim, key=keys[-1])

        # Store normalization as frozen arrays
        self.state_center = jnp.array(
            state_center if state_center is not None else [0.0] * state_dim
        )
        self.state_scale = jnp.array(
            state_scale if state_scale is not None else [1.0] * state_dim
        )
        self.control_center = jnp.array(
            control_center if control_center is not None else [0.0] * control_dim
        )
        self.control_scale = jnp.array(
            control_scale if control_scale is not None else [1.0] * control_dim
        )
        self.param_scale = param_scale

    def encode(
        self,
        state: Float[Array, "state_dim"],
        params: Float[Array, "param_dim"],
        control: Float[Array, "control_dim"],
    ) -> tuple[Float[Array, "latent_dim"], Float[Array, "latent_dim"]]:
        """Encode inputs to latent mean and log-variance."""
        scaled_state = (state - self.state_center) * self.state_scale
        scaled_control = (control - self.control_center) * self.control_scale
        log_params = jnp.sign(params) * jnp.log1p(jnp.abs(params)) * self.param_scale

        x = jnp.concatenate([scaled_state, log_params, scaled_control])

        for layer in self.layers:
            x = jax.nn.silu(layer(x))

        z_mean = jnp.tanh(self.mean_layer(x) * 0.05) * 5.0
        z_logvar = jnp.tanh(self.logvar_layer(x) * 0.1) * 3.0 - 2.0

        return z_mean, z_logvar

    def sample(
        self,
        z_mean: Float[Array, "latent_dim"],
        z_logvar: Float[Array, "latent_dim"],
        key: PRNGKeyArray,
    ) -> Float[Array, "latent_dim"]:
        """Reparameterization-trick sample."""
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
        """Full forward pass with sampling -> (z, z_mean, z_logvar)."""
        z_mean, z_logvar = self.encode(state, params, control)
        z = self.sample(z_mean, z_logvar, key)
        return z, z_mean, z_logvar
