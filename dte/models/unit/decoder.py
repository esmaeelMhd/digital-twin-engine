"""Decoder module for mapping latent states back to physical space."""

from typing import List

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray


def _freeze_constraint(constraint: dict | tuple) -> tuple:
    """Convert a constraint dict into a hashable (type, indices, extras) tuple."""
    if not isinstance(constraint, dict):
        return constraint
    extras = tuple(
        sorted(
            (str(key), value)
            for key, value in constraint.items()
            if key not in ("type", "indices")
        )
    )
    return (
        str(constraint["type"]),
        tuple(int(idx) for idx in constraint["indices"]),
        extras,
    )


def _constraint_fields(constraint: dict | tuple) -> tuple[str, tuple, dict]:
    if isinstance(constraint, dict):
        return str(constraint["type"]), tuple(constraint["indices"]), constraint
    ctype, indices, extras = constraint
    return str(ctype), tuple(indices), dict(extras)


def apply_decoder_constraints(
    state_raw: Float[Array, "state_dim"],
    constraints: list | tuple,
) -> Float[Array, "state_dim"]:
    """Apply physical constraints to raw decoder outputs.

    Each constraint is either a dict::

        {"type": "softplus",     "indices": [0, 1], "bias": 0.5}
        {"type": "sigmoid_range","indices": [2, 3], "low": 0.0, "high": 1.0}
        {"type": "none",         "indices": [4]}

    or the hashable tuple form stored on ``Decoder.constraints``.

    This function is JAX-traceable and works correctly under ``jit`` and
    ``vmap`` provided all constraint metadata is static (no data-dependent
    indexing based on JAX arrays).
    """
    out = state_raw
    for spec in constraints:
        ctype, raw_idxs, extras = _constraint_fields(spec)
        if ctype == "softplus":
            bias = extras.get("bias", 0.5)
            out = out.at[jnp.array(raw_idxs)].set(
                jax.nn.softplus(state_raw[jnp.array(raw_idxs)] + bias)
            )
        elif ctype == "sigmoid_range":
            low = extras.get("low", 0.0)
            high = extras.get("high", 1.0)
            out = out.at[jnp.array(raw_idxs)].set(
                low + (high - low) * jax.nn.sigmoid(state_raw[jnp.array(raw_idxs)])
            )
        # "none" -> no-op
    return out


class Decoder(eqx.Module):
    """Decodes latent state z back to physical state.

    Output constraints are fully driven by the ``constraints`` argument so
    that no system-specific values are hardcoded in this file.
    """

    layers: list
    output_layer: eqx.nn.Linear

    # Normalization for conditioning inputs
    control_scale: Float[Array, "control_dim"]
    param_scale: float = eqx.field(static=True)

    # Constraint specification stored as a hashable tuple of
    # (type, indices, extras) so Equinox can treat it as static metadata.
    constraints: tuple = eqx.field(static=True)

    def __init__(
        self,
        latent_dim: int = 16,
        param_dim: int = 6,
        control_dim: int = 2,
        state_dim: int = 4,
        hidden_dim: int = 128,
        n_layers: int = 3,
        constraints: List[dict] | None = None,
        control_scale: list | None = None,
        param_scale: float = 0.1,
        *,
        key: PRNGKeyArray,
    ):
        """
        Args:
            constraints: List of constraint dicts applied to raw decoder output.
                Each dict must have ``"type"`` and ``"indices"`` keys.
                Defaults to no constraints (pass-through).
            control_scale: Per-element scale applied to control before concat.
            param_scale: Scalar scale applied to log-params before concat.
        """
        keys = jax.random.split(key, n_layers + 1)

        input_dim = latent_dim + param_dim + control_dim

        self.layers = []
        for i in range(n_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            self.layers.append(eqx.nn.Linear(in_dim, hidden_dim, key=keys[i]))

        self.output_layer = eqx.nn.Linear(hidden_dim, state_dim, key=keys[-1])

        self.constraints = tuple(
            _freeze_constraint(item) for item in (constraints or [])
        )
        self.control_scale = jnp.array(
            control_scale if control_scale is not None else [1.0] * control_dim
        )
        self.param_scale = param_scale

    def __call__(
        self,
        z: Float[Array, "latent_dim"],
        params: Float[Array, "param_dim"],
        control: Float[Array, "control_dim"],
    ) -> Float[Array, "state_dim"]:
        """Decode latent vector to physical state."""
        log_params = jnp.sign(params) * jnp.log1p(jnp.abs(params)) * self.param_scale
        scaled_control = control * self.control_scale

        x = jnp.concatenate([z, log_params, scaled_control])

        for layer in self.layers:
            x_out = jax.nn.gelu(layer(x))
            x = x + x_out if x.shape == x_out.shape else x_out

        state_raw = self.output_layer(x)

        return apply_decoder_constraints(state_raw, self.constraints)
