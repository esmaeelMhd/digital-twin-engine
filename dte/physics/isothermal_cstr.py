"""Isothermal CSTR physics loss implementation."""

from typing import Dict

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from dte.physics.base import PhysicsLoss
from dte.simulators.isothermal_cstr import (
    IsothermalCSTRParams,
    _isothermal_cstr_dynamics_with_params,
)


def _isothermal_cstr_param_vector(
    params: IsothermalCSTRParams,
) -> Float[Array, "4"]:
    return jnp.array([params.V, params.k0, params.Ea_over_R, params.T_ref])


def species_mass_balance_residuals_with_params(
    states: Float[Array, "n_steps 2"],
    controls: Float[Array, "n_steps 1"],
    disturbances: Float[Array, "n_steps 1"],
    params: Float[Array, "4"],
    dt: float,
) -> Float[Array, "n_steps-1 2"]:
    numerical = jnp.diff(states, axis=0) / dt
    expected = jax.vmap(
        lambda s, u, d: _isothermal_cstr_dynamics_with_params(s, u, d, params)
    )(states[:-1], controls[:-1], disturbances[:-1])
    return jnp.abs(numerical - expected)


def mass_balance_residual_with_params(
    states: Float[Array, "n_steps 2"],
    controls: Float[Array, "n_steps 1"],
    disturbances: Float[Array, "n_steps 1"],
    params: Float[Array, "4"],
    dt: float,
) -> Float[Array, "n_steps-1"]:
    numerical = jnp.diff(jnp.sum(states, axis=-1)) / dt
    expected = jnp.sum(
        jax.vmap(
            lambda s, u, d: _isothermal_cstr_dynamics_with_params(s, u, d, params)
        )(states[:-1], controls[:-1], disturbances[:-1]),
        axis=-1,
    )
    return jnp.abs(numerical - expected)


class IsothermalCSTRPhysicsLoss(PhysicsLoss):
    """Physics residual losses for the simplified isothermal CSTR."""

    def __init__(self, params: IsothermalCSTRParams):
        self.params = params

    def compute_residuals(
        self,
        states: Float[Array, "batch seq_len state_dim"],
        controls: Float[Array, "batch seq_len control_dim"],
        disturbances: Float[Array, "batch seq_len disturbance_dim"],
        dt,
        params_batch=None,
    ) -> Dict[str, Float[Array, ""]]:
        if params_batch is None:
            nominal_params = _isothermal_cstr_param_vector(self.params)
            params_batch = jnp.broadcast_to(
                nominal_params,
                (states.shape[0], nominal_params.shape[0]),
            )
        mass_vals = jax.vmap(
            lambda s, u, d, p: jnp.mean(mass_balance_residual_with_params(s, u, d, p, dt))
        )(states, controls, disturbances, params_batch)
        species_vals = jax.vmap(
            lambda s, u, d, p: jnp.mean(
                species_mass_balance_residuals_with_params(s, u, d, p, dt)
            )
        )(states, controls, disturbances, params_batch)
        return {
            "mass": jnp.mean(mass_vals),
            "species_mass": jnp.mean(species_vals),
        }

    def residual_names(self) -> list:
        return ["mass", "species_mass"]
