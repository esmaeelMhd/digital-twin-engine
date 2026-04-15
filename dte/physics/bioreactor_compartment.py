"""Bioreactor compartment physics loss implementation."""

from typing import Dict

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from dte.physics.base import PhysicsLoss
from dte.simulators.bioreactor_compartment import (
    BioreactorCompartmentParams,
    _bioreactor_dynamics_with_params,
)


def _bioreactor_param_vector(
    params: BioreactorCompartmentParams,
) -> Float[Array, "4"]:
    return jnp.array([params.mu_max, params.kla, params.decay_rate, params.dilution_rate])


def state_residuals_with_params(
    states: Float[Array, "n_steps 3"],
    controls: Float[Array, "n_steps 1"],
    disturbances: Float[Array, "n_steps 1"],
    params: Float[Array, "4"],
    dt: float,
) -> Float[Array, "n_steps-1 3"]:
    numerical = jnp.diff(states, axis=0) / dt
    expected = jax.vmap(
        lambda s, u, d: _bioreactor_dynamics_with_params(s, u, d, params)
    )(states[:-1], controls[:-1], disturbances[:-1])
    return jnp.abs(numerical - expected)


class BioreactorCompartmentPhysicsLoss(PhysicsLoss):
    """Physics residual losses for the aerobic bioreactor compartment."""

    def __init__(self, params: BioreactorCompartmentParams):
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
            nominal_params = _bioreactor_param_vector(self.params)
            params_batch = jnp.broadcast_to(
                nominal_params,
                (states.shape[0], nominal_params.shape[0]),
            )
        residuals = jax.vmap(
            lambda s, u, d, p: jnp.mean(state_residuals_with_params(s, u, d, p, dt), axis=0)
        )(states, controls, disturbances, params_batch)
        return {
            "substrate": jnp.mean(residuals[:, 0]),
            "biomass": jnp.mean(residuals[:, 1]),
            "oxygen": jnp.mean(residuals[:, 2]),
        }

    def residual_names(self) -> list:
        return ["substrate", "biomass", "oxygen"]
