"""Storage tank physics loss implementation."""

from typing import Dict

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from dte.physics.base import PhysicsLoss
from dte.simulators.storage_tank import (
    StorageTankParams,
    _storage_tank_dynamics_with_params,
)


def _storage_tank_param_vector(
    params: StorageTankParams,
) -> Float[Array, "2"]:
    return jnp.array([params.volume, params.heat_loss])


def state_residuals_with_params(
    states: Float[Array, "n_steps 3"],
    controls: Float[Array, "n_steps 1"],
    disturbances: Float[Array, "n_steps 3"],
    params: Float[Array, "2"],
    dt: float,
) -> Float[Array, "n_steps-1 3"]:
    numerical = jnp.diff(states, axis=0) / dt
    expected = jax.vmap(
        lambda s, u, d: _storage_tank_dynamics_with_params(s, u, d, params)
    )(states[:-1], controls[:-1], disturbances[:-1])
    return jnp.abs(numerical - expected)


class StorageTankPhysicsLoss(PhysicsLoss):
    """Physics residual losses for the synthetic storage tank."""

    def __init__(self, params: StorageTankParams):
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
            nominal_params = _storage_tank_param_vector(self.params)
            params_batch = jnp.broadcast_to(
                nominal_params,
                (states.shape[0], nominal_params.shape[0]),
            )
        residuals = jax.vmap(
            lambda s, u, d, p: jnp.mean(state_residuals_with_params(s, u, d, p, dt), axis=0)
        )(states, controls, disturbances, params_batch)
        return {
            "mass": jnp.mean(residuals[:, 0]),
            "composition": jnp.mean(residuals[:, 1]),
            "energy": jnp.mean(residuals[:, 2]),
        }

    def residual_names(self) -> list:
        return ["mass", "composition", "energy"]
