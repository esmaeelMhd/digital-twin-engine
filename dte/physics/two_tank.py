"""Two-tank physics loss implementation."""

from typing import Dict

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from dte.physics.base import PhysicsLoss
from dte.simulators.two_tank import TwoTankParams


def _two_tank_param_vector(
    params: TwoTankParams,
) -> Float[Array, "5"]:
    return jnp.array([params.A1, params.A2, params.k12, params.kout, params.h_max])


def mass_balance_residual_with_params(
    states: Float[Array, "n_steps 2"],
    controls: Float[Array, "n_steps 2"],
    disturbances: Float[Array, "n_steps 2"],
    params: Float[Array, "5"],
    dt: float,
) -> Float[Array, "n_steps-1"]:
    """Mass-balance residual using a raw per-trajectory parameter vector."""
    A1, A2, k12, kout, _ = params
    h1 = states[:, 0]
    h2 = states[:, 1]
    q_in = controls[:, 0]
    valve = controls[:, 1]
    d1 = disturbances[:, 0]
    d2 = disturbances[:, 1]

    dh1_dt = jnp.diff(h1) / dt
    dh2_dt = jnp.diff(h2) / dt

    interflow = k12 * jnp.sqrt(jnp.maximum(h1[:-1] - h2[:-1], 0.0) + 1e-8)
    outflow = valve[:-1] * kout * jnp.sqrt(jnp.maximum(h2[:-1], 0.0) + 1e-8)

    expected_dh1_dt = (q_in[:-1] + d1[:-1] - interflow) / A1
    expected_dh2_dt = (interflow + d2[:-1] - outflow) / A2

    res1 = jnp.abs(dh1_dt - expected_dh1_dt)
    res2 = jnp.abs(dh2_dt - expected_dh2_dt)
    return 0.5 * (res1 + res2)


def mass_balance_residual(
    states: Float[Array, "n_steps 2"],
    controls: Float[Array, "n_steps 2"],
    disturbances: Float[Array, "n_steps 2"],
    params: TwoTankParams,
    dt: float,
) -> Float[Array, "n_steps-1"]:
    """Mass-balance residual averaged across both tank equations."""
    return mass_balance_residual_with_params(
        states,
        controls,
        disturbances,
        _two_tank_param_vector(params),
        dt,
    )


class TwoTankPhysicsLoss(PhysicsLoss):
    """Physics residual losses for the coupled two-tank process."""

    def __init__(self, params: TwoTankParams):
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
            nominal_params = _two_tank_param_vector(self.params)
            params_batch = jnp.broadcast_to(nominal_params, (states.shape[0], nominal_params.shape[0]))

        mass_vals = jax.vmap(
            lambda s, u, d, p: jnp.mean(
                mass_balance_residual_with_params(s, u, d, p, dt)
            )
        )(states, controls, disturbances, params_batch)
        return {"mass": jnp.mean(mass_vals)}

    def residual_names(self) -> list:
        return ["mass"]
