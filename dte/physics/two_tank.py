"""Two-tank physics loss implementation."""

from typing import Dict

import jax.numpy as jnp
from jaxtyping import Array, Float

from dte.physics.base import PhysicsLoss
from dte.simulators.two_tank import TwoTankParams


def mass_balance_residual(
    states: Float[Array, "n_steps 2"],
    controls: Float[Array, "n_steps 2"],
    disturbances: Float[Array, "n_steps 2"],
    params: TwoTankParams,
    dt: float,
) -> Float[Array, "n_steps-1"]:
    """Mass-balance residual averaged across both tank equations."""
    h1 = states[:, 0]
    h2 = states[:, 1]
    q_in = controls[:, 0]
    valve = controls[:, 1]
    d1 = disturbances[:, 0]
    d2 = disturbances[:, 1]

    dh1_dt = jnp.diff(h1) / dt
    dh2_dt = jnp.diff(h2) / dt

    interflow = params.k12 * jnp.sqrt(jnp.maximum(h1[:-1] - h2[:-1], 0.0) + 1e-8)
    outflow = valve[:-1] * params.kout * jnp.sqrt(jnp.maximum(h2[:-1], 0.0) + 1e-8)

    expected_dh1_dt = (q_in[:-1] + d1[:-1] - interflow) / params.A1
    expected_dh2_dt = (interflow + d2[:-1] - outflow) / params.A2

    res1 = jnp.abs(dh1_dt - expected_dh1_dt)
    res2 = jnp.abs(dh2_dt - expected_dh2_dt)
    return 0.5 * (res1 + res2)


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
    ) -> Dict[str, Float[Array, ""]]:
        mass_vals = []
        for idx in range(states.shape[0]):
            mass_vals.append(
                jnp.mean(
                    mass_balance_residual(
                        states[idx],
                        controls[idx],
                        disturbances[idx],
                        self.params,
                        dt,
                    )
                )
            )
        return {"mass": jnp.mean(jnp.array(mass_vals))}

    def residual_names(self) -> list:
        return ["mass"]
