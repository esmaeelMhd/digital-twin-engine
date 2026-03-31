"""Heat exchanger physics loss implementation.

Provides an energy-balance residual for the counter-current heat exchanger
and a :class:`HeatExchangerPhysicsLoss` that satisfies the
:class:`~dte.physics.base.PhysicsLoss` protocol.
"""

from typing import Dict

import jax.numpy as jnp
from jaxtyping import Array, Float

from dte.simulators.heat_exchanger import HeatExchangerParams
from dte.physics.base import PhysicsLoss


# ---------------------------------------------------------------------------
# Low-level residual functions
# ---------------------------------------------------------------------------

def energy_balance_residual(
    states: Float[Array, "n_steps 2"],
    controls: Float[Array, "n_steps 2"],
    disturbances: Float[Array, "n_steps 2"],
    params: HeatExchangerParams,
    dt: float,
) -> Float[Array, "n_steps-1"]:
    """Energy balance residual for both hot and cold sides (scalar per step).

    Computes the absolute discrepancy between the numerical derivative of each
    side temperature and the ODE right-hand side evaluated at the current step.
    """
    T_hot = states[:, 0]
    T_cold = states[:, 1]
    F_hot = controls[:, 0]
    F_cold = controls[:, 1]
    T_hot_in = disturbances[:, 0]
    T_cold_in = disturbances[:, 1]

    dT_hot_dt = jnp.diff(T_hot) / dt
    dT_cold_dt = jnp.diff(T_cold) / dt

    heat_transfer = (params.UA / (params.rho * params.Cp)) * (T_hot[:-1] - T_cold[:-1])

    expected_dT_hot_dt = (F_hot[:-1] / params.V_hot) * (T_hot_in[:-1] - T_hot[:-1]) \
        - heat_transfer / params.V_hot
    expected_dT_cold_dt = (F_cold[:-1] / params.V_cold) * (T_cold_in[:-1] - T_cold[:-1]) \
        + heat_transfer / params.V_cold

    hot_residual = jnp.abs(dT_hot_dt - expected_dT_hot_dt)
    cold_residual = jnp.abs(dT_cold_dt - expected_dT_cold_dt)

    return (hot_residual + cold_residual) / 2.0


# ---------------------------------------------------------------------------
# PhysicsLoss implementation
# ---------------------------------------------------------------------------

class HeatExchangerPhysicsLoss(PhysicsLoss):
    """Physics residual losses for the counter-current heat exchanger."""

    def __init__(self, params: HeatExchangerParams):
        self.params = params

    def compute_residuals(
        self,
        states: Float[Array, "batch seq_len state_dim"],
        controls: Float[Array, "batch seq_len control_dim"],
        disturbances: Float[Array, "batch seq_len disturbance_dim"],
        dt,
    ) -> Dict[str, Float[Array, ""]]:
        batch_size = states.shape[0]
        energy_vals = []

        for i in range(batch_size):
            energy_vals.append(jnp.mean(
                energy_balance_residual(
                    states[i], controls[i], disturbances[i], self.params, dt
                )
            ))

        return {
            "energy": jnp.mean(jnp.array(energy_vals)),
        }

    def residual_names(self) -> list:
        return ["energy"]
