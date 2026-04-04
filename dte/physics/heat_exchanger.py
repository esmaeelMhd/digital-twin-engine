"""Heat exchanger physics loss implementation.

Provides an energy-balance residual for the counter-current heat exchanger
and a :class:`HeatExchangerPhysicsLoss` that satisfies the
:class:`~dte.physics.base.PhysicsLoss` protocol.
"""

from typing import Dict

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from dte.simulators.heat_exchanger import HeatExchangerParams
from dte.physics.base import PhysicsLoss


# ---------------------------------------------------------------------------
# Low-level residual functions
# ---------------------------------------------------------------------------

def _heat_exchanger_param_vector(
    params: HeatExchangerParams,
) -> Float[Array, "5"]:
    return jnp.array([params.V_hot, params.V_cold, params.UA, params.rho, params.Cp])


def energy_balance_residual_with_params(
    states: Float[Array, "n_steps 2"],
    controls: Float[Array, "n_steps 2"],
    disturbances: Float[Array, "n_steps 2"],
    params: Float[Array, "5"],
    dt: float,
) -> Float[Array, "n_steps-1"]:
    """Energy balance residual using a raw per-trajectory parameter vector."""
    V_hot, V_cold, UA, rho, Cp = params
    T_hot = states[:, 0]
    T_cold = states[:, 1]
    F_hot = controls[:, 0]
    F_cold = controls[:, 1]
    T_hot_in = disturbances[:, 0]
    T_cold_in = disturbances[:, 1]

    dT_hot_dt = jnp.diff(T_hot) / dt
    dT_cold_dt = jnp.diff(T_cold) / dt

    heat_transfer = (UA / (rho * Cp)) * (T_hot[:-1] - T_cold[:-1])

    expected_dT_hot_dt = (F_hot[:-1] / V_hot) * (T_hot_in[:-1] - T_hot[:-1]) - heat_transfer / V_hot
    expected_dT_cold_dt = (F_cold[:-1] / V_cold) * (T_cold_in[:-1] - T_cold[:-1]) + heat_transfer / V_cold

    hot_residual = jnp.abs(dT_hot_dt - expected_dT_hot_dt)
    cold_residual = jnp.abs(dT_cold_dt - expected_dT_cold_dt)

    return 0.5 * (hot_residual + cold_residual)


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
    return energy_balance_residual_with_params(
        states,
        controls,
        disturbances,
        _heat_exchanger_param_vector(params),
        dt,
    )


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
        params_batch=None,
    ) -> Dict[str, Float[Array, ""]]:
        if params_batch is None:
            nominal_params = _heat_exchanger_param_vector(self.params)
            params_batch = jnp.broadcast_to(nominal_params, (states.shape[0], nominal_params.shape[0]))

        energy_vals = jax.vmap(
            lambda s, u, d, p: jnp.mean(
                energy_balance_residual_with_params(s, u, d, p, dt)
            )
        )(states, controls, disturbances, params_batch)

        return {
            "energy": jnp.mean(energy_vals),
        }

    def residual_names(self) -> list:
        return ["energy"]
