"""CSTR-specific physics loss implementation.

This module contains the conservation-law residuals that were previously in
``dte/physics/conservation.py``.  That file is kept for backwards
compatibility and re-exports everything from here.
"""

from typing import Dict

import jax.numpy as jnp
from jaxtyping import Array, Float

from dte.simulators.cstr import CSTRParams
from dte.physics.base import PhysicsLoss


# ---------------------------------------------------------------------------
# Low-level residual functions (unchanged from original conservation.py)
# ---------------------------------------------------------------------------

def mass_balance_residual(
    states: Float[Array, "n_steps 4"],
    controls: Float[Array, "n_steps 2"],
    disturbances: Float[Array, "n_steps 2"],
    params: CSTRParams,
    dt: float,
) -> Float[Array, "n_steps-1"]:
    """Total-moles mass balance residual (scalar per timestep)."""
    Ca = states[:, 0]
    Cb = states[:, 1]
    F_in = controls[:, 0]
    Ca_in = disturbances[:, 0]

    total_moles = Ca + Cb
    d_total_moles_dt = jnp.diff(total_moles) / dt
    expected_change = (F_in[:-1] / params.V) * (Ca_in[:-1] - total_moles[:-1])
    return jnp.abs(d_total_moles_dt - expected_change)


def species_mass_balance_residuals(
    states: Float[Array, "n_steps 4"],
    controls: Float[Array, "n_steps 2"],
    disturbances: Float[Array, "n_steps 2"],
    params: CSTRParams,
    dt: float,
) -> Float[Array, "n_steps-1 2"]:
    """Species-wise (Ca, Cb) mass-balance residuals."""
    Ca = states[:, 0]
    Cb = states[:, 1]
    T = states[:, 2]
    F_in = controls[:, 0]
    Ca_in = disturbances[:, 0]

    dCa_dt = jnp.diff(Ca) / dt
    dCb_dt = jnp.diff(Cb) / dt

    k = params.k0 * jnp.exp(-params.Ea_over_R / T[:-1])
    flow_over_volume = F_in[:-1] / params.V

    expected_dCa_dt = flow_over_volume * (Ca_in[:-1] - Ca[:-1]) - k * Ca[:-1]
    expected_dCb_dt = -flow_over_volume * Cb[:-1] + k * Ca[:-1]

    return jnp.stack(
        [jnp.abs(dCa_dt - expected_dCa_dt), jnp.abs(dCb_dt - expected_dCb_dt)],
        axis=-1,
    )


def energy_balance_residual(
    states: Float[Array, "n_steps 4"],
    controls: Float[Array, "n_steps 2"],
    disturbances: Float[Array, "n_steps 2"],
    params: CSTRParams,
    dt: float,
) -> Float[Array, "n_steps-1"]:
    """Reactor energy balance residual (scalar per timestep)."""
    Ca = states[:, 0]
    T = states[:, 2]
    Tc = states[:, 3]
    F_in = controls[:, 0]
    T_in = disturbances[:, 1]

    dT_dt = jnp.diff(T) / dt
    k = params.k0 * jnp.exp(-params.Ea_over_R / T[:-1])

    flow_term = (F_in[:-1] / params.V) * (T_in[:-1] - T[:-1])
    reaction_term = (-params.dH_rxn / (params.rho * params.Cp)) * k * Ca[:-1]
    heat_transfer_term = (params.UA / (params.V * params.rho * params.Cp)) * (Tc[:-1] - T[:-1])

    expected_dT_dt = flow_term + reaction_term + heat_transfer_term
    return jnp.abs(dT_dt - expected_dT_dt)


def total_conservation_metric(
    states: Float[Array, "n_steps 4"],
    controls: Float[Array, "n_steps 2"],
    disturbances: Float[Array, "n_steps 2"],
    params: CSTRParams,
    dt: float,
) -> Dict[str, float]:
    """Compute all conservation metrics and return a dict of floats."""
    mass_res = mass_balance_residual(states, controls, disturbances, params, dt)
    species_res = species_mass_balance_residuals(states, controls, disturbances, params, dt)
    energy_res = energy_balance_residual(states, controls, disturbances, params, dt)

    return {
        "mass_residual_mean": float(jnp.mean(mass_res)),
        "mass_residual_max": float(jnp.max(mass_res)),
        "ca_mass_residual_mean": float(jnp.mean(species_res[:, 0])),
        "ca_mass_residual_max": float(jnp.max(species_res[:, 0])),
        "cb_mass_residual_mean": float(jnp.mean(species_res[:, 1])),
        "cb_mass_residual_max": float(jnp.max(species_res[:, 1])),
        "energy_residual_mean": float(jnp.mean(energy_res)),
        "energy_residual_max": float(jnp.max(energy_res)),
    }


# ---------------------------------------------------------------------------
# PhysicsLoss implementation
# ---------------------------------------------------------------------------

class CSTRPhysicsLoss(PhysicsLoss):
    """Physics residual losses for the non-isothermal CSTR."""

    def __init__(self, params: CSTRParams):
        self.params = params

    def compute_residuals(
        self,
        states: Float[Array, "batch seq_len state_dim"],
        controls: Float[Array, "batch seq_len control_dim"],
        disturbances: Float[Array, "batch seq_len disturbance_dim"],
        dt,
    ) -> Dict[str, Float[Array, ""]]:
        batch_size = states.shape[0]

        mass_vals = []
        species_vals = []
        energy_vals = []

        for i in range(batch_size):
            mass_vals.append(jnp.mean(
                mass_balance_residual(states[i], controls[i], disturbances[i], self.params, dt)
            ))
            species_vals.append(jnp.mean(
                species_mass_balance_residuals(states[i], controls[i], disturbances[i], self.params, dt)
            ))
            energy_vals.append(jnp.mean(
                energy_balance_residual(states[i], controls[i], disturbances[i], self.params, dt)
            ))

        return {
            "mass": jnp.mean(jnp.array(mass_vals)),
            "species_mass": jnp.mean(jnp.array(species_vals)),
            "energy": jnp.mean(jnp.array(energy_vals)),
        }

    def residual_names(self) -> list:
        return ["mass", "species_mass", "energy"]
