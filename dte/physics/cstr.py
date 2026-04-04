"""CSTR-specific physics loss implementation.

This module contains the conservation-law residuals that were previously in
``dte/physics/conservation.py``.  That file is kept for backwards
compatibility and re-exports everything from here.
"""

from typing import Dict

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from dte.simulators.cstr import CSTRParams
from dte.physics.base import PhysicsLoss


# ---------------------------------------------------------------------------
# Low-level residual functions (unchanged from original conservation.py)
# ---------------------------------------------------------------------------

def _cstr_training_param_vector(
    params: CSTRParams,
) -> Float[Array, "6"]:
    return jnp.array(
        [
            params.V,
            params.k0 / 1e10,
            params.Ea_over_R / 1000.0,
            params.UA / 1e4,
            params.Fc,
            params.Vc,
        ]
    )


def _resolve_cstr_parameters(
    params: Float[Array, "param_dim"],
    defaults: CSTRParams,
):
    """Map either the stored 6D training vector or the packed 11D vector to physics params."""
    if params.shape[0] == 11:
        V = params[0]
        Vc = params[1]
        k0 = params[2]
        Ea_over_R = params[3]
        dH_rxn = params[4]
        rho = params[5]
        Cp = params[6]
        UA = params[7]
        rho_c = params[8]
        Cp_c = params[9]
        Fc = params[10]
    elif params.shape[0] == 6:
        V = params[0]
        k0 = params[1] * 1e10
        Ea_over_R = params[2] * 1000.0
        UA = params[3] * 1e4
        Fc = params[4]
        Vc = params[5]
        dH_rxn = jnp.asarray(defaults.dH_rxn)
        rho = jnp.asarray(defaults.rho)
        Cp = jnp.asarray(defaults.Cp)
        rho_c = jnp.asarray(defaults.rho_c)
        Cp_c = jnp.asarray(defaults.Cp_c)
    else:
        raise ValueError(
            f"Unsupported CSTR parameter vector length {params.shape[0]}; expected 6 or 11."
        )
    return V, Vc, k0, Ea_over_R, dH_rxn, rho, Cp, UA, rho_c, Cp_c, Fc


def mass_balance_residual_with_params(
    states: Float[Array, "n_steps 4"],
    controls: Float[Array, "n_steps 2"],
    disturbances: Float[Array, "n_steps 2"],
    params: Float[Array, "param_dim"],
    dt: float,
    defaults: CSTRParams,
) -> Float[Array, "n_steps-1"]:
    """Total-moles mass balance residual for a raw per-trajectory parameter vector."""
    V, _, _, _, _, _, _, _, _, _, _ = _resolve_cstr_parameters(params, defaults)
    Ca = states[:, 0]
    Cb = states[:, 1]
    F_in = controls[:, 0]
    Ca_in = disturbances[:, 0]

    total_moles = Ca + Cb
    d_total_moles_dt = jnp.diff(total_moles) / dt
    expected_change = (F_in[:-1] / V) * (Ca_in[:-1] - total_moles[:-1])
    return jnp.abs(d_total_moles_dt - expected_change)


def mass_balance_residual(
    states: Float[Array, "n_steps 4"],
    controls: Float[Array, "n_steps 2"],
    disturbances: Float[Array, "n_steps 2"],
    params: CSTRParams,
    dt: float,
) -> Float[Array, "n_steps-1"]:
    """Total-moles mass balance residual (scalar per timestep)."""
    return mass_balance_residual_with_params(
        states,
        controls,
        disturbances,
        _cstr_training_param_vector(params),
        dt,
        params,
    )


def species_mass_balance_residuals_with_params(
    states: Float[Array, "n_steps 4"],
    controls: Float[Array, "n_steps 2"],
    disturbances: Float[Array, "n_steps 2"],
    params: Float[Array, "param_dim"],
    dt: float,
    defaults: CSTRParams,
) -> Float[Array, "n_steps-1 2"]:
    """Species-wise mass-balance residuals for a raw per-trajectory parameter vector."""
    V, _, k0, Ea_over_R, _, _, _, _, _, _, _ = _resolve_cstr_parameters(params, defaults)
    Ca = states[:, 0]
    Cb = states[:, 1]
    T = states[:, 2]
    F_in = controls[:, 0]
    Ca_in = disturbances[:, 0]

    dCa_dt = jnp.diff(Ca) / dt
    dCb_dt = jnp.diff(Cb) / dt

    k = k0 * jnp.exp(-Ea_over_R / T[:-1])
    flow_over_volume = F_in[:-1] / V

    expected_dCa_dt = flow_over_volume * (Ca_in[:-1] - Ca[:-1]) - k * Ca[:-1]
    expected_dCb_dt = -flow_over_volume * Cb[:-1] + k * Ca[:-1]

    return jnp.stack(
        [jnp.abs(dCa_dt - expected_dCa_dt), jnp.abs(dCb_dt - expected_dCb_dt)],
        axis=-1,
    )


def species_mass_balance_residuals(
    states: Float[Array, "n_steps 4"],
    controls: Float[Array, "n_steps 2"],
    disturbances: Float[Array, "n_steps 2"],
    params: CSTRParams,
    dt: float,
) -> Float[Array, "n_steps-1 2"]:
    """Species-wise (Ca, Cb) mass-balance residuals."""
    return species_mass_balance_residuals_with_params(
        states,
        controls,
        disturbances,
        _cstr_training_param_vector(params),
        dt,
        params,
    )


def reactor_energy_balance_residual_with_params(
    states: Float[Array, "n_steps 4"],
    controls: Float[Array, "n_steps 2"],
    disturbances: Float[Array, "n_steps 2"],
    params: Float[Array, "param_dim"],
    dt: float,
    defaults: CSTRParams,
) -> Float[Array, "n_steps-1"]:
    """Reactor energy balance residual for a raw per-trajectory parameter vector."""
    V, _, k0, Ea_over_R, dH_rxn, rho, Cp, UA, _, _, _ = _resolve_cstr_parameters(params, defaults)
    Ca = states[:, 0]
    T = states[:, 2]
    Tc = states[:, 3]
    F_in = controls[:, 0]
    T_in = disturbances[:, 1]

    dT_dt = jnp.diff(T) / dt
    k = k0 * jnp.exp(-Ea_over_R / T[:-1])

    flow_term = (F_in[:-1] / V) * (T_in[:-1] - T[:-1])
    reaction_term = (-dH_rxn / (rho * Cp)) * k * Ca[:-1]
    heat_transfer_term = (UA / (V * rho * Cp)) * (Tc[:-1] - T[:-1])

    expected_dT_dt = flow_term + reaction_term + heat_transfer_term
    return jnp.abs(dT_dt - expected_dT_dt)


def coolant_energy_balance_residual_with_params(
    states: Float[Array, "n_steps 4"],
    controls: Float[Array, "n_steps 2"],
    disturbances: Float[Array, "n_steps 2"],
    params: Float[Array, "param_dim"],
    dt: float,
    defaults: CSTRParams,
) -> Float[Array, "n_steps-1"]:
    """Coolant-jacket energy balance residual for a raw per-trajectory parameter vector."""
    _, Vc, _, _, _, _, _, UA, rho_c, Cp_c, Fc = _resolve_cstr_parameters(params, defaults)
    T = states[:, 2]
    Tc = states[:, 3]
    Tc_in = controls[:, 1]

    dTc_dt = jnp.diff(Tc) / dt
    flow_term = (Fc / Vc) * (Tc_in[:-1] - Tc[:-1])
    heat_transfer_term = (UA / (Vc * rho_c * Cp_c)) * (T[:-1] - Tc[:-1])
    expected_dTc_dt = flow_term + heat_transfer_term
    return jnp.abs(dTc_dt - expected_dTc_dt)


def reactor_energy_balance_residual(
    states: Float[Array, "n_steps 4"],
    controls: Float[Array, "n_steps 2"],
    disturbances: Float[Array, "n_steps 2"],
    params: CSTRParams,
    dt: float,
) -> Float[Array, "n_steps-1"]:
    return reactor_energy_balance_residual_with_params(
        states,
        controls,
        disturbances,
        _cstr_training_param_vector(params),
        dt,
        params,
    )


def coolant_energy_balance_residual(
    states: Float[Array, "n_steps 4"],
    controls: Float[Array, "n_steps 2"],
    disturbances: Float[Array, "n_steps 2"],
    params: CSTRParams,
    dt: float,
) -> Float[Array, "n_steps-1"]:
    return coolant_energy_balance_residual_with_params(
        states,
        controls,
        disturbances,
        _cstr_training_param_vector(params),
        dt,
        params,
    )


def energy_balance_residual_with_params(
    states: Float[Array, "n_steps 4"],
    controls: Float[Array, "n_steps 2"],
    disturbances: Float[Array, "n_steps 2"],
    params: Float[Array, "param_dim"],
    dt: float,
    defaults: CSTRParams,
) -> Float[Array, "n_steps-1"]:
    """Combined thermal residual averaging reactor and coolant balances."""
    reactor_residual = reactor_energy_balance_residual_with_params(
        states,
        controls,
        disturbances,
        params,
        dt,
        defaults,
    )
    coolant_residual = coolant_energy_balance_residual_with_params(
        states,
        controls,
        disturbances,
        params,
        dt,
        defaults,
    )
    return 0.5 * (reactor_residual + coolant_residual)


def energy_balance_residual(
    states: Float[Array, "n_steps 4"],
    controls: Float[Array, "n_steps 2"],
    disturbances: Float[Array, "n_steps 2"],
    params: CSTRParams,
    dt: float,
) -> Float[Array, "n_steps-1"]:
    """Combined reactor and coolant energy balance residual (scalar per timestep)."""
    return energy_balance_residual_with_params(
        states,
        controls,
        disturbances,
        _cstr_training_param_vector(params),
        dt,
        params,
    )


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
    reactor_energy_res = reactor_energy_balance_residual(states, controls, disturbances, params, dt)
    coolant_energy_res = coolant_energy_balance_residual(states, controls, disturbances, params, dt)
    energy_res = 0.5 * (reactor_energy_res + coolant_energy_res)

    return {
        "mass_residual_mean": float(jnp.mean(mass_res)),
        "mass_residual_max": float(jnp.max(mass_res)),
        "ca_mass_residual_mean": float(jnp.mean(species_res[:, 0])),
        "ca_mass_residual_max": float(jnp.max(species_res[:, 0])),
        "cb_mass_residual_mean": float(jnp.mean(species_res[:, 1])),
        "cb_mass_residual_max": float(jnp.max(species_res[:, 1])),
        "reactor_energy_residual_mean": float(jnp.mean(reactor_energy_res)),
        "reactor_energy_residual_max": float(jnp.max(reactor_energy_res)),
        "coolant_energy_residual_mean": float(jnp.mean(coolant_energy_res)),
        "coolant_energy_residual_max": float(jnp.max(coolant_energy_res)),
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
        params_batch=None,
    ) -> Dict[str, Float[Array, ""]]:
        if params_batch is None:
            nominal_params = _cstr_training_param_vector(self.params)
            params_batch = jnp.broadcast_to(nominal_params, (states.shape[0], nominal_params.shape[0]))

        def _trajectory_residuals(s, u, d, p):
            mass = jnp.mean(mass_balance_residual_with_params(s, u, d, p, dt, self.params))
            species = jnp.mean(species_mass_balance_residuals_with_params(s, u, d, p, dt, self.params))
            energy = jnp.mean(energy_balance_residual_with_params(s, u, d, p, dt, self.params))
            return mass, species, energy

        mass_vals, species_vals, energy_vals = jax.vmap(_trajectory_residuals)(
            states, controls, disturbances, params_batch
        )

        return {
            "mass": jnp.mean(mass_vals),
            "species_mass": jnp.mean(species_vals),
            "energy": jnp.mean(energy_vals),
        }

    def residual_names(self) -> list:
        return ["mass", "species_mass", "energy"]
