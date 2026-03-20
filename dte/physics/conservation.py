"""Conservation law computations for CSTR physics."""

from typing import Dict
import jax.numpy as jnp
from jaxtyping import Array, Float

from dte.simulators.cstr import CSTRParams


def mass_balance_residual(
    states: Float[Array, "n_steps 4"],
    controls: Float[Array, "n_steps 2"],
    disturbances: Float[Array, "n_steps 2"],
    params: CSTRParams,
    dt: float,
) -> Float[Array, "n_steps-1"]:
    """Compute mass balance residual at each timestep.
    
    For CSTR: d(V*Ca)/dt = F_in*Ca_in - F_out*Ca - V*k(T)*Ca
              d(V*Cb)/dt = F_in*0    - F_out*Cb + V*k(T)*Ca
    Total moles: d(V*(Ca+Cb))/dt = F_in*Ca_in - F_out*(Ca+Cb)
    
    Residual = |d(Ca+Cb)/dt - (F_in/V)*(Ca_in - Ca - Cb)|
    
    Args:
        states: State trajectory [Ca, Cb, T, Tc]
        controls: Control trajectory [F_in, Tc_in]
        disturbances: Disturbance trajectory [Ca_in, T_in]
        params: CSTR parameters
        dt: Time step
        
    Returns:
        Mass balance residuals (n_steps-1,)
    """
    Ca = states[:, 0]
    Cb = states[:, 1]
    
    F_in = controls[:, 0]
    Ca_in = disturbances[:, 0]
    
    # Total moles
    total_moles = Ca + Cb
    
    # Time derivative (finite difference)
    d_total_moles_dt = jnp.diff(total_moles) / dt
    
    # Expected change from mass balance
    # Assuming F_out = F_in (constant volume)
    expected_change = (F_in[:-1] / params.V) * (Ca_in[:-1] - total_moles[:-1])
    
    # Residual
    residual = jnp.abs(d_total_moles_dt - expected_change)
    
    return residual


def energy_balance_residual(
    states: Float[Array, "n_steps 4"],
    controls: Float[Array, "n_steps 2"],
    disturbances: Float[Array, "n_steps 2"],
    params: CSTRParams,
    dt: float,
) -> Float[Array, "n_steps-1"]:
    """Compute energy balance residual.
    
    For CSTR: rho*Cp*V*dT/dt = F_in*rho*Cp*(T_in-T) + V*(-dH)*k(T)*Ca + UA*(Tc-T)
    
    Residual = |rho*Cp*V*dT/dt - RHS|
    
    Args:
        states: State trajectory [Ca, Cb, T, Tc]
        controls: Control trajectory [F_in, Tc_in]
        disturbances: Disturbance trajectory [Ca_in, T_in]
        params: CSTR parameters
        dt: Time step
        
    Returns:
        Energy balance residuals (n_steps-1,)
    """
    Ca = states[:, 0]
    T = states[:, 2]
    Tc = states[:, 3]
    
    F_in = controls[:, 0]
    T_in = disturbances[:, 1]
    
    # Time derivative of temperature (finite difference)
    dT_dt = jnp.diff(T) / dt
    
    # Reaction rate constant (Arrhenius)
    k = params.k0 * jnp.exp(-params.Ea_over_R / T[:-1])
    
    # Energy balance RHS (all terms)
    flow_term = (F_in[:-1] / params.V) * (T_in[:-1] - T[:-1])
    reaction_term = (-params.dH_rxn / (params.rho * params.Cp)) * k * Ca[:-1]
    heat_transfer_term = (params.UA / (params.V * params.rho * params.Cp)) * (Tc[:-1] - T[:-1])
    
    expected_dT_dt = flow_term + reaction_term + heat_transfer_term
    
    # Residual
    residual = jnp.abs(dT_dt - expected_dT_dt)
    
    return residual


def total_conservation_metric(
    states: Float[Array, "n_steps 4"],
    controls: Float[Array, "n_steps 2"],
    disturbances: Float[Array, "n_steps 2"],
    params: CSTRParams,
    dt: float,
) -> Dict[str, float]:
    """Compute conservation metrics.
    
    Args:
        states: State trajectory
        controls: Control trajectory
        disturbances: Disturbance trajectory
        params: CSTR parameters
        dt: Time step
        
    Returns:
        Dictionary with conservation metrics
    """
    mass_res = mass_balance_residual(states, controls, disturbances, params, dt)
    energy_res = energy_balance_residual(states, controls, disturbances, params, dt)
    
    return {
        "mass_residual_mean": float(jnp.mean(mass_res)),
        "mass_residual_max": float(jnp.max(mass_res)),
        "energy_residual_mean": float(jnp.mean(energy_res)),
        "energy_residual_max": float(jnp.max(energy_res)),
    }
