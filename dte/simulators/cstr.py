"""
Differentiable non-isothermal CSTR simulator using JAX and diffrax.

Implements a continuous stirred-tank reactor with:
- Single first-order reaction A → B
- Non-isothermal operation
- Cooling jacket for temperature control
"""

from dataclasses import dataclass
from typing import Callable, Dict, Tuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray
import diffrax


@dataclass(frozen=True)
class CSTRParams:
    """Parameters for the CSTR model."""
    V: float = 100.0  # Reactor volume (L)
    Vc: float = 20.0  # Coolant volume (L)
    k0: float = 7.2e10  # Pre-exponential factor (1/min)
    Ea_over_R: float = 8750.0  # Activation energy / R (K)
    dH_rxn: float = -5e4  # Heat of reaction (J/mol)
    rho: float = 1000.0  # Density (g/L)
    Cp: float = 0.239  # Specific heat capacity (J/(g*K))
    UA: float = 5e4  # Heat transfer coefficient (J/(min*K))
    rho_c: float = 1000.0  # Coolant density (g/L)
    Cp_c: float = 4.18  # Coolant specific heat (J/(g*K))
    Fc: float = 15.0  # Coolant flow rate (L/min)


class CSTRSimulator:
    """Fully differentiable non-isothermal CSTR simulator."""

    def __init__(self, params: CSTRParams):
        """Initialize simulator with given parameters.
        
        Args:
            params: CSTR parameters
        """
        self.params = params

    def _arrhenius(self, T: Float[Array, ""]) -> Float[Array, ""]:
        """Compute reaction rate constant using Arrhenius equation.
        
        Args:
            T: Temperature (K)
            
        Returns:
            k: Reaction rate constant (1/min)
        """
        k0 = jnp.asarray(self.params.k0)
        Ea_over_R = jnp.asarray(self.params.Ea_over_R)
        return k0 * jnp.exp(-Ea_over_R / T)

    def dynamics(
        self,
        t: float,
        state: Float[Array, "4"],
        control: Float[Array, "2"],
        disturbance: Float[Array, "2"],
    ) -> Float[Array, "4"]:
        """Compute state derivatives (right-hand side of ODEs).
        
        State: [Ca, Cb, T, Tc]
        Control: [F_in, Tc_in]
        Disturbance: [Ca_in, T_in]
        
        Args:
            t: Current time (not used, included for diffrax compatibility)
            state: [Ca, Cb, T, Tc]
            control: [F_in, Tc_in]
            disturbance: [Ca_in, T_in]
            
        Returns:
            state_dot: Time derivatives [dCa/dt, dCb/dt, dT/dt, dTc/dt]
        """
        # Unpack state
        Ca, Cb, T, Tc = state[0], state[1], state[2], state[3]
        
        # Unpack control and disturbance
        F_in, Tc_in = control[0], control[1]
        Ca_in, T_in = disturbance[0], disturbance[1]
        
        # Reaction rate
        k = self._arrhenius(T)
        
        # Convert params to arrays for JAX compatibility
        V = jnp.asarray(self.params.V)
        dH_rxn = jnp.asarray(self.params.dH_rxn)
        rho = jnp.asarray(self.params.rho)
        Cp = jnp.asarray(self.params.Cp)
        UA = jnp.asarray(self.params.UA)
        Fc = jnp.asarray(self.params.Fc)
        Vc = jnp.asarray(self.params.Vc)
        rho_c = jnp.asarray(self.params.rho_c)
        Cp_c = jnp.asarray(self.params.Cp_c)
        
        # Mass balances
        dCa_dt = (F_in / V) * (Ca_in - Ca) - k * Ca
        dCb_dt = (F_in / V) * (0.0 - Cb) + k * Ca
        
        # Energy balance for reactor
        dT_dt = (
            (F_in / V) * (T_in - T)
            + (-dH_rxn / (rho * Cp)) * k * Ca
            + (UA / (V * rho * Cp)) * (Tc - T)
        )
        
        # Energy balance for coolant
        dTc_dt = (
            (Fc / Vc) * (Tc_in - Tc)
            + (UA / (Vc * rho_c * Cp_c)) * (T - Tc)
        )
        
        return jnp.array([dCa_dt, dCb_dt, dT_dt, dTc_dt])

    def simulate(
        self,
        initial_state: Float[Array, "4"],
        control_trajectory: Float[Array, "n_steps 2"],
        disturbance_trajectory: Float[Array, "n_steps 2"],
        t_span: Tuple[float, float],
        dt: float = 0.1,
        n_steps: int = 1000,
    ) -> Dict[str, Float[Array, "..."]]:
        """Simulate CSTR trajectory.
        
        Args:
            initial_state: Initial state [Ca, Cb, T, Tc]
            control_trajectory: Control inputs over time (n_steps, 2)
            disturbance_trajectory: Disturbances over time (n_steps, 2)
            t_span: Time span (t0, tf)
            dt: Time step
            n_steps: Number of steps
            
        Returns:
            Dictionary with keys: time, states, controls
        """
        # Create time points
        ts = jnp.linspace(t_span[0], t_span[1], n_steps)
        
        # Create interpolation for controls and disturbances
        control_interp = diffrax.LinearInterpolation(ts, control_trajectory)
        disturbance_interp = diffrax.LinearInterpolation(ts, disturbance_trajectory)
        
        # Define ODE function compatible with diffrax
        def ode_fn(t, y, args):
            control = control_interp.evaluate(t)
            disturbance = disturbance_interp.evaluate(t)
            return self.dynamics(t, y, control, disturbance)
        
        # Set up ODE term and solver
        term = diffrax.ODETerm(ode_fn)
        solver = diffrax.Tsit5()
        saveat = diffrax.SaveAt(ts=ts)
        
        # Solve
        solution = diffrax.diffeqsolve(
            term,
            solver,
            t0=t_span[0],
            t1=t_span[1],
            dt0=dt,
            y0=initial_state,
            saveat=saveat,
            max_steps=n_steps * 10,  # Safety factor
        )
        
        return {
            "time": ts,
            "states": solution.ys,
            "controls": control_trajectory,
        }

    def steady_state(
        self,
        control: Float[Array, "2"],
        disturbance: Float[Array, "2"],
        initial_guess: Float[Array, "4"] = None,
    ) -> Float[Array, "4"]:
        """Find steady state by simulating for a long time.
        
        Args:
            control: Constant control input [F_in, Tc_in]
            disturbance: Constant disturbance [Ca_in, T_in]
            initial_guess: Initial guess for state (optional)
            
        Returns:
            Steady state [Ca, Cb, T, Tc]
        """
        if initial_guess is None:
            # Default initial guess
            initial_guess = jnp.array([0.5, 0.5, 350.0, 300.0])
        
        # Simulate for long time
        n_steps = 5000
        control_traj = jnp.tile(control[None, :], (n_steps, 1))
        disturbance_traj = jnp.tile(disturbance[None, :], (n_steps, 1))
        
        result = self.simulate(
            initial_guess,
            control_traj,
            disturbance_traj,
            t_span=(0.0, 500.0),
            dt=0.1,
            n_steps=n_steps,
        )
        
        # Return final state
        return result["states"][-1]

    def get_conservation_quantities(
        self,
        states: Float[Array, "n_steps 4"],
        controls: Float[Array, "n_steps 2"],
        disturbances: Float[Array, "n_steps 2"],
    ) -> Dict[str, Float[Array, "n_steps"]]:
        """Compute conservation quantities at each timestep.
        
        Args:
            states: State trajectory
            controls: Control trajectory
            disturbances: Disturbance trajectory
            
        Returns:
            Dictionary with conservation quantities
        """
        Ca = states[:, 0]
        Cb = states[:, 1]
        T = states[:, 2]
        
        # Total moles (Ca + Cb) - should be conserved modulo flow
        total_mass = Ca + Cb
        
        # Total energy (simplified: just temperature for now)
        total_energy = T
        
        return {
            "total_mass": total_mass,
            "total_energy": total_energy,
        }


# JIT-compiled versions of key methods
@jax.jit
def simulate_jit(
    simulator: CSTRSimulator,
    initial_state: Float[Array, "4"],
    control_trajectory: Float[Array, "n_steps 2"],
    disturbance_trajectory: Float[Array, "n_steps 2"],
    t_span: Tuple[float, float],
    dt: float,
    n_steps: int,
) -> Dict[str, Float[Array, "..."]]:
    """JIT-compiled version of simulate."""
    return simulator.simulate(
        initial_state, control_trajectory, disturbance_trajectory, t_span, dt, n_steps
    )


def sample_random_params(key: PRNGKeyArray, n: int = 1) -> CSTRParams:
    """Sample random CSTR parameters for robustness testing.
    
    Args:
        key: PRNG key
        n: Number of parameter sets to sample (currently only supports 1)
        
    Returns:
        CSTRParams with randomized values
    """
    keys = jax.random.split(key, 6)
    
    # Sample parameters with reasonable variations
    V = jax.random.uniform(keys[0], minval=50.0, maxval=200.0)
    k0 = jnp.exp(jax.random.uniform(keys[1], minval=jnp.log(1e9), maxval=jnp.log(1e12)))
    Ea_over_R = jax.random.uniform(keys[2], minval=7000.0, maxval=10000.0)
    UA = jax.random.uniform(keys[3], minval=3e4, maxval=8e4)
    Fc = jax.random.uniform(keys[4], minval=10.0, maxval=25.0)
    Vc = jax.random.uniform(keys[5], minval=10.0, maxval=30.0)
    
    return CSTRParams(
        V=float(V),
        Vc=float(Vc),
        k0=float(k0),
        Ea_over_R=float(Ea_over_R),
        dH_rxn=-5e4,
        rho=1000.0,
        Cp=0.239,
        UA=float(UA),
        rho_c=1000.0,
        Cp_c=4.18,
        Fc=float(Fc),
    )
