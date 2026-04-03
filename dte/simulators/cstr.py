"""
Differentiable non-isothermal CSTR simulator using JAX and diffrax.

Implements a continuous stirred-tank reactor with:
- Single first-order reaction A → B
- Non-isothermal operation
- Cooling jacket for temperature control
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray
import diffrax

from dte.simulators.base import ProcessSimulator, SystemSpec


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


class CSTRSimulator(ProcessSimulator):
    """Fully differentiable non-isothermal CSTR simulator."""

    def __init__(self, params: CSTRParams):
        """Initialize simulator with given parameters.
        
        Args:
            params: CSTR parameters
        """
        self.params = params
        self._spec: Optional[SystemSpec] = None

    @property
    def spec(self) -> SystemSpec:
        if self._spec is None:
            from dte.simulators.registry import _build_cstr_spec

            self._spec = _build_cstr_spec({})
        return self._spec

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

    def simulate_for_data_generation(
        self,
        initial_state: Float[Array, "4"],
        control_trajectory: Float[Array, "n_steps 2"],
        disturbance_trajectory: Float[Array, "n_steps 2"],
        t_span: Tuple[float, float],
        dt: float = 0.1,
        n_steps: int = 1000,
    ) -> Dict[str, Float[Array, "..."]]:
        """Simulate on a fixed grid for offline data generation.

        This path keeps the same public contract as ``simulate`` but avoids the
        adaptive-solver overhead that is useful for reference simulations and less
        useful for large-scale dataset creation on a uniform time grid.
        """
        time, states = simulate_data_generation_jit(
            initial_state,
            control_trajectory,
            disturbance_trajectory,
            _pack_params(self.params),
            t_span[0],
            t_span[1],
        )

        return {
            "time": time,
            "states": states,
            "controls": control_trajectory,
        }

    def simulate_for_data_generation_with_params(
        self,
        initial_state: Float[Array, "4"],
        control_trajectory: Float[Array, "n_steps 2"],
        disturbance_trajectory: Float[Array, "n_steps 2"],
        params: Float[Array, "11"],
        t_span: Tuple[float, float],
        dt: float = 0.1,
        n_steps: int = 1000,
    ) -> Dict[str, Float[Array, "..."]]:
        """Param-aware fixed-grid rollout for offline data generation."""
        del dt, n_steps
        time, states = simulate_data_generation_jit(
            initial_state,
            control_trajectory,
            disturbance_trajectory,
            params,
            t_span[0],
            t_span[1],
        )
        return {
            "time": time,
            "states": states,
            "controls": control_trajectory,
        }

    def _steady_state_residual(
        self,
        T: Float[Array, ""],
        control: Float[Array, "2"],
        disturbance: Float[Array, "2"],
    ) -> Float[Array, ""]:
        """Energy balance residual after eliminating Ca, Cb, and Tc analytically."""
        F_in, Tc_in = control[0], control[1]
        Ca_in, T_in = disturbance[0], disturbance[1]

        V = jnp.asarray(self.params.V)
        dH_rxn = jnp.asarray(self.params.dH_rxn)
        rho = jnp.asarray(self.params.rho)
        Cp = jnp.asarray(self.params.Cp)
        UA = jnp.asarray(self.params.UA)
        Fc = jnp.asarray(self.params.Fc)
        Vc = jnp.asarray(self.params.Vc)
        rho_c = jnp.asarray(self.params.rho_c)
        Cp_c = jnp.asarray(self.params.Cp_c)

        flow_over_volume = F_in / V
        k = self._arrhenius(T)
        Ca = (flow_over_volume * Ca_in) / (flow_over_volume + k)

        coolant_influence = Fc / Vc
        heat_exchange = UA / (Vc * rho_c * Cp_c)
        Tc = (
            coolant_influence * Tc_in + heat_exchange * T
        ) / (coolant_influence + heat_exchange)

        reaction_heating = (-dH_rxn / (rho * Cp)) * k * Ca
        jacket_coupling = (UA / (V * rho * Cp)) * (Tc - T)
        feed_term = flow_over_volume * (T_in - T)

        return feed_term + reaction_heating + jacket_coupling

    def _steady_state_via_simulation(
        self,
        control: Float[Array, "2"],
        disturbance: Float[Array, "2"],
        initial_guess: Float[Array, "4"],
    ) -> Float[Array, "4"]:
        """Fallback steady-state estimate by long-horizon simulation."""
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

        return result["states"][-1]

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

        residual_fn = lambda T: self._steady_state_residual(T, control, disturbance)
        residual_grad = jax.grad(residual_fn)

        T = float(initial_guess[2])
        converged = False

        for _ in range(25):
            residual = float(residual_fn(T))
            if not jnp.isfinite(jnp.asarray(residual)).item():
                break

            if abs(residual) < 1e-6:
                converged = True
                break

            slope = float(residual_grad(T))
            if not jnp.isfinite(jnp.asarray(slope)).item() or abs(slope) < 1e-8:
                break

            # Damped Newton step to keep the iterate in a physically reasonable range.
            step = jnp.clip(residual / slope, -25.0, 25.0)
            next_T = float(jnp.clip(T - step, 250.0, 500.0))

            if abs(next_T - T) < 1e-6:
                T = next_T
                converged = True
                break

            T = next_T

        final_residual = float(residual_fn(T))
        if (
            not converged
            or not jnp.isfinite(jnp.asarray(final_residual)).item()
            or abs(final_residual) > 1e-4
        ):
            return self._steady_state_via_simulation(control, disturbance, initial_guess)

        F_in = control[0]
        Tc_in = control[1]
        Ca_in = disturbance[0]

        V = jnp.asarray(self.params.V)
        flow_over_volume = F_in / V
        k = self._arrhenius(T)
        Ca = (flow_over_volume * Ca_in) / (flow_over_volume + k)
        Cb = Ca_in - Ca

        coolant_influence = jnp.asarray(self.params.Fc) / jnp.asarray(self.params.Vc)
        heat_exchange = (
            jnp.asarray(self.params.UA)
            / (
                jnp.asarray(self.params.Vc)
                * jnp.asarray(self.params.rho_c)
                * jnp.asarray(self.params.Cp_c)
            )
        )
        Tc = (
            coolant_influence * Tc_in + heat_exchange * T
        ) / (coolant_influence + heat_exchange)

        return jnp.array([Ca, Cb, T, Tc])

    def steady_state_for_data_generation_with_params(
        self,
        control: Float[Array, "2"],
        disturbance: Float[Array, "2"],
        params: Float[Array, "11"],
        initial_guess: Optional[Float[Array, "4"]] = None,
    ) -> Float[Array, "4"]:
        """Param-aware steady state used by the generic generation pipeline."""
        if initial_guess is None:
            initial_guess = jnp.array([0.5, 0.5, 350.0, 300.0])
        state = steady_state_from_packed_params_jit(control, disturbance, initial_guess, params)
        residual = jnp.abs(_steady_state_residual_with_params(state[2], control, disturbance, params))
        if bool(jnp.isfinite(residual) & (residual <= 1e-4)):
            return state
        return CSTRSimulator(unpack_params(params)).steady_state(control, disturbance, initial_guess)

    def steady_state_batch_for_data_generation(
        self,
        controls: Float[Array, "batch 2"],
        disturbances: Float[Array, "batch 2"],
        params_batch: Optional[Float[Array, "batch 11"]] = None,
        initial_guesses: Optional[Float[Array, "batch 4"]] = None,
    ) -> Float[Array, "batch 4"]:
        """Vectorized steady-state solve with robust fallback handling."""
        if params_batch is None:
            params_batch = jnp.tile(_pack_params(self.params)[None, :], (controls.shape[0], 1))
        if initial_guesses is None:
            initial_guesses = jnp.tile(jnp.array([[0.5, 0.5, 350.0, 300.0]]), (controls.shape[0], 1))
        alt_temperature_guess = 0.5 * (disturbances[:, 1] + controls[:, 1]) + 20.0
        alternative_initial_guesses = initial_guesses.at[:, 2].set(alt_temperature_guess)
        candidate_states_default, residuals_default = steady_state_batch_with_residuals_jit(
            controls,
            disturbances,
            initial_guesses,
            params_batch,
        )
        candidate_states_alt, residuals_alt = steady_state_batch_with_residuals_jit(
            controls,
            disturbances,
            alternative_initial_guesses,
            params_batch,
        )
        use_alternative_mask = residuals_alt < residuals_default
        residuals = jnp.where(use_alternative_mask, residuals_alt, residuals_default)
        candidate_states = jnp.where(
            use_alternative_mask[:, None],
            candidate_states_alt,
            candidate_states_default,
        )
        valid_fast_mask = jnp.isfinite(residuals) & (residuals <= 1e-4)
        states = []
        for idx in range(controls.shape[0]):
            if bool(valid_fast_mask[idx]):
                states.append(candidate_states[idx])
            else:
                params_obj = unpack_params(params_batch[idx])
                states.append(
                    CSTRSimulator(params_obj).steady_state(
                        controls[idx],
                        disturbances[idx],
                        initial_guesses[idx],
                    )
                )
        return jnp.stack(states)

    def simulate_batch_for_data_generation(
        self,
        initial_states: Float[Array, "batch 4"],
        control_trajectories: Float[Array, "batch n_steps 2"],
        disturbance_trajectories: Float[Array, "batch n_steps 2"],
        t_span: Tuple[float, float],
        params_batch: Optional[Float[Array, "batch 11"]] = None,
        dt: float = 0.1,
        n_steps: int = 1000,
    ) -> Dict[str, Float[Array, "..."]]:
        """Vectorized fixed-grid rollout used by the generic generation pipeline."""
        del dt, n_steps
        if params_batch is None:
            params_batch = jnp.tile(_pack_params(self.params)[None, :], (initial_states.shape[0], 1))
        times, states = simulate_data_generation_batch_jit(
            initial_states,
            control_trajectories,
            disturbance_trajectories,
            params_batch,
            t_span[0],
            t_span[1],
        )
        return {
            "time": times,
            "states": states,
            "controls": control_trajectories,
        }

    def sample_data_generation_params(
        self,
        key: PRNGKeyArray,
    ) -> Float[Array, "11"]:
        """Sample packed simulator parameters for offline data generation."""
        return _pack_params(sample_random_params(key))

    def apply_measurement_noise(
        self,
        key: PRNGKeyArray,
        states: Float[Array, "n_steps 4"],
    ) -> Float[Array, "n_steps 4"]:
        """Add CSTR measurement noise during offline data generation."""
        noise_std = jnp.array([0.01, 0.01, 0.5, 0.5])
        state_noise = jax.random.normal(key, shape=states.shape) * noise_std[None, :]
        return states + state_noise

    def is_valid_trajectory(
        self,
        states: Float[Array, "n_steps 4"],
    ) -> bool:
        """Reject non-finite or physically implausible CSTR trajectories."""
        return bool(
            jnp.all(jnp.isfinite(states))
            & jnp.all(states[:, 0] >= 0.0)
            & jnp.all(states[:, 1] >= -0.5)
        )

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


def _pack_params(params: CSTRParams) -> Float[Array, "11"]:
    """Pack simulator parameters into an array for JIT-friendly helper functions."""
    return jnp.array(
        [
            params.V,
            params.Vc,
            params.k0,
            params.Ea_over_R,
            params.dH_rxn,
            params.rho,
            params.Cp,
            params.UA,
            params.rho_c,
            params.Cp_c,
            params.Fc,
        ]
    )


def pack_params(params: CSTRParams) -> Float[Array, "11"]:
    """Public wrapper for packing simulator parameters into an array."""
    return _pack_params(params)


def unpack_params(params: Float[Array, "11"]) -> CSTRParams:
    """Unpack a packed parameter array into a ``CSTRParams`` object."""
    return CSTRParams(
        V=float(params[0]),
        Vc=float(params[1]),
        k0=float(params[2]),
        Ea_over_R=float(params[3]),
        dH_rxn=float(params[4]),
        rho=float(params[5]),
        Cp=float(params[6]),
        UA=float(params[7]),
        rho_c=float(params[8]),
        Cp_c=float(params[9]),
        Fc=float(params[10]),
    )


def _dynamics_with_params(
    state: Float[Array, "4"],
    control: Float[Array, "2"],
    disturbance: Float[Array, "2"],
    params: Float[Array, "11"],
) -> Float[Array, "4"]:
    """JIT-friendly CSTR dynamics using packed parameter arrays."""
    V, Vc, k0, Ea_over_R, dH_rxn, rho, Cp, UA, rho_c, Cp_c, Fc = params

    Ca, Cb, T, Tc = state[0], state[1], state[2], state[3]
    F_in, Tc_in = control[0], control[1]
    Ca_in, T_in = disturbance[0], disturbance[1]

    k = k0 * jnp.exp(-Ea_over_R / T)

    dCa_dt = (F_in / V) * (Ca_in - Ca) - k * Ca
    dCb_dt = (F_in / V) * (0.0 - Cb) + k * Ca
    dT_dt = (
        (F_in / V) * (T_in - T)
        + (-dH_rxn / (rho * Cp)) * k * Ca
        + (UA / (V * rho * Cp)) * (Tc - T)
    )
    dTc_dt = (
        (Fc / Vc) * (Tc_in - Tc)
        + (UA / (Vc * rho_c * Cp_c)) * (T - Tc)
    )

    return jnp.array([dCa_dt, dCb_dt, dT_dt, dTc_dt])


def _steady_state_residual_with_params(
    T: Float[Array, ""],
    control: Float[Array, "2"],
    disturbance: Float[Array, "2"],
    params: Float[Array, "11"],
) -> Float[Array, ""]:
    """Energy balance residual using packed parameter arrays."""
    V, Vc, k0, Ea_over_R, dH_rxn, rho, Cp, UA, rho_c, Cp_c, Fc = params
    F_in, Tc_in = control[0], control[1]
    Ca_in, T_in = disturbance[0], disturbance[1]

    flow_over_volume = F_in / V
    k = k0 * jnp.exp(-Ea_over_R / T)
    Ca = (flow_over_volume * Ca_in) / (flow_over_volume + k)

    coolant_influence = Fc / Vc
    heat_exchange = UA / (Vc * rho_c * Cp_c)
    Tc = (
        coolant_influence * Tc_in + heat_exchange * T
    ) / (coolant_influence + heat_exchange)

    reaction_heating = (-dH_rxn / (rho * Cp)) * k * Ca
    jacket_coupling = (UA / (V * rho * Cp)) * (Tc - T)
    feed_term = flow_over_volume * (T_in - T)

    return feed_term + reaction_heating + jacket_coupling


@jax.jit
def steady_state_from_packed_params_jit(
    control: Float[Array, "2"],
    disturbance: Float[Array, "2"],
    initial_guess: Float[Array, "4"],
    params: Float[Array, "11"],
) -> Float[Array, "4"]:
    """Fast steady-state estimate using packed parameters and damped Newton iterations."""
    initial_T = jnp.clip(initial_guess[2], 250.0, 500.0)

    def body_fn(_, T):
        residual = _steady_state_residual_with_params(T, control, disturbance, params)
        slope = jax.grad(_steady_state_residual_with_params, argnums=0)(
            T, control, disturbance, params
        )
        safe_slope = jnp.where(jnp.abs(slope) < 1e-8, 1e-8, slope)
        step = jnp.clip(residual / safe_slope, -25.0, 25.0)
        return jnp.clip(T - step, 250.0, 500.0)

    T = jax.lax.fori_loop(0, 25, body_fn, initial_T)

    V, Vc, k0, Ea_over_R, _, _, _, UA, rho_c, Cp_c, Fc = params
    F_in, Tc_in = control[0], control[1]
    Ca_in = disturbance[0]

    flow_over_volume = F_in / V
    k = k0 * jnp.exp(-Ea_over_R / T)
    Ca = (flow_over_volume * Ca_in) / (flow_over_volume + k)
    Cb = Ca_in - Ca

    coolant_influence = Fc / Vc
    heat_exchange = UA / (Vc * rho_c * Cp_c)
    Tc = (
        coolant_influence * Tc_in + heat_exchange * T
    ) / (coolant_influence + heat_exchange)

    return jnp.array([Ca, Cb, T, Tc])


@jax.jit
def steady_state_batch_jit(
    controls: Float[Array, "batch 2"],
    disturbances: Float[Array, "batch 2"],
    initial_guesses: Float[Array, "batch 4"],
    params_batch: Float[Array, "batch 11"],
) -> Float[Array, "batch 4"]:
    """Vectorized fast steady-state solve for a batch of parameter sets."""
    return jax.vmap(steady_state_from_packed_params_jit)(
        controls,
        disturbances,
        initial_guesses,
        params_batch,
    )


@jax.jit
def steady_state_batch_with_residuals_jit(
    controls: Float[Array, "batch 2"],
    disturbances: Float[Array, "batch 2"],
    initial_guesses: Float[Array, "batch 4"],
    params_batch: Float[Array, "batch 11"],
) -> Tuple[Float[Array, "batch 4"], Float[Array, "batch"]]:
    """Vectorized steady-state solve that also returns final residual magnitudes."""
    states = steady_state_batch_jit(
        controls,
        disturbances,
        initial_guesses,
        params_batch,
    )
    residuals = jax.vmap(
        lambda state, control, disturbance, params: jnp.abs(
            _steady_state_residual_with_params(state[2], control, disturbance, params)
        )
    )(states, controls, disturbances, params_batch)
    return states, residuals


@jax.jit
def simulate_data_generation_jit(
    initial_state: Float[Array, "4"],
    control_trajectory: Float[Array, "n_steps 2"],
    disturbance_trajectory: Float[Array, "n_steps 2"],
    params: Float[Array, "11"],
    t0: float,
    t1: float,
) -> Tuple[Float[Array, "n_steps"], Float[Array, "n_steps 4"]]:
    """Fixed-grid RK4 rollout used by the data-generation pipeline."""
    n_steps = control_trajectory.shape[0]
    ts = jnp.linspace(t0, t1, n_steps)

    if n_steps <= 1:
        return ts, initial_state[None, :]

    step_dt = ts[1] - ts[0]
    control_start = control_trajectory[:-1]
    control_end = control_trajectory[1:]
    disturbance_start = disturbance_trajectory[:-1]
    disturbance_end = disturbance_trajectory[1:]

    def step_fn(state, inputs):
        control_0, control_1, disturbance_0, disturbance_1 = inputs
        control_mid = 0.5 * (control_0 + control_1)
        disturbance_mid = 0.5 * (disturbance_0 + disturbance_1)

        k1 = _dynamics_with_params(state, control_0, disturbance_0, params)
        k2 = _dynamics_with_params(
            state + 0.5 * step_dt * k1,
            control_mid,
            disturbance_mid,
            params,
        )
        k3 = _dynamics_with_params(
            state + 0.5 * step_dt * k2,
            control_mid,
            disturbance_mid,
            params,
        )
        k4 = _dynamics_with_params(
            state + step_dt * k3,
            control_1,
            disturbance_1,
            params,
        )

        next_state = state + (step_dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return next_state, next_state

    _, states_tail = jax.lax.scan(
        step_fn,
        initial_state,
        (control_start, control_end, disturbance_start, disturbance_end),
    )
    states = jnp.concatenate([initial_state[None, :], states_tail], axis=0)
    return ts, states


@jax.jit
def simulate_data_generation_batch_jit(
    initial_states: Float[Array, "batch 4"],
    control_trajectories: Float[Array, "batch n_steps 2"],
    disturbance_trajectories: Float[Array, "batch n_steps 2"],
    params_batch: Float[Array, "batch 11"],
    t0: float,
    t1: float,
) -> Tuple[Float[Array, "batch n_steps"], Float[Array, "batch n_steps 4"]]:
    """Vectorized fixed-grid rollout used by the batched data-generation path."""
    return jax.vmap(simulate_data_generation_jit, in_axes=(0, 0, 0, 0, None, None))(
        initial_states,
        control_trajectories,
        disturbance_trajectories,
        params_batch,
        t0,
        t1,
    )


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
