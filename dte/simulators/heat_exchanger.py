"""Counter-current heat exchanger simulator.

Implements a lumped-parameter counter-current heat exchanger with the ODEs:

    dT_hot/dt  = (F_hot / V_hot)  * (T_hot_in  - T_hot)
                 - (UA / (V_hot  * rho * Cp)) * (T_hot - T_cold)

    dT_cold/dt = (F_cold / V_cold) * (T_cold_in - T_cold)
                 + (UA / (V_cold * rho * Cp)) * (T_hot - T_cold)

State:       [T_hot, T_cold]     -- K
Control:     [F_hot, F_cold]     -- L/min
Disturbance: [T_hot_in, T_cold_in]  -- K
Parameters:  [V_hot, V_cold, UA, rho, Cp]
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float
import diffrax

from dte.simulators.base import ProcessSimulator, SystemSpec
from dte.simulators.registry import _build_heat_exchanger_spec


@dataclass(frozen=True)
class HeatExchangerParams:
    """Physical parameters for the counter-current heat exchanger."""

    V_hot: float = 50.0     # Hot-side volume (L)
    V_cold: float = 50.0    # Cold-side volume (L)
    UA: float = 5000.0      # Overall heat transfer coefficient * area (J/(min*K))
    rho: float = 1000.0     # Fluid density (g/L) -- same for both sides
    Cp: float = 4.18        # Specific heat capacity (J/(g*K)) -- same for both sides


class HeatExchangerSimulator(ProcessSimulator):
    """Lumped-parameter counter-current heat exchanger simulator."""

    def __init__(self, params: HeatExchangerParams):
        self._params = params
        self._spec: Optional[SystemSpec] = None

    @property
    def spec(self) -> SystemSpec:
        if self._spec is None:
            from dte.simulators.registry import _build_heat_exchanger_spec
            self._spec = _build_heat_exchanger_spec({})
        return self._spec

    @property
    def params(self) -> HeatExchangerParams:
        return self._params

    def dynamics(
        self,
        t: float,
        state: Float[Array, "2"],
        control: Float[Array, "2"],
        disturbance: Float[Array, "2"],
    ) -> Float[Array, "2"]:
        """ODE right-hand side.

        Args:
            t: Time (unused, kept for diffrax compatibility)
            state: [T_hot, T_cold] in K
            control: [F_hot, F_cold] in L/min
            disturbance: [T_hot_in, T_cold_in] in K
        """
        T_hot = state[0]
        T_cold = state[1]
        F_hot = control[0]
        F_cold = control[1]
        T_hot_in = disturbance[0]
        T_cold_in = disturbance[1]

        p = self._params
        heat_transfer = (p.UA / (p.rho * p.Cp)) * (T_hot - T_cold)

        dT_hot_dt = (F_hot / p.V_hot) * (T_hot_in - T_hot) - heat_transfer / p.V_hot
        dT_cold_dt = (F_cold / p.V_cold) * (T_cold_in - T_cold) + heat_transfer / p.V_cold

        return jnp.array([dT_hot_dt, dT_cold_dt])

    def simulate(
        self,
        initial_state: Float[Array, "2"],
        control_trajectory: Float[Array, "n_steps 2"],
        disturbance_trajectory: Float[Array, "n_steps 2"],
        t_span: Tuple[float, float],
        dt: float = 0.1,
        n_steps: int = 1000,
    ) -> Dict[str, Array]:
        """Simulate a full trajectory using diffrax.

        Args:
            initial_state: [T_hot_0, T_cold_0]
            control_trajectory: (n_steps, 2) array of [F_hot, F_cold]
            disturbance_trajectory: (n_steps, 2) array of [T_hot_in, T_cold_in]
            t_span: (t0, t1)
            dt: Integration timestep
            n_steps: Number of output steps

        Returns:
            dict with 'time', 'states', 'controls'
        """
        t0, t1 = t_span
        ts = jnp.linspace(t0, t1, n_steps)

        control_interp = diffrax.LinearInterpolation(ts, control_trajectory)
        disturbance_interp = diffrax.LinearInterpolation(ts, disturbance_trajectory)

        def _ode(t, y, args):
            u = control_interp.evaluate(t)
            d = disturbance_interp.evaluate(t)
            return self.dynamics(t, y, u, d)

        term = diffrax.ODETerm(_ode)
        solver = diffrax.Heun()
        saveat = diffrax.SaveAt(ts=ts)

        solution = diffrax.diffeqsolve(
            term,
            solver,
            t0=t0,
            t1=t1,
            dt0=dt,
            y0=initial_state,
            saveat=saveat,
            max_steps=4096,
        )
        return {
            "time": ts,
            "states": solution.ys,
            "controls": control_trajectory,
        }

    def simulate_for_data_generation(
        self,
        initial_state: Float[Array, "2"],
        control_trajectory: Float[Array, "n_steps 2"],
        disturbance_trajectory: Float[Array, "n_steps 2"],
        t_span: Tuple[float, float],
        dt: float = 0.1,
        n_steps: int = 1000,
    ) -> Dict[str, Array]:
        """Fixed-grid rollout optimized for offline dataset generation."""
        time, states = simulate_heat_exchanger_data_generation_jit(
            initial_state,
            control_trajectory,
            disturbance_trajectory,
            self.get_params_vector(),
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
        initial_state: Float[Array, "2"],
        control_trajectory: Float[Array, "n_steps 2"],
        disturbance_trajectory: Float[Array, "n_steps 2"],
        params: Float[Array, "5"],
        t_span: Tuple[float, float],
        dt: float = 0.1,
        n_steps: int = 1000,
    ) -> Dict[str, Array]:
        """Param-aware fixed-grid rollout optimized for offline dataset generation."""
        del dt, n_steps
        time, states = simulate_heat_exchanger_data_generation_jit(
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

    def steady_state(
        self,
        control: Float[Array, "2"],
        disturbance: Float[Array, "2"],
        initial_guess: Optional[Float[Array, "2"]] = None,
    ) -> Float[Array, "2"]:
        """Compute the exact steady state for constant inputs via the 2x2 closed form."""
        del initial_guess
        return self.steady_state_for_data_generation(control, disturbance)

    def steady_state_for_data_generation(
        self,
        control: Float[Array, "2"],
        disturbance: Float[Array, "2"],
        initial_guess: Optional[Float[Array, "2"]] = None,
    ) -> Float[Array, "2"]:
        """Closed-form steady state for constant inputs.

        For
            0 = a_h (T_hot_in - T_hot) - b_h (T_hot - T_cold)
            0 = a_c (T_cold_in - T_cold) + b_c (T_hot - T_cold)
        solve the 2x2 linear system directly.
        """
        p = self._params
        F_hot = jnp.maximum(control[0], 1e-6)
        F_cold = jnp.maximum(control[1], 1e-6)
        T_hot_in = disturbance[0]
        T_cold_in = disturbance[1]

        a_h = F_hot / p.V_hot
        a_c = F_cold / p.V_cold
        b_h = p.UA / (p.V_hot * p.rho * p.Cp)
        b_c = p.UA / (p.V_cold * p.rho * p.Cp)

        matrix = jnp.array([
            [a_h + b_h, -b_h],
            [-b_c, a_c + b_c],
        ])
        rhs = jnp.array([
            a_h * T_hot_in,
            a_c * T_cold_in,
        ])
        return jnp.linalg.solve(matrix, rhs)

    def steady_state_for_data_generation_with_params(
        self,
        control: Float[Array, "2"],
        disturbance: Float[Array, "2"],
        params: Float[Array, "5"],
        initial_guess: Optional[Float[Array, "2"]] = None,
    ) -> Float[Array, "2"]:
        """Param-aware closed-form steady state for constant inputs."""
        del initial_guess
        return steady_state_heat_exchanger_jit(control, disturbance, params)

    def steady_state_batch_for_data_generation(
        self,
        controls: Float[Array, "batch 2"],
        disturbances: Float[Array, "batch 2"],
        params_batch: Optional[Float[Array, "batch 5"]] = None,
        initial_guesses: Optional[Float[Array, "batch 2"]] = None,
    ) -> Float[Array, "batch 2"]:
        """Vectorized closed-form steady states for offline data generation."""
        del initial_guesses
        if params_batch is None:
            params_batch = jnp.tile(self.get_params_vector()[None, :], (controls.shape[0], 1))
        return steady_state_heat_exchanger_batch_jit(
            controls,
            disturbances,
            params_batch,
        )

    def simulate_batch_for_data_generation(
        self,
        initial_states: Float[Array, "batch 2"],
        control_trajectories: Float[Array, "batch n_steps 2"],
        disturbance_trajectories: Float[Array, "batch n_steps 2"],
        t_span: Tuple[float, float],
        params_batch: Optional[Float[Array, "batch 5"]] = None,
        dt: float = 0.1,
        n_steps: int = 1000,
    ) -> Dict[str, Array]:
        """Vectorized fixed-grid rollout for offline dataset generation."""
        del dt, n_steps
        if params_batch is None:
            params_batch = jnp.tile(self.get_params_vector()[None, :], (initial_states.shape[0], 1))
        times, states = simulate_heat_exchanger_data_generation_batch_jit(
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
        key,
    ) -> Float[Array, "5"]:
        """Sample physical parameters for one generated trajectory."""
        sampled = self.sample_params(key)
        return jnp.array(
            [sampled.V_hot, sampled.V_cold, sampled.UA, sampled.rho, sampled.Cp]
        )

    def apply_measurement_noise(
        self,
        key,
        states: Float[Array, "n_steps 2"],
    ) -> Float[Array, "n_steps 2"]:
        """Add mild temperature sensor noise during offline data generation."""
        noise_std = jnp.array([0.5, 0.5])
        noise = jax.random.normal(key, shape=states.shape) * noise_std[None, :]
        return states + noise

    def is_valid_trajectory(
        self,
        states: Float[Array, "n_steps 2"],
    ) -> bool:
        """Reject non-finite or physically implausible temperature trajectories."""
        return bool(
            jnp.all(jnp.isfinite(states))
            & jnp.all(states[:, 0] >= 200.0)
            & jnp.all(states[:, 1] >= 200.0)
            & jnp.all(states[:, 0] <= 500.0)
            & jnp.all(states[:, 1] <= 500.0)
        )

    def get_params_vector(self) -> Float[Array, "5"]:
        """Return parameters as a JAX array [V_hot, V_cold, UA, rho, Cp]."""
        p = self._params
        return jnp.array([p.V_hot, p.V_cold, p.UA, p.rho, p.Cp])

    def sample_params(
        self,
        key,
        variation: float = 0.1,
    ) -> HeatExchangerParams:
        """Sample randomised parameters within ±variation of nominals."""
        k1, k2, k3, k4, k5 = jax.random.split(key, 5)
        p = self._params
        scale = lambda k, nom: float(nom * (1.0 + variation * jax.random.uniform(k, minval=-1, maxval=1)))
        return HeatExchangerParams(
            V_hot=scale(k1, p.V_hot),
            V_cold=scale(k2, p.V_cold),
            UA=scale(k3, p.UA),
            rho=float(p.rho),
            Cp=float(p.Cp),
        )


@jax.jit
def _heat_exchanger_dynamics_with_params(
    state: Float[Array, "2"],
    control: Float[Array, "2"],
    disturbance: Float[Array, "2"],
    params: Float[Array, "5"],
) -> Float[Array, "2"]:
    T_hot, T_cold = state
    F_hot, F_cold = control
    T_hot_in, T_cold_in = disturbance
    V_hot, V_cold, UA, rho, Cp = params

    heat_transfer = (UA / (rho * Cp)) * (T_hot - T_cold)
    dT_hot_dt = (F_hot / V_hot) * (T_hot_in - T_hot) - heat_transfer / V_hot
    dT_cold_dt = (F_cold / V_cold) * (T_cold_in - T_cold) + heat_transfer / V_cold
    return jnp.array([dT_hot_dt, dT_cold_dt])


@jax.jit
def steady_state_heat_exchanger_jit(
    control: Float[Array, "2"],
    disturbance: Float[Array, "2"],
    params: Float[Array, "5"],
) -> Float[Array, "2"]:
    """Closed-form steady state for a single constant-input operating point."""
    F_hot = jnp.maximum(control[0], 1e-6)
    F_cold = jnp.maximum(control[1], 1e-6)
    T_hot_in = disturbance[0]
    T_cold_in = disturbance[1]
    V_hot, V_cold, UA, rho, Cp = params

    a_h = F_hot / V_hot
    a_c = F_cold / V_cold
    b_h = UA / (V_hot * rho * Cp)
    b_c = UA / (V_cold * rho * Cp)

    matrix = jnp.array([
        [a_h + b_h, -b_h],
        [-b_c, a_c + b_c],
    ])
    rhs = jnp.array([
        a_h * T_hot_in,
        a_c * T_cold_in,
    ])
    return jnp.linalg.solve(matrix, rhs)


@jax.jit
def steady_state_heat_exchanger_batch_jit(
    controls: Float[Array, "batch 2"],
    disturbances: Float[Array, "batch 2"],
    params_batch: Float[Array, "batch 5"],
) -> Float[Array, "batch 2"]:
    """Vectorized closed-form steady-state solve."""
    return jax.vmap(steady_state_heat_exchanger_jit, in_axes=(0, 0, 0))(
        controls,
        disturbances,
        params_batch,
    )


@jax.jit
def simulate_heat_exchanger_data_generation_jit(
    initial_state: Float[Array, "2"],
    control_trajectory: Float[Array, "n_steps 2"],
    disturbance_trajectory: Float[Array, "n_steps 2"],
    params: Float[Array, "5"],
    t0: float,
    t1: float,
) -> Tuple[Float[Array, "n_steps"], Float[Array, "n_steps 2"]]:
    """Fixed-grid RK4 rollout used by offline data generation."""
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

        k1 = _heat_exchanger_dynamics_with_params(state, control_0, disturbance_0, params)
        k2 = _heat_exchanger_dynamics_with_params(
            state + 0.5 * step_dt * k1,
            control_mid,
            disturbance_mid,
            params,
        )
        k3 = _heat_exchanger_dynamics_with_params(
            state + 0.5 * step_dt * k2,
            control_mid,
            disturbance_mid,
            params,
        )
        k4 = _heat_exchanger_dynamics_with_params(
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
def simulate_heat_exchanger_data_generation_batch_jit(
    initial_states: Float[Array, "batch 2"],
    control_trajectories: Float[Array, "batch n_steps 2"],
    disturbance_trajectories: Float[Array, "batch n_steps 2"],
    params_batch: Float[Array, "batch 5"],
    t0: float,
    t1: float,
) -> Tuple[Float[Array, "batch n_steps"], Float[Array, "batch n_steps 2"]]:
    """Vectorized fixed-grid RK4 rollout used by offline data generation."""
    return jax.vmap(
        simulate_heat_exchanger_data_generation_jit,
        in_axes=(0, 0, 0, 0, None, None),
    )(
        initial_states,
        control_trajectories,
        disturbance_trajectories,
        params_batch,
        t0,
        t1,
    )
