"""Coupled two-tank level process simulator.

State:       [h1, h2]          -- liquid levels in m
Control:     [q_in, valve]     -- inlet flow and outlet valve opening
Disturbance: [d1, d2]          -- external inflows to tank 1 and tank 2
Parameters:  [A1, A2, k12, kout, h_max]
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import diffrax
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from dte.simulators.base import ProcessSimulator, SystemSpec


@dataclass(frozen=True)
class TwoTankParams:
    """Physical parameters for the coupled two-tank process."""

    A1: float = 1.5
    A2: float = 1.2
    k12: float = 0.9
    kout: float = 1.0
    h_max: float = 5.0


class TwoTankSimulator(ProcessSimulator):
    """Coupled two-tank level process with nonlinear gravity-driven flows."""

    def __init__(self, params: TwoTankParams):
        self._params = params
        self._spec: Optional[SystemSpec] = None

    @property
    def spec(self) -> SystemSpec:
        if self._spec is None:
            from dte.simulators.registry import _build_two_tank_spec

            self._spec = _build_two_tank_spec({})
        return self._spec

    @property
    def params(self) -> TwoTankParams:
        return self._params

    def dynamics(
        self,
        t: float,
        state: Float[Array, "2"],
        control: Float[Array, "2"],
        disturbance: Float[Array, "2"],
    ) -> Float[Array, "2"]:
        """ODE right-hand side for the coupled tanks."""
        del t
        return _two_tank_dynamics_with_params(
            state,
            control,
            disturbance,
            self.get_params_vector(),
        )

    def simulate(
        self,
        initial_state: Float[Array, "2"],
        control_trajectory: Float[Array, "n_steps 2"],
        disturbance_trajectory: Float[Array, "n_steps 2"],
        t_span: Tuple[float, float],
        dt: float = 0.1,
        n_steps: int = 1000,
    ) -> Dict[str, Array]:
        """Simulate a full trajectory using diffrax."""
        t0, t1 = t_span
        ts = jnp.linspace(t0, t1, n_steps)

        control_interp = diffrax.LinearInterpolation(ts, control_trajectory)
        disturbance_interp = diffrax.LinearInterpolation(ts, disturbance_trajectory)

        def _ode(t, y, args):
            del args
            u = control_interp.evaluate(t)
            d = disturbance_interp.evaluate(t)
            return self.dynamics(t, y, u, d)

        solution = diffrax.diffeqsolve(
            diffrax.ODETerm(_ode),
            diffrax.Heun(),
            t0=t0,
            t1=t1,
            dt0=dt,
            y0=initial_state,
            saveat=diffrax.SaveAt(ts=ts),
            max_steps=max(4096, n_steps * 4),
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
        del dt, n_steps
        time, states = simulate_two_tank_data_generation_jit(
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
        """Param-aware fixed-grid rollout for offline dataset generation."""
        del dt, n_steps
        time, states = simulate_two_tank_data_generation_jit(
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
        """Compute a steady state for constant inputs."""
        del initial_guess
        return steady_state_two_tank_jit(control, disturbance, self.get_params_vector())

    def steady_state_for_data_generation(
        self,
        control: Float[Array, "2"],
        disturbance: Float[Array, "2"],
        initial_guess: Optional[Float[Array, "2"]] = None,
    ) -> Float[Array, "2"]:
        del initial_guess
        return steady_state_two_tank_jit(control, disturbance, self.get_params_vector())

    def steady_state_for_data_generation_with_params(
        self,
        control: Float[Array, "2"],
        disturbance: Float[Array, "2"],
        params: Float[Array, "5"],
        initial_guess: Optional[Float[Array, "2"]] = None,
    ) -> Float[Array, "2"]:
        del initial_guess
        return steady_state_two_tank_jit(control, disturbance, params)

    def steady_state_batch_for_data_generation(
        self,
        controls: Float[Array, "batch 2"],
        disturbances: Float[Array, "batch 2"],
        params_batch: Optional[Float[Array, "batch 5"]] = None,
        initial_guesses: Optional[Float[Array, "batch 2"]] = None,
    ) -> Float[Array, "batch 2"]:
        del initial_guesses
        if params_batch is None:
            params_batch = jnp.tile(self.get_params_vector()[None, :], (controls.shape[0], 1))
        return steady_state_two_tank_batch_jit(controls, disturbances, params_batch)

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
        del dt, n_steps
        if params_batch is None:
            params_batch = jnp.tile(self.get_params_vector()[None, :], (initial_states.shape[0], 1))
        times, states = simulate_two_tank_data_generation_batch_jit(
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
        sampled = self.sample_params(key)
        return jnp.array([sampled.A1, sampled.A2, sampled.k12, sampled.kout, sampled.h_max])

    def apply_measurement_noise(
        self,
        key,
        states: Float[Array, "n_steps 2"],
    ) -> Float[Array, "n_steps 2"]:
        noise_std = jnp.array([0.01, 0.01])
        noise = jax.random.normal(key, shape=states.shape) * noise_std[None, :]
        return states + noise

    def is_valid_trajectory(
        self,
        states: Float[Array, "n_steps 2"],
    ) -> bool:
        h_max = self._params.h_max * 1.25
        return bool(
            jnp.all(jnp.isfinite(states))
            & jnp.all(states >= -1e-3)
            & jnp.all(states <= h_max)
        )

    def get_params_vector(self) -> Float[Array, "5"]:
        p = self._params
        return jnp.array([p.A1, p.A2, p.k12, p.kout, p.h_max])

    def sample_params(
        self,
        key,
        variation: float = 0.15,
    ) -> TwoTankParams:
        k1, k2, k3, k4 = jax.random.split(key, 4)
        p = self._params
        scale = lambda k, nom: float(
            nom * (1.0 + variation * jax.random.uniform(k, minval=-1.0, maxval=1.0))
        )
        return TwoTankParams(
            A1=scale(k1, p.A1),
            A2=scale(k2, p.A2),
            k12=scale(k3, p.k12),
            kout=scale(k4, p.kout),
            h_max=float(p.h_max),
        )


def _two_tank_dynamics_with_params(
    state: Float[Array, "2"],
    control: Float[Array, "2"],
    disturbance: Float[Array, "2"],
    params: Float[Array, "5"],
) -> Float[Array, "2"]:
    """JIT-friendly dynamics using packed parameter arrays."""
    h1, h2 = state
    q_in, valve = control
    d1, d2 = disturbance
    A1, A2, k12, kout, _ = params

    level_diff = jnp.maximum(h1 - h2, 0.0)
    interflow = k12 * jnp.sqrt(level_diff + 1e-8)
    outflow = valve * kout * jnp.sqrt(jnp.maximum(h2, 0.0) + 1e-8)

    dh1_dt = (q_in + d1 - interflow) / A1
    dh2_dt = (interflow + d2 - outflow) / A2
    return jnp.array([dh1_dt, dh2_dt])


@jax.jit
def steady_state_two_tank_jit(
    control: Float[Array, "2"],
    disturbance: Float[Array, "2"],
    params: Float[Array, "5"],
) -> Float[Array, "2"]:
    """Closed-form steady state for constant inputs."""
    q_in, valve = control
    d1, d2 = disturbance
    A1, A2, k12, kout, h_max = params
    del A1, A2

    q12 = jnp.maximum(q_in + d1, 1e-6)
    qout = jnp.maximum(q12 + d2, 1e-6)
    safe_valve = jnp.maximum(valve, 0.05)
    safe_k12 = jnp.maximum(k12, 1e-6)
    safe_kout = jnp.maximum(kout, 1e-6)

    h2 = (qout / (safe_valve * safe_kout)) ** 2
    h1 = h2 + (q12 / safe_k12) ** 2
    h1 = jnp.clip(h1, 0.0, h_max)
    h2 = jnp.clip(h2, 0.0, h_max)
    return jnp.array([h1, h2])


@jax.jit
def steady_state_two_tank_batch_jit(
    controls: Float[Array, "batch 2"],
    disturbances: Float[Array, "batch 2"],
    params_batch: Float[Array, "batch 5"],
) -> Float[Array, "batch 2"]:
    return jax.vmap(steady_state_two_tank_jit)(controls, disturbances, params_batch)


@jax.jit
def simulate_two_tank_data_generation_jit(
    initial_state: Float[Array, "2"],
    control_trajectory: Float[Array, "n_steps 2"],
    disturbance_trajectory: Float[Array, "n_steps 2"],
    params: Float[Array, "5"],
    t0: float,
    t1: float,
) -> Tuple[Float[Array, "n_steps"], Float[Array, "n_steps 2"]]:
    """Fixed-grid RK4 rollout used by the offline data-generation pipeline."""
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

        k1 = _two_tank_dynamics_with_params(state, control_0, disturbance_0, params)
        k2 = _two_tank_dynamics_with_params(
            state + 0.5 * step_dt * k1, control_mid, disturbance_mid, params
        )
        k3 = _two_tank_dynamics_with_params(
            state + 0.5 * step_dt * k2, control_mid, disturbance_mid, params
        )
        k4 = _two_tank_dynamics_with_params(
            state + step_dt * k3, control_1, disturbance_1, params
        )

        next_state = state + (step_dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        next_state = jnp.maximum(next_state, 0.0)
        return next_state, next_state

    _, states_tail = jax.lax.scan(
        step_fn,
        initial_state,
        (control_start, control_end, disturbance_start, disturbance_end),
    )
    states = jnp.concatenate([initial_state[None, :], states_tail], axis=0)
    return ts, states


@jax.jit
def simulate_two_tank_data_generation_batch_jit(
    initial_states: Float[Array, "batch 2"],
    control_trajectories: Float[Array, "batch n_steps 2"],
    disturbance_trajectories: Float[Array, "batch n_steps 2"],
    params_batch: Float[Array, "batch 5"],
    t0: float,
    t1: float,
) -> Tuple[Float[Array, "batch n_steps"], Float[Array, "batch n_steps 2"]]:
    return jax.vmap(
        simulate_two_tank_data_generation_jit,
        in_axes=(0, 0, 0, 0, None, None),
    )(
        initial_states,
        control_trajectories,
        disturbance_trajectories,
        params_batch,
        t0,
        t1,
    )
