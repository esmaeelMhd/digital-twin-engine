"""Simplified isothermal CSTR for synthetic corpus expansion.

State:       [Ca, Cb]
Control:     [F_in]
Disturbance: [Ca_in]
Parameters:  [V, k0, Ea_over_R, T_ref]
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from dte.simulators.base import ProcessSimulator, SystemSpec


MIN_CONCENTRATION = 0.0
MAX_CONCENTRATION = 3.0


@dataclass(frozen=True)
class IsothermalCSTRParams:
    """Parameters for the isothermal reactor proxy."""

    V: float = 100.0
    k0: float = 7.2e10
    Ea_over_R: float = 8750.0
    T_ref: float = 330.0


class IsothermalCSTRSimulator(ProcessSimulator):
    """Synthetic isothermal CSTR with first-order A -> B kinetics."""

    def __init__(self, params: IsothermalCSTRParams):
        self._params = params
        self._spec: Optional[SystemSpec] = None

    @property
    def spec(self) -> SystemSpec:
        if self._spec is None:
            from dte.simulators.registry import _build_isothermal_cstr_spec

            self._spec = _build_isothermal_cstr_spec({})
        return self._spec

    @property
    def params(self) -> IsothermalCSTRParams:
        return self._params

    def reaction_rate_constant(self, params: Optional[Float[Array, "4"]] = None) -> Float[Array, ""]:
        if params is None:
            params = self.get_params_vector()
        _, k0, ea_over_r, t_ref = params
        return k0 * jnp.exp(-ea_over_r / jnp.maximum(t_ref, 1.0))

    def dynamics(
        self,
        t: float,
        state: Float[Array, "2"],
        control: Float[Array, "1"],
        disturbance: Float[Array, "1"],
    ) -> Float[Array, "2"]:
        del t
        return _isothermal_cstr_dynamics_with_params(
            state,
            control,
            disturbance,
            self.get_params_vector(),
        )

    def simulate(
        self,
        initial_state: Float[Array, "2"],
        control_trajectory: Float[Array, "n_steps 1"],
        disturbance_trajectory: Float[Array, "n_steps 1"],
        t_span: Tuple[float, float],
        dt: float = 0.1,
        n_steps: int = 1000,
    ) -> Dict[str, Array]:
        del dt, n_steps
        time, states = simulate_isothermal_cstr_data_generation_jit(
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

    def simulate_for_data_generation(
        self,
        initial_state: Float[Array, "2"],
        control_trajectory: Float[Array, "n_steps 1"],
        disturbance_trajectory: Float[Array, "n_steps 1"],
        t_span: Tuple[float, float],
        dt: float = 0.1,
        n_steps: int = 1000,
    ) -> Dict[str, Array]:
        return self.simulate(
            initial_state,
            control_trajectory,
            disturbance_trajectory,
            t_span,
            dt=dt,
            n_steps=n_steps,
        )

    def simulate_for_data_generation_with_params(
        self,
        initial_state: Float[Array, "2"],
        control_trajectory: Float[Array, "n_steps 1"],
        disturbance_trajectory: Float[Array, "n_steps 1"],
        params: Float[Array, "4"],
        t_span: Tuple[float, float],
        dt: float = 0.1,
        n_steps: int = 1000,
    ) -> Dict[str, Array]:
        del dt, n_steps
        time, states = simulate_isothermal_cstr_data_generation_jit(
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
        control: Float[Array, "1"],
        disturbance: Float[Array, "1"],
        initial_guess: Optional[Float[Array, "2"]] = None,
    ) -> Float[Array, "2"]:
        del initial_guess
        return steady_state_isothermal_cstr_jit(control, disturbance, self.get_params_vector())

    def steady_state_for_data_generation(
        self,
        control: Float[Array, "1"],
        disturbance: Float[Array, "1"],
        initial_guess: Optional[Float[Array, "2"]] = None,
    ) -> Float[Array, "2"]:
        del initial_guess
        return steady_state_isothermal_cstr_jit(control, disturbance, self.get_params_vector())

    def steady_state_for_data_generation_with_params(
        self,
        control: Float[Array, "1"],
        disturbance: Float[Array, "1"],
        params: Float[Array, "4"],
        initial_guess: Optional[Float[Array, "2"]] = None,
    ) -> Float[Array, "2"]:
        del initial_guess
        return steady_state_isothermal_cstr_jit(control, disturbance, params)

    def steady_state_batch_for_data_generation(
        self,
        controls: Float[Array, "batch 1"],
        disturbances: Float[Array, "batch 1"],
        params_batch: Optional[Float[Array, "batch 4"]] = None,
        initial_guesses: Optional[Float[Array, "batch 2"]] = None,
    ) -> Float[Array, "batch 2"]:
        del initial_guesses
        if params_batch is None:
            params_batch = jnp.tile(self.get_params_vector()[None, :], (controls.shape[0], 1))
        return steady_state_isothermal_cstr_batch_jit(controls, disturbances, params_batch)

    def simulate_batch_for_data_generation(
        self,
        initial_states: Float[Array, "batch 2"],
        control_trajectories: Float[Array, "batch n_steps 1"],
        disturbance_trajectories: Float[Array, "batch n_steps 1"],
        t_span: Tuple[float, float],
        params_batch: Optional[Float[Array, "batch 4"]] = None,
        dt: float = 0.1,
        n_steps: int = 1000,
    ) -> Dict[str, Array]:
        del dt, n_steps
        if params_batch is None:
            params_batch = jnp.tile(self.get_params_vector()[None, :], (initial_states.shape[0], 1))
        times, states = simulate_isothermal_cstr_data_generation_batch_jit(
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
    ) -> Float[Array, "4"]:
        sampled = self.sample_params(key)
        return jnp.array([sampled.V, sampled.k0, sampled.Ea_over_R, sampled.T_ref])

    def apply_measurement_noise(
        self,
        key,
        states: Float[Array, "n_steps 2"],
    ) -> Float[Array, "n_steps 2"]:
        noise_std = jnp.array([0.01, 0.01])
        noise = jax.random.normal(key, shape=states.shape) * noise_std[None, :]
        return jnp.clip(states + noise, MIN_CONCENTRATION, MAX_CONCENTRATION)

    def is_valid_trajectory(
        self,
        states: Float[Array, "n_steps 2"],
    ) -> bool:
        return bool(
            jnp.all(jnp.isfinite(states))
            & jnp.all(states >= MIN_CONCENTRATION)
            & jnp.all(states <= MAX_CONCENTRATION)
        )

    def get_params_vector(self) -> Float[Array, "4"]:
        p = self._params
        return jnp.array([p.V, p.k0, p.Ea_over_R, p.T_ref])

    def sample_params(
        self,
        key,
        volume_variation: float = 0.15,
        kinetics_variation: float = 0.2,
        temperature_variation: float = 0.05,
    ) -> IsothermalCSTRParams:
        k1, k2, k3, k4 = jax.random.split(key, 4)
        p = self._params
        volume = p.V * (
            1.0
            + volume_variation * jax.random.uniform(k1, minval=-1.0, maxval=1.0)
        )
        k0 = p.k0 * (
            1.0
            + kinetics_variation * jax.random.uniform(k2, minval=-1.0, maxval=1.0)
        )
        ea_over_r = p.Ea_over_R * (
            1.0
            + kinetics_variation * jax.random.uniform(k3, minval=-1.0, maxval=1.0)
        )
        t_ref = p.T_ref * (
            1.0
            + temperature_variation * jax.random.uniform(k4, minval=-1.0, maxval=1.0)
        )
        return IsothermalCSTRParams(
            V=float(jnp.maximum(volume, 10.0)),
            k0=float(jnp.maximum(k0, 1.0e6)),
            Ea_over_R=float(jnp.maximum(ea_over_r, 1000.0)),
            T_ref=float(jnp.clip(t_ref, 300.0, 360.0)),
        )


@jax.jit
def _reaction_rate_constant_from_params(
    params: Float[Array, "4"],
) -> Float[Array, ""]:
    _, k0, ea_over_r, t_ref = params
    return k0 * jnp.exp(-ea_over_r / jnp.maximum(t_ref, 1.0))


@jax.jit
def _isothermal_cstr_dynamics_with_params(
    state: Float[Array, "2"],
    control: Float[Array, "1"],
    disturbance: Float[Array, "1"],
    params: Float[Array, "4"],
) -> Float[Array, "2"]:
    ca, cb = state
    f_in = control[0]
    ca_in = disturbance[0]
    v = params[0]
    k = _reaction_rate_constant_from_params(params)
    flow_over_volume = f_in / jnp.maximum(v, 1e-3)
    d_ca_dt = flow_over_volume * (ca_in - ca) - k * ca
    d_cb_dt = flow_over_volume * (0.0 - cb) + k * ca
    return jnp.array([d_ca_dt, d_cb_dt])


@jax.jit
def steady_state_isothermal_cstr_jit(
    control: Float[Array, "1"],
    disturbance: Float[Array, "1"],
    params: Float[Array, "4"],
) -> Float[Array, "2"]:
    f_in = control[0]
    ca_in = disturbance[0]
    v = params[0]
    k = _reaction_rate_constant_from_params(params)
    flow_over_volume = f_in / jnp.maximum(v, 1e-3)
    ca = (flow_over_volume * ca_in) / jnp.maximum(flow_over_volume + k, 1e-6)
    cb = jnp.maximum(ca_in - ca, 0.0)
    return jnp.clip(jnp.array([ca, cb]), MIN_CONCENTRATION, MAX_CONCENTRATION)


@jax.jit
def steady_state_isothermal_cstr_batch_jit(
    controls: Float[Array, "batch 1"],
    disturbances: Float[Array, "batch 1"],
    params_batch: Float[Array, "batch 4"],
) -> Float[Array, "batch 2"]:
    return jax.vmap(steady_state_isothermal_cstr_jit)(controls, disturbances, params_batch)


@jax.jit
def simulate_isothermal_cstr_data_generation_jit(
    initial_state: Float[Array, "2"],
    control_trajectory: Float[Array, "n_steps 1"],
    disturbance_trajectory: Float[Array, "n_steps 1"],
    params: Float[Array, "4"],
    t0: float,
    t1: float,
) -> Tuple[Float[Array, "n_steps"], Float[Array, "n_steps 2"]]:
    """Fixed-grid RK4 rollout used by the offline data-generation pipeline."""
    n_steps = control_trajectory.shape[0]
    ts = jnp.linspace(t0, t1, n_steps)

    if n_steps <= 1:
        clipped = jnp.clip(initial_state, MIN_CONCENTRATION, MAX_CONCENTRATION)
        return ts, clipped[None, :]

    step_dt = ts[1] - ts[0]
    control_start = control_trajectory[:-1]
    control_end = control_trajectory[1:]
    disturbance_start = disturbance_trajectory[:-1]
    disturbance_end = disturbance_trajectory[1:]

    def step_fn(state, inputs):
        control_0, control_1, disturbance_0, disturbance_1 = inputs
        control_mid = 0.5 * (control_0 + control_1)
        disturbance_mid = 0.5 * (disturbance_0 + disturbance_1)

        k1 = _isothermal_cstr_dynamics_with_params(state, control_0, disturbance_0, params)
        k2 = _isothermal_cstr_dynamics_with_params(
            state + 0.5 * step_dt * k1,
            control_mid,
            disturbance_mid,
            params,
        )
        k3 = _isothermal_cstr_dynamics_with_params(
            state + 0.5 * step_dt * k2,
            control_mid,
            disturbance_mid,
            params,
        )
        k4 = _isothermal_cstr_dynamics_with_params(
            state + step_dt * k3,
            control_1,
            disturbance_1,
            params,
        )

        next_state = state + (step_dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        next_state = jnp.clip(next_state, MIN_CONCENTRATION, MAX_CONCENTRATION)
        return next_state, next_state

    initial_state = jnp.clip(initial_state, MIN_CONCENTRATION, MAX_CONCENTRATION)
    _, states_tail = jax.lax.scan(
        step_fn,
        initial_state,
        (control_start, control_end, disturbance_start, disturbance_end),
    )
    states = jnp.concatenate([initial_state[None, :], states_tail], axis=0)
    return ts, states


@jax.jit
def simulate_isothermal_cstr_data_generation_batch_jit(
    initial_states: Float[Array, "batch 2"],
    control_trajectories: Float[Array, "batch n_steps 1"],
    disturbance_trajectories: Float[Array, "batch n_steps 1"],
    params_batch: Float[Array, "batch 4"],
    t0: float,
    t1: float,
) -> Tuple[Float[Array, "batch n_steps"], Float[Array, "batch n_steps 2"]]:
    return jax.vmap(
        simulate_isothermal_cstr_data_generation_jit,
        in_axes=(0, 0, 0, 0, None, None),
    )(
        initial_states,
        control_trajectories,
        disturbance_trajectories,
        params_batch,
        t0,
        t1,
    )
