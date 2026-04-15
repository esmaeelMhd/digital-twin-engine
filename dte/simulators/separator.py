"""Separator simulator for synthetic pretraining corpora.

State:       [light_cut, heavy_cut, tray_temperature]
Control:     [split_fraction]
Disturbance: [feed_quality, feed_temperature]
Parameters:  [holdup, separation_gain]
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from dte.simulators.base import ProcessSimulator, SystemSpec


@dataclass(frozen=True)
class SeparatorParams:
    """Physical parameters for the simplified separator."""

    holdup: float = 1.0
    separation_gain: float = 0.35


class SeparatorSimulator(ProcessSimulator):
    """First-order separator proxy with split-conditioned product qualities."""

    def __init__(self, params: SeparatorParams):
        self._params = params
        self._spec: Optional[SystemSpec] = None

    @property
    def spec(self) -> SystemSpec:
        if self._spec is None:
            from dte.simulators.registry import _build_separator_spec

            self._spec = _build_separator_spec({})
        return self._spec

    @property
    def params(self) -> SeparatorParams:
        return self._params

    def dynamics(
        self,
        t: float,
        state: Float[Array, "3"],
        control: Float[Array, "1"],
        disturbance: Float[Array, "2"],
    ) -> Float[Array, "3"]:
        del t
        return _separator_dynamics_with_params(
            state,
            control,
            disturbance,
            self.get_params_vector(),
        )

    def simulate(
        self,
        initial_state: Float[Array, "3"],
        control_trajectory: Float[Array, "n_steps 1"],
        disturbance_trajectory: Float[Array, "n_steps 2"],
        t_span: Tuple[float, float],
        dt: float = 0.1,
        n_steps: int = 1000,
    ) -> Dict[str, Array]:
        del dt, n_steps
        time, states = simulate_separator_data_generation_jit(
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
        initial_state: Float[Array, "3"],
        control_trajectory: Float[Array, "n_steps 1"],
        disturbance_trajectory: Float[Array, "n_steps 2"],
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
        initial_state: Float[Array, "3"],
        control_trajectory: Float[Array, "n_steps 1"],
        disturbance_trajectory: Float[Array, "n_steps 2"],
        params: Float[Array, "2"],
        t_span: Tuple[float, float],
        dt: float = 0.1,
        n_steps: int = 1000,
    ) -> Dict[str, Array]:
        del dt, n_steps
        time, states = simulate_separator_data_generation_jit(
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
        disturbance: Float[Array, "2"],
        initial_guess: Optional[Float[Array, "3"]] = None,
    ) -> Float[Array, "3"]:
        del initial_guess
        return steady_state_separator_jit(control, disturbance, self.get_params_vector())

    def steady_state_for_data_generation(
        self,
        control: Float[Array, "1"],
        disturbance: Float[Array, "2"],
        initial_guess: Optional[Float[Array, "3"]] = None,
    ) -> Float[Array, "3"]:
        del initial_guess
        return steady_state_separator_jit(control, disturbance, self.get_params_vector())

    def steady_state_for_data_generation_with_params(
        self,
        control: Float[Array, "1"],
        disturbance: Float[Array, "2"],
        params: Float[Array, "2"],
        initial_guess: Optional[Float[Array, "3"]] = None,
    ) -> Float[Array, "3"]:
        del initial_guess
        return steady_state_separator_jit(control, disturbance, params)

    def steady_state_batch_for_data_generation(
        self,
        controls: Float[Array, "batch 1"],
        disturbances: Float[Array, "batch 2"],
        params_batch: Optional[Float[Array, "batch 2"]] = None,
        initial_guesses: Optional[Float[Array, "batch 3"]] = None,
    ) -> Float[Array, "batch 3"]:
        del initial_guesses
        if params_batch is None:
            params_batch = jnp.tile(self.get_params_vector()[None, :], (controls.shape[0], 1))
        return steady_state_separator_batch_jit(controls, disturbances, params_batch)

    def simulate_batch_for_data_generation(
        self,
        initial_states: Float[Array, "batch 3"],
        control_trajectories: Float[Array, "batch n_steps 1"],
        disturbance_trajectories: Float[Array, "batch n_steps 2"],
        t_span: Tuple[float, float],
        params_batch: Optional[Float[Array, "batch 2"]] = None,
        dt: float = 0.1,
        n_steps: int = 1000,
    ) -> Dict[str, Array]:
        del dt, n_steps
        if params_batch is None:
            params_batch = jnp.tile(self.get_params_vector()[None, :], (initial_states.shape[0], 1))
        times, states = simulate_separator_data_generation_batch_jit(
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
    ) -> Float[Array, "2"]:
        sampled = self.sample_params(key)
        return jnp.array([sampled.holdup, sampled.separation_gain])

    def apply_measurement_noise(
        self,
        key,
        states: Float[Array, "n_steps 3"],
    ) -> Float[Array, "n_steps 3"]:
        noise_std = jnp.array([0.01, 0.01, 0.5])
        noise = jax.random.normal(key, shape=states.shape) * noise_std[None, :]
        noisy_states = states + noise
        light_cut = jnp.clip(noisy_states[:, 0], 0.0, 1.0)
        heavy_cut = jnp.clip(noisy_states[:, 1], 0.0, 1.0)
        tray_temperature = jnp.clip(noisy_states[:, 2], 250.0, 420.0)
        return jnp.stack([light_cut, heavy_cut, tray_temperature], axis=-1)

    def is_valid_trajectory(
        self,
        states: Float[Array, "n_steps 3"],
    ) -> bool:
        return bool(
            jnp.all(jnp.isfinite(states))
            & jnp.all(states[:, 0] >= -1e-3)
            & jnp.all(states[:, 0] <= 1.0 + 1e-3)
            & jnp.all(states[:, 1] >= -1e-3)
            & jnp.all(states[:, 1] <= 1.0 + 1e-3)
            & jnp.all(states[:, 2] >= 250.0)
            & jnp.all(states[:, 2] <= 420.0)
        )

    def get_params_vector(self) -> Float[Array, "2"]:
        p = self._params
        return jnp.array([p.holdup, p.separation_gain])

    def sample_params(
        self,
        key,
        holdup_variation: float = 0.2,
        gain_variation: float = 0.25,
    ) -> SeparatorParams:
        k1, k2 = jax.random.split(key, 2)
        p = self._params
        holdup = p.holdup * (
            1.0
            + holdup_variation * jax.random.uniform(k1, minval=-1.0, maxval=1.0)
        )
        separation_gain = p.separation_gain * (
            1.0
            + gain_variation * jax.random.uniform(k2, minval=-1.0, maxval=1.0)
        )
        return SeparatorParams(
            holdup=float(jnp.maximum(holdup, 0.2)),
            separation_gain=float(jnp.maximum(separation_gain, 0.05)),
        )


def _separator_state_targets(
    control: Float[Array, "1"],
    disturbance: Float[Array, "2"],
    params: Float[Array, "2"],
) -> Float[Array, "3"]:
    split_fraction = jnp.clip(control[0], 0.05, 0.95)
    feed_quality = jnp.clip(disturbance[0], 0.0, 1.0)
    feed_temperature = disturbance[1]
    separation_gain = params[1]

    light_cut = jnp.clip(
        feed_quality + separation_gain * (split_fraction - 0.5),
        0.02,
        0.98,
    )
    heavy_cut = 1.0 - light_cut
    tray_temperature = jnp.clip(
        feed_temperature + 8.0 * (0.5 - split_fraction) + 4.0 * (feed_quality - 0.5),
        260.0,
        420.0,
    )
    return jnp.array([light_cut, heavy_cut, tray_temperature])


@jax.jit
def _separator_dynamics_with_params(
    state: Float[Array, "3"],
    control: Float[Array, "1"],
    disturbance: Float[Array, "2"],
    params: Float[Array, "2"],
) -> Float[Array, "3"]:
    tau = jnp.maximum(params[0], 0.2)
    target_state = _separator_state_targets(control, disturbance, params)
    quality_rate = 1.0 / tau
    temperature_rate = 1.0 / (0.75 * tau + 0.1)
    return jnp.array(
        [
            quality_rate * (target_state[0] - state[0]),
            quality_rate * (target_state[1] - state[1]),
            temperature_rate * (target_state[2] - state[2]),
        ]
    )


@jax.jit
def steady_state_separator_jit(
    control: Float[Array, "1"],
    disturbance: Float[Array, "2"],
    params: Float[Array, "2"],
) -> Float[Array, "3"]:
    return _separator_state_targets(control, disturbance, params)


@jax.jit
def steady_state_separator_batch_jit(
    controls: Float[Array, "batch 1"],
    disturbances: Float[Array, "batch 2"],
    params_batch: Float[Array, "batch 2"],
) -> Float[Array, "batch 3"]:
    return jax.vmap(steady_state_separator_jit)(controls, disturbances, params_batch)


@jax.jit
def simulate_separator_data_generation_jit(
    initial_state: Float[Array, "3"],
    control_trajectory: Float[Array, "n_steps 1"],
    disturbance_trajectory: Float[Array, "n_steps 2"],
    params: Float[Array, "2"],
    t0: float,
    t1: float,
) -> Tuple[Float[Array, "n_steps"], Float[Array, "n_steps 3"]]:
    """Fixed-grid RK4 rollout used by the offline data-generation pipeline."""
    n_steps = control_trajectory.shape[0]
    ts = jnp.linspace(t0, t1, n_steps)

    if n_steps <= 1:
        clipped = _clip_separator_state(initial_state)
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

        k1 = _separator_dynamics_with_params(state, control_0, disturbance_0, params)
        k2 = _separator_dynamics_with_params(
            state + 0.5 * step_dt * k1,
            control_mid,
            disturbance_mid,
            params,
        )
        k3 = _separator_dynamics_with_params(
            state + 0.5 * step_dt * k2,
            control_mid,
            disturbance_mid,
            params,
        )
        k4 = _separator_dynamics_with_params(
            state + step_dt * k3,
            control_1,
            disturbance_1,
            params,
        )

        next_state = state + (step_dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        next_state = _clip_separator_state(next_state)
        return next_state, next_state

    initial_state = _clip_separator_state(initial_state)
    _, states_tail = jax.lax.scan(
        step_fn,
        initial_state,
        (control_start, control_end, disturbance_start, disturbance_end),
    )
    states = jnp.concatenate([initial_state[None, :], states_tail], axis=0)
    return ts, states


@jax.jit
def simulate_separator_data_generation_batch_jit(
    initial_states: Float[Array, "batch 3"],
    control_trajectories: Float[Array, "batch n_steps 1"],
    disturbance_trajectories: Float[Array, "batch n_steps 2"],
    params_batch: Float[Array, "batch 2"],
    t0: float,
    t1: float,
) -> Tuple[Float[Array, "batch n_steps"], Float[Array, "batch n_steps 3"]]:
    return jax.vmap(
        simulate_separator_data_generation_jit,
        in_axes=(0, 0, 0, 0, None, None),
    )(
        initial_states,
        control_trajectories,
        disturbance_trajectories,
        params_batch,
        t0,
        t1,
    )


def _clip_separator_state(
    state: Float[Array, "3"],
) -> Float[Array, "3"]:
    light_cut = jnp.clip(state[0], 0.0, 1.0)
    heavy_cut = jnp.clip(state[1], 0.0, 1.0)
    tray_temperature = jnp.clip(state[2], 250.0, 420.0)
    return jnp.array([light_cut, heavy_cut, tray_temperature])
