"""Storage tank simulator for synthetic pretraining corpora.

State:       [inventory, quality, temperature]
Control:     [outlet_flow]
Disturbance: [feed_rate, feed_quality, feed_temperature]
Parameters:  [volume, heat_loss]
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from dte.simulators.base import ProcessSimulator, SystemSpec


AMBIENT_TEMPERATURE = 300.0
MIN_INVENTORY_FRACTION = 0.05
MAX_INVENTORY_MULTIPLE = 6.0


@dataclass(frozen=True)
class StorageTankParams:
    """Physical parameters for the storage tank."""

    volume: float = 1.5
    heat_loss: float = 0.2


class StorageTankSimulator(ProcessSimulator):
    """Simple surge-tank process with mixed quality and temperature dynamics."""

    def __init__(self, params: StorageTankParams):
        self._params = params
        self._spec: Optional[SystemSpec] = None

    @property
    def spec(self) -> SystemSpec:
        if self._spec is None:
            from dte.simulators.registry import _build_storage_tank_spec

            self._spec = _build_storage_tank_spec({})
        return self._spec

    @property
    def params(self) -> StorageTankParams:
        return self._params

    def dynamics(
        self,
        t: float,
        state: Float[Array, "3"],
        control: Float[Array, "1"],
        disturbance: Float[Array, "3"],
    ) -> Float[Array, "3"]:
        del t
        return _storage_tank_dynamics_with_params(
            state,
            control,
            disturbance,
            self.get_params_vector(),
        )

    def simulate(
        self,
        initial_state: Float[Array, "3"],
        control_trajectory: Float[Array, "n_steps 1"],
        disturbance_trajectory: Float[Array, "n_steps 3"],
        t_span: Tuple[float, float],
        dt: float = 0.1,
        n_steps: int = 1000,
    ) -> Dict[str, Array]:
        del dt, n_steps
        time, states = simulate_storage_tank_data_generation_jit(
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
        disturbance_trajectory: Float[Array, "n_steps 3"],
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
        disturbance_trajectory: Float[Array, "n_steps 3"],
        params: Float[Array, "2"],
        t_span: Tuple[float, float],
        dt: float = 0.1,
        n_steps: int = 1000,
    ) -> Dict[str, Array]:
        del dt, n_steps
        time, states = simulate_storage_tank_data_generation_jit(
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
        disturbance: Float[Array, "3"],
        initial_guess: Optional[Float[Array, "3"]] = None,
    ) -> Float[Array, "3"]:
        del initial_guess
        return steady_state_storage_tank_jit(control, disturbance, self.get_params_vector())

    def steady_state_for_data_generation(
        self,
        control: Float[Array, "1"],
        disturbance: Float[Array, "3"],
        initial_guess: Optional[Float[Array, "3"]] = None,
    ) -> Float[Array, "3"]:
        del initial_guess
        return steady_state_storage_tank_jit(control, disturbance, self.get_params_vector())

    def steady_state_for_data_generation_with_params(
        self,
        control: Float[Array, "1"],
        disturbance: Float[Array, "3"],
        params: Float[Array, "2"],
        initial_guess: Optional[Float[Array, "3"]] = None,
    ) -> Float[Array, "3"]:
        del initial_guess
        return steady_state_storage_tank_jit(control, disturbance, params)

    def steady_state_batch_for_data_generation(
        self,
        controls: Float[Array, "batch 1"],
        disturbances: Float[Array, "batch 3"],
        params_batch: Optional[Float[Array, "batch 2"]] = None,
        initial_guesses: Optional[Float[Array, "batch 3"]] = None,
    ) -> Float[Array, "batch 3"]:
        del initial_guesses
        if params_batch is None:
            params_batch = jnp.tile(self.get_params_vector()[None, :], (controls.shape[0], 1))
        return steady_state_storage_tank_batch_jit(controls, disturbances, params_batch)

    def simulate_batch_for_data_generation(
        self,
        initial_states: Float[Array, "batch 3"],
        control_trajectories: Float[Array, "batch n_steps 1"],
        disturbance_trajectories: Float[Array, "batch n_steps 3"],
        t_span: Tuple[float, float],
        params_batch: Optional[Float[Array, "batch 2"]] = None,
        dt: float = 0.1,
        n_steps: int = 1000,
    ) -> Dict[str, Array]:
        del dt, n_steps
        if params_batch is None:
            params_batch = jnp.tile(self.get_params_vector()[None, :], (initial_states.shape[0], 1))
        times, states = simulate_storage_tank_data_generation_batch_jit(
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
        return jnp.array([sampled.volume, sampled.heat_loss])

    def apply_measurement_noise(
        self,
        key,
        states: Float[Array, "n_steps 3"],
    ) -> Float[Array, "n_steps 3"]:
        noise_std = jnp.array([0.01, 0.01, 0.4])
        noise = jax.random.normal(key, shape=states.shape) * noise_std[None, :]
        noisy_states = states + noise
        inventory = jnp.maximum(noisy_states[:, 0], 0.0)
        quality = jnp.clip(noisy_states[:, 1], 0.0, 1.0)
        temperature = jnp.clip(noisy_states[:, 2], 250.0, 420.0)
        return jnp.stack([inventory, quality, temperature], axis=-1)

    def is_valid_trajectory(
        self,
        states: Float[Array, "n_steps 3"],
    ) -> bool:
        max_inventory = MAX_INVENTORY_MULTIPLE * self._params.volume
        return bool(
            jnp.all(jnp.isfinite(states))
            & jnp.all(states[:, 0] >= 0.0)
            & jnp.all(states[:, 0] <= max_inventory)
            & jnp.all(states[:, 1] >= -1e-3)
            & jnp.all(states[:, 1] <= 1.0 + 1e-3)
            & jnp.all(states[:, 2] >= 250.0)
            & jnp.all(states[:, 2] <= 420.0)
        )

    def get_params_vector(self) -> Float[Array, "2"]:
        p = self._params
        return jnp.array([p.volume, p.heat_loss])

    def sample_params(
        self,
        key,
        volume_variation: float = 0.2,
        heat_loss_variation: float = 0.25,
    ) -> StorageTankParams:
        k1, k2 = jax.random.split(key, 2)
        p = self._params
        volume = p.volume * (
            1.0
            + volume_variation * jax.random.uniform(k1, minval=-1.0, maxval=1.0)
        )
        heat_loss = p.heat_loss * (
            1.0
            + heat_loss_variation * jax.random.uniform(k2, minval=-1.0, maxval=1.0)
        )
        return StorageTankParams(
            volume=float(jnp.maximum(volume, 0.25)),
            heat_loss=float(jnp.maximum(heat_loss, 0.02)),
        )


def _storage_tank_state_targets(
    control: Float[Array, "1"],
    disturbance: Float[Array, "3"],
    params: Float[Array, "2"],
) -> Float[Array, "3"]:
    outlet_flow = jnp.maximum(control[0], 1e-3)
    feed_rate, feed_quality, feed_temperature = disturbance
    volume, heat_loss = params

    inventory_ss = volume * jnp.square(feed_rate / outlet_flow)
    inventory_ss = jnp.clip(
        inventory_ss,
        MIN_INVENTORY_FRACTION * volume,
        MAX_INVENTORY_MULTIPLE * volume,
    )
    quality_ss = jnp.clip(feed_quality, 0.0, 1.0)
    temperature_ss = (
        feed_rate * feed_temperature + heat_loss * AMBIENT_TEMPERATURE
    ) / jnp.maximum(feed_rate + heat_loss, 1e-3)
    temperature_ss = jnp.clip(temperature_ss, 250.0, 420.0)
    return jnp.array([inventory_ss, quality_ss, temperature_ss])


@jax.jit
def _storage_tank_dynamics_with_params(
    state: Float[Array, "3"],
    control: Float[Array, "1"],
    disturbance: Float[Array, "3"],
    params: Float[Array, "2"],
) -> Float[Array, "3"]:
    inventory, quality, temperature = state
    feed_rate, feed_quality, feed_temperature = disturbance
    volume, heat_loss = params

    effective_inventory = jnp.maximum(inventory, MIN_INVENTORY_FRACTION * volume)
    outlet_rating = jnp.maximum(control[0], 1e-3)
    effective_outflow = outlet_rating * jnp.sqrt(jnp.maximum(inventory, 0.0) / jnp.maximum(volume, 1e-3))
    residence_rate = feed_rate / effective_inventory

    d_inventory_dt = feed_rate - effective_outflow
    d_quality_dt = residence_rate * (feed_quality - quality)
    d_temperature_dt = residence_rate * (feed_temperature - temperature) - (
        heat_loss / effective_inventory
    ) * (temperature - AMBIENT_TEMPERATURE)
    return jnp.array([d_inventory_dt, d_quality_dt, d_temperature_dt])


@jax.jit
def steady_state_storage_tank_jit(
    control: Float[Array, "1"],
    disturbance: Float[Array, "3"],
    params: Float[Array, "2"],
) -> Float[Array, "3"]:
    return _storage_tank_state_targets(control, disturbance, params)


@jax.jit
def steady_state_storage_tank_batch_jit(
    controls: Float[Array, "batch 1"],
    disturbances: Float[Array, "batch 3"],
    params_batch: Float[Array, "batch 2"],
) -> Float[Array, "batch 3"]:
    return jax.vmap(steady_state_storage_tank_jit)(controls, disturbances, params_batch)


@jax.jit
def simulate_storage_tank_data_generation_jit(
    initial_state: Float[Array, "3"],
    control_trajectory: Float[Array, "n_steps 1"],
    disturbance_trajectory: Float[Array, "n_steps 3"],
    params: Float[Array, "2"],
    t0: float,
    t1: float,
) -> Tuple[Float[Array, "n_steps"], Float[Array, "n_steps 3"]]:
    """Fixed-grid RK4 rollout used by the offline data-generation pipeline."""
    n_steps = control_trajectory.shape[0]
    ts = jnp.linspace(t0, t1, n_steps)

    if n_steps <= 1:
        clipped = _clip_storage_tank_state(initial_state, params)
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

        k1 = _storage_tank_dynamics_with_params(state, control_0, disturbance_0, params)
        k2 = _storage_tank_dynamics_with_params(
            state + 0.5 * step_dt * k1,
            control_mid,
            disturbance_mid,
            params,
        )
        k3 = _storage_tank_dynamics_with_params(
            state + 0.5 * step_dt * k2,
            control_mid,
            disturbance_mid,
            params,
        )
        k4 = _storage_tank_dynamics_with_params(
            state + step_dt * k3,
            control_1,
            disturbance_1,
            params,
        )

        next_state = state + (step_dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        next_state = _clip_storage_tank_state(next_state, params)
        return next_state, next_state

    initial_state = _clip_storage_tank_state(initial_state, params)
    _, states_tail = jax.lax.scan(
        step_fn,
        initial_state,
        (control_start, control_end, disturbance_start, disturbance_end),
    )
    states = jnp.concatenate([initial_state[None, :], states_tail], axis=0)
    return ts, states


@jax.jit
def simulate_storage_tank_data_generation_batch_jit(
    initial_states: Float[Array, "batch 3"],
    control_trajectories: Float[Array, "batch n_steps 1"],
    disturbance_trajectories: Float[Array, "batch n_steps 3"],
    params_batch: Float[Array, "batch 2"],
    t0: float,
    t1: float,
) -> Tuple[Float[Array, "batch n_steps"], Float[Array, "batch n_steps 3"]]:
    return jax.vmap(
        simulate_storage_tank_data_generation_jit,
        in_axes=(0, 0, 0, 0, None, None),
    )(
        initial_states,
        control_trajectories,
        disturbance_trajectories,
        params_batch,
        t0,
        t1,
    )


def _clip_storage_tank_state(
    state: Float[Array, "3"],
    params: Float[Array, "2"],
) -> Float[Array, "3"]:
    volume = params[0]
    inventory = jnp.clip(
        state[0],
        0.0,
        MAX_INVENTORY_MULTIPLE * volume,
    )
    quality = jnp.clip(state[1], 0.0, 1.0)
    temperature = jnp.clip(state[2], 250.0, 420.0)
    return jnp.array([inventory, quality, temperature])
