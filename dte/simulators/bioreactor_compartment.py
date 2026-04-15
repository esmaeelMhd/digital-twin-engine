"""Aerobic bioreactor compartment for synthetic corpus expansion.

State:       [substrate, biomass, dissolved_oxygen]
Control:     [aeration]
Disturbance: [feed_substrate]
Parameters:  [mu_max, kla, decay_rate, dilution_rate]
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from dte.laws.biology import inhibition_factor, monod_growth_rate
from dte.simulators.base import ProcessSimulator, SystemSpec


HALF_SATURATION = 0.2
OXYGEN_HALF_SATURATION = 0.15
YIELD_COEFFICIENT = 0.55
OXYGEN_SATURATION = 1.0
OXYGEN_DEMAND_FACTOR = 0.4
INHIBITION_CONSTANT = 8.0
MAX_SUBSTRATE = 3.0
MAX_BIOMASS = 3.0
MAX_DISSOLVED_OXYGEN = 1.2


@dataclass(frozen=True)
class BioreactorCompartmentParams:
    """Parameters for the simplified aerobic bioreactor compartment."""

    mu_max: float = 0.45
    kla: float = 1.2
    decay_rate: float = 0.03
    dilution_rate: float = 0.12


class BioreactorCompartmentSimulator(ProcessSimulator):
    """Synthetic aerobic bioreactor compartment with Monod-like growth."""

    def __init__(self, params: BioreactorCompartmentParams):
        self._params = params
        self._spec: Optional[SystemSpec] = None

    @property
    def spec(self) -> SystemSpec:
        if self._spec is None:
            from dte.simulators.registry import _build_bioreactor_compartment_spec

            self._spec = _build_bioreactor_compartment_spec({})
        return self._spec

    @property
    def params(self) -> BioreactorCompartmentParams:
        return self._params

    def dynamics(
        self,
        t: float,
        state: Float[Array, "3"],
        control: Float[Array, "1"],
        disturbance: Float[Array, "1"],
    ) -> Float[Array, "3"]:
        del t
        return _bioreactor_dynamics_with_params(
            state,
            control,
            disturbance,
            self.get_params_vector(),
        )

    def simulate(
        self,
        initial_state: Float[Array, "3"],
        control_trajectory: Float[Array, "n_steps 1"],
        disturbance_trajectory: Float[Array, "n_steps 1"],
        t_span: Tuple[float, float],
        dt: float = 0.1,
        n_steps: int = 1000,
    ) -> Dict[str, Array]:
        del dt, n_steps
        time, states = simulate_bioreactor_data_generation_jit(
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
        initial_state: Float[Array, "3"],
        control_trajectory: Float[Array, "n_steps 1"],
        disturbance_trajectory: Float[Array, "n_steps 1"],
        params: Float[Array, "4"],
        t_span: Tuple[float, float],
        dt: float = 0.1,
        n_steps: int = 1000,
    ) -> Dict[str, Array]:
        del dt, n_steps
        time, states = simulate_bioreactor_data_generation_jit(
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
        initial_guess: Optional[Float[Array, "3"]] = None,
    ) -> Float[Array, "3"]:
        del initial_guess
        return steady_state_bioreactor_jit(control, disturbance, self.get_params_vector())

    def steady_state_for_data_generation(
        self,
        control: Float[Array, "1"],
        disturbance: Float[Array, "1"],
        initial_guess: Optional[Float[Array, "3"]] = None,
    ) -> Float[Array, "3"]:
        del initial_guess
        return steady_state_bioreactor_jit(control, disturbance, self.get_params_vector())

    def steady_state_for_data_generation_with_params(
        self,
        control: Float[Array, "1"],
        disturbance: Float[Array, "1"],
        params: Float[Array, "4"],
        initial_guess: Optional[Float[Array, "3"]] = None,
    ) -> Float[Array, "3"]:
        del initial_guess
        return steady_state_bioreactor_jit(control, disturbance, params)

    def steady_state_batch_for_data_generation(
        self,
        controls: Float[Array, "batch 1"],
        disturbances: Float[Array, "batch 1"],
        params_batch: Optional[Float[Array, "batch 4"]] = None,
        initial_guesses: Optional[Float[Array, "batch 3"]] = None,
    ) -> Float[Array, "batch 3"]:
        del initial_guesses
        if params_batch is None:
            params_batch = jnp.tile(self.get_params_vector()[None, :], (controls.shape[0], 1))
        return steady_state_bioreactor_batch_jit(controls, disturbances, params_batch)

    def simulate_batch_for_data_generation(
        self,
        initial_states: Float[Array, "batch 3"],
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
        times, states = simulate_bioreactor_data_generation_batch_jit(
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
        return jnp.array(
            [
                sampled.mu_max,
                sampled.kla,
                sampled.decay_rate,
                sampled.dilution_rate,
            ]
        )

    def apply_measurement_noise(
        self,
        key,
        states: Float[Array, "n_steps 3"],
    ) -> Float[Array, "n_steps 3"]:
        noise_std = jnp.array([0.02, 0.01, 0.01])
        noise = jax.random.normal(key, shape=states.shape) * noise_std[None, :]
        noisy_states = states + noise
        return _clip_bioreactor_states(noisy_states)

    def is_valid_trajectory(
        self,
        states: Float[Array, "n_steps 3"],
    ) -> bool:
        return bool(
            jnp.all(jnp.isfinite(states))
            & jnp.all(states[:, 0] >= 0.0)
            & jnp.all(states[:, 0] <= MAX_SUBSTRATE)
            & jnp.all(states[:, 1] >= 0.0)
            & jnp.all(states[:, 1] <= MAX_BIOMASS)
            & jnp.all(states[:, 2] >= 0.0)
            & jnp.all(states[:, 2] <= MAX_DISSOLVED_OXYGEN)
        )

    def get_params_vector(self) -> Float[Array, "4"]:
        p = self._params
        return jnp.array([p.mu_max, p.kla, p.decay_rate, p.dilution_rate])

    def sample_params(
        self,
        key,
        growth_variation: float = 0.2,
        transfer_variation: float = 0.2,
        decay_variation: float = 0.2,
        dilution_variation: float = 0.15,
    ) -> BioreactorCompartmentParams:
        k1, k2, k3, k4 = jax.random.split(key, 4)
        p = self._params
        mu_max = p.mu_max * (
            1.0
            + growth_variation * jax.random.uniform(k1, minval=-1.0, maxval=1.0)
        )
        kla = p.kla * (
            1.0
            + transfer_variation * jax.random.uniform(k2, minval=-1.0, maxval=1.0)
        )
        decay_rate = p.decay_rate * (
            1.0
            + decay_variation * jax.random.uniform(k3, minval=-1.0, maxval=1.0)
        )
        dilution_rate = p.dilution_rate * (
            1.0
            + dilution_variation * jax.random.uniform(k4, minval=-1.0, maxval=1.0)
        )
        return BioreactorCompartmentParams(
            mu_max=float(jnp.maximum(mu_max, 0.05)),
            kla=float(jnp.maximum(kla, 0.1)),
            decay_rate=float(jnp.maximum(decay_rate, 0.005)),
            dilution_rate=float(jnp.maximum(dilution_rate, 0.02)),
        )


@jax.jit
def _bioreactor_growth_terms(
    state: Float[Array, "3"],
    control: Float[Array, "1"],
    params: Float[Array, "4"],
) -> Tuple[Float[Array, ""], Float[Array, ""], Float[Array, ""]]:
    substrate, biomass, dissolved_oxygen = state
    aeration = jnp.clip(control[0], 0.0, 2.0)
    mu_max, kla, _, _ = params

    specific_growth = monod_growth_rate(
        substrate,
        mu_max=mu_max,
        half_saturation=HALF_SATURATION,
    )
    oxygen_factor = jnp.maximum(dissolved_oxygen, 0.0) / (
        OXYGEN_HALF_SATURATION + jnp.maximum(dissolved_oxygen, 0.0) + 1e-8
    )
    inhibition = inhibition_factor(substrate, inhibition_constant=INHIBITION_CONSTANT)
    mu = specific_growth * oxygen_factor * inhibition
    oxygen_transfer = aeration * kla * (OXYGEN_SATURATION - dissolved_oxygen)
    substrate_uptake = mu * jnp.maximum(biomass, 0.0) / max(YIELD_COEFFICIENT, 1e-6)
    return mu, oxygen_transfer, substrate_uptake


@jax.jit
def _bioreactor_dynamics_with_params(
    state: Float[Array, "3"],
    control: Float[Array, "1"],
    disturbance: Float[Array, "1"],
    params: Float[Array, "4"],
) -> Float[Array, "3"]:
    substrate, biomass, _ = state
    feed_substrate = disturbance[0]
    _, _, decay_rate, dilution_rate = params

    mu, oxygen_transfer, substrate_uptake = _bioreactor_growth_terms(state, control, params)

    d_substrate_dt = dilution_rate * (feed_substrate - substrate) - substrate_uptake
    d_biomass_dt = (mu - decay_rate - dilution_rate) * jnp.maximum(biomass, 0.0)
    d_oxygen_dt = oxygen_transfer - OXYGEN_DEMAND_FACTOR * mu * jnp.maximum(biomass, 0.0)
    return jnp.array([d_substrate_dt, d_biomass_dt, d_oxygen_dt])


@jax.jit
def steady_state_bioreactor_jit(
    control: Float[Array, "1"],
    disturbance: Float[Array, "1"],
    params: Float[Array, "4"],
) -> Float[Array, "3"]:
    feed_substrate = disturbance[0]
    aeration = jnp.clip(control[0], 0.0, 2.0)
    mu_max, kla, decay_rate, dilution_rate = params
    effective_kla = jnp.maximum(aeration * kla, 1e-3)
    inhibition = inhibition_factor(feed_substrate, inhibition_constant=INHIBITION_CONSTANT)

    def _iterate(_, dissolved_oxygen):
        oxygen_factor = dissolved_oxygen / (OXYGEN_HALF_SATURATION + dissolved_oxygen + 1e-8)
        mu_cap = jnp.maximum(mu_max * inhibition * oxygen_factor, 1e-4)
        mu_target = jnp.minimum(decay_rate + dilution_rate, 0.9 * mu_cap)
        substrate = HALF_SATURATION * mu_target / jnp.maximum(mu_cap - mu_target, 1e-4)
        substrate = jnp.clip(substrate, 0.0, feed_substrate)
        biomass = (
            YIELD_COEFFICIENT
            * dilution_rate
            * jnp.maximum(feed_substrate - substrate, 0.0)
            / jnp.maximum(mu_target, 1e-4)
        )
        dissolved_oxygen_next = OXYGEN_SATURATION - (
            OXYGEN_DEMAND_FACTOR * mu_target * biomass / effective_kla
        )
        return jnp.clip(dissolved_oxygen_next, 0.05, OXYGEN_SATURATION)

    dissolved_oxygen = jax.lax.fori_loop(0, 5, _iterate, jnp.asarray(0.7, dtype=jnp.float32))
    oxygen_factor = dissolved_oxygen / (OXYGEN_HALF_SATURATION + dissolved_oxygen + 1e-8)
    mu_cap = jnp.maximum(mu_max * inhibition * oxygen_factor, 1e-4)
    mu_target = jnp.minimum(decay_rate + dilution_rate, 0.9 * mu_cap)
    substrate = HALF_SATURATION * mu_target / jnp.maximum(mu_cap - mu_target, 1e-4)
    substrate = jnp.clip(substrate, 0.0, feed_substrate)
    biomass = (
        YIELD_COEFFICIENT
        * dilution_rate
        * jnp.maximum(feed_substrate - substrate, 0.0)
        / jnp.maximum(mu_target, 1e-4)
    )
    return _clip_bioreactor_states(jnp.array([substrate, biomass, dissolved_oxygen]))


@jax.jit
def steady_state_bioreactor_batch_jit(
    controls: Float[Array, "batch 1"],
    disturbances: Float[Array, "batch 1"],
    params_batch: Float[Array, "batch 4"],
) -> Float[Array, "batch 3"]:
    return jax.vmap(steady_state_bioreactor_jit)(controls, disturbances, params_batch)


@jax.jit
def simulate_bioreactor_data_generation_jit(
    initial_state: Float[Array, "3"],
    control_trajectory: Float[Array, "n_steps 1"],
    disturbance_trajectory: Float[Array, "n_steps 1"],
    params: Float[Array, "4"],
    t0: float,
    t1: float,
) -> Tuple[Float[Array, "n_steps"], Float[Array, "n_steps 3"]]:
    """Fixed-grid RK4 rollout used by the offline data-generation pipeline."""
    n_steps = control_trajectory.shape[0]
    ts = jnp.linspace(t0, t1, n_steps)

    if n_steps <= 1:
        clipped = _clip_bioreactor_states(initial_state)
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

        k1 = _bioreactor_dynamics_with_params(state, control_0, disturbance_0, params)
        k2 = _bioreactor_dynamics_with_params(
            state + 0.5 * step_dt * k1,
            control_mid,
            disturbance_mid,
            params,
        )
        k3 = _bioreactor_dynamics_with_params(
            state + 0.5 * step_dt * k2,
            control_mid,
            disturbance_mid,
            params,
        )
        k4 = _bioreactor_dynamics_with_params(
            state + step_dt * k3,
            control_1,
            disturbance_1,
            params,
        )

        next_state = state + (step_dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        next_state = _clip_bioreactor_states(next_state)
        return next_state, next_state

    initial_state = _clip_bioreactor_states(initial_state)
    _, states_tail = jax.lax.scan(
        step_fn,
        initial_state,
        (control_start, control_end, disturbance_start, disturbance_end),
    )
    states = jnp.concatenate([initial_state[None, :], states_tail], axis=0)
    return ts, states


@jax.jit
def simulate_bioreactor_data_generation_batch_jit(
    initial_states: Float[Array, "batch 3"],
    control_trajectories: Float[Array, "batch n_steps 1"],
    disturbance_trajectories: Float[Array, "batch n_steps 1"],
    params_batch: Float[Array, "batch 4"],
    t0: float,
    t1: float,
) -> Tuple[Float[Array, "batch n_steps"], Float[Array, "batch n_steps 3"]]:
    return jax.vmap(
        simulate_bioreactor_data_generation_jit,
        in_axes=(0, 0, 0, 0, None, None),
    )(
        initial_states,
        control_trajectories,
        disturbance_trajectories,
        params_batch,
        t0,
        t1,
    )


@jax.jit
def _clip_bioreactor_states(
    states: Float[Array, "... 3"],
) -> Float[Array, "... 3"]:
    substrate = jnp.clip(states[..., 0], 0.0, MAX_SUBSTRATE)
    biomass = jnp.clip(states[..., 1], 0.0, MAX_BIOMASS)
    dissolved_oxygen = jnp.clip(states[..., 2], 0.0, MAX_DISSOLVED_OXYGEN)
    return jnp.stack([substrate, biomass, dissolved_oxygen], axis=-1)
