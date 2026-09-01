"""Reusable chemistry law helpers and modules."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from dataclasses import dataclass
from jaxtyping import Array, Float

from dte.laws.base import LawModule


def arrhenius_rate_constant(
    pre_exponential: float,
    activation_energy_over_r: float,
    temperature: Float[Array, ""],
) -> Float[Array, ""]:
    """Return a simple Arrhenius rate constant."""
    safe_temperature = jnp.maximum(temperature, 1e-6)
    return pre_exponential * jnp.exp(-activation_energy_over_r / safe_temperature)


def power_law_rate(
    concentrations: Float[Array, "n_species"],
    orders: Float[Array, "n_species"],
    rate_constant: Float[Array, ""],
) -> Float[Array, ""]:
    """Return a mass-action or power-law reaction rate."""
    safe_concentration = jnp.maximum(concentrations, 0.0) + 1e-8
    return rate_constant * jnp.prod(safe_concentration ** orders)


@dataclass(frozen=True)
class ChemistryLaw(LawModule):
    """Lightweight stoichiometric chemistry module."""

    module_name: str
    state_dim: int
    stoichiometry: Array
    reactant_indices: tuple[int, ...]
    reaction_orders: Array
    temperature_index: int | None = None
    kinetic_family: str = "arrhenius_power_law"
    pre_exponential: float = 1.0
    activation_energy_over_r: float = 0.0
    heat_of_reaction: float = 0.0
    thermal_state_index: int | None = None
    state_gain: float = 1.0
    thermal_gain: float = 0.0
    closed_system: bool = False

    def __post_init__(self):
        object.__setattr__(
            self,
            "stoichiometry",
            jnp.asarray(self.stoichiometry, dtype=jnp.float32),
        )
        object.__setattr__(
            self,
            "reaction_orders",
            jnp.asarray(self.reaction_orders, dtype=jnp.float32),
        )
        if self.stoichiometry.shape != (self.state_dim,):
            raise ValueError(
                f"chemistry law '{self.module_name}' expected stoichiometry of "
                f"shape ({self.state_dim},), got {self.stoichiometry.shape}."
            )
        if len(self.reactant_indices) != int(self.reaction_orders.shape[0]):
            raise ValueError(
                f"chemistry law '{self.module_name}' reaction_orders must match "
                "reactant_indices length."
            )

    @property
    def family_name(self) -> str:
        return "chemistry"

    def feature_names(self) -> tuple[str, ...]:
        return ("rate_constant", "reaction_rate", "heat_release")

    def residual_names(self) -> tuple[str, ...]:
        return ("state_delta_consistency", "nonnegative_rate")

    def rate_constant(self, state: Float[Array, "state_dim"]) -> Float[Array, ""]:
        if self.kinetic_family == "constant":
            return jnp.asarray(self.pre_exponential, dtype=jnp.float32)
        if self.temperature_index is None:
            return jnp.asarray(self.pre_exponential, dtype=jnp.float32)
        return arrhenius_rate_constant(
            self.pre_exponential,
            self.activation_energy_over_r,
            state[self.temperature_index],
        )

    def reaction_rate(self, state: Float[Array, "state_dim"]) -> Float[Array, ""]:
        concentrations = state[jnp.asarray(self.reactant_indices, dtype=jnp.int32)]
        return power_law_rate(concentrations, self.reaction_orders, self.rate_constant(state))

    def species_source_terms(
        self,
        state: Float[Array, "state_dim"],
    ) -> Float[Array, "state_dim"]:
        return self.stoichiometry * self.reaction_rate(state)

    def heat_release(self, state: Float[Array, "state_dim"]) -> Float[Array, ""]:
        return -self.heat_of_reaction * self.reaction_rate(state)

    def feature_vector(
        self,
        state: Float[Array, "state_dim"],
        control,
        disturbance,
        params,
        dt,
    ) -> Float[Array, "3"]:
        del control, disturbance, params, dt
        return jnp.asarray(
            [
                self.rate_constant(state),
                self.reaction_rate(state),
                self.heat_release(state),
            ],
            dtype=jnp.float32,
        )

    def mechanistic_delta(
        self,
        state: Float[Array, "state_dim"],
        control,
        disturbance,
        params,
        dt,
    ) -> Float[Array, "state_dim"]:
        del control, disturbance, params, dt
        delta = self.state_gain * self.species_source_terms(state)
        if self.thermal_state_index is not None:
            delta = delta.at[self.thermal_state_index].add(
                self.thermal_gain * self.heat_release(state)
            )
        return delta

    def trajectory_residuals(
        self,
        states: Float[Array, "n_steps state_dim"],
        controls: Float[Array, "n_steps control_dim"],
        disturbances: Float[Array, "n_steps disturbance_dim"],
        dt: float | Array,
        params: Float[Array, "param_dim"] | None,
    ) -> dict[str, Float[Array, "n_steps_minus_one"]]:
        del params
        if states.shape[0] < 2:
            zeros = jnp.zeros((1,), dtype=states.dtype)
            return {
                "state_delta_consistency": zeros,
                "nonnegative_rate": zeros,
            }

        safe_dt = jnp.maximum(jnp.asarray(dt, dtype=states.dtype), 1e-6)
        rates = jax.vmap(self.reaction_rate)(states[:-1])
        nonnegative_rate = jax.nn.relu(-rates)
        if not self.closed_system:
            # Source-term-only dx/dt balance holds for closed reactors, not
            # open flow systems whose observed dx/dt includes transport.
            zeros = jnp.zeros((states.shape[0] - 1,), dtype=states.dtype)
            return {
                "state_delta_consistency": zeros,
                "nonnegative_rate": nonnegative_rate,
            }

        predicted_delta = jax.vmap(
            lambda s, u, d: self.mechanistic_delta(s, u, d, None, safe_dt)
        )(states[:-1], controls[:-1], disturbances[:-1])
        observed_delta = jnp.diff(states, axis=0) / safe_dt
        active_mask = (jnp.abs(self.stoichiometry) > 0).astype(states.dtype)
        if self.thermal_state_index is not None:
            active_mask = active_mask.at[self.thermal_state_index].set(1.0)
        active_count = jnp.maximum(jnp.sum(active_mask), 1.0)
        state_residual = jnp.sum(
            jnp.abs(observed_delta - predicted_delta) * active_mask[None, :],
            axis=-1,
        ) / active_count
        return {
            "state_delta_consistency": state_residual,
            "nonnegative_rate": nonnegative_rate,
        }
