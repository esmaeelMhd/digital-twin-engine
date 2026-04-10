"""Reusable microbiology law helpers and modules."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
from jaxtyping import Array, Float

from dte.laws.base import LawModule


def monod_growth_rate(
    substrate: Float[Array, ""],
    *,
    mu_max: float,
    half_saturation: float,
) -> Float[Array, ""]:
    """Return a Monod specific growth rate."""
    safe_substrate = jnp.maximum(substrate, 0.0)
    return mu_max * safe_substrate / (half_saturation + safe_substrate + 1e-8)


def inhibition_factor(
    state_value: Float[Array, ""],
    *,
    inhibition_constant: float | None,
) -> Float[Array, ""]:
    """Return a simple hyperbolic inhibition factor."""
    if inhibition_constant is None:
        return jnp.asarray(1.0, dtype=jnp.float32)
    safe_value = jnp.maximum(state_value, 0.0)
    return inhibition_constant / (inhibition_constant + safe_value + 1e-8)


@dataclass(frozen=True)
class BiologyLaw(LawModule):
    """Simple bioprocess module with growth, uptake, and oxygen transfer."""

    module_name: str
    state_dim: int
    substrate_index: int
    biomass_index: int
    oxygen_index: int | None = None
    mu_max: float = 0.5
    half_saturation: float = 0.1
    decay_rate: float = 0.01
    yield_coefficient: float = 0.5
    oxygen_half_saturation: float | None = None
    kla: float = 0.0
    oxygen_saturation: float = 0.0
    oxygen_demand_factor: float = 0.0
    inhibition_kind: str | None = None
    inhibition_constant: float | None = None

    @property
    def family_name(self) -> str:
        return "biology"

    def feature_names(self) -> tuple[str, ...]:
        return (
            "specific_growth_rate",
            "substrate_uptake",
            "oxygen_transfer",
            "inhibition_factor",
        )

    def residual_names(self) -> tuple[str, ...]:
        return ("biomass_consistency", "substrate_consistency", "oxygen_consistency")

    def _inhibition(self, state: Float[Array, "state_dim"]) -> Float[Array, ""]:
        if self.inhibition_kind == "substrate":
            value = state[self.substrate_index]
        elif self.inhibition_kind == "oxygen" and self.oxygen_index is not None:
            value = state[self.oxygen_index]
        else:
            value = jnp.asarray(0.0, dtype=jnp.float32)
        return inhibition_factor(value, inhibition_constant=self.inhibition_constant)

    def specific_growth_rate(self, state: Float[Array, "state_dim"]) -> Float[Array, ""]:
        mu = monod_growth_rate(
            state[self.substrate_index],
            mu_max=self.mu_max,
            half_saturation=self.half_saturation,
        )
        if self.oxygen_index is not None and self.oxygen_half_saturation is not None:
            oxygen = jnp.maximum(state[self.oxygen_index], 0.0)
            oxygen_factor = oxygen / (self.oxygen_half_saturation + oxygen + 1e-8)
            mu = mu * oxygen_factor
        return mu * self._inhibition(state)

    def substrate_uptake(self, state: Float[Array, "state_dim"]) -> Float[Array, ""]:
        biomass = jnp.maximum(state[self.biomass_index], 0.0)
        return self.specific_growth_rate(state) * biomass / max(self.yield_coefficient, 1e-6)

    def oxygen_transfer(self, state: Float[Array, "state_dim"]) -> Float[Array, ""]:
        if self.oxygen_index is None:
            return jnp.asarray(0.0, dtype=jnp.float32)
        oxygen = state[self.oxygen_index]
        return self.kla * (self.oxygen_saturation - oxygen)

    def feature_vector(
        self,
        state: Float[Array, "state_dim"],
        control,
        disturbance,
        params,
        dt,
    ) -> Float[Array, "4"]:
        del control, disturbance, params, dt
        return jnp.asarray(
            [
                self.specific_growth_rate(state),
                self.substrate_uptake(state),
                self.oxygen_transfer(state),
                self._inhibition(state),
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
        delta = jnp.zeros((self.state_dim,), dtype=jnp.float32)
        biomass = jnp.maximum(state[self.biomass_index], 0.0)
        growth = self.specific_growth_rate(state)
        substrate_uptake = self.substrate_uptake(state)
        d_biomass = (growth - self.decay_rate) * biomass
        d_substrate = -substrate_uptake
        delta = delta.at[self.biomass_index].set(d_biomass)
        delta = delta.at[self.substrate_index].set(d_substrate)
        if self.oxygen_index is not None:
            d_oxygen = self.oxygen_transfer(state) - self.oxygen_demand_factor * growth * biomass
            delta = delta.at[self.oxygen_index].set(d_oxygen)
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
                "biomass_consistency": zeros,
                "substrate_consistency": zeros,
                "oxygen_consistency": zeros,
            }

        safe_dt = jnp.maximum(jnp.asarray(dt, dtype=states.dtype), 1e-6)
        predicted_delta = jnp.stack(
            [
                self.mechanistic_delta(states[idx], controls[idx], disturbances[idx], None, safe_dt)
                for idx in range(states.shape[0] - 1)
            ],
            axis=0,
        )
        observed_delta = jnp.diff(states, axis=0) / safe_dt
        biomass_residual = jnp.abs(
            observed_delta[:, self.biomass_index] - predicted_delta[:, self.biomass_index]
        )
        substrate_residual = jnp.abs(
            observed_delta[:, self.substrate_index] - predicted_delta[:, self.substrate_index]
        )
        if self.oxygen_index is None:
            oxygen_residual = jnp.zeros_like(biomass_residual)
        else:
            oxygen_residual = jnp.abs(
                observed_delta[:, self.oxygen_index] - predicted_delta[:, self.oxygen_index]
            )
        return {
            "biomass_consistency": biomass_residual,
            "substrate_consistency": substrate_residual,
            "oxygen_consistency": oxygen_residual,
        }
