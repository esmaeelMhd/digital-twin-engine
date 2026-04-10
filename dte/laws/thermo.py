"""Reusable thermodynamic placeholder modules."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from dte.laws.base import LawModule


def linear_heat_capacity(
    temperature: Float[Array, ""],
    *,
    reference_temperature: float,
    reference_heat_capacity: float,
    slope: float = 0.0,
) -> Float[Array, ""]:
    """Simple linear heat-capacity correlation."""
    return reference_heat_capacity + slope * (temperature - reference_temperature)


def enthalpy_like_transform(
    temperature: Float[Array, ""],
    *,
    reference_temperature: float,
    heat_capacity: Float[Array, ""],
    density: float = 1.0,
) -> Float[Array, ""]:
    """Lightweight enthalpy-like state transform."""
    return density * heat_capacity * (temperature - reference_temperature)


@dataclass(frozen=True)
class ThermoLaw(LawModule):
    """Simple thermodynamic correlation and equilibrium placeholder."""

    module_name: str
    state_dim: int
    temperature_index: int
    reference_temperature: float = 298.15
    heat_capacity_reference: float = 1.0
    heat_capacity_slope: float = 0.0
    density: float = 1.0
    equilibrium_temperature: float | None = None
    equilibrium_sharpness: float = 0.0
    thermal_state_index: int | None = None
    thermal_gain: float = 0.0

    @property
    def family_name(self) -> str:
        return "thermo"

    def feature_names(self) -> tuple[str, ...]:
        return ("heat_capacity", "enthalpy_like", "phase_indicator")

    def residual_names(self) -> tuple[str, ...]:
        return ("enthalpy_transform_consistency",)

    def heat_capacity(self, state: Float[Array, "state_dim"]) -> Float[Array, ""]:
        return linear_heat_capacity(
            state[self.temperature_index],
            reference_temperature=self.reference_temperature,
            reference_heat_capacity=self.heat_capacity_reference,
            slope=self.heat_capacity_slope,
        )

    def enthalpy_like(self, state: Float[Array, "state_dim"]) -> Float[Array, ""]:
        return enthalpy_like_transform(
            state[self.temperature_index],
            reference_temperature=self.reference_temperature,
            heat_capacity=self.heat_capacity(state),
            density=self.density,
        )

    def phase_indicator(self, state: Float[Array, "state_dim"]) -> Float[Array, ""]:
        if self.equilibrium_temperature is None or self.equilibrium_sharpness == 0.0:
            return jnp.asarray(0.0, dtype=jnp.float32)
        return jax.nn.sigmoid(
            self.equilibrium_sharpness * (state[self.temperature_index] - self.equilibrium_temperature)
        )

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
                self.heat_capacity(state),
                self.enthalpy_like(state),
                self.phase_indicator(state),
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
        if self.thermal_state_index is not None:
            delta = delta.at[self.thermal_state_index].set(
                self.thermal_gain * (self.reference_temperature - state[self.temperature_index])
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
        del controls, disturbances, params
        if states.shape[0] < 2:
            return {"enthalpy_transform_consistency": jnp.zeros((1,), dtype=states.dtype)}

        safe_dt = jnp.maximum(jnp.asarray(dt, dtype=states.dtype), 1e-6)
        enthalpy = jax.vmap(self.enthalpy_like)(states)
        cp = jax.vmap(self.heat_capacity)(states[:-1])
        dH_dt = jnp.diff(enthalpy) / safe_dt
        dT_dt = jnp.diff(states[:, self.temperature_index]) / safe_dt
        residual = jnp.abs(dH_dt - self.density * cp * dT_dt)
        return {"enthalpy_transform_consistency": residual}
