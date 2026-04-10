"""Base abstractions for reusable mechanistic law modules."""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


class LawModule(ABC):
    """Base protocol for modular law layers.

    Law modules expose four integration surfaces:

    - explicit features for learned models
    - partial mechanistic state updates
    - residual/constraint series for training or diagnostics
    - metadata naming for configuration and logging
    """

    module_name: str
    state_dim: int

    @property
    def family_name(self) -> str:
        return "generic"

    def feature_names(self) -> tuple[str, ...]:
        return ()

    def residual_names(self) -> tuple[str, ...]:
        return ()

    def feature_vector(
        self,
        state: Float[Array, "state_dim"],
        control: Float[Array, "control_dim"] | None,
        disturbance: Float[Array, "disturbance_dim"] | None,
        params: Float[Array, "param_dim"] | None,
        dt: float | Array,
    ) -> Float[Array, "feature_dim"]:
        del state, control, disturbance, params, dt
        return jnp.zeros((0,), dtype=jnp.float32)

    def mechanistic_delta(
        self,
        state: Float[Array, "state_dim"],
        control: Float[Array, "control_dim"] | None,
        disturbance: Float[Array, "disturbance_dim"] | None,
        params: Float[Array, "param_dim"] | None,
        dt: float | Array,
    ) -> Float[Array, "state_dim"]:
        del state, control, disturbance, params, dt
        return jnp.zeros((self.state_dim,), dtype=jnp.float32)

    def trajectory_residuals(
        self,
        states: Float[Array, "n_steps state_dim"],
        controls: Float[Array, "n_steps control_dim"],
        disturbances: Float[Array, "n_steps disturbance_dim"],
        dt: float | Array,
        params: Float[Array, "param_dim"] | None,
    ) -> dict[str, Float[Array, "n_steps_minus_one"]]:
        del states, controls, disturbances, dt, params
        return {}


@dataclass(frozen=True)
class UnitLawBundle:
    """Composition container for all law modules attached to one unit."""

    spec_name: str
    state_dim: int
    modules: tuple[LawModule, ...]

    def feature_names(self) -> tuple[str, ...]:
        names: list[str] = []
        for module in self.modules:
            names.extend(
                f"{module.family_name}_{module.module_name}_{name}"
                for name in module.feature_names()
            )
        return tuple(names)

    def residual_names(self) -> tuple[str, ...]:
        names: list[str] = []
        for module in self.modules:
            names.extend(
                f"{module.family_name}_{module.module_name}_{name}"
                for name in module.residual_names()
            )
        return tuple(names)

    def feature_vector(
        self,
        state: Float[Array, "state_dim"],
        control: Float[Array, "control_dim"] | None,
        disturbance: Float[Array, "disturbance_dim"] | None,
        params: Float[Array, "param_dim"] | None,
        dt: float | Array,
    ) -> Float[Array, "feature_dim"]:
        if not self.modules:
            return jnp.zeros((0,), dtype=jnp.float32)
        features = [
            module.feature_vector(state, control, disturbance, params, dt)
            for module in self.modules
        ]
        if not features:
            return jnp.zeros((0,), dtype=jnp.float32)
        return jnp.concatenate(features, axis=0)

    def mechanistic_delta(
        self,
        state: Float[Array, "state_dim"],
        control: Float[Array, "control_dim"] | None,
        disturbance: Float[Array, "disturbance_dim"] | None,
        params: Float[Array, "param_dim"] | None,
        dt: float | Array,
    ) -> Float[Array, "state_dim"]:
        if not self.modules:
            return jnp.zeros((self.state_dim,), dtype=jnp.float32)
        deltas = [
            module.mechanistic_delta(state, control, disturbance, params, dt)
            for module in self.modules
        ]
        return jnp.sum(jnp.stack(deltas, axis=0), axis=0)

    def trajectory_residual_series(
        self,
        states: Float[Array, "n_steps state_dim"],
        controls: Float[Array, "n_steps control_dim"],
        disturbances: Float[Array, "n_steps disturbance_dim"],
        dt: float | Array,
        params: Float[Array, "param_dim"] | None = None,
    ) -> dict[str, Float[Array, "n_steps_minus_one"]]:
        residuals: dict[str, Float[Array, "n_steps_minus_one"]] = {}
        for module in self.modules:
            module_residuals = module.trajectory_residuals(
                states,
                controls,
                disturbances,
                dt,
                params,
            )
            for name, values in module_residuals.items():
                residuals[f"{module.family_name}_{module.module_name}_{name}"] = values
        return residuals

    def compute_residuals(
        self,
        states: Float[Array, "batch n_steps state_dim"],
        controls: Float[Array, "batch n_steps control_dim"],
        disturbances: Float[Array, "batch n_steps disturbance_dim"],
        dt: float | Array,
        params_batch: Float[Array, "batch param_dim"] | None = None,
    ) -> dict[str, Float[Array, ""]]:
        if not self.modules:
            return {}

        batch_size = int(states.shape[0])
        if params_batch is None:
            params_batch = jnp.zeros((batch_size, 0), dtype=states.dtype)

        aggregated: dict[str, Float[Array, ""]] = {}
        for module in self.modules:
            def one_trajectory(s, u, d, p):
                return module.trajectory_residuals(s, u, d, dt, p)

            module_residuals = jax.vmap(one_trajectory)(
                states,
                controls,
                disturbances,
                params_batch,
            )
            for name, values in module_residuals.items():
                aggregated[f"{module.family_name}_{module.module_name}_{name}"] = jnp.mean(values)
        return aggregated
