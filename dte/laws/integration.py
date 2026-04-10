"""Config-driven integration hooks for modular law bundles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax.numpy as jnp
from jaxtyping import Array, Float

from dte.laws.base import LawModule, UnitLawBundle
from dte.laws.biology import BiologyLaw
from dte.laws.chemistry import ChemistryLaw
from dte.laws.thermo import ThermoLaw
from dte.physics.base import PhysicsLoss
from dte.simulators.base import ProcessUnitSpec


def _ensure_list(raw):
    if raw is None:
        return []
    if isinstance(raw, list):
        return raw
    return [raw]


def _build_chemistry_module(spec: ProcessUnitSpec, item: dict) -> ChemistryLaw:
    kind = str(item.get("kind", "arrhenius_reaction"))
    if kind not in {"arrhenius_reaction", "mass_action", "constant"}:
        raise ValueError(f"unsupported chemistry law kind '{kind}'.")
    return ChemistryLaw(
        module_name=str(item["name"]),
        state_dim=spec.state_dim,
        stoichiometry=jnp.asarray(
            item.get("stoichiometry", [0.0] * spec.state_dim),
            dtype=jnp.float32,
        ),
        reactant_indices=tuple(int(idx) for idx in item.get("reactant_indices", [])),
        reaction_orders=jnp.asarray(item.get("reaction_orders", []), dtype=jnp.float32),
        temperature_index=(
            None if item.get("temperature_index") is None else int(item["temperature_index"])
        ),
        kinetic_family=str(item.get("kinetic_family", kind)),
        pre_exponential=float(item.get("pre_exponential", 1.0)),
        activation_energy_over_r=float(item.get("activation_energy_over_r", 0.0)),
        heat_of_reaction=float(item.get("heat_of_reaction", 0.0)),
        thermal_state_index=(
            None if item.get("thermal_state_index") is None else int(item["thermal_state_index"])
        ),
        state_gain=float(item.get("state_gain", 1.0)),
        thermal_gain=float(item.get("thermal_gain", 0.0)),
    )


def _build_thermo_module(spec: ProcessUnitSpec, item: dict) -> ThermoLaw:
    kind = str(item.get("kind", "constant_heat_capacity"))
    if kind not in {"constant_heat_capacity", "linear_heat_capacity"}:
        raise ValueError(f"unsupported thermo law kind '{kind}'.")
    return ThermoLaw(
        module_name=str(item["name"]),
        state_dim=spec.state_dim,
        temperature_index=int(item["temperature_index"]),
        reference_temperature=float(item.get("reference_temperature", 298.15)),
        heat_capacity_reference=float(item.get("heat_capacity_reference", 1.0)),
        heat_capacity_slope=float(item.get("heat_capacity_slope", 0.0)),
        density=float(item.get("density", 1.0)),
        equilibrium_temperature=item.get("equilibrium_temperature"),
        equilibrium_sharpness=float(item.get("equilibrium_sharpness", 0.0)),
        thermal_state_index=(
            None if item.get("thermal_state_index") is None else int(item["thermal_state_index"])
        ),
        thermal_gain=float(item.get("thermal_gain", 0.0)),
    )


def _build_biology_module(spec: ProcessUnitSpec, item: dict) -> BiologyLaw:
    kind = str(item.get("kind", "monod_growth"))
    if kind != "monod_growth":
        raise ValueError(f"unsupported biology law kind '{kind}'.")
    return BiologyLaw(
        module_name=str(item["name"]),
        state_dim=spec.state_dim,
        substrate_index=int(item["substrate_index"]),
        biomass_index=int(item["biomass_index"]),
        oxygen_index=(
            None if item.get("oxygen_index") is None else int(item["oxygen_index"])
        ),
        mu_max=float(item.get("mu_max", 0.5)),
        half_saturation=float(item.get("half_saturation", 0.1)),
        decay_rate=float(item.get("decay_rate", 0.01)),
        yield_coefficient=float(item.get("yield_coefficient", 0.5)),
        oxygen_half_saturation=item.get("oxygen_half_saturation"),
        kla=float(item.get("kla", 0.0)),
        oxygen_saturation=float(item.get("oxygen_saturation", 0.0)),
        oxygen_demand_factor=float(item.get("oxygen_demand_factor", 0.0)),
        inhibition_kind=item.get("inhibition_kind"),
        inhibition_constant=item.get("inhibition_constant"),
    )


def build_law_bundle(
    spec: ProcessUnitSpec,
    system_config: dict,
) -> UnitLawBundle | None:
    """Build all configured law modules for a unit spec."""
    law_cfg = system_config.get("laws", {})
    if not law_cfg or not bool(law_cfg.get("enabled", False)):
        return None

    modules: list[LawModule] = []
    for item in _ensure_list(law_cfg.get("chemistry")):
        modules.append(_build_chemistry_module(spec, item))
    for item in _ensure_list(law_cfg.get("thermo")):
        modules.append(_build_thermo_module(spec, item))
    for item in _ensure_list(law_cfg.get("biology")):
        modules.append(_build_biology_module(spec, item))

    if not modules:
        return None
    return UnitLawBundle(
        spec_name=spec.name,
        state_dim=spec.state_dim,
        modules=tuple(modules),
    )


@dataclass(frozen=True)
class LawAugmentedPhysicsLoss(PhysicsLoss):
    """PhysicsLoss wrapper that appends modular-law residuals."""

    base_physics: PhysicsLoss
    law_bundle: UnitLawBundle

    def compute_residuals(
        self,
        states: Float[Array, "batch seq_len state_dim"],
        controls: Float[Array, "batch seq_len control_dim"],
        disturbances: Float[Array, "batch seq_len disturbance_dim"],
        dt: float,
        params_batch=None,
    ) -> dict[str, Float[Array, ""]]:
        residuals = dict(
            self.base_physics.compute_residuals(
                states,
                controls,
                disturbances,
                dt,
                params_batch=params_batch,
            )
        )
        residuals.update(
            self.law_bundle.compute_residuals(
                states,
                controls,
                disturbances,
                dt,
                params_batch=params_batch,
            )
        )
        return residuals

    def residual_names(self) -> list[str]:
        return list(dict.fromkeys([
            *self.base_physics.residual_names(),
            *self.law_bundle.residual_names(),
        ]))


def augment_diagnostic_with_laws(
    base_diagnostic_fn: Callable | None,
    law_bundle: UnitLawBundle,
) -> Callable:
    """Augment a per-trajectory diagnostic function with law residual series."""

    def _diagnose(states, controls, disturbances, dt, params=None):
        residuals = {}
        if base_diagnostic_fn is not None:
            residuals.update(base_diagnostic_fn(states, controls, disturbances, dt, params))
        residuals.update(
            law_bundle.trajectory_residual_series(
                states,
                controls,
                disturbances,
                dt,
                params=params,
            )
        )
        return residuals

    return _diagnose
