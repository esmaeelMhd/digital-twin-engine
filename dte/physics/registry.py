"""Registry helpers for system-specific physics losses and diagnostics."""

from collections.abc import Callable
from dataclasses import fields

import jax.numpy as jnp
from jaxtyping import Array, Float

from dte.laws.integration import (
    LawAugmentedPhysicsLoss,
    augment_diagnostic_with_laws,
    build_law_bundle,
)
from dte.physics.base import NullPhysicsLoss, PhysicsLoss
from dte.simulators.registry import get_system_spec


PhysicsDiagnosticFn = Callable[
    [
        Float[Array, "n_steps state_dim"],
        Float[Array, "n_steps control_dim"],
        Float[Array, "n_steps disturbance_dim"],
        float,
        Float[Array, "param_dim"] | None,
    ],
    dict[str, Array],
]


def _params_from_config(cfg: dict, params_cls):
    """Build a parameter dataclass, skipping unknown keys.

    Known fields that cannot be converted to float raise a clear error
    instead of crashing inside the dataclass constructor.
    """
    allowed = {item.name for item in fields(params_cls)}
    kwargs = {}
    for key, value in cfg.items():
        if key not in allowed:
            continue
        try:
            kwargs[key] = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Physics config key {key!r} for {params_cls.__name__} "
                f"must be numeric, got {value!r}"
            ) from exc
    return params_cls(**kwargs)


def _build_cstr_physics_loss(system_config: dict) -> PhysicsLoss:
    from dte.physics.cstr import CSTRPhysicsLoss
    from dte.simulators.cstr import CSTRParams

    params_obj = _params_from_config(system_config.get("cstr", {}), CSTRParams)
    return CSTRPhysicsLoss(params_obj)


def _build_heat_exchanger_physics_loss(system_config: dict) -> PhysicsLoss:
    from dte.physics.heat_exchanger import HeatExchangerPhysicsLoss
    from dte.simulators.heat_exchanger import HeatExchangerParams

    params_obj = _params_from_config(system_config.get("heat_exchanger", {}), HeatExchangerParams)
    return HeatExchangerPhysicsLoss(params_obj)


def _build_two_tank_physics_loss(system_config: dict) -> PhysicsLoss:
    from dte.physics.two_tank import TwoTankPhysicsLoss
    from dte.simulators.two_tank import TwoTankParams

    params_obj = _params_from_config(system_config.get("two_tank", {}), TwoTankParams)
    return TwoTankPhysicsLoss(params_obj)


def _build_isothermal_cstr_physics_loss(system_config: dict) -> PhysicsLoss:
    from dte.physics.isothermal_cstr import IsothermalCSTRPhysicsLoss
    from dte.simulators.isothermal_cstr import IsothermalCSTRParams

    params_obj = _params_from_config(system_config.get("isothermal_cstr", {}), IsothermalCSTRParams)
    return IsothermalCSTRPhysicsLoss(params_obj)


def _build_storage_tank_physics_loss(system_config: dict) -> PhysicsLoss:
    from dte.physics.storage_tank import StorageTankPhysicsLoss
    from dte.simulators.storage_tank import StorageTankParams

    params_obj = _params_from_config(system_config.get("storage_tank", {}), StorageTankParams)
    return StorageTankPhysicsLoss(params_obj)


def _build_separator_physics_loss(system_config: dict) -> PhysicsLoss:
    from dte.physics.separator import SeparatorPhysicsLoss
    from dte.simulators.separator import SeparatorParams

    params_obj = _params_from_config(system_config.get("separator", {}), SeparatorParams)
    return SeparatorPhysicsLoss(params_obj)


def _build_bioreactor_compartment_physics_loss(system_config: dict) -> PhysicsLoss:
    from dte.physics.bioreactor_compartment import BioreactorCompartmentPhysicsLoss
    from dte.simulators.bioreactor_compartment import BioreactorCompartmentParams

    params_obj = _params_from_config(system_config.get("bioreactor_compartment", {}), BioreactorCompartmentParams)
    return BioreactorCompartmentPhysicsLoss(params_obj)


def _build_cstr_diagnostic_fn(system_config: dict) -> PhysicsDiagnosticFn:
    from dte.physics.cstr import (
        energy_balance_residual,
        energy_balance_residual_with_params,
        mass_balance_residual,
        mass_balance_residual_with_params,
    )
    from dte.simulators.cstr import CSTRParams

    params_obj = _params_from_config(system_config.get("cstr", {}), CSTRParams)

    def _diagnose(states, controls, disturbances, dt, params=None):
        if params is None:
            mass = mass_balance_residual(states, controls, disturbances, params_obj, dt)
            energy = energy_balance_residual(states, controls, disturbances, params_obj, dt)
        else:
            mass = mass_balance_residual_with_params(
                states, controls, disturbances, params, dt, params_obj
            )
            energy = energy_balance_residual_with_params(
                states, controls, disturbances, params, dt, params_obj
            )
        return {
            "mass": mass,
            "energy": energy,
        }

    return _diagnose


def _build_heat_exchanger_diagnostic_fn(system_config: dict) -> PhysicsDiagnosticFn:
    from dte.physics.heat_exchanger import (
        energy_balance_residual,
        energy_balance_residual_with_params,
    )
    from dte.simulators.heat_exchanger import HeatExchangerParams

    params_obj = _params_from_config(system_config.get("heat_exchanger", {}), HeatExchangerParams)

    def _diagnose(states, controls, disturbances, dt, params=None):
        return {
            "energy": (
                energy_balance_residual(states, controls, disturbances, params_obj, dt)
                if params is None
                else energy_balance_residual_with_params(states, controls, disturbances, params, dt)
            ),
        }

    return _diagnose


def _build_two_tank_diagnostic_fn(system_config: dict) -> PhysicsDiagnosticFn:
    from dte.physics.two_tank import (
        mass_balance_residual,
        mass_balance_residual_with_params,
    )
    from dte.simulators.two_tank import TwoTankParams

    params_obj = _params_from_config(system_config.get("two_tank", {}), TwoTankParams)

    def _diagnose(states, controls, disturbances, dt, params=None):
        return {
            "mass": (
                mass_balance_residual(states, controls, disturbances, params_obj, dt)
                if params is None
                else mass_balance_residual_with_params(states, controls, disturbances, params, dt)
            ),
        }

    return _diagnose


def _build_isothermal_cstr_diagnostic_fn(system_config: dict) -> PhysicsDiagnosticFn:
    from dte.physics.isothermal_cstr import (
        mass_balance_residual_with_params,
        species_mass_balance_residuals_with_params,
    )
    from dte.simulators.isothermal_cstr import IsothermalCSTRParams

    params_obj = _params_from_config(system_config.get("isothermal_cstr", {}), IsothermalCSTRParams)
    nominal_params = jnp.array(
        [params_obj.V, params_obj.k0, params_obj.Ea_over_R, params_obj.T_ref],
        dtype=jnp.float32,
    )

    def _diagnose(states, controls, disturbances, dt, params=None):
        p = nominal_params if params is None else params
        return {
            "mass": mass_balance_residual_with_params(states, controls, disturbances, p, dt),
            "species_mass": jnp.mean(
                species_mass_balance_residuals_with_params(states, controls, disturbances, p, dt),
                axis=-1,
            ),
        }

    return _diagnose


def _build_storage_tank_diagnostic_fn(system_config: dict) -> PhysicsDiagnosticFn:
    from dte.physics.storage_tank import state_residuals_with_params
    from dte.simulators.storage_tank import StorageTankParams

    params_obj = _params_from_config(system_config.get("storage_tank", {}), StorageTankParams)
    nominal_params = jnp.array([params_obj.volume, params_obj.heat_loss], dtype=jnp.float32)

    def _diagnose(states, controls, disturbances, dt, params=None):
        p = nominal_params if params is None else params
        residuals = state_residuals_with_params(states, controls, disturbances, p, dt)
        return {
            "mass": residuals[:, 0],
            "composition": residuals[:, 1],
            "energy": residuals[:, 2],
        }

    return _diagnose


def _build_separator_diagnostic_fn(system_config: dict) -> PhysicsDiagnosticFn:
    from dte.physics.separator import state_residuals_with_params
    from dte.simulators.separator import SeparatorParams

    params_obj = _params_from_config(system_config.get("separator", {}), SeparatorParams)
    nominal_params = jnp.array([params_obj.holdup, params_obj.separation_gain], dtype=jnp.float32)

    def _diagnose(states, controls, disturbances, dt, params=None):
        p = nominal_params if params is None else params
        residuals = state_residuals_with_params(states, controls, disturbances, p, dt)
        return {
            "phase_split": 0.5 * (residuals[:, 0] + residuals[:, 1]),
            "energy": residuals[:, 2],
        }

    return _diagnose


def _build_bioreactor_compartment_diagnostic_fn(system_config: dict) -> PhysicsDiagnosticFn:
    from dte.physics.bioreactor_compartment import state_residuals_with_params
    from dte.simulators.bioreactor_compartment import BioreactorCompartmentParams

    params_obj = _params_from_config(system_config.get("bioreactor_compartment", {}), BioreactorCompartmentParams)
    nominal_params = jnp.array(
        [params_obj.mu_max, params_obj.kla, params_obj.decay_rate, params_obj.dilution_rate],
        dtype=jnp.float32,
    )

    def _diagnose(states, controls, disturbances, dt, params=None):
        p = nominal_params if params is None else params
        residuals = state_residuals_with_params(states, controls, disturbances, p, dt)
        return {
            "substrate": residuals[:, 0],
            "biomass": residuals[:, 1],
            "oxygen": residuals[:, 2],
        }

    return _diagnose


_PHYSICS_LOSS_BUILDERS = {
    "bioreactor_compartment": _build_bioreactor_compartment_physics_loss,
    "cstr": _build_cstr_physics_loss,
    "heat_exchanger": _build_heat_exchanger_physics_loss,
    "isothermal_cstr": _build_isothermal_cstr_physics_loss,
    "separator": _build_separator_physics_loss,
    "storage_tank": _build_storage_tank_physics_loss,
    "two_tank": _build_two_tank_physics_loss,
}

_PHYSICS_DIAGNOSTIC_BUILDERS = {
    "bioreactor_compartment": _build_bioreactor_compartment_diagnostic_fn,
    "cstr": _build_cstr_diagnostic_fn,
    "heat_exchanger": _build_heat_exchanger_diagnostic_fn,
    "isothermal_cstr": _build_isothermal_cstr_diagnostic_fn,
    "separator": _build_separator_diagnostic_fn,
    "storage_tank": _build_storage_tank_diagnostic_fn,
    "two_tank": _build_two_tank_diagnostic_fn,
}


def get_physics_loss(system_name: str, system_config: dict) -> PhysicsLoss:
    """Build the registered PhysicsLoss implementation for ``system_name``."""
    builder = _PHYSICS_LOSS_BUILDERS.get(system_name)
    if builder is None:
        base_loss: PhysicsLoss = NullPhysicsLoss()
    else:
        base_loss = builder(system_config)

    try:
        spec = get_system_spec(system_config)
    except Exception:
        return base_loss

    law_bundle = build_law_bundle(spec, system_config)
    if law_bundle is None:
        return base_loss
    return LawAugmentedPhysicsLoss(base_loss, law_bundle)


def get_physics_diagnostic_fn(
    system_name: str,
    system_config: dict,
) -> PhysicsDiagnosticFn | None:
    """Return a low-level residual function for evaluation diagnostics.

    The returned callable produces per-timestep residual arrays keyed by
    residual name. Missing diagnostics should simply be omitted from the dict.
    """
    builder = _PHYSICS_DIAGNOSTIC_BUILDERS.get(system_name)
    base_fn = None if builder is None else builder(system_config)
    try:
        spec = get_system_spec(system_config)
    except Exception:
        return base_fn

    law_bundle = build_law_bundle(spec, system_config)
    if law_bundle is None:
        return base_fn
    return augment_diagnostic_with_laws(base_fn, law_bundle)


def zero_residual(length: int) -> jnp.ndarray:
    """Return a zero residual array with at least one element."""
    return jnp.zeros(max(length, 1))
