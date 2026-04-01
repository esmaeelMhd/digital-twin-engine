"""Registry helpers for system-specific physics losses and diagnostics."""

from collections.abc import Callable

import jax.numpy as jnp
from jaxtyping import Array, Float

from dte.physics.base import NullPhysicsLoss, PhysicsLoss


PhysicsDiagnosticFn = Callable[
    [Float[Array, "n_steps state_dim"], Float[Array, "n_steps control_dim"], Float[Array, "n_steps disturbance_dim"], float],
    dict[str, Array],
]


def _build_cstr_physics_loss(system_config: dict) -> PhysicsLoss:
    from dte.physics.cstr import CSTRPhysicsLoss
    from dte.simulators.cstr import CSTRParams

    cstr_cfg = system_config.get("cstr", {})
    params = CSTRParams(**{k: float(v) for k, v in cstr_cfg.items()})
    return CSTRPhysicsLoss(params)


def _build_heat_exchanger_physics_loss(system_config: dict) -> PhysicsLoss:
    from dte.physics.heat_exchanger import HeatExchangerPhysicsLoss
    from dte.simulators.heat_exchanger import HeatExchangerParams

    hx_cfg = system_config.get("heat_exchanger", {})
    params = HeatExchangerParams(**{k: float(v) for k, v in hx_cfg.items()})
    return HeatExchangerPhysicsLoss(params)


def _build_cstr_diagnostic_fn(system_config: dict) -> PhysicsDiagnosticFn:
    from dte.physics.cstr import energy_balance_residual, mass_balance_residual
    from dte.simulators.cstr import CSTRParams

    cstr_cfg = system_config.get("cstr", {})
    params = CSTRParams(**{k: float(v) for k, v in cstr_cfg.items()})

    def _diagnose(states, controls, disturbances, dt):
        return {
            "mass": mass_balance_residual(states, controls, disturbances, params, dt),
            "energy": energy_balance_residual(states, controls, disturbances, params, dt),
        }

    return _diagnose


def _build_heat_exchanger_diagnostic_fn(system_config: dict) -> PhysicsDiagnosticFn:
    from dte.physics.heat_exchanger import energy_balance_residual
    from dte.simulators.heat_exchanger import HeatExchangerParams

    hx_cfg = system_config.get("heat_exchanger", {})
    params = HeatExchangerParams(**{k: float(v) for k, v in hx_cfg.items()})

    def _diagnose(states, controls, disturbances, dt):
        return {
            "energy": energy_balance_residual(states, controls, disturbances, params, dt),
        }

    return _diagnose


_PHYSICS_LOSS_BUILDERS = {
    "cstr": _build_cstr_physics_loss,
    "heat_exchanger": _build_heat_exchanger_physics_loss,
}

_PHYSICS_DIAGNOSTIC_BUILDERS = {
    "cstr": _build_cstr_diagnostic_fn,
    "heat_exchanger": _build_heat_exchanger_diagnostic_fn,
}


def get_physics_loss(system_name: str, system_config: dict) -> PhysicsLoss:
    """Build the registered PhysicsLoss implementation for ``system_name``."""
    builder = _PHYSICS_LOSS_BUILDERS.get(system_name)
    if builder is None:
        return NullPhysicsLoss()
    return builder(system_config)


def get_physics_diagnostic_fn(
    system_name: str,
    system_config: dict,
) -> PhysicsDiagnosticFn | None:
    """Return a low-level residual function for evaluation diagnostics.

    The returned callable produces per-timestep residual arrays keyed by
    residual name. Missing diagnostics should simply be omitted from the dict.
    """
    builder = _PHYSICS_DIAGNOSTIC_BUILDERS.get(system_name)
    if builder is None:
        return None
    return builder(system_config)


def zero_residual(length: int) -> jnp.ndarray:
    """Return a zero residual array with at least one element."""
    return jnp.zeros(max(length, 1))
