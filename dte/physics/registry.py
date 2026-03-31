"""Registry helpers for system-specific physics losses and diagnostics."""

from collections.abc import Callable

import jax.numpy as jnp
from jaxtyping import Array, Float

from dte.physics.base import NullPhysicsLoss, PhysicsLoss


PhysicsDiagnosticFn = Callable[
    [Float[Array, "n_steps state_dim"], Float[Array, "n_steps control_dim"], Float[Array, "n_steps disturbance_dim"], float],
    dict[str, Array],
]


def get_physics_loss(system_name: str, system_config: dict) -> PhysicsLoss:
    """Build the registered PhysicsLoss implementation for ``system_name``."""
    if system_name == "cstr":
        from dte.physics.cstr import CSTRPhysicsLoss
        from dte.simulators.cstr import CSTRParams

        cstr_cfg = system_config.get("cstr", {})
        params = CSTRParams(**{k: float(v) for k, v in cstr_cfg.items()})
        return CSTRPhysicsLoss(params)

    if system_name == "heat_exchanger":
        from dte.physics.heat_exchanger import HeatExchangerPhysicsLoss
        from dte.simulators.heat_exchanger import HeatExchangerParams

        hx_cfg = system_config.get("heat_exchanger", {})
        params = HeatExchangerParams(**{k: float(v) for k, v in hx_cfg.items()})
        return HeatExchangerPhysicsLoss(params)

    return NullPhysicsLoss()


def get_physics_diagnostic_fn(
    system_name: str,
    system_config: dict,
) -> PhysicsDiagnosticFn | None:
    """Return a low-level residual function for evaluation diagnostics.

    The returned callable produces per-timestep residual arrays keyed by
    residual name. Missing diagnostics should simply be omitted from the dict.
    """
    if system_name == "cstr":
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

    if system_name == "heat_exchanger":
        from dte.physics.heat_exchanger import energy_balance_residual
        from dte.simulators.heat_exchanger import HeatExchangerParams

        hx_cfg = system_config.get("heat_exchanger", {})
        params = HeatExchangerParams(**{k: float(v) for k, v in hx_cfg.items()})

        def _diagnose(states, controls, disturbances, dt):
            return {
                "energy": energy_balance_residual(states, controls, disturbances, params, dt),
            }

        return _diagnose

    return None


def zero_residual(length: int) -> jnp.ndarray:
    """Return a zero residual array with at least one element."""
    return jnp.zeros(max(length, 1))
