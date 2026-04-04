"""Backward-compatibility re-export of CSTR conservation functions.

New code should import from :mod:`dte.physics.cstr` directly.
"""

from dte.physics.cstr import (  # noqa: F401
    CSTRPhysicsLoss,
    coolant_energy_balance_residual,
    energy_balance_residual,
    mass_balance_residual,
    reactor_energy_balance_residual,
    species_mass_balance_residuals,
    total_conservation_metric,
)
