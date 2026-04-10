"""Reusable modular law layers for chemistry, thermo, and biology."""

from dte.laws.base import LawModule, UnitLawBundle
from dte.laws.biology import BiologyLaw, inhibition_factor, monod_growth_rate
from dte.laws.chemistry import ChemistryLaw, arrhenius_rate_constant, power_law_rate
from dte.laws.examples import (
    build_bioreactor_law_bundle_example,
    build_bioreactor_law_example_config,
    build_bioreactor_process_unit_spec,
    build_cstr_law_bundle_example,
    build_cstr_law_example_config,
)
from dte.laws.integration import (
    LawAugmentedPhysicsLoss,
    augment_diagnostic_with_laws,
    build_law_bundle,
)
from dte.laws.thermo import ThermoLaw, enthalpy_like_transform, linear_heat_capacity

__all__ = [
    "BiologyLaw",
    "ChemistryLaw",
    "LawAugmentedPhysicsLoss",
    "LawModule",
    "ThermoLaw",
    "UnitLawBundle",
    "arrhenius_rate_constant",
    "augment_diagnostic_with_laws",
    "build_bioreactor_law_bundle_example",
    "build_bioreactor_law_example_config",
    "build_bioreactor_process_unit_spec",
    "build_cstr_law_bundle_example",
    "build_cstr_law_example_config",
    "build_law_bundle",
    "enthalpy_like_transform",
    "inhibition_factor",
    "linear_heat_capacity",
    "monod_growth_rate",
    "power_law_rate",
]
