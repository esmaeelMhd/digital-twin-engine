"""Tests for the richer Phase 1 ProcessUnitSpec metadata."""

import yaml

from dte.core.process_unit_spec import ProcessUnitSpec
from dte.simulators.registry import get_system_spec


def _load_spec(path: str) -> ProcessUnitSpec:
    with open(path, "r") as f:
        return get_system_spec(yaml.safe_load(f))


def test_cstr_process_unit_spec_exposes_typed_channels():
    spec = _load_spec("configs/cstr_default.yaml")

    assert isinstance(spec, ProcessUnitSpec)
    assert spec.unit_type == "stirred_tank_reactor"
    assert spec.family == "reactor"
    assert spec.law_tags == ["mass_balance", "energy_balance", "reaction_kinetics"]
    assert [channel.name for channel in spec.state_channels] == spec.state_names
    assert [channel.role for channel in spec.state_channels] == [
        "concentration",
        "concentration",
        "temperature",
        "temperature",
    ]
    assert [descriptor.name for descriptor in spec.parameter_descriptors] == [
        "V",
        "Ea_over_R",
        "dH_rxn",
        "UA",
        "Fc",
        "Cp",
    ]


def test_two_tank_process_unit_spec_exposes_bounds_and_roles():
    spec = _load_spec("configs/two_tank_default.yaml")

    assert isinstance(spec, ProcessUnitSpec)
    assert spec.unit_type == "coupled_tanks"
    assert spec.family == "hydraulic"
    assert [channel.role for channel in spec.control_channels] == ["flow", "actuator_state"]
    assert list(spec.state_lower_bounds()) == [0.0, 0.0]
    assert list(spec.state_upper_bounds()) == [5.0, 5.0]
