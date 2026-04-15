"""Tests for simulator registry helpers."""

import yaml

from dte.core.process_unit_spec import ProcessUnitSpec
from dte.simulators.base import ProcessSimulator
from dte.simulators.registry import get_simulator, get_system_spec, list_systems


def _load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def test_get_system_spec_uses_registered_name():
    system_config = _load_yaml("configs/heat_exchanger_default.yaml")

    spec = get_system_spec(system_config)

    assert spec.name == "heat_exchanger"
    assert isinstance(spec, ProcessUnitSpec)
    assert spec.state_dim == 2
    assert len(spec.state_groups) == 1
    assert spec.state_groups[0].kind == "thermal"
    assert spec.state_groups[0].indices == [0, 1]
    assert spec.state_channels[0].role == "temperature"
    assert spec.family == "thermal"


def test_get_simulator_returns_registered_process_simulator():
    system_config = _load_yaml("configs/cstr_default.yaml")

    simulator = get_simulator("cstr", system_config)

    assert isinstance(simulator, ProcessSimulator)
    assert simulator.spec.name == "cstr"


def test_list_systems_exposes_registered_systems():
    systems = list_systems()

    assert "bioreactor_compartment" in systems
    assert "cstr" in systems
    assert "heat_exchanger" in systems
    assert "isothermal_cstr" in systems
    assert "separator" in systems
    assert "storage_tank" in systems
    assert "two_tank" in systems


def test_two_tank_system_spec_and_simulator_are_registered():
    system_config = _load_yaml("configs/two_tank_default.yaml")

    spec = get_system_spec(system_config)
    simulator = get_simulator("two_tank", system_config)

    assert spec.name == "two_tank"
    assert spec.state_dim == 2
    assert spec.control_dim == 2
    assert spec.disturbance_dim == 2
    assert isinstance(simulator, ProcessSimulator)
    assert simulator.spec.name == "two_tank"
    assert spec.state_groups[0].kind == "inventory"
    assert spec.state_groups[0].indices == [0, 1]
    assert spec.state_channels[0].lower_bound == 0.0
    assert spec.control_channels[1].role == "actuator_state"


def test_cstr_state_groups_cover_species_and_temperatures():
    system_config = _load_yaml("configs/cstr_default.yaml")

    spec = get_system_spec(system_config)

    assert [group.kind for group in spec.state_groups] == ["concentration", "thermal"]
    assert [group.indices for group in spec.state_groups] == [[0, 1], [2, 3]]
    assert [channel.role for channel in spec.state_channels] == [
        "concentration",
        "concentration",
        "temperature",
        "temperature",
    ]


def test_storage_tank_system_spec_and_simulator_are_registered():
    system_config = _load_yaml("configs/storage_tank_default.yaml")

    spec = get_system_spec(system_config)
    simulator = get_simulator("storage_tank", system_config)

    assert spec.name == "storage_tank"
    assert spec.state_dim == 3
    assert spec.control_dim == 1
    assert spec.disturbance_dim == 3
    assert spec.param_dim == 2
    assert isinstance(simulator, ProcessSimulator)
    assert simulator.spec.name == "storage_tank"
    assert [group.kind for group in spec.state_groups] == [
        "inventory",
        "concentration",
        "thermal",
    ]
    assert spec.control_channels[0].role == "flow"
    assert spec.parameter_descriptors[0].name == "volume"


def test_bioreactor_compartment_system_spec_and_simulator_are_registered():
    system_config = _load_yaml("configs/bioreactor_compartment_default.yaml")

    spec = get_system_spec(system_config)
    simulator = get_simulator("bioreactor_compartment", system_config)

    assert spec.name == "bioreactor_compartment"
    assert spec.state_dim == 3
    assert spec.control_dim == 1
    assert spec.disturbance_dim == 1
    assert spec.param_dim == 4
    assert isinstance(simulator, ProcessSimulator)
    assert simulator.spec.name == "bioreactor_compartment"
    assert [group.kind for group in spec.state_groups] == [
        "concentration",
        "biological",
        "concentration",
    ]
    assert spec.control_channels[0].role == "flow"
    assert spec.parameter_descriptors[0].name == "mu_max"


def test_isothermal_cstr_system_spec_and_simulator_are_registered():
    system_config = _load_yaml("configs/isothermal_cstr_default.yaml")

    spec = get_system_spec(system_config)
    simulator = get_simulator("isothermal_cstr", system_config)

    assert spec.name == "isothermal_cstr"
    assert spec.state_dim == 2
    assert spec.control_dim == 1
    assert spec.disturbance_dim == 1
    assert spec.param_dim == 4
    assert isinstance(simulator, ProcessSimulator)
    assert simulator.spec.name == "isothermal_cstr"
    assert [group.kind for group in spec.state_groups] == ["concentration"]
    assert spec.control_channels[0].role == "flow"
    assert spec.parameter_descriptors[-1].name == "T_ref"


def test_separator_system_spec_and_simulator_are_registered():
    system_config = _load_yaml("configs/separator_default.yaml")

    spec = get_system_spec(system_config)
    simulator = get_simulator("separator", system_config)

    assert spec.name == "separator"
    assert spec.state_dim == 3
    assert spec.control_dim == 1
    assert spec.disturbance_dim == 2
    assert spec.param_dim == 2
    assert isinstance(simulator, ProcessSimulator)
    assert simulator.spec.name == "separator"
    assert [group.kind for group in spec.state_groups] == ["concentration", "thermal"]
    assert spec.state_channels[0].upper_bound == 1.0
    assert spec.control_channels[0].role == "actuator_state"
