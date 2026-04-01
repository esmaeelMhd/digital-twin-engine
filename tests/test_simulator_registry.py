"""Tests for simulator registry helpers."""

import yaml

from dte.simulators.base import ProcessSimulator
from dte.simulators.registry import get_simulator, get_system_spec, list_systems


def _load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def test_get_system_spec_uses_registered_name():
    system_config = _load_yaml("configs/heat_exchanger_default.yaml")

    spec = get_system_spec(system_config)

    assert spec.name == "heat_exchanger"
    assert spec.state_dim == 2


def test_get_simulator_returns_registered_process_simulator():
    system_config = _load_yaml("configs/cstr_default.yaml")

    simulator = get_simulator("cstr", system_config)

    assert isinstance(simulator, ProcessSimulator)
    assert simulator.spec.name == "cstr"


def test_list_systems_exposes_registered_systems():
    systems = list_systems()

    assert "cstr" in systems
    assert "heat_exchanger" in systems
