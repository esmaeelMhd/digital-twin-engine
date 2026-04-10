"""Example law-layer configurations for Phase 4."""

from __future__ import annotations

from pathlib import Path

import yaml

from dte.core.state_schema import ParameterDescriptor, SignalChannel, StateChannel
from dte.laws.integration import build_law_bundle
from dte.simulators.base import (
    DecoderConstraint,
    NormalizationSpec,
    ProcessUnitSpec,
    StateGroupSpec,
)
from dte.simulators.registry import get_system_spec


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def build_cstr_law_example_config() -> dict:
    """Return a merged CSTR config with chemistry and thermo laws enabled."""
    base = _load_yaml(PROJECT_ROOT / "configs" / "cstr_default.yaml")
    laws = _load_yaml(PROJECT_ROOT / "configs" / "cstr_law_example.yaml")
    merged = dict(base)
    merged["laws"] = laws["laws"]
    return merged


def build_bioreactor_process_unit_spec() -> ProcessUnitSpec:
    """Return a simple law-driven bioreactor spec for biology examples."""
    return ProcessUnitSpec(
        name="bioreactor_example",
        unit_type="aerobic_bioreactor",
        family="bioprocess",
        subtype="monod_growth",
        law_tags=["biology", "oxygen_transfer", "mass_balance"],
        state_dim=3,
        control_dim=1,
        disturbance_dim=1,
        param_dim=2,
        state_names=["substrate", "biomass", "dissolved_oxygen"],
        control_names=["aeration"],
        disturbance_names=["feed_substrate"],
        decoder_constraints=[
            DecoderConstraint(type="softplus", indices=[0, 1, 2], bias=0.2),
        ],
        normalization=NormalizationSpec(
            state_center=[1.0, 0.4, 0.5],
            state_scale=[1.0, 2.0, 2.0],
            control_center=[0.5],
            control_scale=[2.0],
            disturbance_center=[1.0],
            disturbance_scale=[1.0],
            param_scale=0.1,
        ),
        default_initial_state=[1.2, 0.4, 0.6],
        default_nominal_disturbance=[1.0],
        control_ranges={"aeration": [0.1, 1.0]},
        disturbance_ranges={"feed_substrate": [0.0, 2.0]},
        state_groups=[
            StateGroupSpec(name="substrate_group", kind="concentration", indices=[0]),
            StateGroupSpec(name="biomass_group", kind="biological", indices=[1]),
            StateGroupSpec(name="oxygen_group", kind="concentration", indices=[2]),
        ],
        state_channels=[
            StateChannel(name="substrate", role="concentration", lower_bound=0.0),
            StateChannel(name="biomass", role="biological_state", lower_bound=0.0),
            StateChannel(name="dissolved_oxygen", role="concentration", lower_bound=0.0),
        ],
        control_channels=[
            SignalChannel(name="aeration", role="flow", lower_bound=0.1, upper_bound=1.0),
        ],
        disturbance_channels=[
            SignalChannel(name="feed_substrate", role="concentration", lower_bound=0.0, upper_bound=2.0),
        ],
        parameter_descriptors=[
            ParameterDescriptor(name="mu_max", law_tag="biology"),
            ParameterDescriptor(name="kla", law_tag="oxygen_transfer"),
        ],
    )


def build_bioreactor_law_example_config() -> dict:
    """Return a biology-law example config."""
    return _load_yaml(PROJECT_ROOT / "configs" / "bioreactor_law_example.yaml")


def build_cstr_law_bundle_example():
    config = build_cstr_law_example_config()
    spec = get_system_spec(config)
    bundle = build_law_bundle(spec, config)
    if bundle is None:
        raise RuntimeError("expected CSTR law bundle example to be enabled.")
    return spec, config, bundle


def build_bioreactor_law_bundle_example():
    spec = build_bioreactor_process_unit_spec()
    config = build_bioreactor_law_example_config()
    bundle = build_law_bundle(spec, config)
    if bundle is None:
        raise RuntimeError("expected bioreactor law bundle example to be enabled.")
    return spec, config, bundle
