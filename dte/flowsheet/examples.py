"""Example flowsheet specifications for Phase 3 demos and tests."""

from __future__ import annotations

from pathlib import Path

import yaml

from dte.core.state_schema import ParameterDescriptor, SignalChannel, StateChannel
from dte.flowsheet.schema import FlowsheetSpec, StreamSpec
from dte.flowsheet.types import EXTERNAL_SINK, EXTERNAL_SOURCE
from dte.simulators.base import (
    DecoderConstraint,
    NormalizationSpec,
    ProcessUnitSpec,
    StateGroupSpec,
)
from dte.simulators.registry import get_system_spec


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_registered_spec(config_name: str) -> ProcessUnitSpec:
    with (PROJECT_ROOT / "configs" / config_name).open("r", encoding="utf-8") as handle:
        return get_system_spec(yaml.safe_load(handle))


def build_storage_tank_spec() -> ProcessUnitSpec:
    return ProcessUnitSpec(
        name="storage_tank",
        unit_type="tank_like",
        family="hydraulic",
        subtype="surge_tank",
        law_tags=["mass_balance", "energy_balance"],
        state_dim=3,
        control_dim=1,
        disturbance_dim=3,
        param_dim=2,
        state_names=["inventory", "quality", "temperature"],
        control_names=["outlet_flow"],
        disturbance_names=["feed_rate", "feed_quality", "feed_temperature"],
        decoder_constraints=[
            DecoderConstraint(type="softplus", indices=[0, 1], bias=0.2),
            DecoderConstraint(type="sigmoid_range", indices=[2], low=260.0, high=420.0),
        ],
        normalization=NormalizationSpec(
            state_center=[1.0, 0.5, 330.0],
            state_scale=[1.0, 1.0, 0.02],
            control_center=[0.5],
            control_scale=[2.0],
            disturbance_center=[0.5, 0.5, 330.0],
            disturbance_scale=[2.0, 1.0, 0.02],
            param_scale=0.1,
        ),
        default_initial_state=[1.0, 0.5, 330.0],
        default_nominal_disturbance=[0.5, 0.5, 330.0],
        control_ranges={"outlet_flow": [0.1, 1.0]},
        disturbance_ranges={
            "feed_rate": [0.0, 1.0],
            "feed_quality": [0.0, 1.0],
            "feed_temperature": [290.0, 360.0],
        },
        state_groups=[
            StateGroupSpec(name="inventory_group", kind="inventory", indices=[0]),
            StateGroupSpec(name="quality_group", kind="concentration", indices=[1]),
            StateGroupSpec(name="temperature_group", kind="thermal", indices=[2]),
        ],
        state_channels=[
            StateChannel(name="inventory", role="inventory", lower_bound=0.0),
            StateChannel(name="quality", role="concentration", lower_bound=0.0),
            StateChannel(name="temperature", role="temperature", lower_bound=260.0, upper_bound=420.0),
        ],
        control_channels=[
            SignalChannel(name="outlet_flow", role="flow", lower_bound=0.1, upper_bound=1.0)
        ],
        disturbance_channels=[
            SignalChannel(name="feed_rate", role="flow", lower_bound=0.0, upper_bound=1.0),
            SignalChannel(name="feed_quality", role="concentration", lower_bound=0.0, upper_bound=1.0),
            SignalChannel(name="feed_temperature", role="temperature", lower_bound=290.0, upper_bound=360.0),
        ],
        parameter_descriptors=[
            ParameterDescriptor(name="volume", law_tag="geometry"),
            ParameterDescriptor(name="heat_loss", law_tag="energy_balance"),
        ],
    )


def build_separator_like_spec() -> ProcessUnitSpec:
    return ProcessUnitSpec(
        name="separator_like",
        unit_type="separator_like",
        family="separator",
        subtype="flash_like",
        law_tags=["mass_balance", "phase_split"],
        state_dim=3,
        control_dim=1,
        disturbance_dim=2,
        param_dim=2,
        state_names=["light_cut", "heavy_cut", "tray_temperature"],
        control_names=["split_fraction"],
        disturbance_names=["feed_quality", "feed_temperature"],
        decoder_constraints=[
            DecoderConstraint(type="softplus", indices=[0, 1], bias=0.2),
            DecoderConstraint(type="sigmoid_range", indices=[2], low=260.0, high=420.0),
        ],
        normalization=NormalizationSpec(
            state_center=[0.5, 0.5, 330.0],
            state_scale=[1.0, 1.0, 0.02],
            control_center=[0.5],
            control_scale=[2.0],
            disturbance_center=[0.5, 330.0],
            disturbance_scale=[1.0, 0.02],
            param_scale=0.1,
        ),
        default_initial_state=[0.5, 0.5, 330.0],
        default_nominal_disturbance=[0.5, 330.0],
        control_ranges={"split_fraction": [0.1, 0.9]},
        disturbance_ranges={
            "feed_quality": [0.0, 1.0],
            "feed_temperature": [290.0, 360.0],
        },
        state_groups=[
            StateGroupSpec(name="cut_group", kind="concentration", indices=[0, 1]),
            StateGroupSpec(name="temperature_group", kind="thermal", indices=[2]),
        ],
        state_channels=[
            StateChannel(name="light_cut", role="concentration", lower_bound=0.0),
            StateChannel(name="heavy_cut", role="concentration", lower_bound=0.0),
            StateChannel(name="tray_temperature", role="temperature", lower_bound=260.0, upper_bound=420.0),
        ],
        control_channels=[
            SignalChannel(name="split_fraction", role="actuator_state", lower_bound=0.1, upper_bound=0.9)
        ],
        disturbance_channels=[
            SignalChannel(name="feed_quality", role="concentration", lower_bound=0.0, upper_bound=1.0),
            SignalChannel(name="feed_temperature", role="temperature", lower_bound=290.0, upper_bound=360.0),
        ],
        parameter_descriptors=[
            ParameterDescriptor(name="holdup", law_tag="geometry"),
            ParameterDescriptor(name="separation_gain", law_tag="phase_split"),
        ],
    )


def build_exchanger_reactor_tank_flowsheet() -> FlowsheetSpec:
    exchanger = _load_registered_spec("heat_exchanger_default.yaml")
    reactor = _load_registered_spec("cstr_default.yaml")
    tank = build_storage_tank_spec()
    return FlowsheetSpec(
        name="exchanger_reactor_tank",
        units={
            "exchanger": exchanger,
            "reactor": reactor,
            "tank": tank,
        },
        streams=[
            StreamSpec(
                name="utility_hot",
                source_unit=EXTERNAL_SOURCE,
                target_unit="exchanger",
                variables=["T_hot"],
                target_variables=["T_hot"],
                kind="utility",
            ),
            StreamSpec(
                name="fresh_feed",
                source_unit=EXTERNAL_SOURCE,
                target_unit="reactor",
                variables=["Ca"],
                target_variables=["Ca"],
                kind="source",
            ),
            StreamSpec(
                name="heated_feed",
                source_unit="exchanger",
                target_unit="reactor",
                variables=["T_hot"],
                target_variables=["T"],
                kind="process",
            ),
            StreamSpec(
                name="reactor_product",
                source_unit="reactor",
                target_unit="tank",
                variables=["Cb", "T"],
                target_variables=["quality", "temperature"],
                kind="process",
            ),
            StreamSpec(
                name="tank_purge",
                source_unit="tank",
                target_unit=EXTERNAL_SINK,
                variables=["inventory"],
                kind="sink",
            ),
        ],
        global_controls=["section_feed_rate"],
        global_disturbances=["ambient_temperature"],
        description="Heat exchanger feeding a reactor, followed by a storage tank.",
    )


def build_reactor_separator_recycle_flowsheet() -> FlowsheetSpec:
    reactor = _load_registered_spec("cstr_default.yaml")
    separator = build_separator_like_spec()
    return FlowsheetSpec(
        name="reactor_separator_recycle",
        units={
            "reactor": reactor,
            "separator": separator,
        },
        streams=[
            StreamSpec(
                name="fresh_feed",
                source_unit=EXTERNAL_SOURCE,
                target_unit="reactor",
                variables=["Ca"],
                target_variables=["Ca"],
                kind="source",
            ),
            StreamSpec(
                name="reactor_effluent",
                source_unit="reactor",
                target_unit="separator",
                variables=["Cb", "T"],
                target_variables=["light_cut", "tray_temperature"],
                kind="process",
            ),
            StreamSpec(
                name="separator_recycle",
                source_unit="separator",
                target_unit="reactor",
                variables=["light_cut", "tray_temperature"],
                target_variables=["Ca", "T"],
                kind="recycle",
                delay=0.1,
            ),
            StreamSpec(
                name="separator_purge",
                source_unit="separator",
                target_unit=EXTERNAL_SINK,
                variables=["heavy_cut"],
                kind="sink",
            ),
        ],
        global_controls=["fresh_feed_rate"],
        global_disturbances=["cooling_utility_temperature"],
        description="Reactor followed by a separator with an explicit recycle loop.",
    )
