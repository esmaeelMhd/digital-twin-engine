"""Example flowsheet specifications for Phase 3 demos and tests."""

from __future__ import annotations

from pathlib import Path

import yaml

from dte.flowsheet.schema import FlowsheetSpec, StreamSpec
from dte.flowsheet.types import EXTERNAL_SINK, EXTERNAL_SOURCE
from dte.simulators.base import ProcessUnitSpec
from dte.simulators.registry import get_system_spec


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_registered_spec(config_name: str) -> ProcessUnitSpec:
    with (PROJECT_ROOT / "configs" / config_name).open("r", encoding="utf-8") as handle:
        return get_system_spec(yaml.safe_load(handle))


def build_storage_tank_spec() -> ProcessUnitSpec:
    return _load_registered_spec("storage_tank_default.yaml")


def build_separator_spec() -> ProcessUnitSpec:
    return _load_registered_spec("separator_default.yaml")


def build_separator_like_spec() -> ProcessUnitSpec:
    return build_separator_spec()


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
    separator = build_separator_spec()
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
