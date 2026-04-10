"""Tests for Phase 3 flowsheet graph schemas."""

import pytest

from dte.flowsheet.examples import (
    build_exchanger_reactor_tank_flowsheet,
    build_reactor_separator_recycle_flowsheet,
)
from dte.flowsheet.schema import FlowsheetSpec, StreamSpec
from dte.flowsheet.types import EXTERNAL_SOURCE


def test_flowsheet_examples_report_recycle_presence():
    exchanger_flowsheet = build_exchanger_reactor_tank_flowsheet()
    recycle_flowsheet = build_reactor_separator_recycle_flowsheet()

    assert exchanger_flowsheet.has_recycle_loops() is False
    assert recycle_flowsheet.has_recycle_loops() is True


def test_flowsheet_schema_rejects_unknown_stream_variables():
    base = build_exchanger_reactor_tank_flowsheet()

    with pytest.raises(ValueError, match="source variables"):
        FlowsheetSpec(
            name="invalid_flowsheet",
            units=base.units,
            streams=[
                StreamSpec(
                    name="bad_stream",
                    source_unit="exchanger",
                    target_unit="reactor",
                    variables=["not_a_state"],
                )
            ],
            global_controls=base.global_controls,
            global_disturbances=base.global_disturbances,
        )


def test_stream_spec_rejects_invalid_external_endpoint():
    with pytest.raises(ValueError, match="cannot terminate"):
        StreamSpec(
            name="bad_external",
            source_unit="reactor",
            target_unit=EXTERNAL_SOURCE,
            variables=["Ca"],
        )
