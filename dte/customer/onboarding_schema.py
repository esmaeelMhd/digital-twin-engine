"""Customer onboarding schema for adaptation workflows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator

from dte.flowsheet.types import EXTERNAL_SINK, EXTERNAL_SOURCE


class OperatingRange(BaseModel):
    """Simple bounded operating window for one variable."""

    low: float
    high: float
    nominal: float | None = None
    unit: str | None = None

    @model_validator(mode="after")
    def _validate_bounds(self) -> "OperatingRange":
        if self.high < self.low:
            raise ValueError("operating range high must be >= low.")
        if self.nominal is not None and not (self.low <= self.nominal <= self.high):
            raise ValueError("operating range nominal must lie within [low, high].")
        return self


class CustomerSignalSpec(BaseModel):
    """Named control or disturbance channel provided during onboarding."""

    name: str
    role: str = "generic"
    unit_name: str | None = None
    unit: str | None = None
    lower_bound: float | None = None
    upper_bound: float | None = None
    description: str | None = None

    @model_validator(mode="after")
    def _validate_bounds(self) -> "CustomerSignalSpec":
        if (
            self.lower_bound is not None
            and self.upper_bound is not None
            and self.upper_bound < self.lower_bound
        ):
            raise ValueError("signal upper_bound must be >= lower_bound.")
        return self


class CustomerMeasurementSpec(BaseModel):
    """Named measured variable or KPI exposed by the customer."""

    name: str
    role: str = "generic"
    unit_name: str | None = None
    unit: str | None = None
    source: Literal["state", "stream", "lab", "virtual"] = "state"
    description: str | None = None
    lower_bound: float | None = None
    upper_bound: float | None = None

    @model_validator(mode="after")
    def _validate_bounds(self) -> "CustomerMeasurementSpec":
        if (
            self.lower_bound is not None
            and self.upper_bound is not None
            and self.upper_bound < self.lower_bound
        ):
            raise ValueError("measurement upper_bound must be >= lower_bound.")
        return self


class CustomerStreamSpec(BaseModel):
    """Directed stream between customer units or external boundaries."""

    name: str
    source_unit: str
    target_unit: str
    variables: list[str]
    kind: str = "process"
    delay: float | None = None
    description: str | None = None

    @model_validator(mode="after")
    def _validate_stream(self) -> "CustomerStreamSpec":
        if not self.variables:
            raise ValueError("customer stream must define at least one variable.")
        if self.delay is not None and self.delay < 0.0:
            raise ValueError("customer stream delay must be non-negative.")
        if self.source_unit == EXTERNAL_SINK:
            raise ValueError(f"stream '{self.name}' cannot originate from {EXTERNAL_SINK}.")
        if self.target_unit == EXTERNAL_SOURCE:
            raise ValueError(f"stream '{self.name}' cannot terminate at {EXTERNAL_SOURCE}.")
        return self


class CustomerUnitSpec(BaseModel):
    """Customer-facing description of one unit or asset node."""

    name: str
    family: str | None = None
    subtype: str | None = None
    unit_type: str | None = None
    controls: list[str] = Field(default_factory=list)
    disturbances: list[str] = Field(default_factory=list)
    measurements: list[str] = Field(default_factory=list)
    known_laws: list[str] = Field(default_factory=list)
    operating_ranges: dict[str, OperatingRange] = Field(default_factory=dict)
    notes: str | None = None


class CustomerOnboardingSpec(BaseModel):
    """Top-level customer onboarding payload used for adaptation and matching."""

    name: str
    asset_kind: Literal["unit", "flowsheet"] = "unit"
    units: list[CustomerUnitSpec]
    streams: list[CustomerStreamSpec] = Field(default_factory=list)
    controls: list[CustomerSignalSpec] = Field(default_factory=list)
    disturbances: list[CustomerSignalSpec] = Field(default_factory=list)
    measurements: list[CustomerMeasurementSpec] = Field(default_factory=list)
    known_laws: list[str] = Field(default_factory=list)
    operating_ranges: dict[str, OperatingRange] = Field(default_factory=dict)
    notes: str | None = None

    @model_validator(mode="after")
    def _validate_schema(self) -> "CustomerOnboardingSpec":
        if not self.units:
            raise ValueError("customer onboarding must define at least one unit.")

        unit_names = [unit.name for unit in self.units]
        if len(unit_names) != len(set(unit_names)):
            raise ValueError("customer unit names must be unique.")

        control_names = [item.name for item in self.controls]
        disturbance_names = [item.name for item in self.disturbances]
        measurement_names = [item.name for item in self.measurements]
        stream_names = [item.name for item in self.streams]
        for label, names in (
            ("control", control_names),
            ("disturbance", disturbance_names),
            ("measurement", measurement_names),
            ("stream", stream_names),
        ):
            if len(names) != len(set(names)):
                raise ValueError(f"customer {label} names must be unique.")

        known_controls = set(control_names)
        known_disturbances = set(disturbance_names)
        known_measurements = set(measurement_names)
        known_units = set(unit_names)

        for unit in self.units:
            missing_controls = sorted(set(unit.controls) - known_controls)
            missing_disturbances = sorted(set(unit.disturbances) - known_disturbances)
            missing_measurements = sorted(set(unit.measurements) - known_measurements)
            if missing_controls:
                raise ValueError(
                    f"unit '{unit.name}' references unknown controls: {missing_controls}"
                )
            if missing_disturbances:
                raise ValueError(
                    f"unit '{unit.name}' references unknown disturbances: {missing_disturbances}"
                )
            if missing_measurements:
                raise ValueError(
                    f"unit '{unit.name}' references unknown measurements: {missing_measurements}"
                )

        for item in [*self.controls, *self.disturbances, *self.measurements]:
            if item.unit_name is not None and item.unit_name not in known_units:
                raise ValueError(
                    f"signal '{item.name}' references unknown unit '{item.unit_name}'."
                )

        for stream in self.streams:
            if stream.source_unit not in known_units and stream.source_unit != EXTERNAL_SOURCE:
                raise ValueError(
                    f"stream '{stream.name}' source '{stream.source_unit}' is not a known unit."
                )
            if stream.target_unit not in known_units and stream.target_unit != EXTERNAL_SINK:
                raise ValueError(
                    f"stream '{stream.name}' target '{stream.target_unit}' is not a known unit."
                )

        if self.asset_kind == "unit" and len(self.units) > 1 and not self.streams:
            raise ValueError(
                "unit onboarding without streams should describe exactly one unit."
            )
        return self

    @property
    def unit_names(self) -> tuple[str, ...]:
        return tuple(unit.name for unit in self.units)

    @property
    def control_names(self) -> tuple[str, ...]:
        return tuple(item.name for item in self.controls)

    @property
    def disturbance_names(self) -> tuple[str, ...]:
        return tuple(item.name for item in self.disturbances)

    @property
    def measurement_names(self) -> tuple[str, ...]:
        return tuple(item.name for item in self.measurements)

    @property
    def all_known_laws(self) -> tuple[str, ...]:
        ordered: list[str] = []
        for law in self.known_laws:
            if law not in ordered:
                ordered.append(law)
        for unit in self.units:
            for law in unit.known_laws:
                if law not in ordered:
                    ordered.append(law)
        return tuple(ordered)

    def controls_for_unit(self, unit_name: str) -> list[CustomerSignalSpec]:
        explicit = {name for unit in self.units if unit.name == unit_name for name in unit.controls}
        return [
            item
            for item in self.controls
            if item.unit_name in (None, unit_name) and (not explicit or item.name in explicit)
        ]

    def disturbances_for_unit(self, unit_name: str) -> list[CustomerSignalSpec]:
        explicit = {
            name
            for unit in self.units
            if unit.name == unit_name
            for name in unit.disturbances
        }
        return [
            item
            for item in self.disturbances
            if item.unit_name in (None, unit_name) and (not explicit or item.name in explicit)
        ]

    def measurements_for_unit(self, unit_name: str) -> list[CustomerMeasurementSpec]:
        explicit = {
            name
            for unit in self.units
            if unit.name == unit_name
            for name in unit.measurements
        }
        return [
            item
            for item in self.measurements
            if item.unit_name in (None, unit_name) and (not explicit or item.name in explicit)
        ]


def load_onboarding_spec(path: str | Path) -> CustomerOnboardingSpec:
    """Load a customer onboarding payload from YAML or JSON."""

    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        if path.suffix.lower() == ".json":
            payload = json.load(handle)
        else:
            payload = yaml.safe_load(handle)
    return CustomerOnboardingSpec.model_validate(payload)
