"""Template matching utilities for customer onboarding payloads."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from dte.customer.onboarding_schema import CustomerOnboardingSpec, CustomerUnitSpec
from dte.flowsheet.examples import (
    build_exchanger_reactor_tank_flowsheet,
    build_reactor_separator_recycle_flowsheet,
)
from dte.flowsheet.schema import FlowsheetSpec
from dte.simulators.base import ProcessUnitSpec
from dte.simulators.registry import get_system_spec

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REGISTERED_UNIT_CONFIGS = {
    "bioreactor_compartment": PROJECT_ROOT / "configs" / "bioreactor_compartment_default.yaml",
    "cstr": PROJECT_ROOT / "configs" / "cstr_default.yaml",
    "heat_exchanger": PROJECT_ROOT / "configs" / "heat_exchanger_default.yaml",
    "isothermal_cstr": PROJECT_ROOT / "configs" / "isothermal_cstr_default.yaml",
    "separator": PROJECT_ROOT / "configs" / "separator_default.yaml",
    "storage_tank": PROJECT_ROOT / "configs" / "storage_tank_default.yaml",
    "two_tank": PROJECT_ROOT / "configs" / "two_tank_default.yaml",
}


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return float(len(left & right) / len(left | right))


def _count_similarity(left: int, right: int) -> float:
    denom = max(left, right, 1)
    return 1.0 - abs(left - right) / denom


def _counter_overlap(left: Counter, right: Counter) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    keys = set(left) | set(right)
    intersection = sum(min(left[key], right[key]) for key in keys)
    union = sum(max(left[key], right[key]) for key in keys)
    return float(intersection / max(union, 1))


def _rounded(value: float) -> float:
    return round(float(value), 6)


def _load_registered_unit_templates() -> dict[str, ProcessUnitSpec]:
    templates = {}
    for name, path in REGISTERED_UNIT_CONFIGS.items():
        with path.open("r", encoding="utf-8") as handle:
            templates[name] = get_system_spec(yaml.safe_load(handle))
    return templates


def _load_flowsheet_templates() -> dict[str, FlowsheetSpec]:
    return {
        "exchanger_reactor_tank": build_exchanger_reactor_tank_flowsheet(),
        "reactor_separator_recycle": build_reactor_separator_recycle_flowsheet(),
    }


@dataclass(frozen=True)
class TemplateMatch:
    """Single ranked template match."""

    name: str
    kind: str
    score: float
    reasons: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "score": _rounded(self.score),
            "reasons": list(self.reasons),
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class TemplateMatchResult:
    """Ranked candidate templates for each customer unit plus whole-asset flowsheets."""

    unit_matches: dict[str, tuple[TemplateMatch, ...]]
    flowsheet_matches: tuple[TemplateMatch, ...]

    def best_unit_match(self, unit_name: str) -> TemplateMatch | None:
        matches = self.unit_matches.get(unit_name, ())
        return matches[0] if matches else None

    def best_flowsheet_match(self) -> TemplateMatch | None:
        return self.flowsheet_matches[0] if self.flowsheet_matches else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "unit_matches": {
                unit_name: [match.to_dict() for match in matches]
                for unit_name, matches in self.unit_matches.items()
            },
            "flowsheet_matches": [match.to_dict() for match in self.flowsheet_matches],
        }


def _score_components(components: list[tuple[str, float, float, bool]]) -> tuple[float, list[str]]:
    active = [(name, value, weight) for name, value, weight, enabled in components if enabled]
    if not active:
        return 0.0, []
    total_weight = sum(weight for _, _, weight in active)
    score = sum(value * weight for _, value, weight in active) / max(total_weight, 1e-6)
    reasons = [
        f"{name}={value:.2f}"
        for name, value, _ in active
        if value > 0.0
    ]
    return float(score), reasons


def _score_unit_template(
    onboarding: CustomerOnboardingSpec,
    customer_unit: CustomerUnitSpec,
    template_name: str,
    spec: ProcessUnitSpec,
) -> TemplateMatch:
    customer_controls = onboarding.controls_for_unit(customer_unit.name)
    customer_disturbances = onboarding.disturbances_for_unit(customer_unit.name)
    customer_measurements = onboarding.measurements_for_unit(customer_unit.name)

    customer_control_roles = {item.role for item in customer_controls}
    customer_control_names = {item.name for item in customer_controls}
    customer_disturbance_roles = {item.role for item in customer_disturbances}
    customer_disturbance_names = {item.name for item in customer_disturbances}
    customer_measurement_roles = {item.role for item in customer_measurements}
    customer_measurement_names = {item.name for item in customer_measurements}
    customer_laws = set(onboarding.known_laws) | set(customer_unit.known_laws)

    template_control_roles = {channel.role for channel in spec.control_channels}
    template_disturbance_roles = {channel.role for channel in spec.disturbance_channels}
    template_measurement_roles = {channel.role for channel in spec.state_channels}

    components = [
        (
            "family",
            1.0 if customer_unit.family and customer_unit.family == spec.family else 0.0,
            2.0,
            bool(customer_unit.family),
        ),
        (
            "subtype",
            1.0
            if customer_unit.subtype
            and customer_unit.subtype in {spec.subtype, spec.unit_type}
            else 0.0,
            1.0,
            bool(customer_unit.subtype),
        ),
        (
            "laws",
            _jaccard(customer_laws, set(spec.law_tags)),
            2.0,
            bool(customer_laws),
        ),
        (
            "control_roles",
            _jaccard(customer_control_roles, template_control_roles),
            1.5,
            bool(customer_control_roles),
        ),
        (
            "control_names",
            _jaccard(customer_control_names, set(spec.control_names)),
            1.0,
            bool(customer_control_names),
        ),
        (
            "disturbance_roles",
            _jaccard(customer_disturbance_roles, template_disturbance_roles),
            1.0,
            bool(customer_disturbance_roles),
        ),
        (
            "disturbance_names",
            _jaccard(customer_disturbance_names, set(spec.disturbance_names)),
            0.75,
            bool(customer_disturbance_names),
        ),
        (
            "measurement_roles",
            _jaccard(customer_measurement_roles, template_measurement_roles),
            1.5,
            bool(customer_measurement_roles),
        ),
        (
            "measurement_names",
            _jaccard(customer_measurement_names, set(spec.state_names)),
            1.0,
            bool(customer_measurement_names),
        ),
        (
            "shape",
            (
                _count_similarity(len(customer_controls), spec.control_dim)
                + _count_similarity(len(customer_disturbances), spec.disturbance_dim)
                + _count_similarity(len(customer_measurements), spec.state_dim)
            )
            / 3.0,
            0.75,
            True,
        ),
    ]
    score, reasons = _score_components(components)
    metadata = {
        "family": spec.family,
        "subtype": spec.subtype or spec.unit_type,
        "unit_type": spec.unit_type,
        "law_tags": list(spec.law_tags),
        "state_names": list(spec.state_names),
        "control_names": list(spec.control_names),
        "disturbance_names": list(spec.disturbance_names),
    }
    return TemplateMatch(
        name=template_name,
        kind="unit_template",
        score=score,
        reasons=tuple(reasons[:6]),
        metadata=metadata,
    )


def _score_flowsheet_template(
    onboarding: CustomerOnboardingSpec,
    template_name: str,
    spec: FlowsheetSpec,
) -> TemplateMatch:
    customer_family_counter = Counter(
        unit.family or unit.unit_type or "generic"
        for unit in onboarding.units
    )
    template_family_counter = Counter(
        unit.family or unit.unit_type or "generic"
        for unit in spec.units.values()
    )
    customer_laws = set(onboarding.all_known_laws)
    template_laws = {
        law
        for unit in spec.units.values()
        for law in getattr(unit, "law_tags", [])
    }
    customer_stream_kinds = {stream.kind for stream in onboarding.streams}
    template_stream_kinds = {stream.kind for stream in spec.streams}
    customer_has_recycle = any(stream.kind == "recycle" for stream in onboarding.streams)
    template_has_recycle = spec.has_recycle_loops()

    components = [
        (
            "unit_families",
            _counter_overlap(customer_family_counter, template_family_counter),
            2.5,
            bool(customer_family_counter),
        ),
        (
            "laws",
            _jaccard(customer_laws, template_laws),
            1.5,
            bool(customer_laws),
        ),
        (
            "stream_kinds",
            _jaccard(customer_stream_kinds, template_stream_kinds),
            1.25,
            bool(customer_stream_kinds),
        ),
        (
            "unit_count",
            _count_similarity(len(onboarding.units), len(spec.units)),
            1.0,
            True,
        ),
        (
            "stream_count",
            _count_similarity(len(onboarding.streams), len(spec.streams)),
            0.75,
            bool(onboarding.streams),
        ),
        (
            "recycle",
            1.0 if customer_has_recycle == template_has_recycle else 0.0,
            0.75,
            bool(onboarding.streams),
        ),
    ]
    score, reasons = _score_components(components)
    metadata = {
        "unit_names": list(spec.unit_names),
        "unit_families": {
            name: unit.family or unit.unit_type or "generic"
            for name, unit in spec.units.items()
        },
        "stream_names": list(spec.stream_names),
        "has_recycle_loop": template_has_recycle,
    }
    return TemplateMatch(
        name=template_name,
        kind="flowsheet_template",
        score=score,
        reasons=tuple(reasons[:6]),
        metadata=metadata,
    )


def match_customer_templates(
    onboarding: CustomerOnboardingSpec,
    *,
    top_k: int = 3,
) -> TemplateMatchResult:
    """Rank unit-family and flowsheet templates for a customer asset."""

    unit_templates = _load_registered_unit_templates()
    flowsheet_templates = _load_flowsheet_templates()

    unit_matches = {}
    for customer_unit in onboarding.units:
        matches = [
            _score_unit_template(onboarding, customer_unit, template_name, spec)
            for template_name, spec in unit_templates.items()
        ]
        matches.sort(key=lambda match: match.score, reverse=True)
        unit_matches[customer_unit.name] = tuple(matches[:top_k])

    flowsheet_matches: tuple[TemplateMatch, ...] = ()
    if onboarding.asset_kind == "flowsheet" or onboarding.streams:
        ranked = [
            _score_flowsheet_template(onboarding, template_name, spec)
            for template_name, spec in flowsheet_templates.items()
        ]
        ranked.sort(key=lambda match: match.score, reverse=True)
        flowsheet_matches = tuple(ranked[:top_k])

    return TemplateMatchResult(
        unit_matches=unit_matches,
        flowsheet_matches=flowsheet_matches,
    )
