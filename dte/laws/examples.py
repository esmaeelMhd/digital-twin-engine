"""Example law-layer configurations for Phase 4."""

from __future__ import annotations

from pathlib import Path

import yaml

from dte.laws.integration import build_law_bundle
from dte.simulators.base import ProcessUnitSpec
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
    """Return the registered bioreactor spec used by biology examples."""
    return get_system_spec(
        _load_yaml(PROJECT_ROOT / "configs" / "bioreactor_compartment_default.yaml")
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
