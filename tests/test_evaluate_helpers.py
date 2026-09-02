"""Tests for evaluation helper functions."""

from scripts.evaluate import _resolve_uncertainty_source


def test_resolve_uncertainty_source_sde_enabled():
    sde_enabled, source = _resolve_uncertainty_source({"sde_training": {"enabled": True}})
    assert sde_enabled is True
    assert source == "sde_rollout"


def test_resolve_uncertainty_source_sde_disabled():
    sde_enabled, source = _resolve_uncertainty_source({"sde_training": {"enabled": False}})
    assert sde_enabled is False
    assert source == "encoder_sampling"


def test_resolve_uncertainty_source_missing_sde_training():
    sde_enabled, source = _resolve_uncertainty_source({})
    assert sde_enabled is False
    assert source == "encoder_sampling"
