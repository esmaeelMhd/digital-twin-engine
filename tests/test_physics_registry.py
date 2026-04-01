"""Tests for system-specific physics registry helpers."""

import jax.numpy as jnp
import yaml

from dte.physics.base import NullPhysicsLoss
from dte.physics.registry import get_physics_diagnostic_fn, get_physics_loss


def _load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def test_get_physics_loss_for_cstr():
    config = _load_yaml("configs/cstr_default.yaml")
    physics_loss = get_physics_loss("cstr", config)

    assert physics_loss.residual_names() == ["mass", "species_mass", "energy"]


def test_get_physics_loss_for_unknown_system_falls_back_to_null():
    physics_loss = get_physics_loss("unknown_system", {})

    assert isinstance(physics_loss, NullPhysicsLoss)
    assert physics_loss.residual_names() == []


def test_get_physics_diagnostic_fn_for_heat_exchanger_exposes_energy_only():
    config = _load_yaml("configs/heat_exchanger_default.yaml")
    diagnostic_fn = get_physics_diagnostic_fn("heat_exchanger", config)

    assert diagnostic_fn is not None
    residuals = diagnostic_fn(
        states=jnp.array([[350.0, 300.0], [349.0, 301.0]]),
        controls=jnp.array([[5.0, 5.0], [5.0, 5.0]]),
        disturbances=jnp.array([[380.0, 280.0], [380.0, 280.0]]),
        dt=0.1,
    )

    assert "energy" in residuals
    assert "mass" not in residuals
