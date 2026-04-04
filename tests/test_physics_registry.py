"""Tests for system-specific physics registry helpers."""

import jax
import jax.numpy as jnp
import yaml

from dte.physics.base import NullPhysicsLoss
from dte.physics.registry import get_physics_diagnostic_fn, get_physics_loss
from dte.simulators.heat_exchanger import HeatExchangerParams, HeatExchangerSimulator
from dte.simulators.two_tank import steady_state_two_tank_jit


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


def test_get_physics_loss_for_two_tank_exposes_mass_only():
    config = _load_yaml("configs/two_tank_default.yaml")
    physics_loss = get_physics_loss("two_tank", config)

    assert physics_loss.residual_names() == ["mass"]


def test_get_physics_diagnostic_fn_for_two_tank_exposes_mass_only():
    config = _load_yaml("configs/two_tank_default.yaml")
    diagnostic_fn = get_physics_diagnostic_fn("two_tank", config)

    assert diagnostic_fn is not None
    residuals = diagnostic_fn(
        states=jnp.array([[1.8, 1.1], [1.81, 1.11]]),
        controls=jnp.array([[1.0, 0.8], [1.0, 0.8]]),
        disturbances=jnp.array([[0.1, 0.05], [0.1, 0.05]]),
        dt=0.1,
    )

    assert "mass" in residuals
    assert "energy" not in residuals


def test_heat_exchanger_diagnostic_uses_per_trajectory_params():
    config = _load_yaml("configs/heat_exchanger_default.yaml")
    diagnostic_fn = get_physics_diagnostic_fn("heat_exchanger", config)

    assert diagnostic_fn is not None

    simulator = HeatExchangerSimulator(HeatExchangerParams(**config["heat_exchanger"]))
    sampled_params = simulator.sample_data_generation_params(jax.random.PRNGKey(0))

    control = jnp.array([5.0, 5.0])
    disturbance = jnp.array([390.0, 290.0])
    initial_state = simulator.steady_state_for_data_generation_with_params(
        control,
        disturbance,
        sampled_params,
    )
    n_steps = 128
    controls = jnp.tile(control[None, :], (n_steps, 1))
    disturbances = jnp.tile(disturbance[None, :], (n_steps, 1))
    result = simulator.simulate_for_data_generation_with_params(
        initial_state,
        controls,
        disturbances,
        sampled_params,
        (0.0, 12.8),
    )
    dt = float(result["time"][1] - result["time"][0])

    nominal_residuals = diagnostic_fn(
        states=result["states"],
        controls=controls,
        disturbances=disturbances,
        dt=dt,
    )
    param_residuals = diagnostic_fn(
        states=result["states"],
        controls=controls,
        disturbances=disturbances,
        dt=dt,
        params=sampled_params,
    )

    assert float(jnp.mean(param_residuals["energy"])) < 1e-3
    assert float(jnp.mean(param_residuals["energy"])) < float(
        jnp.mean(nominal_residuals["energy"])
    )


def test_two_tank_infeasible_operating_point_returns_nonfinite_steady_state():
    state = steady_state_two_tank_jit(
        control=jnp.array([2.5, 0.4]),
        disturbance=jnp.array([0.4, 0.3]),
        params=jnp.array([1.5, 1.2, 0.9, 1.0, 5.0]),
    )

    assert not bool(jnp.all(jnp.isfinite(state)))
