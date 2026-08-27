"""Tests for system-specific physics registry helpers."""

import jax
import jax.numpy as jnp
import pytest
import yaml

from dte.laws.examples import build_cstr_law_example_config
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


def test_get_physics_loss_for_cstr_can_be_augmented_with_law_layers():
    config = build_cstr_law_example_config()
    physics_loss = get_physics_loss("cstr", config)

    residual_names = physics_loss.residual_names()
    assert "mass" in residual_names
    assert "chemistry_primary_reaction_state_delta_consistency" in residual_names
    assert "thermo_liquid_cp_enthalpy_transform_consistency" in residual_names

    residuals = physics_loss.compute_residuals(
        states=jnp.asarray(
            [
                [[0.8, 0.2, 330.0, 300.0], [0.79, 0.21, 330.2, 300.1]],
                [[0.9, 0.1, 328.0, 299.0], [0.88, 0.12, 328.3, 299.2]],
            ],
            dtype=jnp.float32,
        ),
        controls=jnp.asarray(
            [
                [[50.0, 300.0], [50.0, 300.0]],
                [[45.0, 298.0], [45.0, 298.0]],
            ],
            dtype=jnp.float32,
        ),
        disturbances=jnp.asarray(
            [
                [[1.0, 320.0], [1.0, 320.0]],
                [[1.1, 321.0], [1.1, 321.0]],
            ],
            dtype=jnp.float32,
        ),
        dt=0.1,
    )

    assert "chemistry_primary_reaction_state_delta_consistency" in residuals
    assert "thermo_liquid_cp_enthalpy_transform_consistency" in residuals


def test_get_physics_loss_for_unknown_system_falls_back_to_null():
    physics_loss = get_physics_loss("unknown_system", {})

    assert physics_loss.residual_names() == []


def test_get_physics_loss_for_isothermal_cstr_exposes_mass_terms():
    config = _load_yaml("configs/isothermal_cstr_default.yaml")
    physics_loss = get_physics_loss("isothermal_cstr", config)

    assert physics_loss.residual_names() == ["mass", "species_mass"]


def test_get_physics_loss_for_bioreactor_compartment_exposes_state_terms():
    config = _load_yaml("configs/bioreactor_compartment_default.yaml")
    physics_loss = get_physics_loss("bioreactor_compartment", config)

    assert physics_loss.residual_names() == ["substrate", "biomass", "oxygen"]


def test_get_physics_loss_for_storage_tank_exposes_mass_composition_energy():
    config = _load_yaml("configs/storage_tank_default.yaml")
    physics_loss = get_physics_loss("storage_tank", config)

    assert physics_loss.residual_names() == ["mass", "composition", "energy"]


def test_get_physics_loss_for_separator_exposes_phase_split_and_energy():
    config = _load_yaml("configs/separator_default.yaml")
    physics_loss = get_physics_loss("separator", config)

    assert physics_loss.residual_names() == ["phase_split", "energy"]


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


def test_law_augmented_cstr_diagnostic_fn_includes_law_residual_series():
    config = build_cstr_law_example_config()
    diagnostic_fn = get_physics_diagnostic_fn("cstr", config)

    assert diagnostic_fn is not None
    residuals = diagnostic_fn(
        states=jnp.asarray([[0.8, 0.2, 330.0, 300.0], [0.79, 0.21, 330.1, 300.1]]),
        controls=jnp.asarray([[50.0, 300.0], [50.0, 300.0]]),
        disturbances=jnp.asarray([[1.0, 320.0], [1.0, 320.0]]),
        dt=0.1,
    )

    assert "mass" in residuals
    assert "chemistry_primary_reaction_state_delta_consistency" in residuals
    assert residuals["chemistry_primary_reaction_state_delta_consistency"].shape == (1,)


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


def test_get_physics_diagnostic_fn_for_storage_tank_exposes_mass_composition_energy():
    config = _load_yaml("configs/storage_tank_default.yaml")
    diagnostic_fn = get_physics_diagnostic_fn("storage_tank", config)

    assert diagnostic_fn is not None
    residuals = diagnostic_fn(
        states=jnp.array([[1.2, 0.5, 325.0], [1.22, 0.51, 324.8]]),
        controls=jnp.array([[0.8], [0.8]]),
        disturbances=jnp.array([[0.7, 0.55, 330.0], [0.7, 0.55, 330.0]]),
        dt=0.1,
    )

    assert set(residuals) == {"mass", "composition", "energy"}


def test_get_physics_diagnostic_fn_for_separator_exposes_phase_split_and_energy():
    config = _load_yaml("configs/separator_default.yaml")
    diagnostic_fn = get_physics_diagnostic_fn("separator", config)

    assert diagnostic_fn is not None
    residuals = diagnostic_fn(
        states=jnp.array([[0.55, 0.45, 330.0], [0.56, 0.44, 329.5]]),
        controls=jnp.array([[0.5], [0.5]]),
        disturbances=jnp.array([[0.5, 332.0], [0.5, 332.0]]),
        dt=0.1,
    )

    assert set(residuals) == {"phase_split", "energy"}


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


def test_get_physics_loss_skips_unknown_cstr_config_keys():
    config = _load_yaml("configs/cstr_default.yaml")
    config["cstr"]["notes"] = "not a numeric parameter"
    physics_loss = get_physics_loss("cstr", config)
    assert "mass" in physics_loss.residual_names()


def test_get_physics_loss_rejects_non_numeric_known_keys():
    config = _load_yaml("configs/cstr_default.yaml")
    config["cstr"]["V"] = "large"
    with pytest.raises(ValueError, match="must be numeric"):
        get_physics_loss("cstr", config)
