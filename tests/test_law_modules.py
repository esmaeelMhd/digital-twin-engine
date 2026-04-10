"""Tests for Phase 4 modular law layers."""

import jax.numpy as jnp

from dte.laws import (
    arrhenius_rate_constant,
    build_bioreactor_law_bundle_example,
    build_cstr_law_bundle_example,
    enthalpy_like_transform,
    inhibition_factor,
    linear_heat_capacity,
    monod_growth_rate,
    power_law_rate,
)


def test_low_level_law_helpers_return_finite_values():
    k = arrhenius_rate_constant(2.0, 1200.0, jnp.asarray(320.0))
    rate = power_law_rate(jnp.asarray([0.8]), jnp.asarray([1.0]), k)
    cp = linear_heat_capacity(
        jnp.asarray(320.0),
        reference_temperature=298.15,
        reference_heat_capacity=0.239,
        slope=0.0005,
    )
    enthalpy = enthalpy_like_transform(
        jnp.asarray(320.0),
        reference_temperature=298.15,
        heat_capacity=cp,
        density=1000.0,
    )
    mu = monod_growth_rate(jnp.asarray(1.2), mu_max=0.45, half_saturation=0.2)
    inhib = inhibition_factor(jnp.asarray(1.2), inhibition_constant=8.0)

    assert float(k) > 0.0
    assert float(rate) > 0.0
    assert float(cp) > 0.0
    assert float(enthalpy) > 0.0
    assert 0.0 < float(mu) < 0.45
    assert 0.0 < float(inhib) <= 1.0


def test_cstr_law_bundle_exposes_features_deltas_and_residuals():
    spec, _, bundle = build_cstr_law_bundle_example()

    assert spec.name == "cstr"
    assert len(bundle.modules) == 2
    assert "chemistry_primary_reaction_reaction_rate" in bundle.feature_names()
    assert "thermo_liquid_cp_enthalpy_transform_consistency" in bundle.residual_names()

    state = jnp.asarray([0.8, 0.2, 330.0, 300.0], dtype=jnp.float32)
    control = jnp.asarray([50.0, 300.0], dtype=jnp.float32)
    disturbance = jnp.asarray([1.0, 320.0], dtype=jnp.float32)
    params = jnp.asarray([100.0, 8750.0, -50000.0, 50000.0, 15.0, 0.239], dtype=jnp.float32)

    features = bundle.feature_vector(state, control, disturbance, params, 0.1)
    delta = bundle.mechanistic_delta(state, control, disturbance, params, 0.1)
    residuals = bundle.trajectory_residual_series(
        states=jnp.stack([state, state + 0.01], axis=0),
        controls=jnp.stack([control, control], axis=0),
        disturbances=jnp.stack([disturbance, disturbance], axis=0),
        dt=0.1,
        params=params,
    )

    assert features.shape[0] == len(bundle.feature_names())
    assert delta.shape == (spec.state_dim,)
    assert jnp.all(jnp.isfinite(features))
    assert jnp.all(jnp.isfinite(delta))
    assert set(residuals) == set(bundle.residual_names())


def test_biology_law_bundle_example_is_usable():
    spec, _, bundle = build_bioreactor_law_bundle_example()

    assert spec.name == "bioreactor_example"
    assert len(bundle.modules) == 1
    assert "biology_aerobic_growth_specific_growth_rate" in bundle.feature_names()
    assert "biology_aerobic_growth_biomass_consistency" in bundle.residual_names()

    state = jnp.asarray([1.1, 0.4, 0.6], dtype=jnp.float32)
    control = jnp.asarray([0.6], dtype=jnp.float32)
    disturbance = jnp.asarray([1.0], dtype=jnp.float32)

    features = bundle.feature_vector(state, control, disturbance, None, 0.1)
    delta = bundle.mechanistic_delta(state, control, disturbance, None, 0.1)
    residuals = bundle.trajectory_residual_series(
        states=jnp.stack([state, state + jnp.asarray([-0.02, 0.01, 0.0])], axis=0),
        controls=jnp.stack([control, control], axis=0),
        disturbances=jnp.stack([disturbance, disturbance], axis=0),
        dt=0.1,
        params=None,
    )

    assert features.shape[0] == len(bundle.feature_names())
    assert delta.shape == (spec.state_dim,)
    assert float(delta[1]) != 0.0
    assert set(residuals) == set(bundle.residual_names())
