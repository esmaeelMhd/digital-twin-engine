"""Tests for Phase 1 evaluation utilities."""

import jax.numpy as jnp

from dte.evaluation.control_sensitivity import (
    finite_difference_control_jacobian,
    sensitivity_mismatch_metrics,
)
from dte.evaluation.uncertainty import calibration_gap, empirical_coverage, gaussian_nll


def test_finite_difference_control_jacobian_matches_linear_system():
    matrix = jnp.array([[2.0, -1.0], [0.5, 3.0]], dtype=jnp.float32)

    def step_fn(control):
        return matrix @ control

    jacobian = finite_difference_control_jacobian(
        step_fn,
        jnp.array([0.1, -0.2], dtype=jnp.float32),
        1e-3,
    )

    assert jnp.allclose(jacobian, matrix, atol=1e-4)


def test_sensitivity_and_uncertainty_metrics_are_well_formed():
    pred_jac = jnp.array([[1.0, 0.0], [0.0, 1.0]], dtype=jnp.float32)
    ref_jac = jnp.array([[1.0, 0.1], [0.0, 0.9]], dtype=jnp.float32)
    summary = sensitivity_mismatch_metrics(pred_jac, ref_jac)

    assert summary["rmse"] >= 0.0
    assert -1.0 <= summary["cosine_similarity"] <= 1.0

    mean = jnp.array([0.0, 1.0], dtype=jnp.float32)
    std = jnp.array([1.0, 0.5], dtype=jnp.float32)
    target = jnp.array([0.1, 1.2], dtype=jnp.float32)

    assert 0.0 <= empirical_coverage(mean, std, target, sigma=1.0) <= 1.0
    assert gaussian_nll(mean, std, target) == gaussian_nll(mean, std, target)
    assert calibration_gap(mean, std, target) >= 0.0
