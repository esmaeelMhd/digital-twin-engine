"""Tests for Phase 1 evaluation utilities."""

import jax
import jax.numpy as jnp

from dte.evaluation.control_sensitivity import (
    finite_difference_control_jacobian,
    sensitivity_mismatch_metrics,
)
from dte.evaluation.uncertainty import calibration_gap, empirical_coverage, gaussian_nll
from dte.evaluation.universal import normalize_universal_batch, predict_rollout_samples
from dte.data.datasets.universal_unit_dataset import UniversalSystemMetadata
from dte.models.universal.digital_twin import UniversalDigitalTwin


def _build_metadata() -> UniversalSystemMetadata:
    return UniversalSystemMetadata(
        system_names=("cstr",),
        state_center=jnp.zeros((1, 4), dtype=jnp.float32),
        state_scale=jnp.ones((1, 4), dtype=jnp.float32),
        control_center=jnp.zeros((1, 2), dtype=jnp.float32),
        control_scale=jnp.ones((1, 2), dtype=jnp.float32),
        disturbance_center=jnp.zeros((1, 2), dtype=jnp.float32),
        disturbance_scale=jnp.ones((1, 2), dtype=jnp.float32),
        param_scale=jnp.ones((1, 6), dtype=jnp.float32),
        state_mask=jnp.ones((1, 4), dtype=jnp.float32),
        control_mask=jnp.ones((1, 2), dtype=jnp.float32),
        disturbance_mask=jnp.ones((1, 2), dtype=jnp.float32),
        param_mask=jnp.ones((1, 6), dtype=jnp.float32),
        state_dim=jnp.asarray([4], dtype=jnp.int32),
        control_dim=jnp.asarray([2], dtype=jnp.int32),
        disturbance_dim=jnp.asarray([2], dtype=jnp.int32),
        param_dim=jnp.asarray([6], dtype=jnp.int32),
        system_descriptor=jnp.zeros((1, 25), dtype=jnp.float32),
        state_group_kind_names=("concentration", "temperature"),
        state_group_mask=jnp.asarray([[[1, 1, 0, 0], [0, 0, 1, 1]]], dtype=jnp.float32),
        state_group_active=jnp.asarray([[1, 1]], dtype=jnp.float32),
        state_group_kind_id=jnp.asarray([[0, 1]], dtype=jnp.int32),
        state_role_names=("concentration", "temperature"),
        state_role_id=jnp.asarray([[0, 0, 1, 1]], dtype=jnp.int32),
        state_lower_bound=jnp.asarray([[0.0, 0.0, 250.0, 250.0]], dtype=jnp.float32),
        state_upper_bound=jnp.asarray([[jnp.inf, jnp.inf, 400.0, 400.0]], dtype=jnp.float32),
    )


def _build_config() -> dict:
    return {
        "model": {
            "latent_dim": 8,
            "shared_hidden_dim": 16,
            "system_embedding_dim": 8,
            "state_group_token_dim": 12,
            "state_group_kind_dim": 6,
            "state_group_encoder_layers": 2,
            "state_group_coupling_layers": 2,
            "encoder_layers": 2,
            "decoder_layers": 2,
            "drift_layers": 2,
            "use_system_spec_embedding": True,
            "use_variational_encoder": True,
            "neural_cde": {"enabled": True, "hidden_dim": 12, "n_layers": 2},
        }
    }


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


def test_calibration_gap_rejects_unknown_sigma_levels():
    mean = jnp.array([0.0], dtype=jnp.float32)
    std = jnp.array([1.0], dtype=jnp.float32)
    target = jnp.array([0.1], dtype=jnp.float32)
    try:
        calibration_gap(mean, std, target, sigma_levels=(1.5,))
    except ValueError as exc:
        assert "Unknown sigma level" in str(exc)
    else:
        raise AssertionError("calibration_gap should reject unknown sigma levels")


def test_universal_evaluation_helpers_normalize_and_sample_rollouts():
    metadata = _build_metadata()
    model = UniversalDigitalTwin.from_config(_build_config(), metadata, jax.random.PRNGKey(0))
    batch = {
        "states": jnp.asarray([[[0.4, 0.1, 310.0, 295.0], [0.42, 0.11, 311.0, 296.0]]], dtype=jnp.float32),
        "controls": jnp.ones((1, 2, 2), dtype=jnp.float32),
        "disturbances": jnp.ones((1, 2, 2), dtype=jnp.float32) * 0.5,
        "params": jnp.ones((1, 6), dtype=jnp.float32),
        "t": jnp.asarray([[0.0, 0.1]], dtype=jnp.float32),
        "state_mask": jnp.ones((1, 4), dtype=bool),
        "control_mask": jnp.ones((1, 2), dtype=bool),
        "disturbance_mask": jnp.ones((1, 2), dtype=bool),
        "param_mask": jnp.ones((1, 6), dtype=bool),
        "time_mask": jnp.ones((1, 2), dtype=bool),
        "system_id": jnp.zeros((1,), dtype=jnp.int32),
    }

    normalized = normalize_universal_batch(model, batch)
    rollout_samples, true_states, state_mask, system_id = predict_rollout_samples(
        model,
        batch,
        jax.random.PRNGKey(1),
        n_samples=3,
    )

    assert normalized["states_norm"].shape == (1, 2, 4)
    assert normalized["control_mask"].dtype == jnp.float32
    assert rollout_samples.shape == (3, 2, 4)
    assert true_states.shape == (2, 4)
    assert state_mask.shape == (2, 4)
    assert system_id == 0
