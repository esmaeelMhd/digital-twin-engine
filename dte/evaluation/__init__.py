"""Evaluation helpers for universal and flowsheet diagnostics."""

from dte.evaluation.control_sensitivity import (
    finite_difference_control_jacobian,
    sensitivity_mismatch_metrics,
)
from dte.evaluation.flowsheet_metrics import (
    plant_balance_proxy_loss,
    rollout_stability_penalty,
    source_stream_values_from_states,
    stream_consistency_loss,
    target_stream_values_from_states,
    unit_output_consistency_loss,
)
from dte.evaluation.uncertainty import (
    calibration_gap,
    empirical_coverage,
    gaussian_nll,
    variance_collapse_rate,
)
from dte.evaluation.universal import (
    compute_constraint_summary,
    compute_control_sensitivity_summary,
    compute_forecast_metrics,
    compute_rollout_metrics,
    compute_uncertainty_summary,
    normalize_universal_batch,
    per_system_metrics_key,
    predict_model_next_state,
    predict_rollout_samples,
)

__all__ = [
    "calibration_gap",
    "compute_constraint_summary",
    "compute_control_sensitivity_summary",
    "compute_forecast_metrics",
    "compute_rollout_metrics",
    "compute_uncertainty_summary",
    "empirical_coverage",
    "finite_difference_control_jacobian",
    "gaussian_nll",
    "normalize_universal_batch",
    "per_system_metrics_key",
    "plant_balance_proxy_loss",
    "predict_model_next_state",
    "predict_rollout_samples",
    "rollout_stability_penalty",
    "sensitivity_mismatch_metrics",
    "source_stream_values_from_states",
    "stream_consistency_loss",
    "target_stream_values_from_states",
    "unit_output_consistency_loss",
    "variance_collapse_rate",
]
