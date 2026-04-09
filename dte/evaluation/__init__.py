"""Evaluation helpers for control sensitivity and uncertainty diagnostics."""

from dte.evaluation.control_sensitivity import (
    finite_difference_control_jacobian,
    sensitivity_mismatch_metrics,
)
from dte.evaluation.uncertainty import (
    calibration_gap,
    empirical_coverage,
    gaussian_nll,
    variance_collapse_rate,
)

__all__ = [
    "calibration_gap",
    "empirical_coverage",
    "finite_difference_control_jacobian",
    "gaussian_nll",
    "sensitivity_mismatch_metrics",
    "variance_collapse_rate",
]
