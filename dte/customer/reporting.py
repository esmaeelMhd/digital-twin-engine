"""Validation report generation for customer adaptation runs."""

from __future__ import annotations

from typing import Any

import jax

from dte.customer.onboarding_schema import CustomerOnboardingSpec
from dte.customer.template_matching import TemplateMatchResult
from dte.evaluation.universal import (
    compute_constraint_summary,
    compute_control_sensitivity_summary,
    compute_forecast_metrics,
    compute_rollout_metrics,
    compute_uncertainty_summary,
    predict_rollout_samples,
)
from dte.models.universal.digital_twin import UniversalDigitalTwin
from dte.training.universal.trainer import UniversalTrainer


def generate_customer_validation_report(
    *,
    model: UniversalDigitalTwin,
    trainer: UniversalTrainer,
    onboarding: CustomerOnboardingSpec,
    template_matches: TemplateMatchResult,
    calibration_summary: dict[str, Any],
    key: jax.Array,
) -> dict[str, Any]:
    """Generate a structured validation report for the calibrated customer model."""

    report_dataset = trainer.val_dataset or trainer.train_dataset
    system_idx = int(calibration_summary.get("target_system_id", 0))
    system_name = calibration_summary.get("target_system", report_dataset.system_names[system_idx])
    eval_cfg = trainer.config.get("evaluation", {})
    per_system_batches = int(eval_cfg.get("per_system_batches", 2))
    uncertainty_batches = int(eval_cfg.get("uncertainty_batches", per_system_batches))
    uncertainty_samples = int(eval_cfg.get("uncertainty_samples", 8))
    sensitivity_batches = int(eval_cfg.get("sensitivity_batches", per_system_batches))

    key_forecast, key_rollout, key_uncertainty, key_sensitivity, key_constraints = jax.random.split(
        key,
        5,
    )

    best_unit_match = None
    if onboarding.units:
        best = template_matches.best_unit_match(onboarding.units[0].name)
        best_unit_match = None if best is None else best.to_dict()

    best_flowsheet_match = template_matches.best_flowsheet_match()

    return {
        "report_version": "phase5_v1",
        "customer_asset": {
            "name": onboarding.name,
            "asset_kind": onboarding.asset_kind,
            "unit_names": list(onboarding.unit_names),
        },
        "target_system": {
            "name": system_name,
            "system_id": system_idx,
            "family": getattr(report_dataset.entries[system_idx].spec, "family", "generic"),
            "subtype": getattr(report_dataset.entries[system_idx].spec, "subtype", None),
        },
        "template_matching": {
            "best_unit_match": best_unit_match,
            "best_flowsheet_match": None if best_flowsheet_match is None else best_flowsheet_match.to_dict(),
            "ranked": template_matches.to_dict(),
        },
        "calibration_summary": calibration_summary,
        "forecast_metrics": compute_forecast_metrics(
            model,
            trainer,
            system_idx=system_idx,
            key=key_forecast,
            n_batches=per_system_batches,
        ),
        "rollout_metrics": compute_rollout_metrics(
            model,
            trainer,
            system_idx=system_idx,
            key=key_rollout,
            n_batches=per_system_batches,
            n_samples=uncertainty_samples,
        ),
        "uncertainty_summary": compute_uncertainty_summary(
            model,
            trainer,
            system_idx=system_idx,
            key=key_uncertainty,
            n_batches=uncertainty_batches,
            n_samples=uncertainty_samples,
        ),
        "control_sensitivity_metrics": compute_control_sensitivity_summary(
            model,
            trainer,
            system_idx=system_idx,
            key=key_sensitivity,
            n_batches=sensitivity_batches,
        ),
        "constraints_summary": compute_constraint_summary(
            model,
            trainer,
            system_idx=system_idx,
            key=key_constraints,
            n_batches=per_system_batches,
            n_samples=uncertainty_samples,
        ),
    }


def render_validation_report_markdown(report: dict[str, Any]) -> str:
    """Render a short human-readable markdown report."""

    customer_asset = report["customer_asset"]
    target_system = report["target_system"]
    template_matching = report["template_matching"]
    best_unit_match = template_matching.get("best_unit_match")
    best_flowsheet_match = template_matching.get("best_flowsheet_match")
    calibration = report["calibration_summary"]
    forecast = report["forecast_metrics"]
    rollout = report["rollout_metrics"]
    sensitivity = report["control_sensitivity_metrics"]
    uncertainty = report["uncertainty_summary"]
    constraints = report["constraints_summary"]

    lines = [
        f"# Customer Validation Report: {customer_asset['name']}",
        "",
        "## Overview",
        f"- Asset kind: `{customer_asset['asset_kind']}`",
        f"- Target system: `{target_system['name']}`",
        f"- Target family/subtype: `{target_system['family']}` / `{target_system['subtype']}`",
        f"- Best unit template: `{best_unit_match['name'] if best_unit_match else 'n/a'}`",
        f"- Best flowsheet template: `{best_flowsheet_match['name'] if best_flowsheet_match else 'n/a'}`",
        f"- Trainable mode: `{calibration['trainable_mode']}`",
        f"- Trainable parameter count: `{calibration['trainable_parameter_count']}`",
        f"- Best validation loss: `{calibration['best_val_loss']}`",
        "",
        "## Forecast Metrics",
        f"- One-step MAE: `{forecast['mae']:.6f}`",
        f"- One-step RMSE: `{forecast['rmse']:.6f}`",
        f"- One-step max abs error: `{forecast['max_abs_error']:.6f}`",
        "",
        "## Rollout Metrics",
        f"- Rollout MAE: `{rollout['mae']:.6f}`",
        f"- Rollout RMSE: `{rollout['rmse']:.6f}`",
        f"- Rollout max abs error: `{rollout['max_abs_error']:.6f}`",
        "",
        "## Control Sensitivity",
        f"- Jacobian RMSE: `{sensitivity['rmse']:.6f}`",
        f"- Relative L2: `{sensitivity['relative_l2']:.6f}`",
        f"- Cosine similarity: `{sensitivity['cosine_similarity']:.6f}`",
        "",
        "## Uncertainty",
        f"- Coverage @1 sigma: `{uncertainty['coverage_1sigma']:.6f}`",
        f"- Coverage @2 sigma: `{uncertainty['coverage_2sigma']:.6f}`",
        f"- Calibration gap: `{uncertainty['calibration_gap']:.6f}`",
        f"- Gaussian NLL: `{uncertainty['gaussian_nll']:.6f}`",
        f"- Variance collapse rate: `{uncertainty['variance_collapse_rate']:.6f}`",
        "",
        "## Constraints",
        f"- Below lower bound rate: `{constraints['below_lower_bound_rate']:.6f}`",
        f"- Above upper bound rate: `{constraints['above_upper_bound_rate']:.6f}`",
        f"- Positivity violation rate: `{constraints['positivity_violation_rate']:.6f}`",
        f"- Uncertain bound-cross rate: `{constraints['uncertain_bound_cross_rate']:.6f}`",
        "",
    ]
    return "\n".join(lines)
