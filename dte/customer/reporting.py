"""Validation report generation for customer adaptation runs."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from jaxtyping import Array

from dte.customer.onboarding_schema import CustomerOnboardingSpec
from dte.customer.template_matching import TemplateMatchResult
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
from dte.models.universal_digital_twin import UniversalDigitalTwin
from dte.simulators.registry import get_simulator
from dte.training.universal_trainer import UniversalTrainer


def _masked_mean(values: Array, mask: Array) -> float:
    mask = jnp.asarray(mask, dtype=jnp.float32)
    denom = jnp.maximum(jnp.sum(mask), jnp.asarray(1.0, dtype=jnp.float32))
    return float(jnp.sum(jnp.asarray(values, dtype=jnp.float32) * mask) / denom)


def _masked_metrics(prediction: Array, target: Array, mask: Array) -> dict[str, float]:
    diff = prediction - target
    abs_diff = jnp.abs(diff)
    sq_diff = diff ** 2
    return {
        "mae": _masked_mean(abs_diff, mask),
        "rmse": float(jnp.sqrt(_masked_mean(sq_diff, mask))),
        "max_abs_error": float(jnp.max(abs_diff * mask)),
    }


def predict_rollout_samples(
    model: UniversalDigitalTwin,
    trainer: UniversalTrainer,
    batch: dict[str, Array],
    key: jax.Array,
    n_samples: int,
) -> tuple[Array, Array, Array, int]:
    """Sample stochastic rollout predictions for one batch item."""

    normalized = trainer._normalize_batch(model, batch)
    sample = {name: value[0] for name, value in normalized.items()}
    keys = jax.random.split(key, n_samples)

    def rollout_one(sample_key):
        z0, _, _ = model.encode(
            sample["states_norm"][0],
            sample["params_scaled"],
            sample["controls_norm"][0],
            sample["state_mask"],
            sample["control_mask"],
            sample["param_mask"],
            sample["system_id"],
            sample_key,
        )
        z_traj = model.rollout_latent(
            sample["t"],
            z0,
            sample["controls_norm"],
            sample["disturbances_norm"],
            sample["params_scaled"],
            sample["control_mask"],
            sample["disturbance_mask"],
            sample["param_mask"],
            sample["system_id"],
        )
        pred_norm = jax.vmap(
            lambda z_t, control_t: model.decode(
                z_t,
                sample["params_scaled"],
                control_t,
                sample["state_mask"],
                sample["control_mask"],
                sample["param_mask"],
                sample["system_id"],
            )
        )(z_traj, sample["controls_norm"])
        return model.denormalize_states(pred_norm, sample["system_id"])

    samples = jax.vmap(rollout_one)(keys)
    true_states = batch["states"][0]
    state_mask = batch["state_mask"][0][None, :] * batch["time_mask"][0][:, None]
    return samples, true_states, state_mask, int(batch["system_id"][0])


def _predict_model_next_state(
    model: UniversalDigitalTwin,
    trainer: UniversalTrainer,
    batch: dict[str, Array],
    control_t: Array,
    control_tp1: Array,
) -> Array:
    normalized = trainer._normalize_batch(model, batch)
    sample = {name: value[0] for name, value in normalized.items()}
    system_id = sample["system_id"]
    control_t_norm = model.normalize_controls(control_t, system_id) * sample["control_mask"]
    control_tp1_norm = model.normalize_controls(control_tp1, system_id) * sample["control_mask"]
    z_mean = model.encode(
        sample["states_norm"][0],
        sample["params_scaled"],
        sample["controls_norm"][0],
        sample["state_mask"],
        sample["control_mask"],
        sample["param_mask"],
        system_id,
        None,
    )[1]
    z_next = model.latent_step(
        z_mean,
        control_t_norm,
        control_tp1_norm,
        sample["disturbances_norm"][0],
        sample["disturbances_norm"][1],
        sample["params_scaled"],
        sample["control_mask"],
        sample["disturbance_mask"],
        sample["param_mask"],
        system_id,
        sample["t"][1] - sample["t"][0],
    )
    next_state_norm = model.decode(
        z_next,
        sample["params_scaled"],
        control_tp1_norm,
        sample["state_mask"],
        sample["control_mask"],
        sample["param_mask"],
        system_id,
    )
    return model.denormalize_states(next_state_norm, system_id)


def compute_forecast_metrics(
    model: UniversalDigitalTwin,
    trainer: UniversalTrainer,
    *,
    system_idx: int,
    key: jax.Array,
    n_batches: int,
) -> dict[str, float]:
    """Compute one-step physical-unit forecast metrics."""

    dataset = trainer.val_dataset or trainer.train_dataset
    probs = jnp.zeros(dataset.n_systems, dtype=jnp.float32).at[system_idx].set(1.0)
    metrics = {"mae": [], "rmse": [], "max_abs_error": []}

    for _ in range(n_batches):
        key, batch_key = jax.random.split(key)
        batch = dataset.sample_batch(
            batch_key,
            batch_size=1,
            seq_len=2,
            system_probabilities=probs,
        )
        prediction = _predict_model_next_state(
            model,
            trainer,
            batch,
            batch["controls"][0, 0],
            batch["controls"][0, 1],
        )
        target = batch["states"][0, 1]
        mask = batch["state_mask"][0] * batch["time_mask"][0, 1]
        summary = _masked_metrics(prediction, target, mask)
        for name, value in summary.items():
            metrics[name].append(value)

    return {
        name: float(sum(values) / max(len(values), 1))
        for name, values in metrics.items()
    }


def compute_rollout_metrics(
    model: UniversalDigitalTwin,
    trainer: UniversalTrainer,
    *,
    system_idx: int,
    key: jax.Array,
    n_batches: int,
    n_samples: int,
) -> dict[str, float]:
    """Compute multi-step rollout metrics in physical units."""

    dataset = trainer.val_dataset or trainer.train_dataset
    probs = jnp.zeros(dataset.n_systems, dtype=jnp.float32).at[system_idx].set(1.0)
    metrics = {"mae": [], "rmse": [], "max_abs_error": []}

    for _ in range(n_batches):
        key, batch_key, sample_key = jax.random.split(key, 3)
        batch = dataset.sample_batch(
            batch_key,
            batch_size=1,
            seq_len=int(trainer.config["training"]["seq_len"]),
            system_probabilities=probs,
        )
        rollout_samples, true_states, state_mask, _ = predict_rollout_samples(
            model,
            trainer,
            batch,
            sample_key,
            n_samples=max(int(n_samples), 2),
        )
        mean = jnp.mean(rollout_samples, axis=0)
        summary = _masked_metrics(mean, true_states, state_mask)
        for name, value in summary.items():
            metrics[name].append(value)

    return {
        name: float(sum(values) / max(len(values), 1))
        for name, values in metrics.items()
    }


def compute_uncertainty_summary(
    model: UniversalDigitalTwin,
    trainer: UniversalTrainer,
    *,
    system_idx: int,
    key: jax.Array,
    n_batches: int,
    n_samples: int,
) -> dict[str, float]:
    """Summarize stochastic rollout calibration."""

    dataset = trainer.val_dataset or trainer.train_dataset
    probs = jnp.zeros(dataset.n_systems, dtype=jnp.float32).at[system_idx].set(1.0)
    metrics = {
        "coverage_1sigma": [],
        "coverage_2sigma": [],
        "calibration_gap": [],
        "gaussian_nll": [],
        "variance_collapse_rate": [],
    }

    for _ in range(n_batches):
        key, batch_key, sample_key = jax.random.split(key, 3)
        batch = dataset.sample_batch(
            batch_key,
            batch_size=1,
            seq_len=int(trainer.config["training"]["seq_len"]),
            system_probabilities=probs,
        )
        rollout_samples, true_states, state_mask, _ = predict_rollout_samples(
            model,
            trainer,
            batch,
            sample_key,
            n_samples=max(int(n_samples), 2),
        )
        mean = jnp.mean(rollout_samples, axis=0)
        std = jnp.std(rollout_samples, axis=0)
        metrics["coverage_1sigma"].append(
            empirical_coverage(mean, std, true_states, sigma=1.0, mask=state_mask)
        )
        metrics["coverage_2sigma"].append(
            empirical_coverage(mean, std, true_states, sigma=2.0, mask=state_mask)
        )
        metrics["calibration_gap"].append(calibration_gap(mean, std, true_states, mask=state_mask))
        metrics["gaussian_nll"].append(gaussian_nll(mean, std, true_states, mask=state_mask))
        metrics["variance_collapse_rate"].append(
            variance_collapse_rate(std, mask=state_mask)
        )

    return {
        name: float(sum(values) / max(len(values), 1))
        for name, values in metrics.items()
    }


def compute_control_sensitivity_summary(
    model: UniversalDigitalTwin,
    trainer: UniversalTrainer,
    *,
    system_idx: int,
    key: jax.Array,
    n_batches: int,
) -> dict[str, float]:
    """Compare model local gains to the matched simulator."""

    dataset = trainer.val_dataset or trainer.train_dataset
    entry = dataset.entries[system_idx]
    spec = entry.spec
    simulator = get_simulator(spec.name, entry.system_config)
    probs = jnp.zeros(dataset.n_systems, dtype=jnp.float32).at[system_idx].set(1.0)
    control_lower = jnp.asarray(
        [
            -jnp.inf if channel.lower_bound is None else channel.lower_bound
            for channel in spec.control_channels
        ],
        dtype=jnp.float32,
    )
    control_upper = jnp.asarray(
        [
            jnp.inf if channel.upper_bound is None else channel.upper_bound
            for channel in spec.control_channels
        ],
        dtype=jnp.float32,
    )
    control_delta = jnp.maximum(0.02 * (control_upper - control_lower), 1e-3)
    metrics = {"rmse": [], "relative_l2": [], "cosine_similarity": []}

    for _ in range(n_batches):
        key, batch_key = jax.random.split(key)
        batch = dataset.sample_batch(
            batch_key,
            batch_size=1,
            seq_len=2,
            system_probabilities=probs,
        )
        padded_delta = jnp.zeros_like(batch["controls"][0, 0]).at[: spec.control_dim].set(
            control_delta
        )

        def clip_control(control_probe):
            lower = jnp.where(jnp.isfinite(control_lower), control_lower, -jnp.inf)
            upper = jnp.where(jnp.isfinite(control_upper), control_upper, jnp.inf)
            clipped = jnp.clip(control_probe[: spec.control_dim], lower, upper)
            return batch["controls"][0, 0].at[: spec.control_dim].set(clipped)

        def model_step(control_probe):
            clipped = clip_control(control_probe)
            return _predict_model_next_state(model, trainer, batch, clipped, clipped)

        def simulator_step(control_probe):
            active_control = clip_control(control_probe)[: spec.control_dim]
            state0 = batch["states"][0, 0, : spec.state_dim]
            disturbance0 = batch["disturbances"][0, 0, : spec.disturbance_dim]
            dt = batch["t"][0, 1] - batch["t"][0, 0]
            next_state = state0 + dt * simulator.dynamics(0.0, state0, active_control, disturbance0)
            return jnp.zeros_like(batch["states"][0, 0]).at[: spec.state_dim].set(next_state)

        pred_jac = finite_difference_control_jacobian(
            model_step,
            batch["controls"][0, 0],
            padded_delta,
        )
        ref_jac = finite_difference_control_jacobian(
            simulator_step,
            batch["controls"][0, 0],
            padded_delta,
        )
        summary = sensitivity_mismatch_metrics(
            pred_jac,
            ref_jac,
            state_mask=batch["state_mask"][0],
            control_mask=batch["control_mask"][0],
        )
        for name, value in summary.items():
            metrics[name].append(value)

    return {
        name: float(sum(values) / max(len(values), 1))
        for name, values in metrics.items()
    }


def compute_constraint_summary(
    model: UniversalDigitalTwin,
    trainer: UniversalTrainer,
    *,
    system_idx: int,
    key: jax.Array,
    n_batches: int,
    n_samples: int,
) -> dict[str, float]:
    """Summarize rollout bound and positivity violations."""

    dataset = trainer.val_dataset or trainer.train_dataset
    probs = jnp.zeros(dataset.n_systems, dtype=jnp.float32).at[system_idx].set(1.0)
    lower = trainer.state_lower_bound_table[system_idx]
    upper = trainer.state_upper_bound_table[system_idx]
    lower_mask = jnp.isfinite(lower).astype(jnp.float32)
    upper_mask = jnp.isfinite(upper).astype(jnp.float32)
    positivity_mask = jnp.logical_and(jnp.isfinite(lower), lower >= 0.0).astype(jnp.float32)
    metrics = {
        "below_lower_bound_rate": [],
        "above_upper_bound_rate": [],
        "positivity_violation_rate": [],
        "uncertain_bound_cross_rate": [],
    }

    for _ in range(n_batches):
        key, batch_key, sample_key = jax.random.split(key, 3)
        batch = dataset.sample_batch(
            batch_key,
            batch_size=1,
            seq_len=int(trainer.config["training"]["seq_len"]),
            system_probabilities=probs,
        )
        rollout_samples, _, state_mask, _ = predict_rollout_samples(
            model,
            trainer,
            batch,
            sample_key,
            n_samples=max(int(n_samples), 2),
        )
        mean = jnp.mean(rollout_samples, axis=0)
        std = jnp.std(rollout_samples, axis=0)
        below_lower = (mean < lower[None, :]).astype(jnp.float32)
        above_upper = (mean > upper[None, :]).astype(jnp.float32)
        positivity_violation = (mean < 0.0).astype(jnp.float32)
        uncertain_cross = jnp.logical_or(
            mean - 2.0 * std < lower[None, :],
            mean + 2.0 * std > upper[None, :],
        ).astype(jnp.float32)
        metrics["below_lower_bound_rate"].append(
            _masked_mean(below_lower, state_mask * lower_mask[None, :])
        )
        metrics["above_upper_bound_rate"].append(
            _masked_mean(above_upper, state_mask * upper_mask[None, :])
        )
        metrics["positivity_violation_rate"].append(
            _masked_mean(positivity_violation, state_mask * positivity_mask[None, :])
        )
        finite_bound_mask = state_mask * jnp.maximum(lower_mask, upper_mask)[None, :]
        metrics["uncertain_bound_cross_rate"].append(
            _masked_mean(uncertain_cross, finite_bound_mask)
        )

    return {
        name: float(sum(values) / max(len(values), 1))
        for name, values in metrics.items()
    }


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
