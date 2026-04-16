"""Shared evaluation helpers for universal-model diagnostics."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from jaxtyping import Array

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
from dte.models.universal.digital_twin import UniversalDigitalTwin
from dte.simulators.registry import get_simulator

if TYPE_CHECKING:
    from dte.training.universal.trainer import UniversalTrainer


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


def normalize_universal_batch(
    model: UniversalDigitalTwin,
    batch: dict[str, Array],
) -> dict[str, Array]:
    """Normalize one mixed-system batch using model metadata tables."""

    system_ids = batch["system_id"]
    state_mask = batch["state_mask"].astype(jnp.float32)
    control_mask = batch["control_mask"].astype(jnp.float32)
    disturbance_mask = batch["disturbance_mask"].astype(jnp.float32)
    param_mask = batch["param_mask"].astype(jnp.float32)
    time_mask = batch["time_mask"].astype(jnp.float32)

    states_norm = model.normalize_states(batch["states"], system_ids) * state_mask[:, None, :]
    controls_norm = model.normalize_controls(batch["controls"], system_ids) * control_mask[:, None, :]
    disturbances_norm = (
        model.normalize_disturbances(batch["disturbances"], system_ids)
        * disturbance_mask[:, None, :]
    )
    params_scaled = model.scale_params(batch["params"], system_ids) * param_mask

    return {
        "states_norm": states_norm,
        "controls_norm": controls_norm,
        "disturbances_norm": disturbances_norm,
        "params_scaled": params_scaled,
        "state_mask": state_mask,
        "control_mask": control_mask,
        "disturbance_mask": disturbance_mask,
        "param_mask": param_mask,
        "time_mask": time_mask,
        "system_id": system_ids,
        "t": batch["t"],
    }


def _normalized_sample(
    model: UniversalDigitalTwin,
    batch: dict[str, Array],
) -> dict[str, Array]:
    normalized = normalize_universal_batch(model, batch)
    return {name: value[0] for name, value in normalized.items()}


def _evaluation_dataset(trainer: UniversalTrainer):
    return trainer.val_dataset or trainer.train_dataset


def predict_rollout_samples(
    model: UniversalDigitalTwin,
    batch: dict[str, Array],
    key: jax.Array,
    n_samples: int,
) -> tuple[Array, Array, Array, int]:
    """Sample stochastic rollout predictions for one batch item."""

    sample = _normalized_sample(model, batch)
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


def predict_model_next_state(
    model: UniversalDigitalTwin,
    batch: dict[str, Array],
    control_t: Array,
    control_tp1: Array,
) -> Array:
    """Predict the next physical state from a two-step teacher-forced batch."""

    sample = _normalized_sample(model, batch)
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

    dataset = _evaluation_dataset(trainer)
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
        prediction = predict_model_next_state(
            model,
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

    dataset = _evaluation_dataset(trainer)
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

    dataset = _evaluation_dataset(trainer)
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
        metrics["variance_collapse_rate"].append(variance_collapse_rate(std, mask=state_mask))

    return {
        name: float(sum(values) / max(len(values), 1))
        for name, values in metrics.items()
    }


def _clip_control_to_bounds(control: Array, lower: Array, upper: Array) -> Array:
    lower = jnp.where(jnp.isfinite(lower), lower, -jnp.inf)
    upper = jnp.where(jnp.isfinite(upper), upper, jnp.inf)
    return jnp.clip(control, lower, upper)


def compute_control_sensitivity_summary(
    model: UniversalDigitalTwin,
    trainer: UniversalTrainer,
    *,
    system_idx: int,
    key: jax.Array,
    n_batches: int,
) -> dict[str, float]:
    """Compare model local gains to the matched simulator."""

    dataset = _evaluation_dataset(trainer)
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
        padded_delta = jnp.zeros_like(batch["controls"][0, 0]).at[: spec.control_dim].set(control_delta)

        def clip_control(control_probe: Array) -> Array:
            clipped = _clip_control_to_bounds(
                control_probe[: spec.control_dim],
                control_lower,
                control_upper,
            )
            return batch["controls"][0, 0].at[: spec.control_dim].set(clipped)

        def model_step(control_probe: Array) -> Array:
            clipped = clip_control(control_probe)
            return predict_model_next_state(model, batch, clipped, clipped)

        def simulator_step(control_probe: Array) -> Array:
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

    dataset = _evaluation_dataset(trainer)
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
