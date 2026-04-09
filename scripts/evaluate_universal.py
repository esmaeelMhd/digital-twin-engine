"""Evaluate a shared universal digital twin checkpoint."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

from dte.utils.runtime import configure_runtime_logging

configure_runtime_logging()

import jax
import yaml

from dte.data.multi_system_dataset import MultiSystemTrajectoryDataset, SystemDatasetSource
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
from dte.simulators.registry import get_simulator, get_system_spec
from dte.training.universal_trainer import UniversalTrainer


def _json_safe_float(value):
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def _load_sources(config: dict) -> list[SystemDatasetSource]:
    systems = config.get("data", {}).get("systems", [])
    if not systems:
        raise ValueError("Universal config must define data.systems.")
    return [
        SystemDatasetSource(
            name=item["name"],
            system_config=item["system_config"],
            data_dir=item["data_dir"],
            weight=float(item.get("weight", 1.0)),
        )
        for item in systems
    ]


def _aggregate_geometric_mean(values: list[float]) -> float | None:
    if not values:
        return None
    product = 1.0
    for value in values:
        product *= float(value)
    return product ** (1.0 / len(values))


def _predict_rollout_samples(
    model: UniversalDigitalTwin,
    trainer: UniversalTrainer,
    batch: dict,
    key: jax.Array,
    n_samples: int,
):
    normalized = trainer._normalize_batch(model, batch)
    sample = {
        name: value[0]
        for name, value in normalized.items()
    }
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
        )(
            z_traj,
            sample["controls_norm"],
        )
        return model.denormalize_states(pred_norm, sample["system_id"])

    samples = jax.vmap(rollout_one)(keys)
    true_states = batch["states"][0]
    state_mask = batch["state_mask"][0][None, :] * batch["time_mask"][0][:, None]
    return samples, true_states, state_mask, int(batch["system_id"][0])


def _compute_uncertainty_metrics(
    model: UniversalDigitalTwin,
    trainer: UniversalTrainer,
    dataset: MultiSystemTrajectoryDataset,
    system_idx: int,
    key: jax.Array,
    n_batches: int,
    n_samples: int,
) -> dict[str, float]:
    probs = jax.numpy.zeros(dataset.n_systems, dtype=jax.numpy.float32).at[system_idx].set(1.0)
    metrics = {"coverage_1sigma": [], "coverage_2sigma": [], "calibration_gap": [], "gaussian_nll": [], "variance_collapse_rate": []}
    for _ in range(n_batches):
        key, batch_key, sample_key = jax.random.split(key, 3)
        batch = dataset.sample_batch(
            batch_key,
            batch_size=1,
            seq_len=int(trainer.config["training"]["seq_len"]),
            system_probabilities=probs,
        )
        rollout_samples, true_states, state_mask, _ = _predict_rollout_samples(
            model,
            trainer,
            batch,
            sample_key,
            n_samples,
        )
        mean = jax.numpy.mean(rollout_samples, axis=0)
        std = jax.numpy.std(rollout_samples, axis=0)
        metrics["coverage_1sigma"].append(
            empirical_coverage(mean, std, true_states, sigma=1.0, mask=state_mask)
        )
        metrics["coverage_2sigma"].append(
            empirical_coverage(mean, std, true_states, sigma=2.0, mask=state_mask)
        )
        metrics["calibration_gap"].append(
            calibration_gap(mean, std, true_states, mask=state_mask)
        )
        metrics["gaussian_nll"].append(
            gaussian_nll(mean, std, true_states, mask=state_mask)
        )
        metrics["variance_collapse_rate"].append(
            variance_collapse_rate(std, mask=state_mask)
        )
    return {
        name: float(sum(values) / max(len(values), 1))
        for name, values in metrics.items()
    }


def _clip_control_to_bounds(control, lower, upper):
    lower = jax.numpy.where(jax.numpy.isfinite(lower), lower, -jax.numpy.inf)
    upper = jax.numpy.where(jax.numpy.isfinite(upper), upper, jax.numpy.inf)
    return jax.numpy.clip(control, lower, upper)


def _predict_model_next_state(
    model: UniversalDigitalTwin,
    trainer: UniversalTrainer,
    batch: dict,
    control_t,
    control_tp1,
):
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


def _compute_sensitivity_metrics(
    model: UniversalDigitalTwin,
    trainer: UniversalTrainer,
    dataset: MultiSystemTrajectoryDataset,
    system_idx: int,
    simulator,
    key: jax.Array,
    n_batches: int,
) -> dict[str, float]:
    probs = jax.numpy.zeros(dataset.n_systems, dtype=jax.numpy.float32).at[system_idx].set(1.0)
    spec = dataset.entries[system_idx].spec
    control_lower = jax.numpy.asarray(
        [
            -jax.numpy.inf if channel.lower_bound is None else channel.lower_bound
            for channel in spec.control_channels
        ],
        dtype=jax.numpy.float32,
    )
    control_upper = jax.numpy.asarray(
        [
            jax.numpy.inf if channel.upper_bound is None else channel.upper_bound
            for channel in spec.control_channels
        ],
        dtype=jax.numpy.float32,
    )
    control_delta = jax.numpy.maximum(0.02 * (control_upper - control_lower), 1e-3)
    metrics = {"rmse": [], "relative_l2": [], "cosine_similarity": []}

    for _ in range(n_batches):
        key, batch_key = jax.random.split(key)
        batch = dataset.sample_batch(
            batch_key,
            batch_size=1,
            seq_len=2,
            system_probabilities=probs,
        )
        base_control = batch["controls"][0, 0, : spec.control_dim]
        padded_delta = jax.numpy.zeros_like(batch["controls"][0, 0]).at[: spec.control_dim].set(control_delta)

        def model_step(control_probe):
            clipped = batch["controls"][0, 0].at[: spec.control_dim].set(
                _clip_control_to_bounds(
                    control_probe[: spec.control_dim],
                    control_lower,
                    control_upper,
                )
            )
            return _predict_model_next_state(model, trainer, batch, clipped, clipped)

        def simulator_step(control_probe):
            active_control = _clip_control_to_bounds(
                control_probe[: spec.control_dim],
                control_lower,
                control_upper,
            )
            state0 = batch["states"][0, 0, : spec.state_dim]
            disturbance0 = batch["disturbances"][0, 0, : spec.disturbance_dim]
            dt = batch["t"][0, 1] - batch["t"][0, 0]
            next_state = state0 + dt * simulator.dynamics(0.0, state0, active_control, disturbance0)
            return jax.numpy.zeros_like(batch["states"][0, 0]).at[: spec.state_dim].set(next_state)

        base_control_padded = batch["controls"][0, 0]
        pred_jac = finite_difference_control_jacobian(model_step, base_control_padded, padded_delta)
        ref_jac = finite_difference_control_jacobian(simulator_step, base_control_padded, padded_delta)
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


def main():
    parser = argparse.ArgumentParser(description="Evaluate a universal digital twin")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training_universal.yaml",
        help="Path to universal training config",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained universal checkpoint",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/universal_eval/",
        help="Directory for evaluation artifacts",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    training_cfg = config["training"]
    seq_len = int(training_cfg["seq_len"])
    stride = int(training_cfg["stride"])
    val_split = float(training_cfg.get("val_split", 0.2))

    sources = _load_sources(config)
    system_configs = {}
    simulators = {}
    for source in sources:
        with open(source.system_config, "r") as f:
            system_config = yaml.safe_load(f)
        system_configs[source.name] = system_config
        system_name = get_system_spec(system_config).name
        simulators[source.name] = get_simulator(system_name, system_config)
    dataset = MultiSystemTrajectoryDataset.from_sources(sources, seq_len=seq_len, stride=stride)
    train_dataset, val_dataset = dataset.split(val_split)

    os.makedirs(args.output_dir, exist_ok=True)
    with open(Path(args.output_dir) / "config.yaml", "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    model = UniversalDigitalTwin.load(args.model_path, config, train_dataset.metadata)
    trainer = UniversalTrainer(model, config, train_dataset, val_dataset)

    key = jax.random.PRNGKey(args.seed)
    key_mixed, key_systems, key_uncertainty, key_sensitivity = jax.random.split(key, 4)
    mixed_val = trainer._validate_batches(
        val_dataset,
        key_mixed,
        n_batches=int(config.get("checkpointing", {}).get("max_val_batches", 4)),
    )
    per_system = trainer.evaluate_per_system(
        key_systems,
        n_batches=int(config.get("evaluation", {}).get("per_system_batches", config.get("checkpointing", {}).get("max_val_batches", 4))),
    )
    uncertainty_batches = int(config.get("evaluation", {}).get("uncertainty_batches", 0))
    uncertainty_samples = int(config.get("evaluation", {}).get("uncertainty_samples", 0))
    sensitivity_batches = int(config.get("evaluation", {}).get("sensitivity_batches", 0))

    uncertainty_metrics = {}
    if uncertainty_batches > 0 and uncertainty_samples > 0:
        for idx, name in enumerate(val_dataset.system_names):
            key_uncertainty, subkey = jax.random.split(key_uncertainty)
            uncertainty_metrics[name] = _compute_uncertainty_metrics(
                model,
                trainer,
                val_dataset,
                idx,
                subkey,
                uncertainty_batches,
                uncertainty_samples,
            )

    sensitivity_metrics = {}
    if sensitivity_batches > 0:
        for idx, name in enumerate(val_dataset.system_names):
            key_sensitivity, subkey = jax.random.split(key_sensitivity)
            sensitivity_metrics[name] = _compute_sensitivity_metrics(
                model,
                trainer,
                val_dataset,
                idx,
                simulators[name],
                subkey,
                sensitivity_batches,
            )
    per_system_total = {name: metrics["total"] for name, metrics in per_system.items()}
    aggregate_total = _aggregate_geometric_mean(list(per_system_total.values()))

    summary = {
        "status": "ok",
        "config_path": str(Path(args.config).resolve()),
        "model_path": str(Path(args.model_path).resolve()),
        "output_dir": str(Path(args.output_dir).resolve()),
        "mixed_val_losses": mixed_val,
        "per_system_val_losses": per_system,
        "aggregate_metric_name": "geometric_mean_per_system_total_loss",
        "aggregate_metric_value": _json_safe_float(aggregate_total),
        "val_manifest": val_dataset.manifest(),
        "uncertainty_metrics": uncertainty_metrics,
        "control_sensitivity_metrics": sensitivity_metrics,
    }

    summary_path = Path(args.output_dir) / "summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("UNIVERSAL DIGITAL TWIN EVALUATION")
    print("=" * 60)
    print(f"Model: {args.model_path}")
    print(f"Mixed validation total loss: {mixed_val['total']:.4f}")
    for name, metrics in per_system.items():
        print(f"  {name}: total={metrics['total']:.4f}, traj={metrics['trajectory']:.4f}")
        if name in uncertainty_metrics:
            print(
                f"    uncertainty: cov@2sigma={uncertainty_metrics[name]['coverage_2sigma']:.3f}, "
                f"gap={uncertainty_metrics[name]['calibration_gap']:.3f}"
            )
        if name in sensitivity_metrics:
            print(
                f"    sensitivity: rmse={sensitivity_metrics[name]['rmse']:.4f}, "
                f"rel={sensitivity_metrics[name]['relative_l2']:.4f}"
            )
    if aggregate_total is not None:
        print(f"Aggregate geometric mean total loss: {aggregate_total:.4f}")
    print(f"Evaluation summary: {summary_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
