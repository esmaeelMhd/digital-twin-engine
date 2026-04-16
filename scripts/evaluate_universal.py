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

from dte.data.datasets.universal_unit_dataset import (
    MultiSystemTrajectoryDataset,
    SystemDatasetSource,
)
from dte.evaluation.universal import (
    compute_control_sensitivity_summary,
    compute_uncertainty_summary,
)
from dte.models.universal.digital_twin import UniversalDigitalTwin
from dte.training.universal.trainer import UniversalTrainer


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
    dataset = MultiSystemTrajectoryDataset.from_sources(
        sources,
        seq_len=seq_len,
        stride=stride,
    )
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
            uncertainty_metrics[name] = compute_uncertainty_summary(
                model,
                trainer,
                system_idx=idx,
                key=subkey,
                n_batches=uncertainty_batches,
                n_samples=uncertainty_samples,
            )

    sensitivity_metrics = {}
    if sensitivity_batches > 0:
        for idx, name in enumerate(val_dataset.system_names):
            key_sensitivity, subkey = jax.random.split(key_sensitivity)
            sensitivity_metrics[name] = compute_control_sensitivity_summary(
                model,
                trainer,
                system_idx=idx,
                key=subkey,
                n_batches=sensitivity_batches,
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
