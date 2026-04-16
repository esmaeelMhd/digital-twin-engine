"""Train a shared universal digital twin across multiple systems."""

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
from dte.models.universal.digital_twin import UniversalDigitalTwin
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


def main():
    parser = argparse.ArgumentParser(description="Train a shared universal digital twin")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training_universal.yaml",
        help="Path to universal training config",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/universal_model/",
        help="Output directory",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--n_epochs", type=int, default=None, help="Optional epoch override")
    parser.add_argument("--batch_size", type=int, default=None, help="Optional batch-size override")
    parser.add_argument(
        "--time_budget_minutes",
        type=float,
        default=None,
        help="Optional wall-clock training budget in minutes",
    )
    parser.add_argument(
        "--summary_path",
        type=str,
        default=None,
        help="Optional machine-readable summary path",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    if args.n_epochs is not None:
        config.setdefault("training", {})["n_epochs"] = int(args.n_epochs)
    if args.batch_size is not None:
        config.setdefault("training", {})["batch_size"] = int(args.batch_size)

    training_cfg = config["training"]
    seq_len = int(training_cfg["seq_len"])
    stride = int(training_cfg["stride"])
    val_split = float(training_cfg.get("val_split", 0.2))

    sources = _load_sources(config)
    full_dataset = MultiSystemTrajectoryDataset.from_sources(sources, seq_len=seq_len, stride=stride)
    train_dataset, val_dataset = full_dataset.split(val_split)

    batch_size = min(int(training_cfg["batch_size"]), train_dataset.n_samples)
    full_batches = max(1, train_dataset.n_samples // batch_size)
    max_batches_per_epoch = training_cfg.get("max_batches_per_epoch")
    n_batches = full_batches if max_batches_per_epoch is None else max(1, min(full_batches, int(max_batches_per_epoch)))

    opt_cfg = config.setdefault("optimizer", {})
    opt_cfg.setdefault("peak_lr", 5.0e-4)
    opt_cfg.setdefault("end_lr", 1.0e-5)
    opt_cfg.setdefault("warmup_steps", 50)
    opt_cfg.setdefault(
        "total_steps",
        int(training_cfg["n_epochs"]) * n_batches,
    )
    opt_cfg.setdefault("gradient_clip", 1.0)
    config.setdefault("checkpointing", {})
    config["checkpointing"].setdefault("val_every", 1)
    config["checkpointing"].setdefault("save_every", 5)
    config["checkpointing"].setdefault("max_val_batches", 4)
    config.setdefault("evaluation", {})
    config["evaluation"].setdefault("per_system_batches", config["checkpointing"]["max_val_batches"])

    os.makedirs(args.output_dir, exist_ok=True)
    with open(Path(args.output_dir) / "config.yaml", "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    with open(Path(args.output_dir) / "data_manifest.json", "w") as f:
        json.dump(
            {
                "full_dataset": full_dataset.manifest(),
                "train_dataset": train_dataset.manifest(),
                "val_dataset": val_dataset.manifest(),
            },
            f,
            indent=2,
        )

    key = jax.random.PRNGKey(args.seed)
    key_model, key_train, key_eval = jax.random.split(key, 3)

    model = UniversalDigitalTwin.from_config(config, train_dataset.metadata, key_model)
    trainer = UniversalTrainer(model, config, train_dataset, val_dataset)

    print("\n" + "=" * 60)
    print("UNIVERSAL DIGITAL TWIN TRAINING")
    print("=" * 60)
    print(f"Config: {args.config}")
    print(f"Output directory: {args.output_dir}")
    print(f"Systems: {', '.join(train_dataset.system_names)}")
    print(
        "Max dims: "
        f"state={train_dataset.max_state_dim}, "
        f"control={train_dataset.max_control_dim}, "
        f"disturbance={train_dataset.max_disturbance_dim}, "
        f"param={train_dataset.max_param_dim}"
    )
    print(f"Train samples: {train_dataset.n_samples}")
    print(f"Val samples: {val_dataset.n_samples}")
    print(f"Epochs: {config['training']['n_epochs']}")
    print(f"Batch size: {config['training']['batch_size']}")
    print(f"Batches per epoch: {n_batches}")
    if args.time_budget_minutes is not None:
        print(f"Time budget: {args.time_budget_minutes:.2f} minutes")
    print("=" * 60)

    train_summary = trainer.train(
        n_epochs=int(config["training"]["n_epochs"]),
        output_dir=args.output_dir,
        key=key_train,
        time_budget_seconds=None if args.time_budget_minutes is None else args.time_budget_minutes * 60.0,
    )
    per_system = trainer.evaluate_per_system(
        key_eval,
        n_batches=int(config["evaluation"]["per_system_batches"]),
    )
    per_system_total = {name: metrics["total"] for name, metrics in per_system.items()}
    aggregate_total = _aggregate_geometric_mean(list(per_system_total.values()))

    summary = {
        "status": "ok" if train_summary.get("failure_reason") is None else "failed",
        "config_path": str(Path(args.config).resolve()),
        "output_dir": str(Path(args.output_dir).resolve()),
        "best_val_loss": _json_safe_float(train_summary.get("best_val_loss")),
        "epochs_completed": int(train_summary.get("epochs_completed", 0)),
        "steps_completed": int(train_summary.get("steps_completed", 0)),
        "timed_out": bool(train_summary.get("timed_out", False)),
        "training_seconds": _json_safe_float(train_summary.get("training_seconds")),
        "failure_reason": train_summary.get("failure_reason"),
        "parameter_counts": model.get_parameter_count(),
        "train_manifest": train_dataset.manifest(),
        "val_manifest": val_dataset.manifest(),
        "per_system_val_losses": per_system,
        "aggregate_metric_name": "geometric_mean_per_system_total_loss",
        "aggregate_metric_value": _json_safe_float(aggregate_total),
    }

    summary_path = Path(args.summary_path) if args.summary_path else Path(args.output_dir) / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    print("\nPer-system validation losses:")
    for name, metrics in per_system.items():
        print(f"  {name}: total={metrics['total']:.4f}, traj={metrics['trajectory']:.4f}")
    if aggregate_total is not None:
        print(f"Aggregate geometric mean total loss: {aggregate_total:.4f}")
    print(f"Training summary: {summary_path}")


if __name__ == "__main__":
    main()
