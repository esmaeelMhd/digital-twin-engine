"""Scaffold entrypoint for future shared-checkpoint universal training.

This script does not train a universal model yet. It validates the universal
config, loads mixed-system datasets, samples padded batches, and writes a
machine-readable manifest so the next implementation step can focus on the
shared model/trainer instead of data plumbing.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dte.utils.runtime import configure_runtime_logging

configure_runtime_logging()

import jax
import yaml

from dte.data.multi_system_dataset import MultiSystemTrajectoryDataset, SystemDatasetSource


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


def main():
    parser = argparse.ArgumentParser(description="Scaffold universal multi-system training")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training_universal.yaml",
        help="Path to universal training config",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/universal_scaffold/",
        help="Directory for scaffold artifacts",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for dataset smoke sampling",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Optional batch size override",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    training_cfg = config.get("training", {})
    seq_len = int(training_cfg.get("seq_len", 50))
    stride = int(training_cfg.get("stride", 10))
    batch_size = int(args.batch_size or training_cfg.get("batch_size", 32))
    val_split = float(training_cfg.get("val_split", 0.2))

    sources = _load_sources(config)
    dataset = MultiSystemTrajectoryDataset.from_sources(sources, seq_len=seq_len, stride=stride)
    train_dataset, val_dataset = dataset.split(val_split)

    os.makedirs(args.output_dir, exist_ok=True)
    with open(Path(args.output_dir) / "config.yaml", "w") as f:
        yaml.safe_dump(config, f)

    key = jax.random.PRNGKey(args.seed)
    key_train, key_val = jax.random.split(key)
    train_batch = train_dataset.sample_batch(key_train, batch_size=batch_size, seq_len=seq_len)
    val_batch = val_dataset.sample_batch(key_val, batch_size=min(batch_size, val_dataset.n_samples), seq_len=seq_len)

    summary = {
        "status": "scaffold_ready",
        "message": (
            "Universal dataset plumbing is ready. A shared-checkpoint universal "
            "model/trainer still needs to be implemented."
        ),
        "config_path": str(Path(args.config).resolve()),
        "output_dir": str(Path(args.output_dir).resolve()),
        "train_manifest": train_dataset.manifest(),
        "val_manifest": val_dataset.manifest(),
        "sample_batch_shapes": {
            "train": {name: list(value.shape) for name, value in train_batch.items()},
            "val": {name: list(value.shape) for name, value in val_batch.items()},
        },
        "todos": [
            "Implement a shared universal model with system conditioning and masked inputs/outputs.",
            "Add a universal trainer that consumes mixed-system padded batches.",
            "Add evaluation logic for one checkpoint across all systems.",
        ],
    }

    summary_path = Path(args.output_dir) / "universal_train_scaffold_summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("UNIVERSAL TRAINING SCAFFOLD")
    print("=" * 60)
    print(f"Config: {args.config}")
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
    print(f"Scaffold summary: {summary_path}")
    print("Status: dataset scaffold ready, shared universal trainer not implemented yet.")
    print("=" * 60)


if __name__ == "__main__":
    main()
