"""Scaffold entrypoint for future universal checkpoint evaluation."""

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
    parser = argparse.ArgumentParser(description="Scaffold universal multi-system evaluation")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training_universal.yaml",
        help="Path to universal training config",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Optional future universal model checkpoint path",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/universal_eval_scaffold/",
        help="Directory for scaffold evaluation artifacts",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for dataset smoke sampling",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    training_cfg = config.get("training", {})
    seq_len = int(training_cfg.get("seq_len", 50))
    stride = int(training_cfg.get("stride", 10))
    batch_size = int(training_cfg.get("batch_size", 32))
    val_split = float(training_cfg.get("val_split", 0.2))

    sources = _load_sources(config)
    dataset = MultiSystemTrajectoryDataset.from_sources(sources, seq_len=seq_len, stride=stride)
    _, val_dataset = dataset.split(val_split)

    os.makedirs(args.output_dir, exist_ok=True)
    with open(Path(args.output_dir) / "config.yaml", "w") as f:
        yaml.safe_dump(config, f)

    key = jax.random.PRNGKey(args.seed)
    batch = val_dataset.sample_batch(
        key,
        batch_size=min(batch_size, val_dataset.n_samples),
        seq_len=seq_len,
    )

    model_path = Path(args.model_path).resolve() if args.model_path else None
    summary = {
        "status": "scaffold_ready",
        "message": (
            "Universal evaluation scaffold is ready. It validates the mixed "
            "validation dataset shape contract but does not score a universal "
            "checkpoint yet."
        ),
        "config_path": str(Path(args.config).resolve()),
        "model_path": str(model_path) if model_path else None,
        "model_exists": bool(model_path and model_path.exists()),
        "val_manifest": val_dataset.manifest(),
        "sample_batch_shapes": {name: list(value.shape) for name, value in batch.items()},
        "todos": [
            "Implement checkpoint loading for a shared universal model.",
            "Evaluate one checkpoint across all systems with masked decoding.",
            "Report aggregate cross-system metrics from one shared run.",
        ],
    }

    summary_path = Path(args.output_dir) / "universal_eval_scaffold_summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("UNIVERSAL EVALUATION SCAFFOLD")
    print("=" * 60)
    print(f"Config: {args.config}")
    print(f"Model path: {model_path if model_path else 'not provided'}")
    print(f"Model exists: {summary['model_exists']}")
    print(f"Systems: {', '.join(val_dataset.system_names)}")
    print(f"Validation samples: {val_dataset.n_samples}")
    print(f"Scaffold summary: {summary_path}")
    print("Status: evaluation scaffold ready, universal checkpoint scoring not implemented yet.")
    print("=" * 60)


if __name__ == "__main__":
    main()
