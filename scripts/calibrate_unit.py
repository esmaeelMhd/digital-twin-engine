"""Calibrate a pretrained universal model to a target customer unit."""

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

from dte.calibration.unit_calibration import (
    CalibrationOptions,
    UnitCalibrator,
    initialize_target_model_from_pretrained,
)
from dte.data.datasets.universal_unit_dataset import (
    MultiSystemTrajectoryDataset,
    SystemDatasetSource,
)
from dte.models.universal.digital_twin import UniversalDigitalTwin


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


def _parse_param_indices(raw: str | None) -> tuple[int, ...] | None:
    if raw is None or raw.strip() == "":
        return None
    return tuple(int(token.strip()) for token in raw.split(",") if token.strip())


def main():
    parser = argparse.ArgumentParser(description="Calibrate a universal model to one target unit")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to pretrained universal checkpoint",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training_universal.yaml",
        help="Path to universal training config used for pretraining",
    )
    parser.add_argument(
        "--system_config",
        type=str,
        required=True,
        help="Target system config YAML",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Target unit data directory containing train_data.h5",
    )
    parser.add_argument(
        "--system_name",
        type=str,
        default=None,
        help="Optional target system name override",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/unit_calibration/",
        help="Output directory",
    )
    parser.add_argument(
        "--trainable_mode",
        choices=["adapters", "full"],
        default="adapters",
        help="Which parameter subset to calibrate",
    )
    parser.add_argument(
        "--tune_normalization",
        action="store_true",
        help="Allow calibration of normalization offsets/scales",
    )
    parser.add_argument(
        "--tune_physics_params",
        action="store_true",
        help="Allow calibration of selected physical parameter slots",
    )
    parser.add_argument(
        "--param_indices",
        type=str,
        default=None,
        help="Comma-separated physical parameter indices to calibrate",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--n_epochs", type=int, default=None, help="Optional epoch override")
    parser.add_argument("--batch_size", type=int, default=None, help="Optional batch-size override")
    parser.add_argument(
        "--time_budget_minutes",
        type=float,
        default=None,
        help="Optional wall-clock budget in minutes",
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
    with open(args.system_config, "r") as f:
        target_system_config = yaml.safe_load(f)

    if args.n_epochs is not None:
        config.setdefault("training", {})["n_epochs"] = int(args.n_epochs)
    if args.batch_size is not None:
        config.setdefault("training", {})["batch_size"] = int(args.batch_size)

    seq_len = int(config["training"]["seq_len"])
    stride = int(config["training"]["stride"])
    val_split = float(config["training"].get("val_split", 0.2))

    source_sources = _load_sources(config)
    source_dataset = MultiSystemTrajectoryDataset.from_sources(
        source_sources,
        seq_len=seq_len,
        stride=stride,
    )

    target_name = args.system_name or str(target_system_config["system"]["name"])
    target_source = SystemDatasetSource(
        name=target_name,
        system_config=args.system_config,
        data_dir=args.data_dir,
        weight=1.0,
    )
    full_target_dataset = MultiSystemTrajectoryDataset.from_sources(
        [target_source],
        seq_len=seq_len,
        stride=stride,
    )
    train_dataset, val_dataset = full_target_dataset.split(val_split)

    os.makedirs(args.output_dir, exist_ok=True)
    calibration_config = dict(config)
    calibration_config["data"] = {
        "systems": [
            {
                "name": target_name,
                "system_config": args.system_config,
                "data_dir": args.data_dir,
                "weight": 1.0,
            }
        ]
    }
    calibration_config["calibration"] = {
        "trainable_mode": args.trainable_mode,
        "tune_normalization": bool(args.tune_normalization),
        "tune_physics_params": bool(args.tune_physics_params),
        "param_indices": list(_parse_param_indices(args.param_indices) or ()),
        "source_model_path": str(Path(args.model_path).resolve()),
    }
    with open(Path(args.output_dir) / "config.yaml", "w") as f:
        yaml.safe_dump(calibration_config, f, sort_keys=False)

    key = jax.random.PRNGKey(args.seed)
    key_load, key_init, key_train = jax.random.split(key, 3)

    pretrained_model = UniversalDigitalTwin.load(
        args.model_path,
        config,
        source_dataset.metadata,
    )
    model = initialize_target_model_from_pretrained(
        pretrained_model,
        source_dataset.metadata,
        train_dataset.metadata,
        config,
        key_init,
    )

    options = CalibrationOptions(
        trainable_mode=args.trainable_mode,
        tune_normalization=bool(args.tune_normalization),
        tune_physics_params=bool(args.tune_physics_params),
        active_param_indices=_parse_param_indices(args.param_indices),
    )
    calibrator = UnitCalibrator(
        model,
        calibration_config,
        train_dataset,
        val_dataset,
        options=options,
        target_system_id=0,
    )

    print("\n" + "=" * 60)
    print("UNIT CALIBRATION")
    print("=" * 60)
    print(f"Source checkpoint: {args.model_path}")
    print(f"Target system: {target_name}")
    print(f"Target system config: {args.system_config}")
    print(f"Target data dir: {args.data_dir}")
    print(f"Trainable mode: {args.trainable_mode}")
    print(f"Tune normalization: {bool(args.tune_normalization)}")
    print(f"Tune physics params: {bool(args.tune_physics_params)}")
    print(f"Trainable parameters: {calibrator.trainable_parameter_count:,}")
    if args.time_budget_minutes is not None:
        print(f"Time budget: {args.time_budget_minutes:.2f} minutes")
    print("=" * 60)

    train_summary = calibrator.calibrate(
        args.output_dir,
        key=key_train,
        time_budget_seconds=None
        if args.time_budget_minutes is None
        else args.time_budget_minutes * 60.0,
    )

    summary = {
        "status": "ok" if train_summary.get("failure_reason") is None else "failed",
        "config_path": str(Path(args.config).resolve()),
        "output_dir": str(Path(args.output_dir).resolve()),
        "model_path": str(Path(args.model_path).resolve()),
        "target_system_name": target_name,
        "target_system_config": str(Path(args.system_config).resolve()),
        "target_data_dir": str(Path(args.data_dir).resolve()),
        "trainable_mode": args.trainable_mode,
        "tune_normalization": bool(args.tune_normalization),
        "tune_physics_params": bool(args.tune_physics_params),
        "param_indices": list(_parse_param_indices(args.param_indices) or ()),
        "best_val_loss": _json_safe_float(train_summary.get("best_val_loss")),
        "epochs_completed": int(train_summary.get("epochs_completed", 0)),
        "steps_completed": int(train_summary.get("steps_completed", 0)),
        "timed_out": bool(train_summary.get("timed_out", False)),
        "training_seconds": _json_safe_float(train_summary.get("training_seconds")),
        "failure_reason": train_summary.get("failure_reason"),
        "trainable_parameter_count": int(train_summary["trainable_parameter_count"]),
        "parameter_counts": train_summary["parameter_counts"],
        "target_manifest": {
            "full_dataset": full_target_dataset.manifest(),
            "train_dataset": train_dataset.manifest(),
            "val_dataset": val_dataset.manifest(),
        },
        "per_system_val_losses": train_summary.get("per_system_val_losses"),
        "per_system_train_fallback": train_summary.get("per_system_train_fallback"),
    }

    summary_path = Path(args.summary_path) if args.summary_path else Path(args.output_dir) / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"Calibration summary: {summary_path}")


if __name__ == "__main__":
    main()
