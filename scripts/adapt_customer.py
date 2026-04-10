"""Adapt a pretrained universal checkpoint to a customer unit from onboarding data."""

from __future__ import annotations

import argparse
from pathlib import Path

from dte.utils.runtime import configure_runtime_logging

configure_runtime_logging()

import yaml

from dte.calibration.unit_calibration import CalibrationOptions
from dte.customer.adaptation import run_customer_adaptation
from dte.customer.onboarding_schema import load_onboarding_spec


def _parse_param_indices(raw: str | None) -> tuple[int, ...] | None:
    if raw is None or raw.strip() == "":
        return None
    return tuple(int(token.strip()) for token in raw.split(",") if token.strip())


def main():
    parser = argparse.ArgumentParser(
        description="Adapt a pretrained universal model to a customer unit and generate a validation report"
    )
    parser.add_argument(
        "--onboarding",
        type=str,
        required=True,
        help="Path to customer onboarding YAML or JSON",
    )
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
        help="Universal training config used for the pretrained checkpoint",
    )
    parser.add_argument(
        "--system_config",
        type=str,
        required=True,
        help="Target system config YAML for the customer unit",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Target customer data directory containing train_data.h5",
    )
    parser.add_argument(
        "--system_name",
        type=str,
        default=None,
        help="Optional override for the target dataset system name",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/customer_adaptation/",
        help="Directory for adaptation outputs",
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
        help="Allow calibration of normalization offsets and scales",
    )
    parser.add_argument(
        "--tune_physics_params",
        action="store_true",
        help="Allow calibration of selected parameter slots",
    )
    parser.add_argument(
        "--param_indices",
        type=str,
        default=None,
        help="Comma-separated parameter indices to calibrate when physics tuning is enabled",
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

    onboarding = load_onboarding_spec(args.onboarding)
    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if args.n_epochs is not None:
        config.setdefault("training", {})["n_epochs"] = int(args.n_epochs)
    if args.batch_size is not None:
        config.setdefault("training", {})["batch_size"] = int(args.batch_size)

    options = CalibrationOptions(
        trainable_mode=args.trainable_mode,
        tune_normalization=bool(args.tune_normalization),
        tune_physics_params=bool(args.tune_physics_params),
        active_param_indices=_parse_param_indices(args.param_indices),
    )

    summary = run_customer_adaptation(
        model_path=args.model_path,
        config=config,
        onboarding=onboarding,
        system_config_path=args.system_config,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        system_name=args.system_name,
        options=options,
        seed=args.seed,
        time_budget_seconds=None
        if args.time_budget_minutes is None
        else float(args.time_budget_minutes) * 60.0,
        summary_path=args.summary_path,
    )

    print("\n" + "=" * 60)
    print("CUSTOMER ADAPTATION")
    print("=" * 60)
    print(f"Customer asset: {onboarding.name}")
    print(f"Target system: {summary['target_system_name']}")
    print(f"Status: {summary['status']}")
    print(f"Best val loss: {summary['best_val_loss']}")
    print(f"Report JSON: {summary['report_json_path']}")
    print(f"Report Markdown: {summary['report_markdown_path']}")
    print(f"Summary: {Path(args.summary_path).resolve() if args.summary_path else Path(args.output_dir).resolve() / 'summary.json'}")


if __name__ == "__main__":
    main()
