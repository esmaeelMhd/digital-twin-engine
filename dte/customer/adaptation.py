"""Customer adaptation orchestration built on the universal calibration stack."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import jax
import yaml

from dte.calibration.unit_calibration import (
    CalibrationOptions,
    UnitCalibrator,
    initialize_target_model_from_pretrained,
)
from dte.customer.onboarding_schema import CustomerOnboardingSpec
from dte.customer.reporting import (
    generate_customer_validation_report,
    render_validation_report_markdown,
)
from dte.customer.template_matching import match_customer_templates
from dte.data.datasets.universal_unit_dataset import (
    MultiSystemTrajectoryDataset,
    SystemDatasetSource,
)
from dte.models.universal.digital_twin import UniversalDigitalTwin


def load_universal_sources(config: dict[str, Any]) -> list[SystemDatasetSource]:
    """Parse universal training config sources into dataset descriptors."""

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


def run_customer_adaptation(
    *,
    model_path: str,
    config: dict[str, Any],
    onboarding: CustomerOnboardingSpec,
    system_config_path: str,
    data_dir: str,
    output_dir: str,
    system_name: str | None = None,
    options: CalibrationOptions | None = None,
    seed: int = 42,
    time_budget_seconds: float | None = None,
    summary_path: str | None = None,
) -> dict[str, Any]:
    """Adapt a pretrained universal model to one customer unit and generate a report."""

    if onboarding.asset_kind != "unit":
        raise ValueError(
            "Phase 5 adaptation currently supports asset_kind='unit'. "
            "Flowsheet onboarding/template matching is available, but flowsheet calibration "
            "is not yet orchestrated through this entry point."
        )

    config = copy.deepcopy(config)
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    template_matches = match_customer_templates(onboarding)
    onboarding_json_path = output_dir_path / "onboarding.json"
    template_matches_path = output_dir_path / "template_matches.json"
    onboarding_json_path.write_text(
        json.dumps(onboarding.model_dump(mode="json"), indent=2),
        encoding="utf-8",
    )
    template_matches_path.write_text(
        json.dumps(template_matches.to_dict(), indent=2),
        encoding="utf-8",
    )

    seq_len = int(config["training"]["seq_len"])
    stride = int(config["training"]["stride"])
    val_split = float(config["training"].get("val_split", 0.2))

    source_sources = load_universal_sources(config)
    source_dataset = MultiSystemTrajectoryDataset.from_sources(
        source_sources,
        seq_len=seq_len,
        stride=stride,
    )

    target_name = system_name or onboarding.name
    target_source = SystemDatasetSource(
        name=target_name,
        system_config=system_config_path,
        data_dir=data_dir,
        weight=1.0,
    )
    full_target_dataset = MultiSystemTrajectoryDataset.from_sources(
        [target_source],
        seq_len=seq_len,
        stride=stride,
    )
    train_dataset, val_dataset = full_target_dataset.split(val_split)

    calibration_config = copy.deepcopy(config)
    calibration_config["data"] = {
        "systems": [
            {
                "name": target_name,
                "system_config": system_config_path,
                "data_dir": data_dir,
                "weight": 1.0,
            }
        ]
    }
    calibration_config["customer"] = {
        "name": onboarding.name,
        "asset_kind": onboarding.asset_kind,
        "onboarding_path": str(onboarding_json_path.resolve()),
        "template_matches_path": str(template_matches_path.resolve()),
        "source_model_path": str(Path(model_path).resolve()),
    }
    with (output_dir_path / "config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(calibration_config, handle, sort_keys=False)

    key = jax.random.PRNGKey(seed)
    key_init, key_train, key_report = jax.random.split(key, 3)

    pretrained_model = UniversalDigitalTwin.load(
        model_path,
        config,
        source_dataset.metadata,
    )
    target_model = initialize_target_model_from_pretrained(
        pretrained_model,
        source_dataset.metadata,
        train_dataset.metadata,
        config,
        key_init,
    )

    calibrator = UnitCalibrator(
        target_model,
        calibration_config,
        train_dataset,
        val_dataset,
        options=options or CalibrationOptions(),
        target_system_id=0,
    )
    train_summary = calibrator.calibrate(
        str(output_dir_path),
        key=key_train,
        time_budget_seconds=time_budget_seconds,
    )

    report = generate_customer_validation_report(
        model=calibrator.model,
        trainer=calibrator.trainer,
        onboarding=onboarding,
        template_matches=template_matches,
        calibration_summary=train_summary,
        key=key_report,
    )
    report_json_path = output_dir_path / "validation_report.json"
    report_md_path = output_dir_path / "validation_report.md"
    report_json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report_md_path.write_text(render_validation_report_markdown(report), encoding="utf-8")

    summary = {
        "status": "ok" if train_summary.get("failure_reason") is None else "failed",
        "model_path": str(Path(model_path).resolve()),
        "output_dir": str(output_dir_path.resolve()),
        "config_path": str((output_dir_path / "config.yaml").resolve()),
        "target_system_name": target_name,
        "target_system_config": str(Path(system_config_path).resolve()),
        "target_data_dir": str(Path(data_dir).resolve()),
        "trainable_mode": train_summary["trainable_mode"],
        "tune_normalization": bool(train_summary["tune_normalization"]),
        "tune_physics_params": bool(train_summary["tune_physics_params"]),
        "best_val_loss": train_summary.get("best_val_loss"),
        "epochs_completed": int(train_summary.get("epochs_completed", 0)),
        "steps_completed": int(train_summary.get("steps_completed", 0)),
        "timed_out": bool(train_summary.get("timed_out", False)),
        "training_seconds": train_summary.get("training_seconds"),
        "failure_reason": train_summary.get("failure_reason"),
        "trainable_parameter_count": int(train_summary["trainable_parameter_count"]),
        "parameter_counts": train_summary["parameter_counts"],
        "template_matches_path": str(template_matches_path.resolve()),
        "report_json_path": str(report_json_path.resolve()),
        "report_markdown_path": str(report_md_path.resolve()),
        "target_manifest": {
            "full_dataset": full_target_dataset.manifest(),
            "train_dataset": train_dataset.manifest(),
            "val_dataset": val_dataset.manifest(),
        },
        "template_matching": template_matches.to_dict(),
        "validation_report": report,
        "per_system_val_losses": train_summary["per_system_val_losses"],
    }

    summary_path = Path(summary_path) if summary_path else output_dir_path / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
