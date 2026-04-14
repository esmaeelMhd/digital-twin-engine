"""Helpers for customer-onboarding API flows."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import uuid
import warnings
from pathlib import Path
from typing import Any

import yaml

from dte.customer.onboarding_schema import (
    CustomerMeasurementSpec,
    CustomerOnboardingSpec,
    CustomerSignalSpec,
    CustomerUnitSpec,
)
from dte.customer.template_matching import REGISTERED_UNIT_CONFIGS
from dte.data.real_data import RealDataIngestion
from dte.demo.engine import load_demo_config, serialize_system_spec
from dte.simulators.base import ProcessUnitSpec
from dte.simulators.registry import get_system_spec

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ONBOARDING_ROOT = PROJECT_ROOT / "outputs" / "customer_jobs"


def onboarding_root() -> Path:
    return Path(os.environ.get("DTE_ONBOARDING_ROOT", str(DEFAULT_ONBOARDING_ROOT)))


def uploads_root() -> Path:
    path = onboarding_root() / "uploads"
    path.mkdir(parents=True, exist_ok=True)
    return path


def previews_root() -> Path:
    path = onboarding_root() / "previews"
    path.mkdir(parents=True, exist_ok=True)
    return path


def jobs_root() -> Path:
    path = onboarding_root() / "jobs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def upload_dir(upload_id: str) -> Path:
    return uploads_root() / upload_id


def preview_dir(preview_id: str) -> Path:
    return previews_root() / preview_id


def job_dir(job_id: str) -> Path:
    return jobs_root() / job_id


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "tolist"):
        return _json_safe(value.tolist())
    if isinstance(value, Path):
        return str(value)
    return value


def _resolve_path(raw_path: str | None, *, anchor: Path | None = None) -> Path | None:
    if raw_path is None:
        return None
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate
    if anchor is not None:
        anchored = anchor.parent / candidate
        if anchored.exists():
            return anchored
    return PROJECT_ROOT / candidate


def _load_upload_frame(upload_path: Path, detected_format: str):
    import pandas as pd

    if detected_format == "parquet":
        return pd.read_parquet(upload_path)
    return pd.read_csv(upload_path)


def persist_upload(filename: str, content: bytes) -> dict[str, Any]:
    suffix = Path(filename).suffix.lower()
    if suffix not in {".csv", ".parquet"}:
        raise ValueError("Only .csv and .parquet uploads are supported.")

    upload_id = new_id("upl")
    directory = upload_dir(upload_id)
    directory.mkdir(parents=True, exist_ok=True)

    detected_format = "parquet" if suffix == ".parquet" else "csv"
    source_path = directory / f"source{suffix}"
    source_path.write_bytes(content)

    frame = _load_upload_frame(source_path, detected_format)
    metadata = {
        "upload_id": upload_id,
        "filename": filename,
        "detected_format": detected_format,
        "columns": [str(column) for column in frame.columns.tolist()],
        "row_count": int(len(frame)),
        "size_bytes": int(source_path.stat().st_size),
        "path": str(source_path.resolve()),
        "created_at": time.time(),
    }
    _write_json(directory / "metadata.json", metadata)
    return metadata


def load_upload_metadata(upload_id: str) -> dict[str, Any] | None:
    return _read_json(upload_dir(upload_id) / "metadata.json")


def load_preview_record(preview_id: str) -> dict[str, Any] | None:
    return _read_json(preview_dir(preview_id) / "preview.json")


def load_job_status(job_id: str) -> dict[str, Any] | None:
    return _read_json(job_dir(job_id) / "status.json")


def update_job_status(job_id: str, **updates: Any) -> dict[str, Any]:
    status_path = job_dir(job_id) / "status.json"
    payload = _read_json(status_path) or {}
    payload.update(_json_safe(updates))
    payload["updated_at"] = time.time()
    _write_json(status_path, payload)
    return payload


def build_onboarding_templates(
    system_configs: dict[str, dict[str, Any]],
    *,
    demo_config: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    demo_config = demo_config or {}
    demos_by_system = {
        str(item.get("system")): item
        for item in demo_config.get("demos", [])
        if item.get("system")
    }
    templates: list[dict[str, Any]] = []
    for template_id, config_path in REGISTERED_UNIT_CONFIGS.items():
        system_config = system_configs.get(template_id)
        if system_config is None and config_path.exists():
            with config_path.open("r", encoding="utf-8") as handle:
                system_config = yaml.safe_load(handle) or {}
        if system_config is None:
            continue

        spec = get_system_spec(system_config)
        demo = demos_by_system.get(spec.name, {})
        templates.append(
            {
                "id": spec.name,
                "title": str(demo.get("title", spec.name.replace("_", " ").title())),
                "description": str(
                    demo.get(
                        "description",
                        f"{spec.family.replace('_', ' ').title()} pilot onboarding template.",
                    )
                ),
                "system_spec": serialize_system_spec(spec),
                "suggested_objectives": list(demo.get("highlight_states", spec.state_names)),
                "suggested_controls": list(
                    demo.get("candidate_profiles", [{}])[0].get("profile", {}).get("channels", {}).keys()
                )
                or list(spec.control_names),
            }
        )
    return templates


def _build_signal_specs(
    channels: list[Any],
    *,
    unit_name: str,
) -> list[dict[str, Any]]:
    payload = []
    for channel in channels:
        payload.append(
            {
                "name": str(channel.name),
                "role": str(channel.role or "generic"),
                "unit_name": unit_name,
                "unit": channel.unit,
                "lower_bound": channel.lower_bound,
                "upper_bound": channel.upper_bound,
                "description": channel.description,
            }
        )
    return payload


def _build_measurement_specs(
    channels: list[Any],
    *,
    unit_name: str,
) -> list[dict[str, Any]]:
    payload = []
    for channel in channels:
        payload.append(
            {
                "name": str(channel.name),
                "role": str(channel.role or "generic"),
                "unit_name": unit_name,
                "unit": channel.unit,
                "lower_bound": channel.lower_bound,
                "upper_bound": channel.upper_bound,
                "description": channel.description,
                "source": "state",
            }
        )
    return payload


def _validate_mapping(
    *,
    available_columns: set[str],
    expected_names: list[str],
    provided_mapping: dict[str, str],
    label: str,
    blocking_errors: list[str],
) -> list[str]:
    ordered_columns: list[str] = []
    missing_expected = [name for name in expected_names if name not in provided_mapping]
    if missing_expected:
        blocking_errors.append(f"Missing {label} mapping entries for: {missing_expected}.")
    for expected_name in expected_names:
        mapped_column = provided_mapping.get(expected_name)
        if not mapped_column:
            continue
        if mapped_column not in available_columns:
            blocking_errors.append(
                f"Mapped {label} column '{mapped_column}' for '{expected_name}' was not found in the upload."
            )
        ordered_columns.append(mapped_column)
    return ordered_columns


def run_onboarding_preview(
    payload: dict[str, Any],
    *,
    system_configs: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    upload_metadata = load_upload_metadata(str(payload["upload_id"]))
    if upload_metadata is None:
        raise FileNotFoundError(f"Upload '{payload['upload_id']}' was not found.")

    template_id = str(payload["template_id"])
    template_path = REGISTERED_UNIT_CONFIGS.get(template_id)
    if template_path is None:
        raise ValueError(f"Unsupported template '{template_id}'.")

    system_config = system_configs.get(template_id)
    if system_config is None:
        with template_path.open("r", encoding="utf-8") as handle:
            system_config = yaml.safe_load(handle) or {}
    spec = get_system_spec(system_config)

    available_columns = set(upload_metadata["columns"])
    blocking_errors: list[str] = []
    warnings_list: list[str] = []

    state_columns = _validate_mapping(
        available_columns=available_columns,
        expected_names=list(spec.state_names),
        provided_mapping=dict(payload.get("state_column_map") or {}),
        label="state",
        blocking_errors=blocking_errors,
    )
    control_columns = _validate_mapping(
        available_columns=available_columns,
        expected_names=list(spec.control_names),
        provided_mapping=dict(payload.get("control_column_map") or {}),
        label="control",
        blocking_errors=blocking_errors,
    )
    disturbance_columns = _validate_mapping(
        available_columns=available_columns,
        expected_names=list(spec.disturbance_names),
        provided_mapping=dict(payload.get("disturbance_column_map") or {}),
        label="disturbance",
        blocking_errors=blocking_errors,
    )

    timestamp_column = payload.get("timestamp_column")
    if timestamp_column and timestamp_column not in available_columns:
        blocking_errors.append(
            f"Timestamp column '{timestamp_column}' was not found in the upload."
        )

    objective_state_names = [str(name) for name in payload.get("objective_state_names") or []]
    invalid_objectives = sorted(set(objective_state_names) - set(spec.state_names))
    if invalid_objectives:
        blocking_errors.append(
            f"Objective states must be drawn from {list(spec.state_names)}; got {invalid_objectives}."
        )
    if not objective_state_names:
        blocking_errors.append("Select at least one objective state.")

    control_variable_names = [str(name) for name in payload.get("control_variable_names") or []]
    invalid_controls = sorted(set(control_variable_names) - set(spec.control_names))
    if invalid_controls:
        blocking_errors.append(
            f"Control variables must be drawn from {list(spec.control_names)}; got {invalid_controls}."
        )
    if not control_variable_names:
        blocking_errors.append("Select at least one control variable.")

    if blocking_errors:
        return {
            "preview_id": None,
            "upload_id": str(payload["upload_id"]),
            "template_id": template_id,
            "valid": False,
            "blocking_errors": blocking_errors,
            "warnings": warnings_list,
            "ingestion_summary": None,
            "onboarding_spec": None,
            "objective_state_names": objective_state_names,
            "control_variable_names": control_variable_names,
        }

    preview_id = new_id("prv")
    directory = preview_dir(preview_id)
    processed_dir = directory / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    output_path = processed_dir / "train_data.h5"
    unit_name = f"{template_id}_unit"

    onboarding_spec = CustomerOnboardingSpec(
        name=str(payload["customer_name"]).strip(),
        asset_kind="unit",
        units=[
            CustomerUnitSpec(
                name=unit_name,
                family=getattr(spec, "family", None),
                subtype=getattr(spec, "subtype", None),
                unit_type=getattr(spec, "unit_type", None),
                controls=list(spec.control_names),
                disturbances=list(spec.disturbance_names),
                measurements=list(spec.state_names),
                known_laws=list(getattr(spec, "law_tags", ())),
            )
        ],
        controls=[
            CustomerSignalSpec.model_validate(item)
            for item in _build_signal_specs(
                list(getattr(spec, "control_channels", [])),
                unit_name=unit_name,
            )
        ],
        disturbances=[
            CustomerSignalSpec.model_validate(item)
            for item in _build_signal_specs(
                list(getattr(spec, "disturbance_channels", [])),
                unit_name=unit_name,
            )
        ],
        measurements=[
            CustomerMeasurementSpec.model_validate(item)
            for item in _build_measurement_specs(
                list(getattr(spec, "state_channels", [])),
                unit_name=unit_name,
            )
        ],
        known_laws=list(getattr(spec, "law_tags", ())),
    )

    ingestor = RealDataIngestion(
        spec=spec,
        state_columns=state_columns,
        control_columns=control_columns,
        disturbance_columns=disturbance_columns,
        timestamp_column=str(timestamp_column) if timestamp_column else "timestamp",
        dt=float(payload.get("dt", 0.1)),
        max_gap_fill=float(payload.get("max_gap_fill", 10.0)),
        outlier_sigma=float(payload.get("outlier_sigma", 5.0)),
        drop_large_gaps=bool(payload.get("drop_large_gaps", False)),
    )

    source_path = Path(upload_metadata["path"])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if upload_metadata["detected_format"] == "parquet":
            summary = ingestor.ingest_parquet(
                source_path,
                output_path,
                trajectory_duration=float(payload.get("trajectory_duration", 100.0)),
                trajectory_stride=float(payload.get("trajectory_stride", 10.0)),
            )
        else:
            summary = ingestor.ingest_csv(
                source_path,
                output_path,
                trajectory_duration=float(payload.get("trajectory_duration", 100.0)),
                trajectory_stride=float(payload.get("trajectory_stride", 10.0)),
            )
    warnings_list = [str(item.message) for item in caught]

    onboarding_json_path = directory / "onboarding.json"
    onboarding_json_path.write_text(
        json.dumps(onboarding_spec.model_dump(mode="json"), indent=2),
        encoding="utf-8",
    )

    preview_record = {
        "preview_id": preview_id,
        "upload_id": str(payload["upload_id"]),
        "template_id": template_id,
        "template_config_path": str(template_path.resolve()),
        "template_system_name": spec.name,
        "valid": True,
        "blocking_errors": [],
        "warnings": warnings_list,
        "ingestion_summary": _json_safe(summary),
        "onboarding_spec": onboarding_spec.model_dump(mode="json"),
        "objective_state_names": objective_state_names,
        "control_variable_names": control_variable_names,
        "column_maps": {
            "state": dict(payload.get("state_column_map") or {}),
            "control": dict(payload.get("control_column_map") or {}),
            "disturbance": dict(payload.get("disturbance_column_map") or {}),
        },
        "timestamp_column": str(timestamp_column) if timestamp_column else None,
        "artifacts": {
            "uploaded_file": str(source_path.resolve()),
            "onboarding_json": str(onboarding_json_path.resolve()),
            "processed_data_dir": str(processed_dir.resolve()),
            "processed_data_path": str(output_path.resolve()),
        },
        "created_at": time.time(),
    }
    _write_json(directory / "preview.json", preview_record)
    _write_json(directory / "summary.json", preview_record)
    return preview_record


def _resolve_default_runtime_from_demo(demo_config_path: str | Path | None) -> tuple[Path | None, Path | None]:
    config = load_demo_config(demo_config_path)
    anchor = Path(demo_config_path) if demo_config_path is not None else PROJECT_ROOT / "configs" / "demo_app.yaml"
    runtime_cfg = config.get("runtime", {})
    model_path = _resolve_path(runtime_cfg.get("model_path"), anchor=anchor)
    config_path = _resolve_path(runtime_cfg.get("config_path"), anchor=anchor)
    return model_path, config_path


def resolve_adaptation_runtime(
    *,
    requested_model_path: str | None,
    requested_config_path: str | None,
    demo_config_path: str | Path | None,
) -> tuple[Path | None, Path | None]:
    model_path = _resolve_path(requested_model_path, anchor=PROJECT_ROOT) if requested_model_path else None
    config_path = _resolve_path(requested_config_path, anchor=PROJECT_ROOT) if requested_config_path else None
    if model_path is not None and config_path is not None:
        return model_path, config_path
    default_model_path, default_config_path = _resolve_default_runtime_from_demo(demo_config_path)
    return model_path or default_model_path, config_path or default_config_path


def initialize_job_status(
    *,
    job_id: str,
    preview_id: str,
    artifacts: dict[str, Any],
) -> dict[str, Any]:
    now = time.time()
    payload = {
        "job_id": job_id,
        "preview_id": preview_id,
        "status": "queued",
        "stage": "pending",
        "progress_message": "Queued for customer adaptation.",
        "created_at": now,
        "updated_at": now,
        "artifacts": _json_safe(artifacts),
        "metrics": {},
        "error": None,
    }
    _write_json(job_dir(job_id) / "status.json", payload)
    return payload


def _extract_job_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    report = summary.get("validation_report") or {}
    template_matches = summary.get("template_matching") or {}
    return {
        "best_val_loss": summary.get("best_val_loss"),
        "forecast_rmse": (report.get("forecast_metrics") or {}).get("rmse"),
        "rollout_rmse": (report.get("rollout_metrics") or {}).get("rmse"),
        "best_unit_template": (template_matches.get("unit_matches") or {}),
    }


def run_onboarding_job(
    *,
    job_id: str,
    preview_record: dict[str, Any],
    request_payload: dict[str, Any],
    demo_config_path: str | Path | None,
) -> dict[str, Any]:
    directory = job_dir(job_id)
    directory.mkdir(parents=True, exist_ok=True)
    logs_dir = directory / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    output_dir = directory / "adaptation"
    summary_path = directory / "summary.json"
    report_json_path = directory / "validation_report.json"
    report_md_path = directory / "validation_report.md"
    log_path = logs_dir / "adapt_customer.log"

    model_path, config_path = resolve_adaptation_runtime(
        requested_model_path=request_payload.get("model_path"),
        requested_config_path=request_payload.get("config_path"),
        demo_config_path=demo_config_path,
    )
    if model_path is None or not model_path.exists():
        raise FileNotFoundError("Could not resolve a valid universal model_path for adaptation.")
    if config_path is None or not config_path.exists():
        raise FileNotFoundError("Could not resolve a valid universal config_path for adaptation.")

    update_job_status(
        job_id,
        status="running",
        stage="adaptation",
        progress_message="Running customer adaptation against the uploaded unit data.",
        artifacts={
            **dict(load_job_status(job_id).get("artifacts", {}) if load_job_status(job_id) else {}),
            "summary_json": str(summary_path.resolve()),
            "report_json": str(report_json_path.resolve()),
            "report_markdown": str(report_md_path.resolve()),
            "log_path": str(log_path.resolve()),
        },
    )

    command = [
        sys.executable,
        "scripts/adapt_customer.py",
        "--onboarding",
        str(preview_record["artifacts"]["onboarding_json"]),
        "--model_path",
        str(model_path),
        "--config",
        str(config_path),
        "--system_config",
        str(preview_record["template_config_path"]),
        "--data_dir",
        str(preview_record["artifacts"]["processed_data_dir"]),
        "--output_dir",
        str(output_dir),
        "--trainable_mode",
        str(request_payload.get("trainable_mode", "adapters")),
        "--seed",
        str(int(request_payload.get("seed", 42))),
        "--summary_path",
        str(summary_path),
    ]
    if bool(request_payload.get("tune_normalization", True)):
        command.append("--tune_normalization")
    if bool(request_payload.get("tune_physics_params", False)):
        command.append("--tune_physics_params")
    if request_payload.get("param_indices"):
        command.extend(
            ["--param_indices", ",".join(str(int(item)) for item in request_payload["param_indices"])]
        )
    if request_payload.get("time_budget_minutes") is not None:
        command.extend(
            [
                "--time_budget_minutes",
                str(float(request_payload["time_budget_minutes"])),
            ]
        )

    with log_path.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )

    if completed.returncode != 0:
        update_job_status(
            job_id,
            status="failed",
            stage="adaptation",
            progress_message="Customer adaptation failed.",
            error=f"adapt_customer.py exited with code {completed.returncode}",
        )
        return load_job_status(job_id) or {}

    summary = _read_json(summary_path)
    if summary is None:
        update_job_status(
            job_id,
            status="failed",
            stage="adaptation",
            progress_message="Customer adaptation finished without a summary artifact.",
            error="summary.json was not produced by adapt_customer.py",
        )
        return load_job_status(job_id) or {}

    summary_report_json = summary.get("report_json_path")
    summary_report_md = summary.get("report_markdown_path")
    if summary_report_json and Path(summary_report_json).exists():
        shutil.copy2(summary_report_json, report_json_path)
    if summary_report_md and Path(summary_report_md).exists():
        shutil.copy2(summary_report_md, report_md_path)

    metrics = _extract_job_metrics(summary)
    if isinstance(metrics.get("best_unit_template"), dict):
        first_key = next(iter(metrics["best_unit_template"].values()), [])
        best_match = first_key[0]["name"] if first_key else None
        metrics["best_unit_template"] = best_match

    update_job_status(
        job_id,
        status="completed",
        stage="completed",
        progress_message="Customer adaptation completed.",
        metrics=metrics,
        error=None,
    )
    return load_job_status(job_id) or {}


def load_job_report(job_id: str) -> tuple[dict[str, Any], str | None]:
    directory = job_dir(job_id)
    summary = _read_json(directory / "summary.json")
    if summary is None:
        raise FileNotFoundError(f"Job '{job_id}' does not have a completed summary.")
    report_markdown = None
    report_path = directory / "validation_report.md"
    if report_path.exists():
        report_markdown = report_path.read_text(encoding="utf-8")
    elif summary.get("report_markdown_path"):
        candidate = Path(summary["report_markdown_path"])
        if candidate.exists():
            report_markdown = candidate.read_text(encoding="utf-8")
    return summary, report_markdown
