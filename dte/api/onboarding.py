"""Helpers for customer-onboarding API flows."""

from __future__ import annotations

import copy
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

import numpy as np
import yaml

from dte.data.multi_system_dataset import MultiSystemTrajectoryDataset, SystemDatasetSource
from dte.customer.onboarding_schema import (
    CustomerMeasurementSpec,
    CustomerOnboardingSpec,
    CustomerSignalSpec,
    CustomerUnitSpec,
)
from dte.customer.template_matching import REGISTERED_UNIT_CONFIGS
from dte.data.real_data import RealDataIngestion
from dte.demo.engine import (
    UniversalDemoRuntime,
    build_signal_sequence,
    default_disturbance_sequence,
    load_demo_config,
    serialize_demo_definition,
    serialize_system_spec,
)
from dte.models.universal_digital_twin import UniversalDigitalTwin
from dte.simulators.base import ProcessUnitSpec
from dte.simulators.registry import get_system_spec

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ONBOARDING_ROOT = PROJECT_ROOT / "outputs" / "customer_jobs"
_JOB_RUNTIME_CACHE: dict[str, UniversalDemoRuntime] = {}


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
        "--system_name",
        str(preview_record["template_system_name"]),
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


def load_completed_job_context(job_id: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Load the completed job, preview, and summary artifacts for one onboarding job."""

    job_payload = load_job_status(job_id)
    if job_payload is None:
        raise FileNotFoundError(f"Job '{job_id}' was not found.")
    if job_payload.get("status") != "completed":
        raise ValueError(f"Job '{job_id}' is not completed.")

    preview_id = str(job_payload.get("preview_id") or "")
    preview_record = load_preview_record(preview_id)
    if preview_record is None:
        raise FileNotFoundError(f"Preview '{preview_id}' for job '{job_id}' was not found.")

    summary = _read_json(job_dir(job_id) / "summary.json")
    if summary is None:
        raise FileNotFoundError(f"Job '{job_id}' does not have a completed summary.")
    return job_payload, preview_record, summary


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _summary_vector(
    summary: dict[str, Any],
    key: str,
    names: list[str],
    fallback: list[float] | np.ndarray,
) -> np.ndarray:
    values = summary.get(key)
    fallback_arr = np.asarray(fallback, dtype=np.float32)
    if values is None:
        return fallback_arr.copy()
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if arr.shape[0] < len(names):
        padded = fallback_arr.copy()
        padded[: arr.shape[0]] = arr
        return padded
    return arr[: len(names)]


def _named_map(names: list[str], values: list[float] | np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    return {
        name: float(arr[idx])
        for idx, name in enumerate(names)
    }


def _constant_profile(names: list[str], values: list[float] | np.ndarray) -> dict[str, Any]:
    return {
        "type": "constant",
        "channels": _named_map(names, values),
    }


def _midpoint_controls(spec: ProcessUnitSpec) -> np.ndarray:
    return np.asarray(
        [0.5 * sum(spec.control_ranges[name]) for name in spec.control_names],
        dtype=np.float32,
    )


def _recenter_named_values(
    values: dict[str, Any],
    template_center: dict[str, float],
    customer_center: dict[str, float],
) -> dict[str, float]:
    recentered: dict[str, float] = {}
    for name, value in values.items():
        template_value = float(template_center.get(name, value))
        customer_value = float(customer_center.get(name, template_value))
        recentered[name] = float(customer_value + (float(value) - template_value))
    return recentered


def _recenter_profile(
    profile: dict[str, Any] | None,
    *,
    template_center: dict[str, float],
    customer_center: dict[str, float],
) -> dict[str, Any] | None:
    if profile is None:
        return None

    recentered = copy.deepcopy(profile)
    for field_name in ("channels", "values", "start", "end", "base", "pulse"):
        raw_values = recentered.get(field_name)
        if isinstance(raw_values, dict):
            recentered[field_name] = _recenter_named_values(
                raw_values,
                template_center,
                customer_center,
            )
    return recentered


def _range_shift(std_value: float, bounds: list[float] | tuple[float, float]) -> float:
    lower, upper = float(bounds[0]), float(bounds[1])
    return max(float(std_value), 0.1 * (upper - lower), 1e-3)


def _build_generic_customer_workspace(
    *,
    job_id: str,
    customer_name: str,
    spec: ProcessUnitSpec,
    preview_summary: dict[str, Any],
    initial_state: dict[str, float],
    target_state: dict[str, float],
    control_center: np.ndarray,
    disturbance_center: np.ndarray,
    editable_control_names: list[str],
    highlight_states: list[str],
) -> dict[str, Any]:
    n_steps = max(12, min(int(preview_summary.get("n_steps_per_trajectory", 25)), 40))
    dt = float(preview_summary.get("dt", 0.1))
    control_std = _summary_vector(
        preview_summary,
        "control_std",
        list(spec.control_names),
        np.zeros(spec.control_dim, dtype=np.float32),
    )
    disturbance_std = _summary_vector(
        preview_summary,
        "disturbance_std",
        list(spec.disturbance_names),
        np.zeros(spec.disturbance_dim, dtype=np.float32),
    )

    candidate_profiles: list[dict[str, Any]] = []
    lead_control = editable_control_names[0] if editable_control_names else spec.control_names[0]
    lead_control_idx = spec.control_names.index(lead_control)
    lead_control_shift = _range_shift(
        float(control_std[lead_control_idx]),
        spec.control_ranges[lead_control],
    )
    control_center_map = _named_map(list(spec.control_names), control_center)
    disturbance_center_map = _named_map(list(spec.disturbance_names), disturbance_center)

    candidate_profiles.append(
        {
            "id": "customer-ramp",
            "title": f"{lead_control} ramp",
            "description": "Ramp the primary customer control around the observed operating mean.",
            "profile": {
                "type": "ramp",
                "start": {
                    lead_control: float(control_center_map[lead_control] - lead_control_shift),
                },
                "end": {
                    lead_control: float(control_center_map[lead_control] + lead_control_shift),
                },
            },
        }
    )

    disturbance_presets: list[dict[str, Any]] = [
        {
            "id": "customer-nominal",
            "title": "Observed nominal",
            "description": "Hold disturbances at the observed mean operating point.",
            "profile": _constant_profile(list(spec.disturbance_names), disturbance_center),
        }
    ]
    if spec.disturbance_names:
        lead_disturbance = spec.disturbance_names[0]
        lead_disturbance_idx = spec.disturbance_names.index(lead_disturbance)
        disturbance_shift = _range_shift(
            float(disturbance_std[lead_disturbance_idx]),
            spec.disturbance_ranges[lead_disturbance],
        )
        disturbance_presets.append(
            {
                "id": "customer-pulse",
                "title": f"{lead_disturbance} pulse",
                "description": "Pulse the lead disturbance around the observed mean.",
                "profile": {
                    "type": "pulse",
                    "base": disturbance_center_map,
                    "pulse": {
                        lead_disturbance: float(
                            disturbance_center_map[lead_disturbance] + disturbance_shift
                        ),
                    },
                    "start_step": max(n_steps // 3, 1),
                    "duration": max(n_steps // 5, 1),
                },
            }
        )

    return {
        "id": f"customer-{job_id}",
        "title": f"{customer_name} Planning Workspace",
        "system": spec.name,
        "kind": "customer_workspace",
        "description": (
            f"Customer-specific planning surface for {customer_name}. "
            "Forecasts come from the adapted checkpoint built from the uploaded plant history."
        ),
        "operator_goal": "Compare plan options against the adapted customer checkpoint.",
        "dt": dt,
        "n_steps": n_steps,
        "highlight_states": highlight_states,
        "target_state": target_state,
        "initial_state": initial_state,
        "baseline_control_profile": _constant_profile(list(spec.control_names), control_center),
        "disturbance_presets": disturbance_presets,
        "candidate_profiles": candidate_profiles,
        "optimization": {
            "n_candidates": 48,
            "seed": 0,
        },
        "run_button_label": "Compare customer plan",
        "optimize_button_label": "Recommend stabilization plan",
        "editable_control_names": editable_control_names,
        "system_spec": serialize_system_spec(spec),
    }


def build_job_workspace(
    job_id: str,
    *,
    demo_config_path: str | Path | None,
    system_configs: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Build a customer planning workspace centered on one completed onboarding job."""

    job_payload, preview_record, summary = load_completed_job_context(job_id)
    template_system_name = str(preview_record["template_system_name"])
    customer_name = str(
        (preview_record.get("onboarding_spec") or {}).get("name")
        or template_system_name
    )
    system_config = system_configs.get(template_system_name)
    if system_config is None:
        system_config = _load_yaml(preview_record["template_config_path"])
    spec = get_system_spec(system_config)

    preview_summary = preview_record.get("ingestion_summary") or {}
    demo_config = load_demo_config(demo_config_path)
    template_demo = next(
        (
            item
            for item in demo_config.get("demos", [])
            if str(item.get("system")) == template_system_name
        ),
        None,
    )
    template_workspace = (
        serialize_demo_definition(template_demo, spec)
        if template_demo is not None
        else None
    )

    template_control_center = (
        np.mean(
            build_signal_sequence(
                spec,
                int(template_workspace["n_steps"]),
                signal_kind="control",
                profile=template_workspace.get("baseline_control_profile"),
            ),
            axis=0,
        )
        if template_workspace is not None
        else _midpoint_controls(spec)
    )
    template_disturbance_center = (
        np.mean(
            default_disturbance_sequence(spec, int(template_workspace["n_steps"])),
            axis=0,
        )
        if template_workspace is not None
        else np.asarray(spec.default_nominal_disturbance, dtype=np.float32)
    )
    state_center = _summary_vector(
        preview_summary,
        "state_mean",
        list(spec.state_names),
        spec.default_initial_state,
    )
    control_center = _summary_vector(
        preview_summary,
        "control_mean",
        list(spec.control_names),
        template_control_center,
    )
    disturbance_center = _summary_vector(
        preview_summary,
        "disturbance_mean",
        list(spec.disturbance_names),
        template_disturbance_center,
    )
    initial_state = _named_map(list(spec.state_names), state_center)
    editable_control_names = [
        name
        for name in preview_record.get("control_variable_names") or []
        if name in spec.control_names
    ] or list(spec.control_names)
    highlight_states = [
        name
        for name in preview_record.get("objective_state_names") or []
        if name in spec.state_names
    ]
    if not highlight_states:
        highlight_states = (
            list(template_workspace["highlight_states"])
            if template_workspace is not None
            else list(spec.state_names[: min(2, spec.state_dim)])
        )

    if template_workspace is None:
        target_state = _named_map(list(spec.state_names), spec.default_initial_state)
        return build_job_workspace_response(
            job_payload,
            preview_record,
            summary,
            _build_generic_customer_workspace(
                job_id=job_id,
                customer_name=customer_name,
                spec=spec,
                preview_summary=preview_summary,
                initial_state=initial_state,
                target_state=target_state,
                control_center=control_center,
                disturbance_center=disturbance_center,
                editable_control_names=editable_control_names,
                highlight_states=highlight_states,
            ),
        )

    template_initial = template_workspace["initial_state"]
    template_target = template_workspace["target_state"]
    target_state = {
        name: float(
            initial_state[name]
            + (
                float(template_target.get(name, template_initial[name]))
                - float(template_initial[name])
            )
        )
        for name in spec.state_names
    }

    template_control_center_map = _named_map(list(spec.control_names), template_control_center)
    customer_control_center_map = _named_map(list(spec.control_names), control_center)
    template_disturbance_center_map = _named_map(
        list(spec.disturbance_names),
        template_disturbance_center,
    )
    customer_disturbance_center_map = _named_map(
        list(spec.disturbance_names),
        disturbance_center,
    )

    workspace = copy.deepcopy(template_workspace)
    workspace.update(
        {
            "id": f"customer-{job_id}",
            "title": f"{customer_name} Planning Workspace",
            "kind": "customer_workspace",
            "description": (
                f"Customer-specific planning surface for {customer_name}. "
                "Forecasts come from the adapted checkpoint built from the uploaded plant history."
            ),
            "initial_state": initial_state,
            "target_state": target_state,
            "highlight_states": highlight_states,
            "baseline_control_profile": _constant_profile(list(spec.control_names), control_center),
            "disturbance_presets": [
                {
                    **preset,
                    "profile": _recenter_profile(
                        preset.get("profile"),
                        template_center=template_disturbance_center_map,
                        customer_center=customer_disturbance_center_map,
                    ),
                }
                for preset in template_workspace["disturbance_presets"]
            ],
            "candidate_profiles": [
                {
                    **preset,
                    "profile": _recenter_profile(
                        preset.get("profile"),
                        template_center=template_control_center_map,
                        customer_center=customer_control_center_map,
                    ),
                }
                for preset in template_workspace["candidate_profiles"]
            ],
            "editable_control_names": editable_control_names,
        }
    )
    return build_job_workspace_response(job_payload, preview_record, summary, workspace)


def build_job_workspace_gate(
    job_payload: dict[str, Any],
    preview_record: dict[str, Any],
    summary: dict[str, Any],
    workspace: dict[str, Any],
) -> dict[str, Any]:
    """Classify the soft-gate state for the customer planning workspace."""

    preview_summary = preview_record.get("ingestion_summary") or {}
    metrics = job_payload.get("metrics") or {}
    summary_status = str(summary.get("status") or "").strip().lower()
    state_std = np.asarray(preview_summary.get("state_std") or [], dtype=np.float32).reshape(-1)
    state_names = list(workspace["system_spec"]["state_names"])
    highlight_states = [
        name for name in workspace.get("highlight_states", []) if name in state_names
    ]
    highlight_indices = [state_names.index(name) for name in highlight_states]

    if not highlight_indices or state_std.shape[0] < len(state_names):
        return {
            "status": "warning",
            "message": (
                "Preview variability is incomplete for the selected objective states. "
                "Review the validation report before trusting the planning outputs."
            ),
        }

    mean_highlight_std = float(np.mean(state_std[highlight_indices]))
    if not np.isfinite(mean_highlight_std) or mean_highlight_std <= 0.0:
        return {
            "status": "warning",
            "message": (
                "Preview variability could not be estimated for the selected objective states. "
                "Use the workspace as a diagnostic aid, not a production recommendation."
            ),
        }

    if summary_status and summary_status not in {"ok", "completed", "success"}:
        return {
            "status": "warning",
            "message": (
                "The adaptation summary indicates this job did not complete cleanly. "
                "Use the workspace for investigation, not operational decisions."
            ),
        }

    forecast_rmse = metrics.get("forecast_rmse")
    rollout_rmse = metrics.get("rollout_rmse")
    if forecast_rmse is None or rollout_rmse is None:
        return {
            "status": "warning",
            "message": (
                "Fit metrics are incomplete for this job. The workspace is available, "
                "but forecasts should be reviewed alongside the validation report."
            ),
        }

    forecast_ratio = float(forecast_rmse) / mean_highlight_std
    rollout_ratio = float(rollout_rmse) / mean_highlight_std
    if forecast_ratio > 1.0 or rollout_ratio > 1.0:
        return {
            "status": "warning",
            "message": (
                "The adapted model error is larger than the preview variability band for "
                "the selected objective states. Use this workspace for inspection, not direct action."
            ),
            "forecast_rmse_ratio": forecast_ratio,
            "rollout_rmse_ratio": rollout_ratio,
        }

    return {
        "status": "ready",
        "message": (
            "The adapted model fit is within the preview variability band for the selected "
            "objective states. Compare plans here before escalating to operations."
        ),
        "forecast_rmse_ratio": forecast_ratio,
        "rollout_rmse_ratio": rollout_ratio,
    }


def build_job_workspace_response(
    job_payload: dict[str, Any],
    preview_record: dict[str, Any],
    summary: dict[str, Any],
    workspace: dict[str, Any],
) -> dict[str, Any]:
    return {
        "job": job_payload,
        "gate": build_job_workspace_gate(job_payload, preview_record, summary, workspace),
        "workspace": workspace,
    }


def load_job_demo_runtime(job_id: str) -> UniversalDemoRuntime:
    """Load or reuse a cached adapted runtime for one completed onboarding job."""

    runtime = _JOB_RUNTIME_CACHE.get(job_id)
    if runtime is not None:
        return runtime

    _job_payload, preview_record, summary = load_completed_job_context(job_id)
    adaptation_dir = job_dir(job_id) / "adaptation"
    model_path = adaptation_dir / "best_model.eqx"
    if not model_path.exists():
        model_path = adaptation_dir / "final_model.eqx"
    if not model_path.exists():
        raise FileNotFoundError(f"Job '{job_id}' does not have an adapted model checkpoint.")

    config_path = adaptation_dir / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Job '{job_id}' does not have an adaptation config.")
    config = _load_yaml(config_path)
    sources = [
        SystemDatasetSource(
            name=str(item["name"]),
            system_config=str(item["system_config"]),
            data_dir=str(item["data_dir"]),
            weight=float(item.get("weight", 1.0)),
        )
        for item in config.get("data", {}).get("systems", [])
    ]
    if not sources:
        raise ValueError(f"Job '{job_id}' adaptation config does not define data.systems.")

    metadata = MultiSystemTrajectoryDataset.metadata_from_sources(sources)
    model = UniversalDigitalTwin.load(str(model_path), config, metadata)
    system_ids = {
        name: idx for idx, name in enumerate(metadata.system_names)
    }
    if len(metadata.system_names) == 1:
        template_system_name = str(preview_record["template_system_name"])
        target_system_name = str(summary.get("target_system_name") or template_system_name)
        system_ids[template_system_name] = 0
        system_ids[target_system_name] = 0

    runtime = UniversalDemoRuntime(
        model=model,
        system_ids=system_ids,
        model_path=str(model_path),
        config_path=str(config_path),
    )
    _JOB_RUNTIME_CACHE[job_id] = runtime
    return runtime
