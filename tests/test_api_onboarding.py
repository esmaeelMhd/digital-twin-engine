"""Tests for customer onboarding API routes."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml
from fastapi.testclient import TestClient

from dte.api import onboarding, service

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _demo_env(monkeypatch):
    monkeypatch.setenv(
        "DTE_SYSTEM_CONFIG",
        ",".join(
            [
                str(PROJECT_ROOT / "configs" / "cstr_default.yaml"),
                str(PROJECT_ROOT / "configs" / "heat_exchanger_default.yaml"),
                str(PROJECT_ROOT / "configs" / "two_tank_default.yaml"),
            ]
        ),
    )
    monkeypatch.setenv("DTE_MODEL_PATH", str(PROJECT_ROOT / "outputs" / "definitely_missing.eqx"))
    monkeypatch.setenv("DTE_TRAINING_CONFIG", str(PROJECT_ROOT / "configs" / "training_default.yaml"))
    monkeypatch.setenv("DTE_DEMO_CONFIG", str(PROJECT_ROOT / "configs" / "demo_app.yaml"))


def _clear_service_state():
    service._models.clear()
    service._specs.clear()
    service._system_configs.clear()
    service._universal_runtime = None


def _write_historian_csv(path: Path) -> None:
    rows = ["timestamp,Ca,Cb,T,Tc,F_in,Tc_in,Ca_in,T_in"]
    for step in range(40):
        timestamp = step * 0.1
        rows.append(
            ",".join(
                [
                    f"{timestamp:.2f}",
                    f"{0.45 + 0.001 * step:.6f}",
                    f"{0.55 + 0.0015 * step:.6f}",
                    f"{345.0 + 0.05 * step:.6f}",
                    f"{301.0 + 0.03 * step:.6f}",
                    "55.0",
                    "302.0",
                    "1.25",
                    "320.0",
                ]
            )
        )
    path.write_text("\n".join(rows), encoding="utf-8")


def _load_system(name: str):
    with (PROJECT_ROOT / "configs" / f"{name}_default.yaml").open("r", encoding="utf-8") as handle:
        system_config = yaml.safe_load(handle)
    spec = service.get_system_spec(system_config)
    simulator = service.get_simulator(name, system_config)
    return spec, simulator


def _create_completed_job(
    client: TestClient,
    monkeypatch,
    tmp_path: Path,
    *,
    summary_overrides: dict | None = None,
    status_metrics: dict | None = None,
) -> tuple[dict, dict, dict]:
    model_path = tmp_path / "model.eqx"
    config_path = tmp_path / "config.yaml"
    model_path.write_text("placeholder", encoding="utf-8")
    config_path.write_text("training: {}\n", encoding="utf-8")

    default_metrics = {
        "best_val_loss": 0.0123,
        "forecast_rmse": 1e-4,
        "rollout_rmse": 1e-4,
        "best_unit_template": "cstr",
    }
    if status_metrics is not None:
        default_metrics.update(status_metrics)

    def _fake_start(job_id: str, preview_record: dict, request_payload: dict, demo_config_path: str):
        job_directory = service.job_dir(job_id)
        job_directory.mkdir(parents=True, exist_ok=True)
        summary = {
            "status": "ok",
            "best_val_loss": default_metrics.get("best_val_loss"),
            "template_matching": {
                "unit_matches": {
                    "cstr_unit": [
                        {"name": "cstr", "score": 0.98},
                    ]
                }
            },
            "validation_report": {
                "forecast_metrics": {"rmse": default_metrics.get("forecast_rmse")},
                "rollout_metrics": {"rmse": default_metrics.get("rollout_rmse")},
            },
            "report_json_path": str((job_directory / "validation_report.json").resolve()),
            "report_markdown_path": str((job_directory / "validation_report.md").resolve()),
        }
        if summary_overrides:
            summary.update(summary_overrides)
        (job_directory / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
        (job_directory / "validation_report.json").write_text(
            json.dumps({"report_version": "test"}),
            encoding="utf-8",
        )
        (job_directory / "validation_report.md").write_text(
            "# Customer Validation Report\n\nAll checks passed.",
            encoding="utf-8",
        )
        service.update_job_status(
            job_id,
            status="completed",
            stage="completed",
            progress_message="Customer adaptation completed.",
            metrics=default_metrics,
            error=None,
        )

    monkeypatch.setattr(service, "_start_onboarding_job", _fake_start)

    historian_path = tmp_path / "historian.csv"
    _write_historian_csv(historian_path)

    upload_response = client.post(
        "/onboarding/uploads",
        files={
            "file": (
                historian_path.name,
                historian_path.read_bytes(),
                "text/csv",
            )
        },
    )
    assert upload_response.status_code == 200
    upload_json = upload_response.json()

    preview_response = client.post(
        "/onboarding/preview",
        json={
            "upload_id": upload_json["upload_id"],
            "template_id": "cstr",
            "customer_name": "Reactor Pilot",
            "timestamp_column": "timestamp",
            "dt": 0.1,
            "trajectory_duration": 1.0,
            "trajectory_stride": 0.5,
            "state_column_map": {"Ca": "Ca", "Cb": "Cb", "T": "T", "Tc": "Tc"},
            "control_column_map": {"F_in": "F_in", "Tc_in": "Tc_in"},
            "disturbance_column_map": {"Ca_in": "Ca_in", "T_in": "T_in"},
            "objective_state_names": ["Cb", "T", "Tc"],
            "control_variable_names": ["F_in", "Tc_in"],
        },
    )
    assert preview_response.status_code == 200
    preview_json = preview_response.json()
    assert preview_json["valid"] is True

    create_job_response = client.post(
        "/onboarding/jobs",
        json={
            "preview_id": preview_json["preview_id"],
            "model_path": str(model_path),
            "config_path": str(config_path),
            "trainable_mode": "adapters",
            "time_budget_minutes": 5,
        },
    )
    assert create_job_response.status_code == 200
    job_json = create_job_response.json()
    return upload_json, preview_json, job_json


def test_onboarding_routes_cover_templates_upload_preview_and_report(monkeypatch, tmp_path: Path):
    _demo_env(monkeypatch)
    monkeypatch.setenv("DTE_DISABLE_UNIVERSAL_RUNTIME", "1")
    monkeypatch.setenv("DTE_ONBOARDING_ROOT", str(tmp_path / "customer_jobs"))
    _clear_service_state()

    with TestClient(service.app) as client:
        templates_response = client.get("/onboarding/templates")
        assert templates_response.status_code == 200
        templates_json = templates_response.json()
        assert len(templates_json["templates"]) == 3
        assert templates_json["templates"][0]["system_spec"]["state_names"]

        upload_json, preview_json, job_json = _create_completed_job(client, monkeypatch, tmp_path)
        assert upload_json["detected_format"] == "csv"
        assert "Ca" in upload_json["columns"]
        assert preview_json["valid"] is True
        assert preview_json["preview_id"] is not None
        assert preview_json["ingestion_summary"]["n_trajectories"] > 0
        assert preview_json["onboarding_spec"]["asset_kind"] == "unit"
        assert job_json["status"] == "queued"

        status_response = client.get(f"/onboarding/jobs/{job_json['job_id']}")
        assert status_response.status_code == 200
        status_json = status_response.json()
        assert status_json["status"] == "completed"
        assert status_json["metrics"]["best_unit_template"] == "cstr"
        assert status_json["metrics"]["forecast_rmse"] == 1e-4

        report_response = client.get(f"/onboarding/jobs/{job_json['job_id']}/report")
        assert report_response.status_code == 200
        report_json = report_response.json()
        assert report_json["report_markdown"].startswith("# Customer Validation Report")
        assert report_json["summary"]["best_val_loss"] == 0.0123

        workspace_response = client.get(f"/onboarding/jobs/{job_json['job_id']}/workspace")
        assert workspace_response.status_code == 200
        workspace_json = workspace_response.json()
        assert workspace_json["job"]["job_id"] == job_json["job_id"]
        assert workspace_json["gate"]["status"] == "ready"
        assert workspace_json["workspace"]["highlight_states"] == ["Cb", "T", "Tc"]
        assert workspace_json["workspace"]["editable_control_names"] == ["F_in", "Tc_in"]
        assert workspace_json["workspace"]["baseline_control_profile"]["channels"] == {
            "F_in": 55.0,
            "Tc_in": 302.0,
        }
        assert workspace_json["workspace"]["initial_state"]["Cb"] > 0.0
        assert (
            workspace_json["workspace"]["target_state"]["Cb"]
            > workspace_json["workspace"]["initial_state"]["Cb"]
        )
        assert (
            workspace_json["workspace"]["target_state"]["T"]
            < workspace_json["workspace"]["initial_state"]["T"]
        )
        assert (
            workspace_json["workspace"]["target_state"]["Tc"]
            > workspace_json["workspace"]["initial_state"]["Tc"]
        )


def test_workspace_gate_warns_when_fit_metrics_are_missing(monkeypatch, tmp_path: Path):
    _demo_env(monkeypatch)
    monkeypatch.setenv("DTE_DISABLE_UNIVERSAL_RUNTIME", "1")
    monkeypatch.setenv("DTE_ONBOARDING_ROOT", str(tmp_path / "customer_jobs"))
    _clear_service_state()

    with TestClient(service.app) as client:
        _upload_json, _preview_json, job_json = _create_completed_job(
            client,
            monkeypatch,
            tmp_path,
            status_metrics={
                "forecast_rmse": None,
                "rollout_rmse": None,
            },
        )
        workspace_response = client.get(f"/onboarding/jobs/{job_json['job_id']}/workspace")
        assert workspace_response.status_code == 200
        workspace_json = workspace_response.json()
        assert workspace_json["gate"]["status"] == "warning"
        assert "Fit metrics are incomplete" in workspace_json["gate"]["message"]


def test_onboarding_compare_endpoint_uses_job_runtime(monkeypatch):
    _demo_env(monkeypatch)
    monkeypatch.setenv("DTE_DISABLE_UNIVERSAL_RUNTIME", "1")
    _clear_service_state()

    spec, simulator = _load_system("cstr")
    sentinel_model = object()

    monkeypatch.setattr(
        service,
        "_get_onboarding_job_demo_runtime",
        lambda job_id: (
            {"job_id": job_id, "status": "completed"},
            {"template_system_name": "cstr"},
            spec,
            simulator,
            sentinel_model,
        ),
    )

    def _fake_compare(spec_arg, simulator_arg, **kwargs):
        assert spec_arg.name == "cstr"
        assert simulator_arg is simulator
        assert kwargs["model"] is sentinel_model
        n_steps = kwargs["baseline_controls"].shape[0]
        zeros = np.zeros((n_steps, spec.state_dim), dtype=np.float32)
        constraints = {
            "above_upper_bound_rate": 0.0,
            "below_lower_bound_rate": 0.0,
        }
        return {
            "baseline": {
                "source": "universal_model",
                "times": np.arange(n_steps, dtype=np.float32),
                "mean": zeros,
                "p05": zeros,
                "p95": zeros,
                "constraint_summary": constraints,
            },
            "candidate": {
                "source": "universal_model",
                "times": np.arange(n_steps, dtype=np.float32),
                "mean": zeros,
                "p05": zeros,
                "p95": zeros,
                "constraint_summary": constraints,
            },
            "summary": {
                "final_state_delta_norm": 0.0,
                "mean_abs_delta": {name: 0.0 for name in spec.state_names},
                "candidate_advantage": {name: 0.0 for name in spec.state_names},
            },
        }

    monkeypatch.setattr(service, "compare_scenarios", _fake_compare)

    with TestClient(service.app) as client:
        response = client.post(
            "/onboarding/jobs/job_runtime/compare_scenarios",
            json={
                "system": "cstr",
                "initial_state": [0.5, 0.5, 350.0, 300.0],
                "baseline_controls": [[55.0, 302.0]] * 6,
                "candidate_controls": [[60.0, 300.0]] * 6,
                "disturbances": [[1.25, 320.0]] * 6,
                "dt": 0.1,
                "n_samples": 4,
            },
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["baseline_source"] == "universal_model"
        assert payload["candidate_source"] == "universal_model"
        assert len(payload["times"]) == 6


def test_onboarding_optimize_endpoint_keeps_non_active_controls_fixed(monkeypatch):
    _demo_env(monkeypatch)
    monkeypatch.setenv("DTE_DISABLE_UNIVERSAL_RUNTIME", "1")
    _clear_service_state()

    spec, simulator = _load_system("two_tank")
    reference_controls = [[0.9, 0.75]] * 6

    monkeypatch.setattr(
        service,
        "_get_onboarding_job_demo_runtime",
        lambda job_id: (
            {"job_id": job_id, "status": "completed"},
            {"template_system_name": "two_tank"},
            spec,
            simulator,
            object(),
        ),
    )

    with TestClient(service.app) as client:
        response = client.post(
            "/onboarding/jobs/job_runtime/optimize_control",
            json={
                "system": "two_tank",
                "initial_state": [1.0, 0.8],
                "disturbances": [[0.05, 0.03]] * 6,
                "reference_controls": reference_controls,
                "active_control_names": ["q_in"],
                "target_state": [1.2, 1.0],
                "tracked_state_names": ["h1", "h2"],
                "dt": 0.1,
                "n_candidates": 8,
            },
        )
        assert response.status_code == 200
        payload = response.json()
        assert len(payload["control_sequence"]) == 6
        assert all(abs(row[1] - 0.75) < 1e-6 for row in payload["control_sequence"])


def test_load_job_demo_runtime_aliases_template_and_legacy_system_names(
    monkeypatch,
    tmp_path: Path,
):
    onboarding._JOB_RUNTIME_CACHE.clear()
    monkeypatch.setenv("DTE_ONBOARDING_ROOT", str(tmp_path / "customer_jobs"))

    preview_id = "prv_legacy"
    job_id = "job_legacy"
    preview_directory = onboarding.preview_dir(preview_id)
    preview_directory.mkdir(parents=True, exist_ok=True)
    onboarding._write_json(
        preview_directory / "preview.json",
        {
            "preview_id": preview_id,
            "template_system_name": "cstr",
        },
    )

    job_directory = onboarding.job_dir(job_id)
    adaptation_directory = job_directory / "adaptation"
    adaptation_directory.mkdir(parents=True, exist_ok=True)
    onboarding._write_json(
        job_directory / "status.json",
        {
            "job_id": job_id,
            "preview_id": preview_id,
            "status": "completed",
            "stage": "completed",
            "created_at": 0.0,
            "updated_at": 0.0,
            "artifacts": {},
            "metrics": {},
            "error": None,
        },
    )
    onboarding._write_json(
        job_directory / "summary.json",
        {
            "status": "ok",
            "target_system_name": "legacy-reactor-1",
        },
    )
    (adaptation_directory / "best_model.eqx").write_text("placeholder", encoding="utf-8")
    (adaptation_directory / "config.yaml").write_text(
        "\n".join(
            [
                "data:",
                "  systems:",
                "    - name: legacy-reactor-1",
                f"      system_config: {PROJECT_ROOT / 'configs' / 'cstr_default.yaml'}",
                f"      data_dir: {tmp_path / 'unused_data'}",
            ]
        ),
        encoding="utf-8",
    )

    sentinel_model = object()

    def _fake_load(model_path, config, metadata):
        assert metadata.system_names == ("legacy-reactor-1",)
        return sentinel_model

    monkeypatch.setattr(onboarding.UniversalDigitalTwin, "load", staticmethod(_fake_load))

    runtime = onboarding.load_job_demo_runtime(job_id)
    assert runtime.model is sentinel_model
    assert runtime.system_ids["legacy-reactor-1"] == 0
    assert runtime.system_ids["cstr"] == 0
