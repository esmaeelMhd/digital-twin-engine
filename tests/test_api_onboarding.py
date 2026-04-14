"""Tests for customer onboarding API routes."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from dte.api import service


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


def test_onboarding_routes_cover_templates_upload_preview_and_report(monkeypatch, tmp_path: Path):
    _demo_env(monkeypatch)
    monkeypatch.setenv("DTE_DISABLE_UNIVERSAL_RUNTIME", "1")
    monkeypatch.setenv("DTE_ONBOARDING_ROOT", str(tmp_path / "customer_jobs"))
    _clear_service_state()

    model_path = tmp_path / "model.eqx"
    config_path = tmp_path / "config.yaml"
    model_path.write_text("placeholder", encoding="utf-8")
    config_path.write_text("training: {}\n", encoding="utf-8")

    def _fake_start(job_id: str, preview_record: dict, request_payload: dict, demo_config_path: str):
        job_directory = service.job_dir(job_id)
        job_directory.mkdir(parents=True, exist_ok=True)
        summary = {
            "status": "ok",
            "best_val_loss": 0.0123,
            "template_matching": {
                "unit_matches": {
                    "cstr_unit": [
                        {"name": "cstr", "score": 0.98},
                    ]
                }
            },
            "validation_report": {
                "forecast_metrics": {"rmse": 0.41},
                "rollout_metrics": {"rmse": 0.56},
            },
            "report_json_path": str((job_directory / "validation_report.json").resolve()),
            "report_markdown_path": str((job_directory / "validation_report.md").resolve()),
        }
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
            metrics={
                "best_val_loss": 0.0123,
                "forecast_rmse": 0.41,
                "rollout_rmse": 0.56,
                "best_unit_template": "cstr",
            },
            error=None,
        )

    monkeypatch.setattr(service, "_start_onboarding_job", _fake_start)

    historian_path = tmp_path / "historian.csv"
    _write_historian_csv(historian_path)

    with TestClient(service.app) as client:
        templates_response = client.get("/onboarding/templates")
        assert templates_response.status_code == 200
        templates_json = templates_response.json()
        assert len(templates_json["templates"]) == 3
        assert templates_json["templates"][0]["system_spec"]["state_names"]

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
        assert upload_json["detected_format"] == "csv"
        assert "Ca" in upload_json["columns"]

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
        assert preview_json["preview_id"] is not None
        assert preview_json["ingestion_summary"]["n_trajectories"] > 0
        assert preview_json["onboarding_spec"]["asset_kind"] == "unit"

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
        assert job_json["status"] == "queued"

        status_response = client.get(f"/onboarding/jobs/{job_json['job_id']}")
        assert status_response.status_code == 200
        status_json = status_response.json()
        assert status_json["status"] == "completed"
        assert status_json["metrics"]["best_unit_template"] == "cstr"

        report_response = client.get(f"/onboarding/jobs/{job_json['job_id']}/report")
        assert report_response.status_code == 200
        report_json = report_response.json()
        assert report_json["report_markdown"].startswith("# Customer Validation Report")
        assert report_json["summary"]["best_val_loss"] == 0.0123
