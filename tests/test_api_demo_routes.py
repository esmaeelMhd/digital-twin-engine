"""Tests for the Phase 6 demo API routes."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import jax
import yaml
from fastapi.testclient import TestClient

from dte.api import service
from dte.data.multi_system_dataset import MultiSystemTrajectoryDataset, SystemDatasetSource
from dte.demo.engine import load_demo_config
from dte.models.universal_digital_twin import UniversalDigitalTwin


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


def _build_temp_universal_runtime(tmp_path: Path) -> Path:
    with open(PROJECT_ROOT / "configs" / "training_universal_baseline_fast.yaml", "r", encoding="utf-8") as handle:
        universal_cfg = yaml.safe_load(handle)

    absolute_cfg = deepcopy(universal_cfg)
    for item in absolute_cfg["data"]["systems"]:
        item["system_config"] = str(PROJECT_ROOT / item["system_config"])
        item["data_dir"] = str(PROJECT_ROOT / item["data_dir"])

    temp_universal_cfg = tmp_path / "training_universal_api_test.yaml"
    with temp_universal_cfg.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(absolute_cfg, handle, sort_keys=False)

    sources = [
        SystemDatasetSource(
            name=item["name"],
            system_config=item["system_config"],
            data_dir=item["data_dir"],
            weight=float(item.get("weight", 1.0)),
        )
        for item in absolute_cfg["data"]["systems"]
    ]
    metadata = MultiSystemTrajectoryDataset.metadata_from_sources(sources)
    model = UniversalDigitalTwin.from_config(absolute_cfg, metadata, jax.random.PRNGKey(0))
    model_path = tmp_path / "universal_api_test.eqx"
    model.save(str(model_path))

    demo_cfg = load_demo_config(PROJECT_ROOT / "configs" / "demo_app.yaml")
    demo_cfg["runtime"]["model_path"] = str(model_path)
    demo_cfg["runtime"]["config_path"] = str(temp_universal_cfg)
    demo_cfg["runtime"]["train_summary_path"] = ""
    demo_cfg["runtime"]["eval_summary_path"] = ""
    demo_cfg["runtime"]["milestone_summary_path"] = ""
    demo_cfg["runtime"]["customer_pilot_summary_path"] = ""
    demo_cfg["runtime"]["customer_report_path"] = ""
    temp_demo_cfg = tmp_path / "demo_app_universal_test.yaml"
    with temp_demo_cfg.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(demo_cfg, handle, sort_keys=False)
    return temp_demo_cfg


def test_demo_catalog_and_routes_work_without_loaded_models(monkeypatch):
    _demo_env(monkeypatch)
    monkeypatch.setenv("DTE_DISABLE_UNIVERSAL_RUNTIME", "1")
    _clear_service_state()

    with TestClient(service.app) as client:
        page = client.get("/demo/page")
        assert page.status_code == 200
        page_json = page.json()
        assert page_json["release"]["runtime_loaded"] is False
        assert len(page_json["demos"]) == 3
        assert page_json["demos"][0]["system_spec"]["control_names"]

        catalog = client.get("/demo/catalog")
        assert catalog.status_code == 200
        catalog_json = catalog.json()
        assert len(catalog_json["demos"]) == 3

        simulate_response = client.post(
            "/demo/simulate",
            json={
                "system": "two_tank",
                "initial_state": [1.0, 0.8],
                "controls": [[0.7, 0.8]] * 8,
                "disturbances": [[0.6, 0.0]] * 8,
                "dt": 0.1,
            },
        )
        assert simulate_response.status_code == 200
        assert len(simulate_response.json()["states"]) == 8

        rollout_response = client.post(
            "/demo/rollout",
            json={
                "system": "heat_exchanger",
                "initial_state": [360.0, 310.0],
                "controls": [[4.5, 4.0]] * 10,
                "disturbances": [[365.0, 295.0]] * 10,
                "dt": 0.1,
                "n_samples": 4,
            },
        )
        assert rollout_response.status_code == 200
        assert rollout_response.json()["source"] == "simulator_ensemble"

        optimize_response = client.post(
            "/demo/optimize_control",
            json={
                "system": "cstr",
                "initial_state": [0.5, 0.5, 350.0, 300.0],
                "disturbances": [[1.0, 320.0]] * 10,
                "target_state": [0.3, 0.8, 338.0, 304.0],
                "tracked_state_names": ["Cb", "T"],
                "dt": 0.1,
                "n_candidates": 8,
            },
        )
        assert optimize_response.status_code == 200
        assert len(optimize_response.json()["control_sequence"]) == 10

        compare_response = client.post(
            "/demo/compare_scenarios",
            json={
                "system": "cstr",
                "initial_state": [0.5, 0.5, 350.0, 300.0],
                "baseline_controls": [[50.0, 300.0]] * 10,
                "candidate_controls": [[60.0, 295.0]] * 10,
                "disturbances": [[1.0, 320.0]] * 10,
                "dt": 0.1,
                "n_samples": 4,
            },
        )
        assert compare_response.status_code == 200
        compare_json = compare_response.json()
        assert "summary" in compare_json
        assert len(compare_json["times"]) == 10
        assert compare_json["candidate_source"] == "simulator_ensemble"
        assert len(compare_json["baseline_mean"]) == 10


def test_demo_and_inference_routes_use_universal_runtime_when_configured(monkeypatch, tmp_path):
    _demo_env(monkeypatch)
    monkeypatch.delenv("DTE_DISABLE_UNIVERSAL_RUNTIME", raising=False)
    monkeypatch.setenv("DTE_DEMO_CONFIG", str(_build_temp_universal_runtime(tmp_path)))
    _clear_service_state()

    with TestClient(service.app) as client:
        rollout_response = client.post(
            "/demo/rollout",
            json={
                "system": "cstr",
                "initial_state": [0.5, 0.5, 350.0, 300.0],
                "controls": [[55.0, 302.0]] * 10,
                "disturbances": [[1.25, 320.0]] * 10,
                "dt": 0.1,
                "n_samples": 4,
            },
        )
        assert rollout_response.status_code == 200
        assert rollout_response.json()["source"] == "universal_model"

        predict_response = client.post(
            "/predict",
            json={
                "system": "heat_exchanger",
                "initial_state": [350.0, 300.0],
                "controls": [[5.0, 5.0]] * 8,
                "disturbances": [[390.0, 290.0]] * 8,
                "dt": 0.1,
                "return_latent": True,
            },
        )
        assert predict_response.status_code == 200
        predict_json = predict_response.json()
        assert len(predict_json["predicted_states"]) == 8
        assert len(predict_json["latent_trajectory"]) == 8

        ensemble_response = client.post(
            "/ensemble",
            json={
                "system": "two_tank",
                "initial_state": [1.8, 1.1],
                "controls": [[0.72, 0.84]] * 8,
                "disturbances": [[0.05, 0.03]] * 8,
                "dt": 0.1,
                "n_samples": 4,
            },
        )
        assert ensemble_response.status_code == 200
        ensemble_json = ensemble_response.json()
        assert len(ensemble_json["mean"]) == 8
        assert len(ensemble_json["p95"]) == 8
