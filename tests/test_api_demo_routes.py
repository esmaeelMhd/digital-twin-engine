"""Tests for the Phase 6 demo API routes."""

from __future__ import annotations

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


def test_demo_catalog_and_routes_work_without_loaded_models(monkeypatch):
    _demo_env(monkeypatch)
    service._models.clear()
    service._specs.clear()
    service._system_configs.clear()

    with TestClient(service.app) as client:
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
        assert len(compare_json["baseline_mean"]) == 10
