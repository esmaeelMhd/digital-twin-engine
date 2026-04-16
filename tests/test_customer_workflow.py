"""Tests for Phase 5 customer onboarding and adaptation workflow."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import jax
import numpy as np
import yaml

from dte.calibration.unit_calibration import CalibrationOptions
from dte.customer.adaptation import run_customer_adaptation
from dte.customer.onboarding_schema import load_onboarding_spec
from dte.customer.template_matching import match_customer_templates
from dte.data.datasets.universal_unit_dataset import (
    MultiSystemTrajectoryDataset,
    SystemDatasetSource,
)
from dte.models.universal.digital_twin import UniversalDigitalTwin
from dte.simulators.registry import get_system_spec


def _load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _write_dataset(data_dir: Path, *, system_config_path: str, offset: float) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    spec = get_system_spec(_load_yaml(system_config_path))
    n_trajectories = 3
    n_steps = 6
    time = np.linspace(0.0, 0.5, n_steps, dtype=np.float32)
    states = np.zeros((n_trajectories, n_steps, spec.state_dim), dtype=np.float32)
    controls = np.zeros((n_trajectories, n_steps, spec.control_dim), dtype=np.float32)
    disturbances = np.zeros((n_trajectories, n_steps, spec.disturbance_dim), dtype=np.float32)
    params = np.ones((n_trajectories, spec.param_dim), dtype=np.float32)

    default_state = np.asarray(spec.default_initial_state, dtype=np.float32)
    default_control = np.asarray(
        [
            sum(spec.control_ranges[name]) / 2.0
            for name in spec.control_names
        ],
        dtype=np.float32,
    )
    default_disturbance = np.asarray(spec.default_nominal_disturbance, dtype=np.float32)

    state_slope = np.linspace(0.01, 0.04, spec.state_dim, dtype=np.float32)
    control_slope = np.linspace(0.01, 0.02, spec.control_dim, dtype=np.float32)
    disturbance_slope = np.linspace(0.01, 0.015, spec.disturbance_dim, dtype=np.float32)

    for traj_idx in range(n_trajectories):
        traj_shift = offset + 0.01 * traj_idx
        params[traj_idx] = 1.0 + traj_shift + 0.05 * np.arange(spec.param_dim, dtype=np.float32)
        for step_idx, t_value in enumerate(time):
            states[traj_idx, step_idx] = default_state + traj_shift + t_value * state_slope
            controls[traj_idx, step_idx] = default_control + traj_shift + t_value * control_slope
            disturbances[traj_idx, step_idx] = (
                default_disturbance + traj_shift + t_value * disturbance_slope
            )

    normalization = {
        "state_mean": states.reshape(-1, spec.state_dim).mean(axis=0),
        "state_std": states.reshape(-1, spec.state_dim).std(axis=0) + 1e-3,
        "control_mean": controls.reshape(-1, spec.control_dim).mean(axis=0),
        "control_std": controls.reshape(-1, spec.control_dim).std(axis=0) + 1e-3,
        "disturbance_mean": disturbances.reshape(-1, spec.disturbance_dim).mean(axis=0),
        "disturbance_std": disturbances.reshape(-1, spec.disturbance_dim).std(axis=0) + 1e-3,
        "param_mean": params.mean(axis=0),
        "param_std": params.std(axis=0) + 1e-3,
    }

    with h5py.File(data_dir / "train_data.h5", "w") as handle:
        handle.create_dataset("states", data=states)
        handle.create_dataset("controls", data=controls)
        handle.create_dataset("disturbances", data=disturbances)
        handle.create_dataset("params", data=params)
        handle.create_dataset("time", data=np.tile(time[None, :], (n_trajectories, 1)))
        norm = handle.create_group("normalization")
        for key, value in normalization.items():
            norm.create_dataset(key, data=value)


def _build_config(source_dir: Path) -> dict:
    return {
        "model": {
            "latent_dim": 8,
            "shared_hidden_dim": 16,
            "system_embedding_dim": 8,
            "state_group_token_dim": 12,
            "state_group_kind_dim": 6,
            "state_group_encoder_layers": 2,
            "state_group_coupling_layers": 2,
            "encoder_layers": 2,
            "decoder_layers": 2,
            "drift_layers": 2,
            "use_system_spec_embedding": True,
            "use_variational_encoder": True,
            "adapters": {
                "enabled": True,
                "bottleneck_dim": 4,
                "residual_scale": 0.1,
                "encoder": True,
                "drift": True,
                "decoder": True,
            },
            "neural_cde": {"enabled": True, "hidden_dim": 12, "n_layers": 2},
        },
        "training": {
            "batch_size": 2,
            "seq_len": 4,
            "stride": 1,
            "val_split": 0.34,
            "n_epochs": 1,
        },
        "optimizer": {
            "peak_lr": 1e-3,
            "end_lr": 1e-4,
            "warmup_steps": 1,
            "total_steps": 8,
            "gradient_clip": 1.0,
        },
        "loss_weights": {
            "reconstruction": 1.0,
            "trajectory": 1.0,
            "one_step": 0.5,
            "k_step": 0.0,
            "kl": 1e-4,
            "state_bounds": 0.0,
            "positivity": 0.0,
        },
        "data": {
            "systems": [
                {
                    "name": "cstr",
                    "system_config": "configs/cstr_default.yaml",
                    "data_dir": str(source_dir),
                    "weight": 1.0,
                }
            ]
        },
        "evaluation": {
            "per_system_batches": 1,
            "uncertainty_samples": 3,
            "uncertainty_batches": 1,
            "sensitivity_batches": 1,
        },
    }


def _write_onboarding(path: Path, payload: dict) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def test_onboarding_unit_template_matching_prefers_cstr(tmp_path: Path):
    onboarding_path = tmp_path / "customer_unit.yaml"
    _write_onboarding(
        onboarding_path,
        {
            "name": "customer_reactor",
            "asset_kind": "unit",
            "units": [
                {
                    "name": "reactor_section",
                    "family": "reactor",
                    "subtype": "nonisothermal_cstr",
                    "controls": ["F_in", "Tc_in"],
                    "disturbances": ["Ca_in", "T_in"],
                    "measurements": ["Ca", "Cb", "T", "Tc"],
                    "known_laws": ["mass_balance", "energy_balance", "reaction_kinetics"],
                }
            ],
            "controls": [
                {"name": "F_in", "role": "flow", "unit_name": "reactor_section"},
                {"name": "Tc_in", "role": "temperature", "unit_name": "reactor_section"},
            ],
            "disturbances": [
                {"name": "Ca_in", "role": "concentration", "unit_name": "reactor_section"},
                {"name": "T_in", "role": "temperature", "unit_name": "reactor_section"},
            ],
            "measurements": [
                {"name": "Ca", "role": "concentration", "unit_name": "reactor_section"},
                {"name": "Cb", "role": "concentration", "unit_name": "reactor_section"},
                {"name": "T", "role": "temperature", "unit_name": "reactor_section"},
                {"name": "Tc", "role": "temperature", "unit_name": "reactor_section"},
            ],
            "known_laws": ["mass_balance", "energy_balance", "reaction_kinetics"],
        },
    )

    onboarding = load_onboarding_spec(onboarding_path)
    matches = match_customer_templates(onboarding)

    assert onboarding.unit_names == ("reactor_section",)
    assert matches.best_unit_match("reactor_section") is not None
    assert matches.best_unit_match("reactor_section").name == "cstr"


def test_onboarding_flowsheet_template_matching_prefers_recycle_template(tmp_path: Path):
    onboarding_path = tmp_path / "customer_flowsheet.yaml"
    _write_onboarding(
        onboarding_path,
        {
            "name": "customer_recycle_section",
            "asset_kind": "flowsheet",
            "units": [
                {
                    "name": "reactor",
                    "family": "reactor",
                    "known_laws": ["mass_balance", "energy_balance", "reaction_kinetics"],
                },
                {
                    "name": "separator",
                    "family": "separator",
                    "known_laws": ["mass_balance", "phase_split"],
                },
            ],
            "streams": [
                {
                    "name": "fresh_feed",
                    "source_unit": "__source__",
                    "target_unit": "reactor",
                    "variables": ["Ca"],
                    "kind": "source",
                },
                {
                    "name": "reactor_effluent",
                    "source_unit": "reactor",
                    "target_unit": "separator",
                    "variables": ["Cb", "T"],
                    "kind": "process",
                },
                {
                    "name": "recycle",
                    "source_unit": "separator",
                    "target_unit": "reactor",
                    "variables": ["light_cut", "tray_temperature"],
                    "kind": "recycle",
                    "delay": 0.1,
                },
            ],
            "known_laws": ["mass_balance", "phase_split", "reaction_kinetics"],
        },
    )

    matches = match_customer_templates(load_onboarding_spec(onboarding_path))

    assert matches.best_flowsheet_match() is not None
    assert matches.best_flowsheet_match().name == "reactor_separator_recycle"


def test_run_customer_adaptation_generates_validation_report(tmp_path: Path):
    source_dir = tmp_path / "data" / "source_cstr"
    target_dir = tmp_path / "data" / "customer_cstr"
    _write_dataset(source_dir, system_config_path="configs/cstr_default.yaml", offset=0.0)
    _write_dataset(target_dir, system_config_path="configs/cstr_default.yaml", offset=0.08)

    config = _build_config(source_dir)
    source_dataset = MultiSystemTrajectoryDataset.from_sources(
        [
            SystemDatasetSource(
                name="cstr",
                system_config="configs/cstr_default.yaml",
                data_dir=str(source_dir),
            )
        ],
        seq_len=int(config["training"]["seq_len"]),
        stride=int(config["training"]["stride"]),
    )
    model = UniversalDigitalTwin.from_config(
        config,
        source_dataset.metadata,
        jax.random.PRNGKey(0),
    )
    model_path = tmp_path / "pretrained.eqx"
    model.save(str(model_path))

    onboarding_path = tmp_path / "customer_unit.yaml"
    _write_onboarding(
        onboarding_path,
        {
            "name": "customer_cstr_variant",
            "asset_kind": "unit",
            "units": [
                {
                    "name": "reactor_section",
                    "family": "reactor",
                    "subtype": "nonisothermal_cstr",
                    "controls": ["F_in", "Tc_in"],
                    "disturbances": ["Ca_in", "T_in"],
                    "measurements": ["Ca", "Cb", "T", "Tc"],
                    "known_laws": ["mass_balance", "energy_balance", "reaction_kinetics"],
                }
            ],
            "controls": [
                {"name": "F_in", "role": "flow", "unit_name": "reactor_section"},
                {"name": "Tc_in", "role": "temperature", "unit_name": "reactor_section"},
            ],
            "disturbances": [
                {"name": "Ca_in", "role": "concentration", "unit_name": "reactor_section"},
                {"name": "T_in", "role": "temperature", "unit_name": "reactor_section"},
            ],
            "measurements": [
                {"name": "Ca", "role": "concentration", "unit_name": "reactor_section"},
                {"name": "Cb", "role": "concentration", "unit_name": "reactor_section"},
                {"name": "T", "role": "temperature", "unit_name": "reactor_section"},
                {"name": "Tc", "role": "temperature", "unit_name": "reactor_section"},
            ],
            "known_laws": ["mass_balance", "energy_balance", "reaction_kinetics"],
        },
    )
    onboarding = load_onboarding_spec(onboarding_path)

    summary = run_customer_adaptation(
        model_path=str(model_path),
        config=config,
        onboarding=onboarding,
        system_config_path="configs/cstr_default.yaml",
        data_dir=str(target_dir),
        output_dir=str(tmp_path / "outputs" / "customer_adaptation"),
        options=CalibrationOptions(
            trainable_mode="adapters",
            tune_normalization=True,
            tune_physics_params=False,
        ),
        seed=7,
    )

    assert summary["status"] == "ok"
    assert Path(summary["report_json_path"]).exists()
    assert Path(summary["report_markdown_path"]).exists()
    report = json.loads(Path(summary["report_json_path"]).read_text(encoding="utf-8"))
    assert report["template_matching"]["best_unit_match"]["name"] == "cstr"
    assert "forecast_metrics" in report
    assert "rollout_metrics" in report
    assert "control_sensitivity_metrics" in report
    assert "uncertainty_summary" in report
    assert "constraints_summary" in report
