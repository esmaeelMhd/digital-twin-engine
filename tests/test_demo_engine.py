"""Tests for the Phase 6 demo engine."""

from __future__ import annotations

import warnings

import numpy as np
import yaml

from dte.demo.engine import (
    build_signal_sequence,
    compare_scenarios,
    constraint_summary,
    default_control_sequence,
    default_disturbance_sequence,
    demo_catalog_from_config,
    demo_page_from_config,
    load_demo_config,
    optimize_control_sequence,
    simulate_open_loop,
)
from dte.simulators.registry import get_simulator, get_system_spec


def _load_system(system_name: str):
    with open(f"configs/{system_name}_default.yaml", "r", encoding="utf-8") as handle:
        system_config = yaml.safe_load(handle)
    spec = get_system_spec(system_config)
    simulator = get_simulator(system_name, system_config)
    return spec, simulator, system_config


def test_demo_catalog_exposes_three_unit_demos():
    config = load_demo_config("configs/demo_app.yaml")
    system_configs = {}
    for system_name in ("cstr", "heat_exchanger", "two_tank"):
        with open(f"configs/{system_name}_default.yaml", "r", encoding="utf-8") as handle:
            system_configs[system_name] = yaml.safe_load(handle)

    catalog = demo_catalog_from_config(config, system_configs)

    assert catalog["product_name"] == "Digital Twin Engine"
    assert len(catalog["demos"]) == 3
    assert {item["system"] for item in catalog["demos"]} == {"cstr", "heat_exchanger", "two_tank"}
    assert len(catalog["flowsheets"]) == 2


def test_demo_page_exposes_browser_bootstrap_payload():
    config = load_demo_config("configs/demo_app.yaml")
    system_configs = {}
    for system_name in ("cstr", "heat_exchanger", "two_tank"):
        with open(f"configs/{system_name}_default.yaml", "r", encoding="utf-8") as handle:
            system_configs[system_name] = yaml.safe_load(handle)

    page = demo_page_from_config(
        config,
        system_configs,
        config_path="configs/demo_app.yaml",
        runtime_loaded=True,
    )

    assert page["release"]["release_label"] == "V1 milestone release"
    assert page["release"]["runtime_loaded"] is True
    assert len(page["demos"]) == 3
    first_demo = page["demos"][0]
    assert first_demo["baseline_control_profile"]["type"] == "constant"
    assert first_demo["disturbance_presets"]
    assert first_demo["candidate_profiles"]
    assert first_demo["system_spec"]["name"] == first_demo["system"]
    assert first_demo["system_spec"]["control_channels"]


def test_simulate_open_loop_and_constraints_return_expected_shapes():
    spec, simulator, _ = _load_system("two_tank")
    controls = default_control_sequence(spec, 10)
    disturbances = default_disturbance_sequence(spec, 10)
    states = simulate_open_loop(
        spec,
        simulator,
        np.asarray(spec.default_initial_state, dtype=np.float32),
        controls,
        disturbances,
        0.1,
    )
    summary = constraint_summary(spec, states)

    assert states.shape == (10, spec.state_dim)
    assert summary["positivity_violation_rate"] >= 0.0
    assert summary["below_lower_bound_rate"] >= 0.0


def test_build_signal_sequence_supports_configured_profiles():
    spec, _, _ = _load_system("heat_exchanger")

    constant_controls = build_signal_sequence(
        spec,
        8,
        signal_kind="control",
        profile={
            "type": "constant",
            "channels": {"F_hot": 6.0, "F_cold": 5.5},
        },
    )
    ramp_disturbances = build_signal_sequence(
        spec,
        8,
        signal_kind="disturbance",
        profile={
            "type": "ramp",
            "start": {"T_hot_in": 385.0, "T_cold_in": 288.0},
            "end": {"T_hot_in": 405.0, "T_cold_in": 296.0},
        },
    )
    pulse_controls = build_signal_sequence(
        spec,
        8,
        signal_kind="control",
        profile={
            "type": "pulse",
            "base": {"F_hot": 5.0, "F_cold": 5.0},
            "pulse": {"F_hot": 8.0, "F_cold": 6.0},
            "start_step": 2,
            "duration": 3,
        },
    )

    assert constant_controls.shape == (8, spec.control_dim)
    assert np.allclose(constant_controls[0], [6.0, 5.5])
    assert np.allclose(ramp_disturbances[0], [385.0, 288.0])
    assert np.allclose(ramp_disturbances[-1], [405.0, 296.0])
    assert np.allclose(pulse_controls[1], [5.0, 5.0])
    assert np.allclose(pulse_controls[2], [8.0, 6.0])
    assert np.allclose(pulse_controls[5], [5.0, 5.0])


def test_compare_and_optimize_control_sequences():
    spec, simulator, _ = _load_system("heat_exchanger")
    baseline_controls = default_control_sequence(spec, 12)
    candidate_controls = baseline_controls.copy()
    candidate_controls[:, 0] = np.clip(
        candidate_controls[:, 0] * 1.1,
        spec.control_ranges["F_hot"][0],
        spec.control_ranges["F_hot"][1],
    )
    disturbances = default_disturbance_sequence(spec, 12)

    comparison = compare_scenarios(
        spec,
        simulator,
        initial_state=np.asarray(spec.default_initial_state, dtype=np.float32),
        baseline_controls=baseline_controls,
        candidate_controls=candidate_controls,
        disturbances=disturbances,
        dt=0.1,
        model=None,
        n_samples=6,
        seed=3,
    )

    target_state = np.asarray(spec.default_initial_state, dtype=np.float32) + np.array([5.0, 3.0], dtype=np.float32)
    optimized = optimize_control_sequence(
        spec,
        simulator,
        initial_state=np.asarray(spec.default_initial_state, dtype=np.float32),
        disturbances=disturbances,
        dt=0.1,
        target_state=target_state,
        tracked_state_names=["T_hot", "T_cold"],
        n_candidates=12,
        seed=5,
    )

    assert comparison["baseline"]["mean"].shape == (12, spec.state_dim)
    assert comparison["candidate"]["mean"].shape == (12, spec.state_dim)
    assert "final_state_delta_norm" in comparison["summary"]
    assert optimized["control_sequence"].shape == (12, spec.control_dim)
    assert optimized["predicted_states"].shape == (12, spec.state_dim)


def test_cstr_optimizer_rejects_unstable_candidates_without_runtime_warnings():
    spec, simulator, _ = _load_system("cstr")
    disturbances = default_disturbance_sequence(spec, 25)
    target_state = np.asarray(spec.default_initial_state, dtype=np.float32).copy()
    target_state[0] = 0.35
    target_state[1] = 0.85
    target_state[2] = 338.0
    target_state[3] = 304.0

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        optimized = optimize_control_sequence(
            spec,
            simulator,
            initial_state=np.asarray(spec.default_initial_state, dtype=np.float32),
            disturbances=disturbances,
            dt=0.1,
            target_state=target_state,
            tracked_state_names=["Cb", "T", "Tc"],
            n_candidates=72,
            seed=17,
        )

    runtime_warnings = [item for item in caught if issubclass(item.category, RuntimeWarning)]
    assert not runtime_warnings
    assert np.isfinite(optimized["objective"])
    assert np.all(np.isfinite(optimized["predicted_states"]))
