"""Tests for the Phase 6 demo engine."""

from __future__ import annotations

import numpy as np
import yaml

from dte.demo.engine import (
    compare_scenarios,
    constraint_summary,
    default_control_sequence,
    default_disturbance_sequence,
    demo_catalog_from_config,
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
