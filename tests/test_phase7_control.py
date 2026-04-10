"""Tests for the Phase 7 control interfaces and metrics."""

from __future__ import annotations

import numpy as np
import yaml
import jax

from dte.control.mpc_interface import MPCInterfaceConfig, ProcessMPCInterface
from dte.control.rl_env import ProcessControlEnv, ProcessControlEnvConfig
from dte.control.state_correction import StateCorrectionConfig, StateCorrectionHook
from dte.evaluation.control_metrics import disturbance_sensitivity, mismatch_robustness
from dte.models.digital_twin import DigitalTwin
from dte.simulators.registry import get_simulator, get_system_spec


def _load_system(system_name: str):
    with open(f"configs/{system_name}_default.yaml", "r", encoding="utf-8") as handle:
        system_config = yaml.safe_load(handle) or {}
    spec = get_system_spec(system_config)
    simulator = get_simulator(system_name, system_config)
    return spec, simulator, system_config


def _tiny_model_config() -> dict:
    return {
        "model": {
            "latent_dim": 8,
            "hidden_dim": 16,
            "n_layers": 1,
            "drift_layers": 1,
            "diffusion_layers": 1,
            "diffusion_hidden_dim": 8,
            "initial_diffusion_scale": 0.05,
            "simulator_prior": {"enabled": False},
            "learned_solver": {"enabled": False},
            "self_correcting_policy": {"enabled": False},
            "neural_cde": {"enabled": False},
            "grouped_encoder": {"enabled": False},
        }
    }


def test_state_correction_hook_updates_latent_estimate():
    spec, _, _ = _load_system("cstr")
    model = DigitalTwin.from_config(_tiny_model_config(), jax.random.PRNGKey(0), system_spec=spec)
    hook = StateCorrectionHook(
        spec,
        model=model,
        config=StateCorrectionConfig(assimilation_gain=0.5, filter_alpha=1.0),
    )

    prior = np.asarray(spec.default_initial_state, dtype=np.float32)
    measurement = prior + np.asarray([-0.08, 0.07, -3.0, 1.5], dtype=np.float32)
    control = np.asarray(
        [0.5 * sum(spec.control_ranges[name]) for name in spec.control_names],
        dtype=np.float32,
    )

    result = hook.correct(
        prior_state=prior,
        measurement=measurement,
        control=control,
        params=np.ones(spec.param_dim, dtype=np.float32),
        timestamp=0.1,
        seed=3,
    )

    assert result.corrected_state.shape == (spec.state_dim,)
    assert result.latent_mean is not None
    assert result.latent_logvar is not None
    assert result.latent_mean.shape == (8,)
    assert np.linalg.norm(result.corrected_state - measurement) < np.linalg.norm(prior - measurement)

    predicted = hook.predict(
        control=control,
        disturbance=np.asarray(spec.default_nominal_disturbance, dtype=np.float32),
        params=np.ones(spec.param_dim, dtype=np.float32),
        dt=0.1,
    )
    assert predicted is not None
    assert predicted.shape == (spec.state_dim,)


def test_process_mpc_interface_rollout_and_random_shooting():
    spec, simulator, _ = _load_system("two_tank")
    interface = ProcessMPCInterface(
        spec,
        simulator,
        config=MPCInterfaceConfig(dt=0.1, horizon=10, constraint_penalty=2.0),
    )

    initial_state = interface.reset()
    disturbances = np.tile(
        np.asarray(spec.default_nominal_disturbance, dtype=np.float32)[None, :],
        (10, 1),
    )
    target_state = initial_state + np.asarray([0.2, 0.1], dtype=np.float32)

    best = interface.optimize_random_shooting(
        target_state=target_state,
        disturbances=disturbances,
        horizon=10,
        n_candidates=10,
        seed=5,
    )

    assert best["source"] == "simulator"
    assert np.asarray(best["states"]).shape == (10, spec.state_dim)
    assert np.asarray(best["controls"]).shape == (10, spec.control_dim)
    assert np.isfinite(float(best["objective"]))

    custom = interface.evaluate_candidate(
        np.asarray(best["controls"]),
        disturbances=disturbances,
        target_state=target_state,
        cost_hook=lambda states, controls, target, disturbance: float(np.mean(states[:, 0] ** 2)),
        constraint_hook=lambda spec, states, controls: {"penalty": 1.25},
    )
    assert float(custom["objective"]) >= 1.25

    update = interface.assimilate_measurement(
        np.asarray(best["states"])[1],
        control=np.asarray(best["controls"])[0],
        timestamp=0.1,
    )
    assert np.asarray(update["corrected_state"]).shape == (spec.state_dim,)


def test_process_control_env_runs_gymnasium_style_loop():
    spec, simulator, _ = _load_system("two_tank")
    env = ProcessControlEnv(
        spec,
        simulator,
        target_state=np.asarray(spec.default_initial_state, dtype=np.float32) + np.asarray([0.15, 0.05], dtype=np.float32),
        config=ProcessControlEnvConfig(horizon=5, dt=0.1),
    )

    observation, info = env.reset(seed=11)
    assert observation.shape == (spec.state_dim,)
    assert "target_state" in info

    terminated = False
    truncated = False
    step_count = 0
    while not (terminated or truncated):
        action = env.action_space.sample(seed=step_count)
        observation, reward, terminated, truncated, info = env.step(action)
        assert observation.shape == (spec.state_dim,)
        assert np.isfinite(reward)
        assert "metrics" in info
        step_count += 1

    assert truncated or terminated
    assert step_count >= 1


def test_control_metrics_capture_sensitivity_and_mismatch():
    spec, simulator, _ = _load_system("heat_exchanger")
    interface = ProcessMPCInterface(spec, simulator, config=MPCInterfaceConfig(dt=0.1, horizon=8))
    controls = np.tile(
        np.asarray([0.5 * sum(spec.control_ranges[name]) for name in spec.control_names], dtype=np.float32)[None, :],
        (8, 1),
    )
    nominal_disturbances = np.tile(
        np.asarray(spec.default_nominal_disturbance, dtype=np.float32)[None, :],
        (8, 1),
    )
    shifted_disturbances = nominal_disturbances.copy()
    shifted_disturbances[:, 0] += 8.0

    nominal = interface.rollout_candidate(controls, disturbances=nominal_disturbances)
    shifted = interface.rollout_candidate(controls, disturbances=shifted_disturbances)

    sensitivity = disturbance_sensitivity(nominal["states"], shifted["states"])
    robustness = mismatch_robustness(nominal["states"], shifted["states"])

    assert sensitivity["mean_abs_state_delta"] >= 0.0
    assert sensitivity["max_abs_state_delta"] > 0.0
    assert robustness["normalized_rmse"] >= 0.0
