"""Tests for the Phase 7 control interfaces and metrics."""

from __future__ import annotations

import numpy as np
import yaml
import jax
import jax.numpy as jnp
import pytest

from dte.control.mpc import SamplingMPC
from dte.control.mpc_interface import MPCInterfaceConfig, ProcessMPCInterface
from dte.control.rl_env import ProcessControlEnv, ProcessControlEnvConfig
from dte.control.state_correction import StateCorrectionConfig, StateCorrectionHook
from dte.data.datasets.universal_unit_dataset import UniversalSystemMetadata
from dte.evaluation.control_metrics import disturbance_sensitivity, mismatch_robustness
from dte.models.unit.digital_twin import DigitalTwin
from dte.models.universal.digital_twin import UniversalDigitalTwin
from dte.simulators.registry import get_simulator, get_system_spec
from scripts.run_mpc import _build_pid_controller


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


def _tiny_universal_metadata() -> UniversalSystemMetadata:
    return UniversalSystemMetadata(
        system_names=("cstr",),
        state_center=jnp.zeros((1, 4), dtype=jnp.float32),
        state_scale=jnp.ones((1, 4), dtype=jnp.float32),
        control_center=jnp.zeros((1, 2), dtype=jnp.float32),
        control_scale=jnp.ones((1, 2), dtype=jnp.float32),
        disturbance_center=jnp.zeros((1, 2), dtype=jnp.float32),
        disturbance_scale=jnp.ones((1, 2), dtype=jnp.float32),
        param_scale=jnp.ones((1, 6), dtype=jnp.float32),
        state_mask=jnp.ones((1, 4), dtype=jnp.float32),
        control_mask=jnp.ones((1, 2), dtype=jnp.float32),
        disturbance_mask=jnp.ones((1, 2), dtype=jnp.float32),
        param_mask=jnp.ones((1, 6), dtype=jnp.float32),
        state_dim=jnp.asarray([4], dtype=jnp.int32),
        control_dim=jnp.asarray([2], dtype=jnp.int32),
        disturbance_dim=jnp.asarray([2], dtype=jnp.int32),
        param_dim=jnp.asarray([6], dtype=jnp.int32),
        system_descriptor=jnp.zeros((1, 25), dtype=jnp.float32),
        state_group_kind_names=("concentration", "temperature"),
        state_group_mask=jnp.asarray(
            [[[1, 1, 0, 0], [0, 0, 1, 1]]],
            dtype=jnp.float32,
        ),
        state_group_active=jnp.asarray([[1, 1]], dtype=jnp.float32),
        state_group_kind_id=jnp.asarray([[0, 1]], dtype=jnp.int32),
        state_role_names=("concentration", "temperature"),
        state_role_id=jnp.asarray([[0, 0, 1, 1]], dtype=jnp.int32),
        state_lower_bound=jnp.asarray([[0.0, 0.0, 250.0, 250.0]], dtype=jnp.float32),
        state_upper_bound=jnp.asarray([[jnp.inf, jnp.inf, 400.0, 400.0]], dtype=jnp.float32),
    )


def _tiny_universal_model_config() -> dict:
    return {
        "model": {
            "family": "universal_backbone",
            "latent_dim": 8,
            "shared_hidden_dim": 16,
            "system_embedding_dim": 8,
            "state_group_token_dim": 12,
            "state_group_kind_dim": 6,
            "state_group_encoder_layers": 1,
            "state_group_coupling_layers": 1,
            "encoder_layers": 1,
            "decoder_layers": 1,
            "drift_layers": 1,
            "use_system_spec_embedding": True,
            "use_variational_encoder": True,
            "neural_cde": {"enabled": False},
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


def test_state_correction_hook_supports_universal_foundation_model():
    spec, _, _ = _load_system("cstr")
    model = UniversalDigitalTwin.from_config(
        _tiny_universal_model_config(),
        _tiny_universal_metadata(),
        jax.random.PRNGKey(0),
    )
    hook = StateCorrectionHook(
        spec,
        model=model,
        config=StateCorrectionConfig(assimilation_gain=0.5, filter_alpha=1.0),
    )

    prior = np.asarray(spec.default_initial_state, dtype=np.float32)
    measurement = prior + np.asarray([-0.05, 0.04, -2.0, 1.0], dtype=np.float32)
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
        seed=7,
    )

    assert result.corrected_state.shape == (spec.state_dim,)
    assert result.latent_mean is not None
    assert result.latent_logvar is not None
    assert result.latent_mean.shape == (8,)

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


def test_process_mpc_interface_previous_control_updates_on_assimilate():
    spec, simulator, _ = _load_system("two_tank")
    interface = ProcessMPCInterface(
        spec,
        simulator,
        config=MPCInterfaceConfig(dt=0.1, horizon=8),
    )
    midpoint = np.asarray(
        [0.5 * sum(spec.control_ranges[name]) for name in spec.control_names],
        dtype=np.float32,
    )
    last_applied = np.asarray(
        [spec.control_ranges[name][1] for name in spec.control_names],
        dtype=np.float32,
    )
    candidate = np.tile(midpoint[None, :], (8, 1))

    before = interface.evaluate_candidate(candidate)
    interface.assimilate_measurement(
        np.asarray(spec.default_initial_state, dtype=np.float32),
        control=last_applied,
    )
    after = interface.evaluate_candidate(candidate)
    assert float(after["metrics"]["control_effort_cost"]) > float(
        before["metrics"]["control_effort_cost"]
    )

    interface.reset()
    after_reset = interface.evaluate_candidate(candidate)
    assert np.isclose(
        float(after_reset["metrics"]["control_effort_cost"]),
        float(before["metrics"]["control_effort_cost"]),
    )


def test_process_mpc_interface_rollout_supports_universal_foundation_model():
    spec, simulator, _ = _load_system("cstr")
    model = UniversalDigitalTwin.from_config(
        _tiny_universal_model_config(),
        _tiny_universal_metadata(),
        jax.random.PRNGKey(1),
    )
    interface = ProcessMPCInterface(
        spec,
        simulator,
        model=model,
        config=MPCInterfaceConfig(dt=0.05, horizon=6, rollout_samples=3),
    )

    controls = np.tile(
        np.asarray(
            [0.5 * sum(spec.control_ranges[name]) for name in spec.control_names],
            dtype=np.float32,
        )[None, :],
        (6, 1),
    )
    disturbances = np.tile(
        np.asarray(spec.default_nominal_disturbance, dtype=np.float32)[None, :],
        (6, 1),
    )

    rollout = interface.rollout_candidate(
        controls,
        disturbances=disturbances,
        use_model=True,
        n_samples=3,
        seed=9,
    )

    assert rollout["source"] == "model"
    assert np.asarray(rollout["states"]).shape == (6, spec.state_dim)
    assert np.asarray(rollout["std"]).shape == (6, spec.state_dim)
    assert np.isfinite(np.asarray(rollout["states"])).all()


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


def test_run_mpc_pid_builder_uses_requested_cstr_setpoints():
    spec, _, _ = _load_system("cstr")
    setpoints = np.asarray([0.82, 0.0, 338.0, 300.0], dtype=np.float32)

    controller = _build_pid_controller(spec, setpoints, dt=0.2)

    assert controller is not None
    assert controller.pid_Ca.setpoint == pytest.approx(0.82)
    assert controller.pid_T.setpoint == pytest.approx(338.0)
    assert controller.dt == pytest.approx(0.2)


def test_sampling_mpc_step_returns_finite_control_within_bounds():
    spec, _, _ = _load_system("cstr")
    model = DigitalTwin.from_config(_tiny_model_config(), jax.random.PRNGKey(0), system_spec=spec)
    controller = SamplingMPC(
        model,
        {
            "mpc": {
                "horizon": 4,
                "n_candidates": 8,
                "n_elite": 2,
                "n_iterations": 2,
                "initial_std": 0.3,
                "control_bounds": {"F_in": [10.0, 100.0], "Tc_in": [280.0, 320.0]},
                "cost_weights": {
                    "state": [0.0, 0.0, 10.0, 0.0],
                    "control_effort": [0.01, 0.1],
                    "terminal": 5.0,
                },
            }
        },
    )
    current_state = jnp.asarray(spec.default_initial_state, dtype=jnp.float32)
    params = jnp.ones(spec.param_dim, dtype=jnp.float32)
    disturbance_forecast = jnp.tile(
        jnp.asarray(spec.default_nominal_disturbance, dtype=jnp.float32)[None, :],
        (4, 1),
    )
    u_prev = jnp.asarray([50.0, 300.0], dtype=jnp.float32)

    control = controller.step(
        current_state,
        params,
        current_state,
        disturbance_forecast,
        0.1,
        jax.random.PRNGKey(1),
        u_prev,
    )

    assert control.shape == (2,)
    assert bool(jnp.all(jnp.isfinite(control)))
    assert float(control[0]) >= 10.0 - 1e-5
    assert float(control[0]) <= 100.0 + 1e-5
    assert float(control[1]) >= 280.0 - 1e-5
    assert float(control[1]) <= 320.0 + 1e-5
