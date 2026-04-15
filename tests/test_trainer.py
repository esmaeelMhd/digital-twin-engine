"""Tests for trainer failure detection helpers."""

import jax
import jax.numpy as jnp
import yaml

from dte.data.dataset import TrajectoryDataset
from dte.models.digital_twin import DigitalTwin
from dte.simulators.registry import get_system_spec
from dte.training.losses import LossComputer
from dte.training.trainer import Trainer, _format_non_finite_reason, _non_finite_loss_names
from scripts.train import _json_safe_float


def _load_training_config() -> dict:
    with open("configs/training_default.yaml", "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["model"].update(
        {
            "latent_dim": 8,
            "hidden_dim": 16,
            "n_layers": 1,
            "drift_layers": 1,
            "diffusion_layers": 1,
            "diffusion_hidden_dim": 8,
        }
    )
    config["sde_training"] = {
        "enabled": True,
        "warmup_steps": 0,
        "sde_kl_weight": 1e-3,
    }
    return config


def _load_cstr_spec():
    with open("configs/cstr_default.yaml", "r", encoding="utf-8") as handle:
        return get_system_spec(yaml.safe_load(handle))


def _build_dataset(spec) -> TrajectoryDataset:
    n_trajectories = 2
    n_steps = 6
    time = jnp.tile(jnp.linspace(0.0, 0.5, n_steps, dtype=jnp.float32), (n_trajectories, 1))

    base_state = jnp.asarray(spec.default_initial_state, dtype=jnp.float32)
    base_control = jnp.asarray(
        [0.5 * sum(spec.control_ranges[name]) for name in spec.control_names],
        dtype=jnp.float32,
    )
    base_disturbance = jnp.asarray(spec.default_nominal_disturbance, dtype=jnp.float32)

    states = jnp.stack(
        [
            jnp.stack(
                [
                    base_state + 0.01 * traj_idx + 0.02 * step_idx * jnp.arange(1, spec.state_dim + 1)
                    for step_idx in range(n_steps)
                ]
            )
            for traj_idx in range(n_trajectories)
        ]
    )
    controls = jnp.stack(
        [
            jnp.stack(
                [
                    base_control + 0.01 * traj_idx + 0.005 * step_idx * jnp.arange(1, spec.control_dim + 1)
                    for step_idx in range(n_steps)
                ]
            )
            for traj_idx in range(n_trajectories)
        ]
    )
    disturbances = jnp.stack(
        [
            jnp.stack(
                [
                    base_disturbance
                    + 0.01 * traj_idx
                    + 0.003 * step_idx * jnp.arange(1, spec.disturbance_dim + 1)
                    for step_idx in range(n_steps)
                ]
            )
            for traj_idx in range(n_trajectories)
        ]
    )
    params = jnp.stack(
        [
            1.0 + 0.05 * traj_idx + 0.02 * jnp.arange(spec.param_dim, dtype=jnp.float32)
            for traj_idx in range(n_trajectories)
        ]
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

    return TrajectoryDataset(
        {
            "states": states,
            "controls": controls,
            "disturbances": disturbances,
            "params": params,
            "time": time,
            "normalization": normalization,
        },
        seq_len=n_steps,
        stride=1,
    )


def test_non_finite_loss_names_reports_nan_and_inf():
    """NaN and Inf losses should be detected explicitly."""

    losses = {
        "total": float("nan"),
        "reconstruction": 1.0,
        "kl": float("inf"),
    }

    assert _non_finite_loss_names(losses) == ["total", "kl"]


def test_format_non_finite_reason_includes_stage_and_location():
    """Failure reasons should include enough context for debugging."""

    losses = {
        "total": float("nan"),
        "trajectory": float("inf"),
    }

    reason = _format_non_finite_reason(
        "val",
        losses,
        step=42,
        batch_index=3,
        n_batches=7,
        epoch=5,
    )

    assert "non_finite_val_loss" in reason
    assert "epoch=5" in reason
    assert "step=42" in reason
    assert "batch=3/7" in reason
    assert "total=nan" in reason
    assert "trajectory=inf" in reason


def test_json_safe_float_drops_non_finite_values():
    """Training summaries should not emit NaN or Inf into JSON."""

    assert _json_safe_float(1.25) == 1.25
    assert _json_safe_float(float("nan")) is None
    assert _json_safe_float(float("inf")) is None
    assert _json_safe_float(None) is None


def test_single_system_trainer_sde_kl_branch_computes_loss():
    """The optional SDE-KL branch should run without referencing an undefined dt."""

    config = _load_training_config()
    spec = _load_cstr_spec()
    dataset = _build_dataset(spec)
    model = DigitalTwin.from_config(config, jax.random.PRNGKey(0), system_spec=spec)
    loss_computer = LossComputer(
        config,
        dataset.get_normalization_stats(),
        physics_loss=None,
        state_names=spec.state_names,
    )
    trainer = Trainer(model, loss_computer, config, dataset, dataset)

    batch = dataset.sample_batch(jax.random.PRNGKey(1), batch_size=2)
    total_loss, loss_dict = trainer.compute_loss(model, batch, jax.random.PRNGKey(2))

    assert jnp.isfinite(total_loss)
    assert "sde_kl" in loss_dict
    assert jnp.isfinite(loss_dict["sde_kl"])
