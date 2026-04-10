"""Tests for the Phase 3 flowsheet trainer."""

import jax
import jax.numpy as jnp

from dte.flowsheet.examples import (
    build_exchanger_reactor_tank_flowsheet,
    build_reactor_separator_recycle_flowsheet,
)
from dte.flowsheet.synthetic import build_synthetic_flowsheet_dataset
from dte.models.flowsheet_model import FlowsheetModel
from dte.training.flowsheet_trainer import FlowsheetTrainer


def _build_config() -> dict:
    return {
        "model": {
            "hidden_dim": 32,
            "message_dim": 12,
            "family_embedding_dim": 8,
            "n_layers": 2,
            "graph_layers": 2,
            "message_passing_steps": 2,
        },
        "optimizer": {
            "peak_lr": 1e-3,
            "end_lr": 1e-4,
            "warmup_steps": 1,
            "total_steps": 8,
            "gradient_clip": 1.0,
        },
        "training": {
            "batch_size": 2,
            "max_batches_per_epoch": 2,
        },
        "checkpointing": {
            "max_val_batches": 1,
        },
        "loss_weights": {
            "trajectory": 1.0,
            "stream_consistency": 1.0,
            "unit_consistency": 0.25,
            "plant_balance": 0.1,
            "rollout_stability": 0.01,
        },
    }


def test_flowsheet_trainer_computes_finite_losses():
    flowsheet = build_exchanger_reactor_tank_flowsheet()
    dataset = build_synthetic_flowsheet_dataset(
        flowsheet,
        n_trajectories=4,
        n_steps=16,
        seq_len=8,
        stride=4,
        seed=0,
    )
    model = FlowsheetModel.from_config(_build_config(), dataset.metadata, jax.random.PRNGKey(0))
    trainer = FlowsheetTrainer(model, _build_config(), dataset)

    batch = dataset.sample_batch(jax.random.PRNGKey(1), batch_size=2)
    total_loss, loss_dict = trainer.compute_loss(model, batch)

    assert jnp.isfinite(total_loss)
    assert jnp.isfinite(loss_dict["trajectory"])
    assert jnp.isfinite(loss_dict["stream_consistency"])


def test_flowsheet_trainer_runs_one_epoch_on_recycle_example():
    flowsheet = build_reactor_separator_recycle_flowsheet()
    dataset = build_synthetic_flowsheet_dataset(
        flowsheet,
        n_trajectories=4,
        n_steps=16,
        seq_len=8,
        stride=4,
        seed=1,
    )
    train_dataset, val_dataset = dataset.split(0.25)
    config = _build_config()
    model = FlowsheetModel.from_config(config, train_dataset.metadata, jax.random.PRNGKey(0))
    trainer = FlowsheetTrainer(model, config, train_dataset, val_dataset)

    summary = trainer.fit(jax.random.PRNGKey(2), n_epochs=1)

    assert summary["failure_reason"] is None
    assert summary["epochs_completed"] == 1
    assert summary["best_val_loss"] is not None
    assert summary["step"] == 2
