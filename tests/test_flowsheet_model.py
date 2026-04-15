"""Tests for Phase 3 flowsheet rollouts and recycle behavior."""

import numpy as np
import jax
import jax.numpy as jnp

from dte.flowsheet.examples import (
    build_exchanger_reactor_tank_flowsheet,
    build_reactor_separator_recycle_flowsheet,
)
from dte.flowsheet.synthetic import build_synthetic_flowsheet_dataset
from dte.models.flowsheet_model import FlowsheetModel


def _build_config() -> dict:
    return {
        "model": {
            "hidden_dim": 32,
            "message_dim": 12,
            "family_embedding_dim": 8,
            "n_layers": 2,
            "graph_layers": 2,
            "message_passing_steps": 2,
            "channel_conditioning": {"enabled": True},
            "law_conditioning": {"enabled": True},
        }
    }


def test_flowsheet_model_rolls_out_both_demo_graphs():
    for seed, builder in enumerate(
        (build_exchanger_reactor_tank_flowsheet, build_reactor_separator_recycle_flowsheet)
    ):
        flowsheet = builder()
        dataset = build_synthetic_flowsheet_dataset(
            flowsheet,
            n_trajectories=2,
            n_steps=12,
            seq_len=8,
            stride=4,
            seed=seed,
        )
        model = FlowsheetModel.from_config(
            _build_config(),
            dataset.metadata,
            jax.random.PRNGKey(seed),
        )
        batch = dataset.sample_batch(jax.random.PRNGKey(seed + 10), batch_size=1)
        predicted_states, predicted_streams = model.rollout(
            batch["states"][0, 0],
            batch["controls"][0],
            batch["disturbances"][0],
            batch["params"][0],
            batch["t"][0],
            global_controls=batch["global_controls"][0],
            global_disturbances=batch["global_disturbances"][0],
            external_stream_sequence=batch["stream_values"][0],
        )

        assert predicted_states.shape == batch["states"][0].shape
        assert predicted_streams.shape == batch["stream_values"][0].shape
        assert model.get_parameter_count()["total"] > 0


def test_recycle_stream_uses_delayed_source_values():
    flowsheet = build_reactor_separator_recycle_flowsheet()
    dataset = build_synthetic_flowsheet_dataset(
        flowsheet,
        n_trajectories=2,
        n_steps=10,
        seq_len=8,
        stride=4,
        seed=0,
    )
    model = FlowsheetModel.from_config(_build_config(), dataset.metadata, jax.random.PRNGKey(0))
    batch = dataset.sample_batch(jax.random.PRNGKey(1), batch_size=1)
    predicted_states, predicted_streams = model.rollout(
        batch["states"][0, 0],
        batch["controls"][0],
        batch["disturbances"][0],
        batch["params"][0],
        batch["t"][0],
        global_controls=batch["global_controls"][0],
        global_disturbances=batch["global_disturbances"][0],
        external_stream_sequence=batch["stream_values"][0],
    )

    recycle_idx = flowsheet.stream_names.index("separator_recycle")
    separator_idx = flowsheet.unit_names.index("separator")
    separator_spec = flowsheet.units["separator"]
    source_vars = [
        separator_spec.state_names.index("light_cut"),
        separator_spec.state_names.index("tray_temperature"),
    ]

    np.testing.assert_allclose(
        np.asarray(predicted_streams[1, recycle_idx, :2]),
        np.asarray(predicted_states[0, separator_idx, jnp.asarray(source_vars)]),
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        np.asarray(predicted_streams[2, recycle_idx, :2]),
        np.asarray(predicted_states[1, separator_idx, jnp.asarray(source_vars)]),
        rtol=1e-5,
        atol=1e-5,
    )
