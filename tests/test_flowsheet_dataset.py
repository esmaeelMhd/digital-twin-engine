"""Tests for Phase 3 flowsheet datasets and metadata."""

import numpy as np
import jax

from dte.data.datasets.flowsheet_dataset import FlowsheetTrajectoryDataset
from dte.flowsheet.examples import build_exchanger_reactor_tank_flowsheet
from dte.flowsheet.synthetic import (
    build_synthetic_flowsheet_dataset,
    generate_synthetic_flowsheet_data,
)


def test_flowsheet_dataset_batch_shapes_and_seq_override():
    flowsheet = build_exchanger_reactor_tank_flowsheet()
    dataset = build_synthetic_flowsheet_dataset(
        flowsheet,
        n_trajectories=4,
        n_steps=18,
        seq_len=10,
        stride=4,
        seed=0,
    )

    batch = dataset.sample_batch(jax.random.PRNGKey(0), batch_size=2, seq_len=6)

    assert batch["states"].shape == (2, 6, 3, dataset.metadata.unit_state_center.shape[1])
    assert batch["stream_values"].shape[1] == 6
    assert batch["params"].shape[1:] == (3, dataset.metadata.unit_param_scale.shape[1])
    assert batch["time_mask"].shape == (2, 6)


def test_flowsheet_dataset_hdf5_roundtrip(tmp_path):
    flowsheet = build_exchanger_reactor_tank_flowsheet()
    dataset = build_synthetic_flowsheet_dataset(
        flowsheet,
        n_trajectories=3,
        n_steps=12,
        seq_len=6,
        stride=3,
        seed=1,
    )

    path = tmp_path / "flowsheet_data.h5"
    dataset.save_hdf5(path)
    loaded = FlowsheetTrajectoryDataset(str(path))

    assert loaded.metadata.flowsheet_name == dataset.metadata.flowsheet_name
    assert loaded.metadata.unit_names == dataset.metadata.unit_names
    assert loaded.n_samples == dataset.n_samples
    np.testing.assert_allclose(np.asarray(loaded.data["states"]), np.asarray(dataset.data["states"]))


def test_synthetic_flowsheet_generator_builds_expected_keys():
    flowsheet = build_exchanger_reactor_tank_flowsheet()
    data = generate_synthetic_flowsheet_data(
        flowsheet,
        n_trajectories=2,
        n_steps=8,
        seed=2,
    )

    assert set(data) == {
        "states",
        "controls",
        "disturbances",
        "params",
        "stream_values",
        "global_controls",
        "global_disturbances",
        "time",
    }
    assert data["states"].shape[:3] == (2, 8, 3)
