"""Tests for the real-data ingestion pipeline."""

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import yaml

from dte.data.ingestion.real_data import RealDataIngestion
from dte.simulators.registry import get_system_spec


def _load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def test_real_data_ingestion_writes_parameter_normalization_stats(tmp_path: Path):
    spec = get_system_spec(_load_yaml("configs/cstr_default.yaml"))
    df = pd.DataFrame(
        {
            "timestamp": np.arange(12, dtype=float),
            "Ca": np.linspace(0.7, 1.1, 12),
            "Cb": np.linspace(0.2, 0.6, 12),
            "T": np.linspace(330.0, 338.0, 12),
            "Tc": np.linspace(300.0, 306.0, 12),
            "F_in": np.linspace(40.0, 60.0, 12),
            "Tc_in": np.linspace(292.0, 304.0, 12),
            "Ca_in": np.linspace(0.9, 1.1, 12),
            "T_in": np.linspace(318.0, 322.0, 12),
        }
    )

    ingestor = RealDataIngestion(
        spec=spec,
        state_columns=["Ca", "Cb", "T", "Tc"],
        control_columns=["F_in", "Tc_in"],
        disturbance_columns=["Ca_in", "T_in"],
        timestamp_column="timestamp",
        dt=1.0,
    )
    output_path = tmp_path / "ingested.h5"
    summary = ingestor.ingest_dataframe(
        df,
        output_path,
        trajectory_duration=4.0,
        trajectory_stride=2.0,
    )

    assert summary["params_shape"][1] == spec.param_dim
    assert output_path.exists()

    with h5py.File(output_path, "r") as handle:
        assert "param_mean" in handle["normalization"]
        assert "param_std" in handle["normalization"]


def _historian_frame(n_steps: int = 40) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": np.arange(n_steps, dtype=float),
            "Ca": np.linspace(0.7, 1.1, n_steps),
            "Cb": np.linspace(0.2, 0.6, n_steps),
            "T": np.linspace(330.0, 338.0, n_steps),
            "Tc": np.linspace(300.0, 306.0, n_steps),
            "F_in": np.linspace(40.0, 60.0, n_steps),
            "Tc_in": np.linspace(292.0, 304.0, n_steps),
            "Ca_in": np.linspace(0.9, 1.1, n_steps),
            "T_in": np.linspace(318.0, 322.0, n_steps),
        }
    )


def test_real_data_ingestion_drops_windows_with_long_nan_outages(tmp_path: Path):
    spec = get_system_spec(_load_yaml("configs/cstr_default.yaml"))
    kwargs = dict(
        spec=spec,
        state_columns=["Ca", "Cb", "T", "Tc"],
        control_columns=["F_in", "Tc_in"],
        disturbance_columns=["Ca_in", "T_in"],
        timestamp_column="timestamp",
        dt=1.0,
        max_gap_fill=5.0,
        drop_large_gaps=True,
    )
    baseline = RealDataIngestion(**kwargs).ingest_dataframe(
        _historian_frame(40),
        tmp_path / "baseline.h5",
        trajectory_duration=8.0,
        trajectory_stride=4.0,
    )
    df = _historian_frame(40)
    df.loc[10:25, "Ca"] = np.nan
    gapped = RealDataIngestion(**kwargs).ingest_dataframe(
        df,
        tmp_path / "long_nan.h5",
        trajectory_duration=8.0,
        trajectory_stride=4.0,
    )
    assert gapped["n_trajectories"] < baseline["n_trajectories"]
    with h5py.File(tmp_path / "long_nan.h5", "r") as handle:
        assert not np.isnan(handle["states"][:]).any()


def test_real_data_ingestion_interpolates_short_nan_runs(tmp_path: Path):
    spec = get_system_spec(_load_yaml("configs/cstr_default.yaml"))
    df = _historian_frame(20)
    df.loc[8:9, "Ca"] = np.nan

    ingestor = RealDataIngestion(
        spec=spec,
        state_columns=["Ca", "Cb", "T", "Tc"],
        control_columns=["F_in", "Tc_in"],
        disturbance_columns=["Ca_in", "T_in"],
        timestamp_column="timestamp",
        dt=1.0,
        max_gap_fill=10.0,
        drop_large_gaps=True,
    )
    output_path = tmp_path / "short_nan.h5"
    summary = ingestor.ingest_dataframe(
        df,
        output_path,
        trajectory_duration=8.0,
        trajectory_stride=4.0,
    )
    assert summary["n_trajectories"] >= 1
    with h5py.File(output_path, "r") as handle:
        states = handle["states"][:]
        assert not np.isnan(states).any()
        assert np.all(states[:, :, 0] > 0.0)
