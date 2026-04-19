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
