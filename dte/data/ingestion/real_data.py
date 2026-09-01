"""Real-world data ingestion pipeline.

Handles loading time-series data from CSV/Parquet files and converting it into
the format expected by :class:`~dte.data.datasets.unit_dataset.TrajectoryDataset`.

Typical use-case: a CSV exported from a historian / DCS with irregular
timestamps, potential missing values, and sensor noise.

Pipeline steps:
    1. Load CSV or Parquet file.
    2. Align to a uniform time grid via interpolation.
    3. Fill remaining gaps (short gaps: linear interp; long gaps: flag/drop).
    4. Detect and optionally replace outliers.
    5. Characterise sensor noise (std per channel).
    6. Write to an HDF5 file in the standard DTE format.

Example
-------
::

    from dte.data.ingestion.real_data import RealDataIngestion
    from dte.simulators.registry import get_system_spec
    import yaml

    with open("configs/cstr_default.yaml") as f:
        sys_cfg = yaml.safe_load(f)
    spec = get_system_spec(sys_cfg)

    ingestor = RealDataIngestion(
        spec,
        state_columns=["Ca", "Cb", "T", "Tc"],
        control_columns=["F_in", "Tc_in"],
        disturbance_columns=["Ca_in", "T_in"],
        timestamp_column="timestamp",
        dt=0.1,
    )
    summary = ingestor.ingest_csv("data/plant_run_01.csv", "data/processed/plant_01.h5")
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Union

import h5py
import numpy as np

from dte.simulators.base import SystemSpec


def _nan_run_gap_mask(
    ts_seconds: np.ndarray,
    is_nan: np.ndarray,
    t_uniform: np.ndarray,
    max_gap_fill: float,
) -> np.ndarray:
    """Flag uniform-grid points that fall inside long NaN outages.

    Timestamp-spacing gaps are handled separately. This covers the case where
    samples keep arriving on a regular clock but a sensor is NaN for longer
    than ``max_gap_fill``.
    """
    mask = np.zeros(len(t_uniform), dtype=bool)
    if ts_seconds.size == 0 or not np.any(is_nan):
        return mask

    padded = np.concatenate([[False], np.asarray(is_nan, dtype=bool), [False]])
    edges = np.diff(padded.astype(np.int8))
    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0]
    n = len(ts_seconds)
    for start, end in zip(starts, ends):
        prev_idx = start - 1
        next_idx = end
        t_left = ts_seconds[prev_idx] if prev_idx >= 0 else ts_seconds[0]
        t_right = ts_seconds[next_idx] if next_idx < n else ts_seconds[-1]
        if (t_right - t_left) > max_gap_fill:
            left_ok = t_uniform >= t_left if prev_idx < 0 else t_uniform > t_left
            right_ok = t_uniform <= t_right if next_idx >= n else t_uniform < t_right
            mask |= left_ok & right_ok
    return mask


class RealDataIngestion:
    """Ingest real-world plant data and convert to DTE HDF5 format.

    Parameters
    ----------
    spec:
        :class:`~dte.simulators.base.SystemSpec` for the system.
    state_columns:
        Column names in the source file corresponding to each state variable.
    control_columns:
        Column names for control inputs.
    disturbance_columns:
        Column names for disturbance inputs.
    timestamp_column:
        Column containing timestamps (datetime strings or seconds-since-epoch
        floats).
    dt:
        Target uniform sampling interval (seconds).
    max_gap_fill:
        Maximum gap (seconds) to fill by linear interpolation.  Larger gaps
        are flagged and the segment is either dropped or the flanking value is
        held constant depending on ``drop_large_gaps``.
    outlier_sigma:
        Z-score threshold for outlier detection (per channel).  Set to ``inf``
        to disable.
    drop_large_gaps:
        If True, trajectory segments containing unfillable gaps are dropped.
        If False, interior large gaps are linearly interpolated by ``np.interp``
        (not zero-order hold).
    """

    def __init__(
        self,
        spec: SystemSpec,
        state_columns: List[str],
        control_columns: List[str],
        disturbance_columns: List[str],
        timestamp_column: str = "timestamp",
        dt: float = 1.0,
        max_gap_fill: float = 10.0,
        outlier_sigma: float = 5.0,
        drop_large_gaps: bool = False,
    ):
        self.spec = spec
        self.state_columns = state_columns
        self.control_columns = control_columns
        self.disturbance_columns = disturbance_columns
        self.timestamp_column = timestamp_column
        self.dt = dt
        self.max_gap_fill = max_gap_fill
        self.outlier_sigma = outlier_sigma
        self.drop_large_gaps = drop_large_gaps

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ingest_csv(
        self,
        source_path: Union[str, Path],
        output_path: Union[str, Path],
        trajectory_duration: float = 100.0,
        trajectory_stride: float = 10.0,
    ) -> Dict:
        """Load a CSV, process it, and save to HDF5.

        Parameters
        ----------
        source_path:
            Path to input CSV file.
        output_path:
            Destination HDF5 path.
        trajectory_duration:
            Window length in seconds extracted as one "trajectory".
        trajectory_stride:
            Stride in seconds between window starts.  The default duration of
            100 s and stride of 10 s yields 90% overlapping windows; they are
            not independent samples.  Use a stride close to ``trajectory_duration``
            if you need approximately i.i.d. trajectories for train/val splits.

        Returns
        -------
        dict
            Summary statistics and metadata.
        """
        try:
            import pandas as pd
        except ImportError as e:
            raise ImportError("pandas is required for CSV ingestion: pip install pandas") from e

        df = pd.read_csv(source_path)
        return self._process_dataframe(df, output_path, trajectory_duration, trajectory_stride)

    def ingest_parquet(
        self,
        source_path: Union[str, Path],
        output_path: Union[str, Path],
        trajectory_duration: float = 100.0,
        trajectory_stride: float = 10.0,
    ) -> Dict:
        """Load a Parquet file, process it, and save to HDF5."""
        try:
            import pandas as pd
        except ImportError as e:
            raise ImportError("pandas is required for Parquet ingestion: pip install pandas") from e

        df = pd.read_parquet(source_path)
        return self._process_dataframe(df, output_path, trajectory_duration, trajectory_stride)

    def ingest_dataframe(
        self,
        df,
        output_path: Union[str, Path],
        trajectory_duration: float = 100.0,
        trajectory_stride: float = 10.0,
    ) -> Dict:
        """Process an already-loaded pandas DataFrame."""
        return self._process_dataframe(df, output_path, trajectory_duration, trajectory_stride)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _process_dataframe(self, df, output_path, trajectory_duration, trajectory_stride):
        """Core processing pipeline."""
        try:
            import pandas as pd
        except ImportError as e:
            raise ImportError("pandas is required") from e

        # -- 1. Parse timestamps ----------------------------------------
        if self.timestamp_column in df.columns:
            raw_col = df[self.timestamp_column]
            # Try numeric (seconds-since-epoch or relative seconds) first
            try:
                ts_seconds = raw_col.astype(float).values
                # If all values are identical after astype, it's likely a parse
                # failure (e.g. non-numeric strings coerced to NaN). Fall through.
                if np.all(np.isnan(ts_seconds)):
                    raise ValueError("All NaN after float cast")
            except (ValueError, TypeError):
                # Fall back to datetime parsing
                ts_raw = pd.to_datetime(raw_col, errors="coerce")
                if ts_raw.isna().all():
                    warnings.warn(
                        f"Could not parse timestamp column '{self.timestamp_column}'; "
                        "assuming uniform spacing."
                    )
                    ts_seconds = np.arange(len(df), dtype=float) * self.dt
                else:
                    ts_seconds = (ts_raw - ts_raw.iloc[0]).dt.total_seconds().values
        else:
            ts_seconds = np.arange(len(df), dtype=float) * self.dt

        # Sort by time in case data is out of order
        sort_idx = np.argsort(ts_seconds)
        ts_seconds = ts_seconds[sort_idx]
        df = df.iloc[sort_idx].reset_index(drop=True)

        # Remove duplicate timestamps (keep first occurrence)
        _, unique_idx = np.unique(ts_seconds, return_index=True)
        ts_seconds = ts_seconds[unique_idx]
        df = df.iloc[unique_idx].reset_index(drop=True)

        # -- 2. Build uniform time grid ----------------------------------
        t_start = ts_seconds[0]
        t_end = ts_seconds[-1]
        t_uniform = np.arange(t_start, t_end, self.dt)

        all_columns = self.state_columns + self.control_columns + self.disturbance_columns
        data_uniform = {}
        missing_mask = {}

        for col in all_columns:
            if col not in df.columns:
                warnings.warn(f"Column '{col}' not found; filling with zeros.")
                data_uniform[col] = np.zeros(len(t_uniform))
                missing_mask[col] = np.zeros(len(t_uniform), dtype=bool)
                continue

            raw_vals = df[col].values.astype(float)
            is_nan = np.isnan(raw_vals)

            # Identify large gaps in original timestamps
            dt_raw = np.diff(ts_seconds)
            large_gap_starts = np.where(dt_raw > self.max_gap_fill)[0]

            # Interpolate to uniform grid (NaN-aware: interpolate over small gaps)
            valid = ~is_nan
            if valid.sum() < 2:
                warnings.warn(f"Column '{col}' has too few valid values; filling with zeros.")
                data_uniform[col] = np.zeros(len(t_uniform))
                missing_mask[col] = np.ones(len(t_uniform), dtype=bool)
                continue

            interp_vals = np.interp(t_uniform, ts_seconds[valid], raw_vals[valid])

            # Mark uniform grid points that fall inside large gaps
            large_gap_mask = np.zeros(len(t_uniform), dtype=bool)
            for gap_idx in large_gap_starts:
                gap_start_t = ts_seconds[gap_idx]
                gap_end_t = ts_seconds[gap_idx + 1]
                in_gap = (t_uniform > gap_start_t) & (t_uniform < gap_end_t)
                large_gap_mask |= in_gap
            large_gap_mask |= _nan_run_gap_mask(
                ts_seconds, is_nan, t_uniform, self.max_gap_fill
            )

            if self.drop_large_gaps:
                # Replace with NaN, handled during segment extraction
                interp_vals[large_gap_mask] = np.nan
            # else: np.interp linearly bridges interior gaps; ZOH applies
            # only outside the observed time range, not across interior holes.

            data_uniform[col] = interp_vals
            missing_mask[col] = large_gap_mask

        # -- 3. Outlier detection and replacement -----------------------
        noise_std: Dict[str, float] = {}
        for col in all_columns:
            vals = data_uniform[col]
            valid_vals = vals[~np.isnan(vals)]
            if len(valid_vals) < 3:
                noise_std[col] = 0.0
            else:
                # First-difference estimator of sensor noise, not total signal std.
                noise_std[col] = float(np.std(np.diff(valid_vals)) / np.sqrt(2.0))
            if len(valid_vals) == 0:
                continue

            col_mean = np.nanmean(vals)
            col_std = np.nanstd(vals)

            if col_std > 0 and not np.isinf(self.outlier_sigma):
                z_scores = np.abs((vals - col_mean) / col_std)
                outlier_mask = z_scores > self.outlier_sigma
                n_outliers = int(np.sum(outlier_mask))
                if n_outliers > 0:
                    warnings.warn(
                        f"Column '{col}': {n_outliers} outliers detected "
                        f"(|z| > {self.outlier_sigma:.1f}); replacing with column median."
                    )
                    median_val = float(np.nanmedian(vals))
                    vals[outlier_mask] = median_val
                    data_uniform[col] = vals

        # -- 4. Extract trajectory windows -----------------------------
        n_steps_per_traj = int(trajectory_duration / self.dt)
        stride_steps = max(1, int(trajectory_stride / self.dt))

        trajectories = []
        n_points = len(t_uniform)

        for start in range(0, n_points - n_steps_per_traj + 1, stride_steps):
            end = start + n_steps_per_traj
            window_t = t_uniform[start:end]

            # Check for large gap contamination
            skip = False
            for col in all_columns:
                if np.any(missing_mask[col][start:end]) and self.drop_large_gaps:
                    skip = True
                    break
                if np.any(np.isnan(data_uniform[col][start:end])):
                    skip = True
                    break
            if skip:
                continue

            states = np.stack([data_uniform[c][start:end] for c in self.state_columns], axis=-1)
            controls = np.stack([data_uniform[c][start:end] for c in self.control_columns], axis=-1)
            disturbances = np.stack([data_uniform[c][start:end] for c in self.disturbance_columns], axis=-1)
            params = np.ones(self.spec.param_dim, dtype=np.float32)
            rel_t = window_t - window_t[0]

            trajectories.append({
                "states": states.astype(np.float32),
                "controls": controls.astype(np.float32),
                "disturbances": disturbances.astype(np.float32),
                "params": params,
                "time": rel_t.astype(np.float32),
            })

        if not trajectories:
            raise ValueError(
                "No valid trajectory windows could be extracted. "
                "Consider reducing trajectory_duration or max_gap_fill."
            )

        # -- 5. Stack and compute normalization stats -------------------
        states_arr = np.stack([t["states"] for t in trajectories])
        controls_arr = np.stack([t["controls"] for t in trajectories])
        disturbances_arr = np.stack([t["disturbances"] for t in trajectories])
        params_arr = np.stack([t["params"] for t in trajectories])
        time_arr = np.stack([t["time"] for t in trajectories])

        state_mean = states_arr.mean(axis=(0, 1))
        state_std = np.clip(states_arr.std(axis=(0, 1)), 1e-8, None)
        control_mean = controls_arr.mean(axis=(0, 1))
        control_std = np.clip(controls_arr.std(axis=(0, 1)), 1e-8, None)
        disturbance_mean = disturbances_arr.mean(axis=(0, 1))
        disturbance_std = np.clip(disturbances_arr.std(axis=(0, 1)), 1e-8, None)
        param_mean = params_arr.mean(axis=0)
        param_std = np.clip(params_arr.std(axis=0), 1e-8, None)

        # -- 6. Write HDF5 ---------------------------------------------
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(output_path, "w") as f:
            f.create_dataset("states", data=states_arr)
            f.create_dataset("controls", data=controls_arr)
            f.create_dataset("disturbances", data=disturbances_arr)
            f.create_dataset("params", data=params_arr)
            f.create_dataset("time", data=time_arr)

            norm = f.create_group("normalization")
            norm.create_dataset("state_mean", data=state_mean)
            norm.create_dataset("state_std", data=state_std)
            norm.create_dataset("control_mean", data=control_mean)
            norm.create_dataset("control_std", data=control_std)
            norm.create_dataset("disturbance_mean", data=disturbance_mean)
            norm.create_dataset("disturbance_std", data=disturbance_std)
            norm.create_dataset("param_mean", data=param_mean)
            norm.create_dataset("param_std", data=param_std)

        summary = {
            "n_trajectories": len(trajectories),
            "n_steps_per_trajectory": n_steps_per_traj,
            "dt": self.dt,
            "t_total_seconds": float(t_end - t_start),
            "states_shape": states_arr.shape,
            "controls_shape": controls_arr.shape,
            "disturbances_shape": disturbances_arr.shape,
            "params_shape": params_arr.shape,
            "state_mean": state_mean.tolist(),
            "state_std": state_std.tolist(),
            "control_mean": control_mean.tolist(),
            "control_std": control_std.tolist(),
            "disturbance_mean": disturbance_mean.tolist(),
            "disturbance_std": disturbance_std.tolist(),
            "param_mean": param_mean.tolist(),
            "param_std": param_std.tolist(),
            "noise_std": noise_std,
            "output_path": str(output_path),
        }
        return summary

    # ------------------------------------------------------------------
    # Static utilities
    # ------------------------------------------------------------------

    @staticmethod
    def characterize_noise(
        data: np.ndarray,
        window: int = 20,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Estimate noise std per channel via moving-window differencing.

        Returns
        -------
        noise_mean : (n_channels,) mean of local stds
        noise_std  : (n_channels,) std of local stds (for non-stationarity)
        """
        n_steps, n_channels = data.shape
        local_stds = []
        for i in range(0, n_steps - window, window // 2):
            seg = data[i : i + window]
            local_stds.append(seg.std(axis=0))
        local_stds = np.array(local_stds)
        return local_stds.mean(axis=0), local_stds.std(axis=0)
