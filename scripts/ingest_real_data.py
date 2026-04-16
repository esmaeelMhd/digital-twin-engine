"""Script to ingest real-world plant data into DTE HDF5 format.

Handles CSV and Parquet files with:
- Irregular timestamps
- Missing values / sensor gaps
- Outlier detection and replacement
- Automatic extraction of overlapping trajectory windows
- Optional normalization stats output (for seeding SystemSpec)

Usage examples
--------------
Basic CSV ingestion for CSTR:
    python scripts/ingest_real_data.py \\
        --source data/raw/plant_run_01.csv \\
        --output data/cstr_real/train_data.h5 \\
        --system_config configs/cstr_default.yaml \\
        --state_columns Ca Cb T Tc \\
        --control_columns F_in Tc_in \\
        --disturbance_columns Ca_in T_in \\
        --timestamp_column time \\
        --dt 0.1 \\
        --trajectory_duration 100.0

Parquet with custom outlier threshold:
    python scripts/ingest_real_data.py \\
        --source data/raw/historian_export.parquet \\
        --output data/processed/train_data.h5 \\
        --system_config configs/heat_exchanger_default.yaml \\
        --state_columns T_hot T_cold \\
        --control_columns F_hot F_cold \\
        --disturbance_columns T_hot_in T_cold_in \\
        --outlier_sigma 4.0
"""

import argparse
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Ingest real-world plant data (CSV/Parquet) into DTE HDF5 format.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Path to input CSV or Parquet file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path for output HDF5 file.",
    )
    parser.add_argument(
        "--system_config",
        type=str,
        default="configs/cstr_default.yaml",
        help="Path to system YAML config (determines system name and spec).",
    )
    parser.add_argument(
        "--state_columns",
        nargs="+",
        required=True,
        help="Column names for state variables (order must match SystemSpec.state_names).",
    )
    parser.add_argument(
        "--control_columns",
        nargs="+",
        required=True,
        help="Column names for control inputs.",
    )
    parser.add_argument(
        "--disturbance_columns",
        nargs="*",
        default=[],
        help="Column names for disturbance inputs (can be empty).",
    )
    parser.add_argument(
        "--timestamp_column",
        type=str,
        default="timestamp",
        help="Column containing timestamps (datetime strings or float seconds).",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=1.0,
        help="Target uniform sampling interval in seconds.",
    )
    parser.add_argument(
        "--trajectory_duration",
        type=float,
        default=100.0,
        help="Duration of each extracted trajectory window (seconds).",
    )
    parser.add_argument(
        "--trajectory_stride",
        type=float,
        default=10.0,
        help="Step size between consecutive trajectory windows (seconds).",
    )
    parser.add_argument(
        "--max_gap_fill",
        type=float,
        default=10.0,
        help="Maximum gap (seconds) to fill by interpolation; larger gaps are handled per --drop_large_gaps.",
    )
    parser.add_argument(
        "--outlier_sigma",
        type=float,
        default=5.0,
        help="Z-score threshold for outlier detection. Use 'inf' to disable.",
    )
    parser.add_argument(
        "--drop_large_gaps",
        action="store_true",
        default=False,
        help="Drop trajectory windows containing gaps larger than --max_gap_fill.",
    )
    parser.add_argument(
        "--print_summary",
        action="store_true",
        default=True,
        help="Print ingestion summary statistics.",
    )
    parser.add_argument(
        "--save_summary",
        type=str,
        default=None,
        help="Optional path to save summary statistics as JSON.",
    )
    args = parser.parse_args()

    # -- Load system config -------------------------------------------------
    import yaml
    with open(args.system_config, "r") as f:
        config = yaml.safe_load(f)

    from dte.simulators.registry import get_system_spec
    spec = get_system_spec(config)
    print(f"System: {spec.name}")
    print(f"  States ({spec.state_dim}): {spec.state_names}")
    print(f"  Controls ({spec.control_dim}): {spec.control_names}")
    print(f"  Disturbances ({spec.disturbance_dim}): {spec.disturbance_names}")

    # -- Validate column count matches spec ---------------------------------
    if len(args.state_columns) != spec.state_dim:
        print(
            f"ERROR: --state_columns has {len(args.state_columns)} columns "
            f"but system '{spec.name}' has state_dim={spec.state_dim}."
        )
        sys.exit(1)
    if len(args.control_columns) != spec.control_dim:
        print(
            f"ERROR: --control_columns has {len(args.control_columns)} columns "
            f"but system '{spec.name}' has control_dim={spec.control_dim}."
        )
        sys.exit(1)
    if args.disturbance_columns and len(args.disturbance_columns) != spec.disturbance_dim:
        print(
            f"ERROR: --disturbance_columns has {len(args.disturbance_columns)} columns "
            f"but system '{spec.name}' has disturbance_dim={spec.disturbance_dim}."
        )
        sys.exit(1)

    # Fill disturbance columns with placeholders if not provided
    disturbance_columns = args.disturbance_columns
    if not disturbance_columns:
        disturbance_columns = [f"__zero_disturbance_{i}__" for i in range(spec.disturbance_dim)]

    # -- Run ingestion -------------------------------------------------------
    from dte.data.ingestion.real_data import RealDataIngestion

    ingestor = RealDataIngestion(
        spec=spec,
        state_columns=args.state_columns,
        control_columns=args.control_columns,
        disturbance_columns=disturbance_columns,
        timestamp_column=args.timestamp_column,
        dt=args.dt,
        max_gap_fill=args.max_gap_fill,
        outlier_sigma=args.outlier_sigma,
        drop_large_gaps=args.drop_large_gaps,
    )

    source_path = Path(args.source)
    if source_path.suffix.lower() == ".parquet":
        summary = ingestor.ingest_parquet(
            source_path,
            args.output,
            trajectory_duration=args.trajectory_duration,
            trajectory_stride=args.trajectory_stride,
        )
    else:
        summary = ingestor.ingest_csv(
            source_path,
            args.output,
            trajectory_duration=args.trajectory_duration,
            trajectory_stride=args.trajectory_stride,
        )

    # -- Print / save summary ------------------------------------------------
    if args.print_summary:
        print("\n" + "=" * 60)
        print("INGESTION SUMMARY")
        print("=" * 60)
        print(f"Trajectories extracted : {summary['n_trajectories']}")
        print(f"Steps per trajectory   : {summary['n_steps_per_trajectory']}")
        print(f"dt (s)                 : {summary['dt']}")
        print(f"Total plant data (s)   : {summary['t_total_seconds']:.1f}")
        print(f"States shape           : {summary['states_shape']}")
        print(f"Controls shape         : {summary['controls_shape']}")
        print(f"\nState channel stats (across all windows):")
        for i, name in enumerate(args.state_columns):
            print(
                f"  {name:20s}  mean={summary['state_mean'][i]:.4g}  "
                f"std={summary['state_std'][i]:.4g}  "
                f"noise_std={summary['noise_std'].get(name, float('nan')):.4g}"
            )
        print(f"\nSaved to: {summary['output_path']}")
        print("=" * 60)

    if args.save_summary:
        save_summary_path = Path(args.save_summary)
        save_summary_path.parent.mkdir(parents=True, exist_ok=True)
        with save_summary_path.open("w") as f:
            json.dump(summary, f, indent=2)
        print(f"Summary saved to: {save_summary_path}")

    print("Done.")


if __name__ == "__main__":
    main()
