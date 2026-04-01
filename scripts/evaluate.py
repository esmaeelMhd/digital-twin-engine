"""Evaluation script for Digital Twin model."""

import argparse
import json
import os

from dte.utils.runtime import configure_runtime_logging

configure_runtime_logging()

import yaml
import jax
import jax.numpy as jnp
import numpy as np
from pathlib import Path

from dte.models.digital_twin import DigitalTwin
from dte.physics.registry import get_physics_diagnostic_fn, zero_residual
from dte.data.dataset import TrajectoryDataset
from dte.simulators.registry import get_system_spec
from dte.utils.plotting import (
    plot_trajectory_comparison,
    plot_conservation_violation,
    plot_latent_space,
    plot_prediction_error,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _resolve_repo_path(path_value: str | None) -> Path | None:
    """Resolve absolute or repo-relative paths."""
    if path_value is None:
        return None
    path = Path(path_value)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def _candidate_run_dir(run_value: str) -> Path:
    """Resolve a run name like ``cstr_v3`` or a direct run directory path."""
    candidate = Path(run_value)
    if candidate.is_absolute() or "/" in run_value or candidate.exists():
        return _resolve_repo_path(run_value)
    return REPO_ROOT / "outputs" / run_value


def _pick_first_existing(paths: list[Path]) -> Path | None:
    """Return the first existing path from a list."""
    for path in paths:
        if path.exists():
            return path
    return None


def _load_yaml(path: Path) -> dict:
    """Load YAML config from disk."""
    with path.open("r") as f:
        return yaml.safe_load(f)


def _infer_data_dir_from_system_config(system_config_path: Path) -> Path | None:
    """Map the saved system config to the conventional dataset directory."""
    try:
        system_config = _load_yaml(system_config_path)
    except Exception:
        return None
    system_name = system_config.get("system", {}).get("name")
    if not system_name:
        return None
    return REPO_ROOT / "data" / system_name


def _resolve_evaluation_paths(args: argparse.Namespace) -> dict[str, Path]:
    """Infer evaluation paths from a run name or model path when possible."""
    run_dir = _candidate_run_dir(args.run) if args.run else None

    model_path = _resolve_repo_path(args.model_path)
    if model_path is None and run_dir is not None:
        model_path = _pick_first_existing(
            [run_dir / "best_model.eqx", run_dir / "final_model.eqx"]
        )
    if model_path is None:
        raise ValueError("Provide either --run or --model_path.")
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    inferred_run_dir = run_dir or model_path.parent
    config_path = _resolve_repo_path(args.config) or inferred_run_dir / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}. Pass --config explicitly."
        )

    system_config_path = _resolve_repo_path(args.system_config)
    if system_config_path is None:
        system_config_path = _pick_first_existing(
            [
                inferred_run_dir / "system_config.yaml",
                inferred_run_dir / "cstr_config.yaml",
            ]
        )
    if system_config_path is None or not system_config_path.exists():
        raise FileNotFoundError(
            "System config not found next to the model. Pass --system_config explicitly."
        )

    data_dir = _resolve_repo_path(args.data_dir)
    if data_dir is None:
        data_dir = _infer_data_dir_from_system_config(system_config_path)
    if data_dir is None:
        raise ValueError(
            "Could not infer data directory from system config. Pass --data_dir explicitly."
        )

    output_dir = _resolve_repo_path(args.output_dir)
    if output_dir is None:
        suffix = "eval_det" if args.predict_mode == "deterministic" else "eval_stoch"
        output_dir = inferred_run_dir / suffix

    return {
        "run_dir": inferred_run_dir,
        "model_path": model_path,
        "config_path": config_path,
        "system_config_path": system_config_path,
        "data_dir": data_dir,
        "output_dir": output_dir,
    }


def _init_metric_store():
    """Create a metric accumulator for scalar and per-state metrics."""
    return {
        "mse_1step": [],
        "mse_10step": [],
        "mse_laststep": [],
        "mse_fullseq": [],
        "mass_violation_mean": [],
        "mass_violation_max": [],
        "energy_violation_mean": [],
        "energy_violation_max": [],
        "rmse_per_state": [],
        "nrmse_per_state": [],
    }


def _predict_states(
    model: DigitalTwin,
    sample: dict,
    key,
    predict_mode: str,
):
    """Predict a trajectory for one validation sample."""
    initial_state = sample["states"][0]
    controls = sample["controls"]
    disturbances = sample["disturbances"]
    params = sample["params"]
    ts = sample["t"]

    if predict_mode == "deterministic":
        # Match the training/validation rollout path: use z_mean and mean_trajectory.
        _, z_mean, _ = model.encode(initial_state, params, controls[0], None)
        z_traj = model.latent_sde.mean_trajectory(
            ts,
            z_mean,
            controls,
            params,
            disturbances=disturbances,
        )
        return jax.vmap(lambda z, u: model.decode(z, params, u))(z_traj, controls)

    result = model.predict(initial_state, controls, disturbances, params, ts, key)
    return result["states"]


def _record_prediction_metrics(metric_store, true_states, pred_states, state_std):
    """Accumulate scalar and per-state prediction metrics."""
    rollout_len = int(true_states.shape[0])
    ten_step_horizon = min(10, rollout_len)

    squared_error = (true_states - pred_states) ** 2
    metric_store["mse_1step"].append(float(jnp.mean((true_states[1] - pred_states[1]) ** 2)))
    metric_store["mse_10step"].append(
        float(jnp.mean((true_states[:ten_step_horizon] - pred_states[:ten_step_horizon]) ** 2))
    )
    metric_store["mse_laststep"].append(
        float(jnp.mean((true_states[-1] - pred_states[-1]) ** 2))
    )
    metric_store["mse_fullseq"].append(float(jnp.mean(squared_error)))

    rmse_per_state = jnp.sqrt(jnp.mean(squared_error, axis=0))
    nrmse_per_state = jnp.sqrt(jnp.mean(((true_states - pred_states) / state_std) ** 2, axis=0))
    metric_store["rmse_per_state"].append(np.asarray(rmse_per_state))
    metric_store["nrmse_per_state"].append(np.asarray(nrmse_per_state))


def _record_physics_metrics(metric_store, pred_states, controls, disturbances, physics_diagnostic_fn, dt):
    """Accumulate conservation-law metrics in physical units.

    ``physics_diagnostic_fn`` is a callable with the signature
    ``(states, controls, disturbances, dt) -> dict[str, residual_array]``
    or ``None`` when no physics diagnostics are available.
    """
    if physics_diagnostic_fn is None:
        residuals = {}
    else:
        residuals = physics_diagnostic_fn(pred_states, controls, disturbances, dt)

    n = pred_states.shape[0] - 1
    mass_res = residuals.get("mass", zero_residual(n))
    energy_res = residuals.get("energy", zero_residual(n))

    metric_store["mass_violation_mean"].append(float(jnp.mean(mass_res)))
    metric_store["mass_violation_max"].append(float(jnp.max(mass_res)))
    metric_store["energy_violation_mean"].append(float(jnp.mean(energy_res)))
    metric_store["energy_violation_max"].append(float(jnp.max(energy_res)))

    return mass_res, energy_res


def _plot_priority_score(
    selection: str,
    true_states,
    pred_states,
    controls,
    disturbances,
    state_std,
) -> float:
    """Rank samples so saved plots focus on informative trajectories."""
    true_states = np.asarray(true_states)
    pred_states = np.asarray(pred_states)
    controls = np.asarray(controls)
    disturbances = np.asarray(disturbances)
    state_std = np.asarray(state_std)

    if selection == "prediction_error":
        return float(np.mean((true_states - pred_states) ** 2))
    if selection == "input_variation":
        return float(
            np.sum(np.ptp(controls, axis=0)) + np.sum(np.ptp(disturbances, axis=0))
        )

    # Default: favor windows with the richest state motion after normalizing by
    # dataset scale so temperatures do not dominate concentrations.
    normalized_state_range = np.ptp(true_states, axis=0) / state_std
    return float(np.sum(normalized_state_range))


def _summarize_metric_store(metric_store, state_names):
    """Convert metric lists into mean/std summary values."""
    summary = {}
    scalar_metric_names = (
        "mse_1step",
        "mse_10step",
        "mse_laststep",
        "mse_fullseq",
        "mass_violation_mean",
        "mass_violation_max",
        "energy_violation_mean",
        "energy_violation_max",
    )

    for name in scalar_metric_names:
        values = np.asarray(metric_store[name], dtype=float)
        summary[name] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
        }

    rmse_per_state = np.mean(np.stack(metric_store["rmse_per_state"]), axis=0)
    nrmse_per_state = np.mean(np.stack(metric_store["nrmse_per_state"]), axis=0)
    summary["rmse_per_state"] = {
        state_name: float(value)
        for state_name, value in zip(state_names, rmse_per_state)
    }
    summary["nrmse_per_state"] = {
        state_name: float(value)
        for state_name, value in zip(state_names, nrmse_per_state)
    }

    return summary


def _print_metric_summary(title: str, summary: dict):
    """Print a human-readable metric summary."""
    print(f"\n{title}:")
    print(
        f"  1-step MSE:   {summary['mse_1step']['mean']:.6f} ± "
        f"{summary['mse_1step']['std']:.6f}"
    )
    print(
        f"  10-step MSE:  {summary['mse_10step']['mean']:.6f} ± "
        f"{summary['mse_10step']['std']:.6f}"
    )
    print(
        f"  Last-step MSE:{summary['mse_laststep']['mean']:.6f} ± "
        f"{summary['mse_laststep']['std']:.6f}"
    )
    print(
        f"  Full-seq MSE: {summary['mse_fullseq']['mean']:.6f} ± "
        f"{summary['mse_fullseq']['std']:.6f}"
    )
    print("  Per-state RMSE:")
    for state_name, value in summary["rmse_per_state"].items():
        print(f"    {state_name}: {value:.6f}")
    print("  Per-state normalized RMSE:")
    for state_name, value in summary["nrmse_per_state"].items():
        print(f"    {state_name}: {value:.6f}")
    print("  Mass balance violation:")
    print(
        f"    Mean: {summary['mass_violation_mean']['mean']:.6f} ± "
        f"{summary['mass_violation_mean']['std']:.6f}"
    )
    print(
        f"    Max:  {summary['mass_violation_max']['mean']:.6f} ± "
        f"{summary['mass_violation_max']['std']:.6f}"
    )
    print("  Energy balance violation:")
    print(
        f"    Mean: {summary['energy_violation_mean']['mean']:.2e} ± "
        f"{summary['energy_violation_mean']['std']:.2e}"
    )
    print(
        f"    Max:  {summary['energy_violation_max']['mean']:.2e} ± "
        f"{summary['energy_violation_max']['std']:.2e}"
    )


def main():
    parser = argparse.ArgumentParser(description="Evaluate Digital Twin model")
    parser.add_argument(
        "--run",
        type=str,
        default=None,
        help=(
            "Run name under outputs/ (for example cstr_v3) or a direct run "
            "directory path. When supplied, model/config/system_config/output_dir "
            "are inferred automatically."
        ),
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help=(
            "Path to trained model. If omitted, --run is used to infer "
            "best_model.eqx/final_model.eqx."
        )
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to model config. Defaults to <run_dir>/config.yaml when inferable."
    )
    parser.add_argument(
        "--system_config",
        "--cstr_config",
        type=str,
        default=None,
        dest="system_config",
        help="Path to system config. Defaults to <run_dir>/system_config.yaml when inferable."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help=(
            "Data directory. Defaults to data/<system_name>/ based on the saved "
            "system config when inferable."
        )
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help=(
            "Output directory for plots. Defaults to <run_dir>/eval_det or "
            "<run_dir>/eval_stoch based on predict_mode."
        )
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=20,
        help="Number of ensemble samples"
    )
    parser.add_argument(
        "--n_trajectories",
        type=int,
        default=10,
        help="Number of validation subsequences to evaluate"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--predict_mode",
        type=str,
        choices=("deterministic", "stochastic"),
        default="deterministic",
        help="Trajectory rollout mode for metric computation"
    )
    parser.add_argument(
        "--plot_count",
        type=int,
        default=3,
        help="Number of sampled subsequences to plot"
    )
    parser.add_argument(
        "--plot_selection",
        type=str,
        choices=("sample_order", "state_variation", "prediction_error", "input_variation"),
        default="state_variation",
        help="How to prioritize which sampled subsequences get saved as plots"
    )
    parser.add_argument(
        "--summary_path",
        type=str,
        default=None,
        help="Path to write machine-readable evaluation summary JSON"
    )
    parser.add_argument(
        "--skip_ensemble",
        action="store_true",
        help="Skip ensemble uncertainty plots and calibration"
    )
    args = parser.parse_args()

    try:
        resolved_paths = _resolve_evaluation_paths(args)
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))

    args.model_path = str(resolved_paths["model_path"])
    args.config = str(resolved_paths["config_path"])
    args.system_config = str(resolved_paths["system_config_path"])
    args.data_dir = str(resolved_paths["data_dir"])
    args.output_dir = str(resolved_paths["output_dir"])
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("\n" + "="*60)
    print("DIGITAL TWIN EVALUATION")
    print("="*60)
    print(f"Model: {args.model_path}")
    print(f"Config: {args.config}")
    print(f"System config: {args.system_config}")
    print(f"Data: {args.data_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Predict mode: {args.predict_mode}")
    print("="*60)
    
    # Load configs
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    with open(args.system_config, "r") as f:
        system_config = yaml.safe_load(f)

    system_spec = get_system_spec(system_config)
    state_names = system_spec.state_names
    control_names = system_spec.control_names

    physics_diagnostic_fn = get_physics_diagnostic_fn(system_spec.name, system_config)

    # Initialize PRNG
    key = jax.random.PRNGKey(args.seed)

    # Load model
    print("\nLoading model...")
    model = DigitalTwin.load(args.model_path, config, system_spec=system_spec)
    param_counts = model.get_parameter_count()
    print(f"Model parameters: {param_counts['total']:,}")
    
    # Load validation dataset
    print("\nLoading validation dataset...")
    data_path = os.path.join(args.data_dir, "train_data.h5")
    full_dataset = TrajectoryDataset(
        data_path,
        seq_len=config["training"]["seq_len"],
        stride=config["training"]["stride"],
    )
    
    # Use last 20% as validation
    val_split = 0.2
    _, val_dataset = full_dataset.split(val_split)
    
    print(f"Validation samples: {val_dataset.n_samples}")
    
    model_metrics = _init_metric_store()
    baseline_metrics = _init_metric_store()
    state_std = jnp.asarray(val_dataset.get_normalization_stats()["state_std"]) + 1e-8
    plot_candidates = []
    
    # Evaluate on random validation subsequences
    print(f"\nEvaluating on {args.n_trajectories} validation subsequences...")
    n_eval = min(args.n_trajectories, val_dataset.n_samples)
    key, index_key = jax.random.split(key)
    sample_indices = np.asarray(
        jax.random.choice(index_key, val_dataset.n_samples, shape=(n_eval,), replace=False)
    )

    for sample_idx in sample_indices:
        sample = val_dataset[int(sample_idx)]
        
        initial_state = sample["states"][0]
        true_states = sample["states"]
        controls = sample["controls"]
        disturbances = sample["disturbances"]
        ts = sample["t"]
        
        # Predict with selected evaluation mode
        key, subkey = jax.random.split(key)
        pred_states = _predict_states(model, sample, subkey, args.predict_mode)
        _record_prediction_metrics(model_metrics, true_states, pred_states, state_std)

        baseline_states = jnp.broadcast_to(initial_state[None, :], true_states.shape)
        _record_prediction_metrics(baseline_metrics, true_states, baseline_states, state_std)

        # Samples from TrajectoryDataset are already in physical units.
        dt = float(ts[1] - ts[0])
        mass_res, energy_res = _record_physics_metrics(
            model_metrics, pred_states, controls, disturbances, physics_diagnostic_fn, dt
        )
        _record_physics_metrics(
            baseline_metrics, baseline_states, controls, disturbances, physics_diagnostic_fn, dt
        )

        plot_candidates.append(
            {
                "sample_idx": int(sample_idx),
                "score": _plot_priority_score(
                    args.plot_selection,
                    true_states,
                    pred_states,
                    controls,
                    disturbances,
                    state_std,
                ),
                "true_states": np.asarray(true_states),
                "pred_states": np.asarray(pred_states),
                "ts": np.asarray(ts),
                "controls": np.asarray(controls),
                "mass_res": np.asarray(mass_res),
                "energy_res": np.asarray(energy_res),
            }
        )

    if args.plot_selection == "sample_order":
        selected_plots = plot_candidates[:args.plot_count]
    else:
        selected_plots = sorted(
            plot_candidates,
            key=lambda candidate: candidate["score"],
            reverse=True,
        )[:args.plot_count]

    for plot_rank, candidate in enumerate(selected_plots):
        file_stub = f"{plot_rank}_idx_{candidate['sample_idx']}"

        fig = plot_trajectory_comparison(
            candidate["true_states"],
            candidate["pred_states"],
            candidate["ts"],
            state_names=state_names,
            controls=candidate["controls"],
            control_names=control_names,
            save_path=os.path.join(args.output_dir, f"trajectory_{file_stub}.png")
        )
        plt.close(fig)

        fig = plot_prediction_error(
            candidate["true_states"],
            candidate["pred_states"],
            candidate["ts"],
            state_names=state_names,
            save_path=os.path.join(args.output_dir, f"error_{file_stub}.png")
        )
        plt.close(fig)

        fig = plot_conservation_violation(
            candidate["mass_res"],
            candidate["energy_res"],
            candidate["ts"][:-1],
            save_path=os.path.join(args.output_dir, f"conservation_{file_stub}.png")
        )
        plt.close(fig)
    
    model_summary = _summarize_metric_store(model_metrics, state_names)
    baseline_summary = _summarize_metric_store(baseline_metrics, state_names)

    # Compute summary statistics
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    _print_metric_summary("Model Metrics", model_summary)
    _print_metric_summary("Persistence Baseline", baseline_summary)

    ensemble_summary = None
    if not args.skip_ensemble:
        print(f"\nGenerating ensemble predictions with {args.n_samples} samples...")
        sample = val_dataset[int(sample_indices[0])]
        key, subkey = jax.random.split(key)
        
        ensemble_result = model.predict_ensemble(
            sample["states"][0],
            sample["controls"],
            sample["disturbances"],
            sample["params"],
            sample["t"],
            subkey,
            n_samples=args.n_samples,
        )
        
        # Plot ensemble prediction
        fig = plot_trajectory_comparison(
            np.array(sample["states"]),
            np.array(ensemble_result["states_samples"]),
            np.array(sample["t"]),
            state_names=state_names,
            controls=np.array(sample["controls"]),
            control_names=control_names,
            pred_std=np.array(ensemble_result["states_std"]),
            save_path=os.path.join(args.output_dir, "ensemble_prediction.png")
        )
        plt.close(fig)
        
        # Compute calibration (what % of true values fall within ±2σ)
        true_states = sample["states"]
        pred_mean = ensemble_result["states_mean"]
        pred_std = ensemble_result["states_std"]
        
        within_2sigma = np.abs(true_states - pred_mean) <= 2 * pred_std
        calibration = float(np.mean(within_2sigma) * 100.0)

        ensemble_summary = {
            "sample_index": int(sample_indices[0]),
            "n_samples": args.n_samples,
            "within_2sigma_percent": calibration,
        }
        
        print(f"\nUncertainty Calibration:")
        print(f"  % within ±2σ: {calibration:.1f}% (ideal: 95%)")

    summary = {
        "model_path": args.model_path,
        "config_path": args.config,
        "data_dir": args.data_dir,
        "predict_mode": args.predict_mode,
        "sample_count": int(n_eval),
        "sample_indices": [int(idx) for idx in sample_indices.tolist()],
        "plot_selection": args.plot_selection,
        "plot_sample_indices": [candidate["sample_idx"] for candidate in selected_plots],
        "model_metrics": model_summary,
        "persistence_baseline": baseline_summary,
        "ensemble": ensemble_summary,
    }
    summary_path = args.summary_path or os.path.join(args.output_dir, "evaluation_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "="*60)
    print(f"Evaluation complete! Plots saved to {args.output_dir}")
    print(f"Summary saved to {summary_path}")
    print("="*60)


import matplotlib.pyplot as plt

if __name__ == "__main__":
    main()
