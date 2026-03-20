"""Evaluation script for Digital Twin model."""

import argparse
import os
import yaml
import jax
import jax.numpy as jnp
import numpy as np
from pathlib import Path

from dte.models.digital_twin import DigitalTwin
from dte.data.dataset import TrajectoryDataset
from dte.physics.conservation import mass_balance_residual, energy_balance_residual
from dte.simulators.cstr import CSTRParams
from dte.utils.plotting import (
    plot_trajectory_comparison,
    plot_conservation_violation,
    plot_latent_space,
    plot_prediction_error,
)


def main():
    parser = argparse.ArgumentParser(description="Evaluate Digital Twin model")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to trained model"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to model config"
    )
    parser.add_argument(
        "--cstr_config",
        type=str,
        default="configs/cstr_default.yaml",
        help="Path to CSTR config"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/test/",
        help="Data directory"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/evaluation/",
        help="Output directory for plots"
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
        help="Number of validation trajectories to evaluate"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("\n" + "="*60)
    print("DIGITAL TWIN EVALUATION")
    print("="*60)
    print(f"Model: {args.model_path}")
    print(f"Config: {args.config}")
    print(f"Data: {args.data_dir}")
    print(f"Output: {args.output_dir}")
    print("="*60)
    
    # Load configs
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    with open(args.cstr_config, "r") as f:
        cstr_config = yaml.safe_load(f)
    
    # Initialize PRNG
    key = jax.random.PRNGKey(args.seed)
    
    # Load model
    print("\nLoading model...")
    model = DigitalTwin.load(args.model_path, config)
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
    
    # Create CSTR params for physics evaluation
    cstr_params_dict = {k: float(v) for k, v in cstr_config["cstr"].items()}
    cstr_params = CSTRParams(**cstr_params_dict)
    
    # Evaluation metrics
    all_metrics = {
        "mse_1step": [],
        "mse_10step": [],
        "mse_fullseq": [],
        "mass_violation_mean": [],
        "mass_violation_max": [],
        "energy_violation_mean": [],
        "energy_violation_max": [],
    }
    
    # Evaluate on random trajectories
    print(f"\nEvaluating on {args.n_trajectories} trajectories...")
    n_eval = min(args.n_trajectories, val_dataset.n_samples)
    
    for i in range(n_eval):
        # Get sample
        sample = val_dataset[i]
        
        initial_state = sample["states"][0]
        true_states = sample["states"]
        controls = sample["controls"]
        disturbances = sample["disturbances"]
        params = sample["params"]
        ts = sample["t"]
        
        # Predict
        key, subkey = jax.random.split(key)
        result = model.predict(initial_state, controls, disturbances, params, ts, subkey)
        pred_states = result["states"]
        
        # Compute MSE at different horizons
        mse_1step = float(jnp.mean((true_states[1] - pred_states[1])**2))
        mse_10step = float(jnp.mean((true_states[:10] - pred_states[:10])**2))
        mse_fullseq = float(jnp.mean((true_states - pred_states)**2))
        
        all_metrics["mse_1step"].append(mse_1step)
        all_metrics["mse_10step"].append(mse_10step)
        all_metrics["mse_fullseq"].append(mse_fullseq)
        
        # Denormalize for physics evaluation
        pred_states_denorm = val_dataset.denormalize_states(pred_states)
        controls_denorm = val_dataset.denormalize_controls(controls)
        disturbances_denorm = val_dataset.denormalize_disturbances(disturbances)
        
        # Compute conservation violations
        dt = float(ts[1] - ts[0])
        mass_res = mass_balance_residual(
            pred_states_denorm, controls_denorm, disturbances_denorm, cstr_params, dt
        )
        energy_res = energy_balance_residual(
            pred_states_denorm, controls_denorm, disturbances_denorm, cstr_params, dt
        )
        
        all_metrics["mass_violation_mean"].append(float(jnp.mean(mass_res)))
        all_metrics["mass_violation_max"].append(float(jnp.max(mass_res)))
        all_metrics["energy_violation_mean"].append(float(jnp.mean(energy_res)))
        all_metrics["energy_violation_max"].append(float(jnp.max(energy_res)))
        
        # Plot first few trajectories
        if i < 3:
            # Trajectory comparison
            fig = plot_trajectory_comparison(
                np.array(true_states),
                np.array(pred_states),
                np.array(ts),
                controls=np.array(controls),
                save_path=os.path.join(args.output_dir, f"trajectory_{i}.png")
            )
            plt.close(fig)
            
            # Prediction error
            fig = plot_prediction_error(
                np.array(true_states),
                np.array(pred_states),
                np.array(ts),
                save_path=os.path.join(args.output_dir, f"error_{i}.png")
            )
            plt.close(fig)
            
            # Conservation violations
            fig = plot_conservation_violation(
                np.array(mass_res),
                np.array(energy_res),
                np.array(ts[:-1]),
                save_path=os.path.join(args.output_dir, f"conservation_{i}.png")
            )
            plt.close(fig)
    
    # Compute summary statistics
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"\nPrediction Accuracy (MSE in normalized space):")
    print(f"  1-step MSE:     {np.mean(all_metrics['mse_1step']):.6f} ± {np.std(all_metrics['mse_1step']):.6f}")
    print(f"  10-step MSE:    {np.mean(all_metrics['mse_10step']):.6f} ± {np.std(all_metrics['mse_10step']):.6f}")
    print(f"  Full-seq MSE:   {np.mean(all_metrics['mse_fullseq']):.6f} ± {np.std(all_metrics['mse_fullseq']):.6f}")
    
    print(f"\nPhysics Constraint Satisfaction:")
    print(f"  Mass balance violation:")
    print(f"    Mean: {np.mean(all_metrics['mass_violation_mean']):.6f} ± {np.std(all_metrics['mass_violation_mean']):.6f}")
    print(f"    Max:  {np.mean(all_metrics['mass_violation_max']):.6f} ± {np.std(all_metrics['mass_violation_max']):.6f}")
    print(f"  Energy balance violation:")
    print(f"    Mean: {np.mean(all_metrics['energy_violation_mean']):.2e} ± {np.std(all_metrics['energy_violation_mean']):.2e}")
    print(f"    Max:  {np.mean(all_metrics['energy_violation_max']):.2e} ± {np.std(all_metrics['energy_violation_max']):.2e}")
    
    # Test ensemble prediction
    print(f"\nGenerating ensemble predictions with {args.n_samples} samples...")
    sample = val_dataset[0]
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
        controls=np.array(sample["controls"]),
        pred_std=np.array(ensemble_result["states_std"]),
        save_path=os.path.join(args.output_dir, "ensemble_prediction.png")
    )
    plt.close(fig)
    
    # Compute calibration (what % of true values fall within ±2σ)
    true_states = sample["states"]
    pred_mean = ensemble_result["states_mean"]
    pred_std = ensemble_result["states_std"]
    
    within_2sigma = np.abs(true_states - pred_mean) <= 2 * pred_std
    calibration = np.mean(within_2sigma) * 100
    
    print(f"\nUncertainty Calibration:")
    print(f"  % within ±2σ: {calibration:.1f}% (ideal: 95%)")
    
    print("\n" + "="*60)
    print(f"Evaluation complete! Plots saved to {args.output_dir}")
    print("="*60)


import matplotlib.pyplot as plt

if __name__ == "__main__":
    main()
