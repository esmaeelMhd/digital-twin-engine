"""Script to generate training data for CSTR digital twin."""

import argparse
import os
import yaml
import jax
import jax.numpy as jnp

from dte.simulators.cstr import CSTRSimulator, CSTRParams
from dte.data.generation import DataGenerator


def main():
    parser = argparse.ArgumentParser(description="Generate CSTR training data")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/cstr_default.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--n_trajectories",
        type=int,
        default=10000,
        help="Number of trajectories to generate"
    )
    parser.add_argument(
        "--n_steps",
        type=int,
        default=1000,
        help="Number of steps per trajectory"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/cstr/",
        help="Output directory"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    args = parser.parse_args()
    
    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize simulator with default parameters
    params = CSTRParams(**config["cstr"])
    simulator = CSTRSimulator(params)
    
    # Initialize data generator
    generator = DataGenerator(simulator, config)
    
    # Generate dataset
    print(f"\nGenerating dataset with {args.n_trajectories} trajectories...")
    print(f"Steps per trajectory: {args.n_steps}")
    print(f"Random seed: {args.seed}")
    
    key = jax.random.PRNGKey(args.seed)
    dataset = generator.generate_dataset(
        key, n_trajectories=args.n_trajectories, n_steps=args.n_steps
    )
    
    # Save dataset
    output_path = os.path.join(args.output_dir, "train_data.h5")
    generator.save_dataset(dataset, output_path)
    
    # Print summary statistics
    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    print(f"Total trajectories: {args.n_trajectories}")
    print(f"Steps per trajectory: {args.n_steps}")
    print(f"Total timesteps: {args.n_trajectories * args.n_steps}")
    print(f"\nData shapes:")
    print(f"  States: {dataset['states'].shape}")
    print(f"  Controls: {dataset['controls'].shape}")
    print(f"  Disturbances: {dataset['disturbances'].shape}")
    print(f"  Params: {dataset['params'].shape}")
    
    print(f"\nState statistics (before normalization):")
    print(f"  Mean: {dataset['normalization']['state_mean']}")
    print(f"  Std:  {dataset['normalization']['state_std']}")
    
    print(f"\nControl statistics:")
    print(f"  Mean: {dataset['normalization']['control_mean']}")
    print(f"  Std:  {dataset['normalization']['control_std']}")
    
    print(f"\nSaved to: {output_path}")
    print(f"File size: {os.path.getsize(output_path) / 1024**2:.2f} MB")
    print("=" * 60)


if __name__ == "__main__":
    main()
