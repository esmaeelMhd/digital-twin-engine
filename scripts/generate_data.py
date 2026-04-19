"""Script to generate training data for a process system digital twin."""

import argparse
import os

from dte.utils.runtime import configure_runtime_logging

configure_runtime_logging()

import yaml
import jax
import jax.numpy as jnp

from dte.simulators.registry import get_system_spec, get_simulator

def main():
    parser = argparse.ArgumentParser(description="Generate process system training data")
    parser.add_argument(
        "--config",
        "--system_config",
        type=str,
        default="configs/cstr_default.yaml",
        dest="config",
        help="Path to system config file (CSTR, heat exchanger, etc.)"
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
    parser.add_argument(
        "--simulation_mode",
        type=str,
        default="dataset",
        choices=["dataset", "reference"],
        help="Rollout path to use during generation"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Number of trajectories to process together in dataset mode (auto if omitted)"
    )
    args = parser.parse_args()
    
    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Build system spec and simulator from registry
    system_spec = get_system_spec(config)
    system_name = system_spec.name

    simulator = get_simulator(system_name, config)
    from dte.data.generators.generic import GenericDataGenerator
    generator = GenericDataGenerator(simulator, config, system_spec)
    batch_size = (
        args.batch_size
        if args.batch_size is not None
        else generator.recommend_batch_size(jax.default_backend())
    )
    
    # Generate dataset
    print(f"\nGenerating dataset with {args.n_trajectories} trajectories...")
    print(f"Steps per trajectory: {args.n_steps}")
    print(f"Random seed: {args.seed}")
    print(f"Simulation mode: {args.simulation_mode}")
    print(f"Batch size: {batch_size}")
    
    key = jax.random.PRNGKey(args.seed)
    output_path = os.path.join(args.output_dir, "train_data.h5")
    dataset_summary = generator.generate_dataset_to_hdf5(
        key,
        output_path,
        n_trajectories=args.n_trajectories,
        n_steps=args.n_steps,
        simulation_mode=args.simulation_mode,
        batch_size=batch_size,
    )
    
    # Print summary statistics
    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    print(f"Total trajectories: {args.n_trajectories}")
    print(f"Steps per trajectory: {args.n_steps}")
    print(f"Total timesteps: {args.n_trajectories * args.n_steps}")
    print(f"\nData shapes:")
    print(f"  States: {dataset_summary['states_shape']}")
    print(f"  Controls: {dataset_summary['controls_shape']}")
    print(f"  Disturbances: {dataset_summary['disturbances_shape']}")
    print(f"  Params: {dataset_summary['params_shape']}")
    
    print(f"\nState statistics (before normalization):")
    print(f"  Mean: {dataset_summary['normalization']['state_mean']}")
    print(f"  Std:  {dataset_summary['normalization']['state_std']}")
    
    print(f"\nControl statistics:")
    print(f"  Mean: {dataset_summary['normalization']['control_mean']}")
    print(f"  Std:  {dataset_summary['normalization']['control_std']}")
    
    print(f"\nSaved to: {output_path}")
    print(f"File size: {os.path.getsize(output_path) / 1024**2:.2f} MB")
    if generator.last_profile:
        profile = generator.last_profile
        print("\nGeneration timing:")
        print(f"  Total: {profile['total_generation_seconds']:.2f} s")
        print(f"  Signal generation: {profile['signal_generation_seconds']:.2f} s")
        print(f"  Steady state: {profile['steady_state_seconds']:.2f} s")
        print(f"  Rollout: {profile['rollout_seconds']:.2f} s")
        print(f"  Measurement noise: {profile['measurement_noise_seconds']:.2f} s")
        print(f"  Validation: {profile['validation_seconds']:.2f} s")
        print(f"  Attempts: {int(profile['attempts'])}")
        print(f"  Invalid trajectories: {int(profile['invalid_trajectories'])}")
        print(f"  Exceptions: {int(profile['exceptions'])}")
        print(
            f"  Fast steady states: "
            f"{int(profile.get('steady_state_fast_successes', 0))}"
        )
        print(
            f"  Steady-state fallbacks: "
            f"{int(profile.get('steady_state_fallbacks', 0))}"
        )
        if "batch_size" in profile:
            print(f"  Profile batch size: {int(profile['batch_size'])}")
    print("=" * 60)


if __name__ == "__main__":
    main()
