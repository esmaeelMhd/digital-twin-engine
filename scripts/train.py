"""Training script for Digital Twin model."""

import argparse
import os
import yaml
import jax
import json

from dte.models.digital_twin import DigitalTwin
from dte.training.trainer import Trainer
from dte.training.losses import LossComputer
from dte.data.dataset import TrajectoryDataset
from dte.simulators.cstr import CSTRParams


def main():
    parser = argparse.ArgumentParser(description="Train Digital Twin model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training_default.yaml",
        help="Path to training config"
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
        default="outputs/cstr_v1/",
        help="Output directory"
    )
    parser.add_argument(
        "--n_epochs",
        type=int,
        default=None,
        help="Number of epochs (overrides config)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size (overrides config)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Use Weights & Biases logging"
    )
    args = parser.parse_args()
    
    # Load configs
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    with open(args.cstr_config, "r") as f:
        cstr_config = yaml.safe_load(f)
    
    # Override config if specified
    if args.n_epochs is not None:
        config["training"]["n_epochs"] = args.n_epochs
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save configs
    with open(os.path.join(args.output_dir, "config.yaml"), "w") as f:
        yaml.dump(config, f)
    with open(os.path.join(args.output_dir, "cstr_config.yaml"), "w") as f:
        yaml.dump(cstr_config, f)
    
    print("\n" + "="*60)
    print("DIGITAL TWIN TRAINING")
    print("="*60)
    print(f"Training config: {args.config}")
    print(f"CSTR config: {args.cstr_config}")
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Seed: {args.seed}")
    print(f"Epochs: {config['training']['n_epochs']}")
    print(f"Batch size: {config['training']['batch_size']}")
    print("="*60)
    
    # Initialize PRNG
    key = jax.random.PRNGKey(args.seed)
    key_model, key_train = jax.random.split(key)
    
    # Load dataset
    print("\nLoading dataset...")
    data_path = os.path.join(args.data_dir, "train_data.h5")
    full_dataset = TrajectoryDataset(
        data_path,
        seq_len=config["training"]["seq_len"],
        stride=config["training"]["stride"],
    )
    
    # Split train/val
    val_split = config["training"]["val_split"]
    n_val = int(full_dataset.n_samples * val_split)
    n_train = full_dataset.n_samples - n_val
    
    print(f"Total samples: {full_dataset.n_samples}")
    print(f"Train samples: {n_train}")
    print(f"Val samples: {n_val}")
    
    # Create split datasets (simple approach - first n_train for train, rest for val)
    train_dataset, val_dataset = full_dataset.split(val_split)
    
    # Create CSTR params for physics losses
    cstr_params_dict = {k: float(v) for k, v in cstr_config["cstr"].items()}
    cstr_params = CSTRParams(**cstr_params_dict)
    
    # Create loss computer
    norm_stats = train_dataset.get_normalization_stats()
    loss_computer = LossComputer(config, norm_stats, cstr_params)
    
    # Create model
    print("\nInitializing model...")
    model = DigitalTwin.from_config(config, key_model)
    
    param_counts = model.get_parameter_count()
    print(f"Parameter counts:")
    for k, v in param_counts.items():
        print(f"  {k}: {v:,}")
    
    # Create trainer
    print("\nCreating trainer...")
    trainer = Trainer(
        model=model,
        loss_computer=loss_computer,
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )
    
    # Initialize wandb if requested
    if args.wandb:
        try:
            import wandb
            wandb.init(
                project="digital-twin-engine",
                config=config,
                name=os.path.basename(args.output_dir),
            )
            print("✓ Weights & Biases initialized")
        except ImportError:
            print("⚠ wandb not available, skipping W&B logging")
    
    # Train
    history = trainer.train(
        n_epochs=config["training"]["n_epochs"],
        output_dir=args.output_dir,
        key=key_train,
    )
    
    # Save training history
    history_path = os.path.join(args.output_dir, "training_history.json")
    with open(history_path, "w") as f:
        # Convert numpy arrays to lists for JSON
        history_json = {
            k: [float(x) for x in v] for k, v in history.items()
        }
        json.dump(history_json, f, indent=2)
    print(f"\n✓ Training history saved to {history_path}")
    
    print("\n" + "="*60)
    print("Training complete!")
    print("="*60)


if __name__ == "__main__":
    main()
