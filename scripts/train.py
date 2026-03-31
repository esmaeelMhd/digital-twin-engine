"""Training script for Digital Twin model."""

import argparse
import math
import os

from dte.utils.runtime import configure_runtime_logging

configure_runtime_logging()

import yaml
import jax
import json

from dte.models.digital_twin import DigitalTwin
from dte.training.trainer import Trainer
from dte.training.losses import LossComputer
from dte.data.dataset import TrajectoryDataset
from dte.simulators.registry import get_system_spec, get_simulator


def _json_safe_float(value):
    """Return a JSON-safe float or None for NaN/Inf/non-numeric values."""

    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def main():
    parser = argparse.ArgumentParser(description="Train Digital Twin model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training_default.yaml",
        help="Path to training config"
    )
    parser.add_argument(
        "--system_config",
        "--cstr_config",  # backwards-compatible alias
        type=str,
        default="configs/cstr_default.yaml",
        dest="system_config",
        help="Path to system config (CSTR, heat exchanger, etc.)"
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
        "--time_budget_minutes",
        type=float,
        default=None,
        help="Optional wall-clock training budget in minutes"
    )
    parser.add_argument(
        "--summary_path",
        type=str,
        default=None,
        help="Path to write machine-readable training summary"
    )
    parser.add_argument(
        "--val_every",
        type=int,
        default=None,
        help="Validation cadence in epochs (overrides config checkpointing.val_every)"
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Use Weights & Biases logging"
    )
    parser.add_argument(
        "--finetune",
        type=str,
        default=None,
        metavar="CHECKPOINT_PATH",
        help=(
            "Path to a pre-trained model checkpoint.  When supplied, the "
            "encoder and latent-SDE are frozen and only the decoder is updated "
            "(few-shot transfer learning mode).  Supports --n_epochs to control "
            "the number of fine-tuning epochs."
        ),
    )
    parser.add_argument(
        "--finetune_part",
        type=str,
        default="decoder",
        choices=["decoder", "encoder", "all"],
        help=(
            "Which part of the model to fine-tune when --finetune is used. "
            "'decoder' (default): freeze encoder+SDE, update only decoder. "
            "'encoder': freeze decoder+SDE, update only encoder. "
            "'all': update all parameters (standard transfer with warm start)."
        ),
    )
    args = parser.parse_args()
    
    # Load configs
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    config.setdefault("model", {})["initial_diffusion_scale"] = 0.0001
    config.setdefault("model", {}).setdefault("disturbance_dim", 2)
    config.setdefault("training", {})["peak_lr"] = 2.0e-3
    config.setdefault("training", {})["gradient_clip"] = 0.5
    config.setdefault("loss_weights", {})
    config["loss_weights"]["kl"] = 0.0

    with open(args.system_config, "r") as f:
        system_config = yaml.safe_load(f)

    # Build generic SystemSpec from config
    system_spec = get_system_spec(system_config)
    
    # Override config if specified
    if args.n_epochs is not None:
        config["training"]["n_epochs"] = args.n_epochs
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
    config.setdefault("checkpointing", {})
    if args.time_budget_minutes is not None:
        config["training"].setdefault("max_batches_per_epoch", 8)
        config["checkpointing"].setdefault("max_val_batches", 4)
    if args.val_every is not None:
        config["checkpointing"]["val_every"] = args.val_every
    else:
        config["checkpointing"].setdefault("val_every", 5)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Save configs
    with open(os.path.join(args.output_dir, "config.yaml"), "w") as f:
        yaml.dump(config, f)
    with open(os.path.join(args.output_dir, "system_config.yaml"), "w") as f:
        yaml.dump(system_config, f)
    # Write legacy alias so existing evaluation scripts still find the file
    with open(os.path.join(args.output_dir, "cstr_config.yaml"), "w") as f:
        yaml.dump(system_config, f)

    print("\n" + "="*60)
    print("DIGITAL TWIN TRAINING")
    print("="*60)
    print(f"Training config: {args.config}")
    print(f"System config: {args.system_config} (system={system_spec.name})")
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Seed: {args.seed}")
    print(f"Epochs: {config['training']['n_epochs']}")
    print(f"Batch size: {config['training']['batch_size']}")
    if config["training"].get("max_batches_per_epoch") is not None:
        print(f"Max batches per epoch: {config['training']['max_batches_per_epoch']}")
    if config["checkpointing"].get("max_val_batches") is not None:
        print(f"Max validation batches: {config['checkpointing']['max_val_batches']}")
    print(f"Validation every: {config['checkpointing']['val_every']} epoch(s)")
    if args.time_budget_minutes is not None:
        print(f"Time budget: {args.time_budget_minutes:.2f} minutes")
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
    train_dataset, val_dataset = full_dataset.split(val_split)
    
    print(f"Total samples: {full_dataset.n_samples}")
    print(f"Train samples: {train_dataset.n_samples}")
    print(f"Val samples: {val_dataset.n_samples}")
    
    # Build system-specific physics loss
    system_name = system_spec.name
    if system_name == "cstr":
        from dte.physics.cstr import CSTRPhysicsLoss
        from dte.simulators.cstr import CSTRParams
        cstr_cfg = system_config.get("cstr", {})
        cstr_params = CSTRParams(**{k: float(v) for k, v in cstr_cfg.items()})
        physics_loss = CSTRPhysicsLoss(cstr_params)
    elif system_name == "heat_exchanger":
        from dte.physics.heat_exchanger import HeatExchangerPhysicsLoss
        from dte.simulators.heat_exchanger import HeatExchangerParams
        hx_cfg = system_config.get("heat_exchanger", {})
        hx_params = HeatExchangerParams(**{k: float(v) for k, v in hx_cfg.items()})
        physics_loss = HeatExchangerPhysicsLoss(hx_params)
    else:
        from dte.physics.base import NullPhysicsLoss
        physics_loss = NullPhysicsLoss()
        print(f"Note: no physics loss implementation found for system '{system_name}', using null.")

    # Create loss computer
    norm_stats = train_dataset.get_normalization_stats()
    loss_computer = LossComputer(
        config,
        norm_stats,
        physics_loss=physics_loss,
        state_names=system_spec.state_names,
    )

    # Create model (from scratch or from pre-trained checkpoint)
    print("\nInitializing model...")
    is_finetune = args.finetune is not None
    if is_finetune:
        from dte.training.transfer import apply_finetune_mask
        print(f"Fine-tune mode: loading checkpoint from {args.finetune}")
        model = DigitalTwin.load(args.finetune, config, system_spec=system_spec)
        model = apply_finetune_mask(model, part=args.finetune_part)
        print(f"Fine-tune part: {args.finetune_part} (other components frozen)")
    else:
        model = DigitalTwin.from_config(config, key_model, system_spec=system_spec)

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
        time_budget_seconds=(
            args.time_budget_minutes * 60.0 if args.time_budget_minutes is not None else None
        ),
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

    summary = {
        "best_val_loss": _json_safe_float(trainer.last_train_summary.get("best_val_loss")),
        "final_train_loss": _json_safe_float(history["train_loss"][-1] if history["train_loss"] else None),
        "final_val_loss": _json_safe_float(history["val_loss"][-1] if history["val_loss"] else None),
        "epochs_completed": trainer.last_train_summary.get("epochs_completed", 0),
        "steps_completed": trainer.last_train_summary.get("steps_completed", trainer.step),
        "timed_out": trainer.last_train_summary.get("timed_out", False),
        "training_seconds": _json_safe_float(trainer.last_train_summary.get("training_seconds")),
        "time_budget_seconds": (
            _json_safe_float(args.time_budget_minutes * 60.0) if args.time_budget_minutes is not None else None
        ),
        "seed": args.seed,
        "batch_size": config["training"]["batch_size"],
        "max_batches_per_epoch": config["training"].get("max_batches_per_epoch"),
        "max_val_batches": config["checkpointing"].get("max_val_batches"),
        "n_epochs_requested": config["training"]["n_epochs"],
        "val_every": config["checkpointing"]["val_every"],
        "failure_reason": trainer.last_train_summary.get("failure_reason"),
        "non_finite_detected": trainer.last_train_summary.get("non_finite_detected", False),
        "output_dir": args.output_dir,
    }
    summary_path = args.summary_path or os.path.join(args.output_dir, "training_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Training summary saved to {summary_path}")
    print(f"best_val_loss: {summary['best_val_loss']}")
    print(f"training_seconds: {summary['training_seconds']}")
    print(f"epochs_completed: {summary['epochs_completed']}")
    print(f"timed_out: {summary['timed_out']}")
    if summary["failure_reason"] is not None:
        print(f"failure_reason: {summary['failure_reason']}")
    
    print("\n" + "="*60)
    print("Training complete!")
    print("="*60)

    return summary


if __name__ == "__main__":
    main()
