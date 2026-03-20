"""Training loop for Digital Twin model."""

from typing import Dict, Optional
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
from jaxtyping import Array, PRNGKeyArray
from tqdm import tqdm
import os

from dte.models.digital_twin import DigitalTwin
from dte.training.losses import LossComputer
from dte.data.dataset import TrajectoryDataset


class Trainer:
    """Trainer for Digital Twin model."""
    
    def __init__(
        self,
        model: DigitalTwin,
        loss_computer: LossComputer,
        config: dict,
        train_dataset: TrajectoryDataset,
        val_dataset: Optional[TrajectoryDataset] = None,
    ):
        """Initialize trainer.
        
        Args:
            model: Digital Twin model
            loss_computer: Loss computer
            config: Configuration dictionary
            train_dataset: Training dataset
            val_dataset: Validation dataset (optional)
        """
        self.model = model
        self.loss_computer = loss_computer
        self.config = config
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.step = 0
        
        # Setup optimizer
        opt_config = config["optimizer"]
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=1e-5,
            peak_value=opt_config["peak_lr"],
            warmup_steps=opt_config["warmup_steps"],
            decay_steps=opt_config["total_steps"],
            end_value=opt_config["end_lr"],
        )
        
        self.optimizer = optax.chain(
            optax.clip_by_global_norm(opt_config["gradient_clip"]),
            optax.adam(schedule),
        )
        
        self.opt_state = self.optimizer.init(eqx.filter(model, eqx.is_array))
        
        # Training history
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "step": [],
        }
        
        self.best_val_loss = float("inf")
    
    def compute_loss(
        self,
        model: DigitalTwin,
        batch: Dict[str, Array],
        key: PRNGKeyArray,
    ) -> tuple[float, Dict[str, float]]:
        """Compute all losses for a batch.
        
        Args:
            model: Digital Twin model
            batch: Batch data
            key: PRNG key
            
        Returns:
            Tuple of (total_loss, loss_dict)
        """
        batch_size = batch["states"].shape[0]
        seq_len = batch["states"].shape[1]
        
        # Extract data
        states = batch["states"]
        controls = batch["controls"]
        disturbances = batch["disturbances"]
        params_batch = batch["params"]
        ts = batch["t"]
        
        # Get loss weights (with KL annealing)
        weights = self.loss_computer.get_loss_weights(self.step)
        
        # Process each sequence in batch
        keys = jax.random.split(key, batch_size)
        
        def process_one(idx, k):
            # Get initial state
            initial_state = states[idx, 0]
            
            # Encode initial state
            key_enc, key_sde = jax.random.split(k)
            z0, z_mean, z_logvar = model.encode(
                initial_state, params_batch[idx], controls[idx, 0], key_enc
            )
            
            # Roll out latent SDE
            z_traj = model.latent_sde.mean_trajectory(
                ts[idx], z0, controls[idx], params_batch[idx]
            )
            
            # Decode all timesteps
            decode_fn = jax.vmap(
                lambda z, u: model.decode(z, params_batch[idx], u),
                in_axes=(0, 0)
            )
            pred_states = decode_fn(z_traj, controls[idx])
            
            return pred_states, z_mean, z_logvar
        
        # Process batch
        results = jax.vmap(process_one, in_axes=(0, 0))(jnp.arange(batch_size), keys)
        pred_states_batch, z_means, z_logvars = results
        
        # Normalize states for losses
        norm_stats = self.train_dataset.get_normalization_stats()
        pred_states_norm = (pred_states_batch - norm_stats["state_mean"]) / (norm_stats["state_std"] + 1e-8)
        true_states_norm = (states - norm_stats["state_mean"]) / (norm_stats["state_std"] + 1e-8)
        
        # Compute losses in normalized space
        loss_recon = self.loss_computer.reconstruction_loss(pred_states_norm, true_states_norm)
        loss_traj = self.loss_computer.trajectory_loss(pred_states_norm, true_states_norm)
        loss_kl = self.loss_computer.kl_divergence_loss(z_means, z_logvars)
        
        # Denormalize for physics losses
        pred_states_denorm = self.loss_computer.denormalize_states(pred_states_batch)
        controls_denorm = self.loss_computer.denormalize_controls(controls)
        disturbances_denorm = self.loss_computer.denormalize_disturbances(disturbances)
        
        # Compute physics losses
        dt = ts[0, 1] - ts[0, 0]  # Keep as JAX array, don't convert to float
        loss_mass = self.loss_computer.physics_mass_loss(
            pred_states_denorm, controls_denorm, disturbances_denorm, dt
        )
        loss_energy = self.loss_computer.physics_energy_loss(
            pred_states_denorm, controls_denorm, disturbances_denorm, dt
        )
        
        # Total loss
        total_loss = (
            weights["reconstruction"] * loss_recon
            + weights["kl"] * loss_kl
            + weights["trajectory"] * loss_traj
            + weights["mass_balance"] * loss_mass
            + weights["energy_balance"] * loss_energy
        )
        
        # Keep as JAX arrays, don't convert to Python floats inside JIT
        loss_dict = {
            "total": total_loss,
            "reconstruction": loss_recon,
            "kl": loss_kl,
            "trajectory": loss_traj,
            "mass_balance": loss_mass,
            "energy_balance": loss_energy,
        }
        
        return total_loss, loss_dict
    
    @eqx.filter_jit
    def train_step(
        self,
        model: DigitalTwin,
        opt_state,
        batch: Dict[str, Array],
        key: PRNGKeyArray,
    ):
        """Single training step.
        
        Args:
            model: Current model
            opt_state: Optimizer state
            batch: Batch data
            key: PRNG key
            
        Returns:
            Tuple of (new_model, new_opt_state, loss_dict)
        """
        # Compute loss and gradients
        (loss, loss_dict), grads = eqx.filter_value_and_grad(
            lambda m: self.compute_loss(m, batch, key),
            has_aux=True
        )(model)
        
        # Update model
        updates, new_opt_state = self.optimizer.update(
            grads, opt_state, eqx.filter(model, eqx.is_array)
        )
        new_model = eqx.apply_updates(model, updates)
        
        return new_model, new_opt_state, loss_dict
    
    def train_epoch(self, key: PRNGKeyArray) -> Dict[str, float]:
        """Train for one epoch.
        
        Args:
            key: PRNG key
            
        Returns:
            Dictionary of mean losses
        """
        batch_size = self.config["training"]["batch_size"]
        n_batches = self.train_dataset.n_samples // batch_size
        
        epoch_losses = {
            "total": [],
            "reconstruction": [],
            "kl": [],
            "trajectory": [],
            "mass_balance": [],
            "energy_balance": [],
        }
        
        pbar = tqdm(range(n_batches), desc="Training")
        for i in pbar:
            # Sample batch
            key, subkey = jax.random.split(key)
            batch = self.train_dataset.sample_batch(subkey, batch_size)
            
            # Training step
            key, subkey = jax.random.split(key)
            self.model, self.opt_state, loss_dict = self.train_step(
                self.model, self.opt_state, batch, subkey
            )
            
            # Record losses (convert to Python floats now)
            for k, v in loss_dict.items():
                epoch_losses[k].append(float(v))
            
            self.step += 1
            
            # Update progress bar
            pbar.set_postfix({"loss": f"{loss_dict['total']:.4f}"})
        
        # Compute mean losses
        mean_losses = {k: float(jnp.mean(jnp.array(v))) for k, v in epoch_losses.items()}
        
        return mean_losses
    
    def validate(self, key: PRNGKeyArray, n_batches: int = 50) -> Dict[str, float]:
        """Validate on validation set.
        
        Args:
            key: PRNG key
            n_batches: Number of batches to validate on
            
        Returns:
            Dictionary of validation losses
        """
        if self.val_dataset is None:
            return {}
        
        batch_size = self.config["training"]["batch_size"]
        n_batches = min(n_batches, self.val_dataset.n_samples // batch_size)
        
        val_losses = {
            "total": [],
            "reconstruction": [],
            "kl": [],
            "trajectory": [],
            "mass_balance": [],
            "energy_balance": [],
        }
        
        for i in range(n_batches):
            # Sample batch
            key, subkey = jax.random.split(key)
            batch = self.val_dataset.sample_batch(subkey, batch_size)
            
            # Compute loss (no gradients)
            key, subkey = jax.random.split(key)
            _, loss_dict = self.compute_loss(self.model, batch, subkey)
            
            # Record losses (convert to Python floats)
            for k, v in loss_dict.items():
                val_losses[k].append(float(v))
        
        # Compute mean losses
        mean_losses = {k: float(jnp.mean(jnp.array(v))) for k, v in val_losses.items()}
        
        return mean_losses
    
    def train(self, n_epochs: int, output_dir: str, key: PRNGKeyArray):
        """Full training loop.
        
        Args:
            n_epochs: Number of epochs
            output_dir: Output directory for checkpoints
            key: PRNG key
        """
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\nStarting training for {n_epochs} epochs...")
        print(f"Output directory: {output_dir}")
        
        for epoch in range(n_epochs):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch+1}/{n_epochs}")
            print(f"{'='*60}")
            
            # Train epoch
            key, subkey = jax.random.split(key)
            train_losses = self.train_epoch(subkey)
            
            print(f"\nTrain losses: {train_losses}")
            
            # Validate
            if self.val_dataset is not None and (epoch + 1) % 5 == 0:
                key, subkey = jax.random.split(key)
                val_losses = self.validate(subkey)
                print(f"Val losses: {val_losses}")
                
                # Save best model
                if val_losses["total"] < self.best_val_loss:
                    self.best_val_loss = val_losses["total"]
                    best_path = os.path.join(output_dir, "best_model.eqx")
                    self.model.save(best_path)
                    print(f"✓ Saved best model (val_loss={self.best_val_loss:.4f})")
            
            # Save checkpoint
            if (epoch + 1) % self.config["checkpointing"]["save_every"] == 0:
                ckpt_path = os.path.join(output_dir, f"checkpoint_epoch_{epoch+1}.eqx")
                self.model.save(ckpt_path)
            
            # Record history
            self.history["train_loss"].append(train_losses["total"])
            if self.val_dataset is not None and (epoch + 1) % 5 == 0:
                self.history["val_loss"].append(val_losses["total"])
            self.history["step"].append(self.step)
        
        # Save final model
        final_path = os.path.join(output_dir, "final_model.eqx")
        self.model.save(final_path)
        
        print(f"\n{'='*60}")
        print("Training complete!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"{'='*60}\n")
        
        return self.history
