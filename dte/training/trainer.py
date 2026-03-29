"""Training loop for Digital Twin model."""

import math
from typing import Dict, Optional
import time
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


def _non_finite_loss_names(losses: Dict[str, float]) -> list[str]:
    """Return loss keys whose values are NaN or Inf."""

    return [
        name for name, value in losses.items()
        if not math.isfinite(float(value))
    ]


def _format_non_finite_reason(
    stage: str,
    losses: Dict[str, float],
    *,
    step: int,
    batch_index: int,
    n_batches: int,
    epoch: int | None = None,
) -> str:
    """Build a concise summary of non-finite losses."""

    bad_names = _non_finite_loss_names(losses)
    details = ", ".join(f"{name}={losses[name]}" for name in bad_names)
    location = f"step={step} batch={batch_index}/{n_batches}"
    if epoch is not None:
        location = f"epoch={epoch} " + location
    return f"non_finite_{stage}_loss: {location}; {details}"


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
            optax.adam(schedule, b1=0.95, b2=0.99),
        )
        
        self.opt_state = self.optimizer.init(eqx.filter(model, eqx.is_array))
        
        # Training history
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "step": [],
        }
        
        self.best_val_loss = float("inf")
        self.last_train_summary: Dict[str, float | int | bool | str | None] = {}
    
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
                ts[idx], z_mean, controls[idx], params_batch[idx]
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
        
        # TrajectoryDataset batches are already stored in physical units.
        pred_states_phys = pred_states_batch
        controls_phys = controls
        disturbances_phys = disturbances
        
        # Compute physics losses
        dt = ts[0, 1] - ts[0, 0]  # Keep as JAX array, don't convert to float
        loss_mass = self.loss_computer.physics_mass_loss(
            pred_states_phys, controls_phys, disturbances_phys, dt
        )
        loss_energy = self.loss_computer.physics_energy_loss(
            pred_states_phys, controls_phys, disturbances_phys, dt
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
    
    def train_epoch(
        self,
        key: PRNGKeyArray,
        deadline: Optional[float] = None,
    ) -> tuple[Optional[Dict[str, float]], bool, Optional[str]]:
        """Train for one epoch.
        
        Args:
            key: PRNG key
            deadline: Optional wall-clock deadline in perf_counter seconds
            
        Returns:
            Tuple of (mean losses, whether the time budget expired, failure reason)
        """
        batch_size = min(
            self.config["training"]["batch_size"],
            self.train_dataset.n_samples,
        )
        full_n_batches = max(1, self.train_dataset.n_samples // batch_size)
        max_batches_per_epoch = self.config["training"].get("max_batches_per_epoch")
        if max_batches_per_epoch is None:
            n_batches = full_n_batches
        else:
            n_batches = max(1, min(full_n_batches, int(max_batches_per_epoch)))
        
        epoch_losses = {
            "total": [],
            "reconstruction": [],
            "kl": [],
            "trajectory": [],
            "mass_balance": [],
            "energy_balance": [],
        }
        
        pbar = tqdm(range(n_batches), desc="Training")
        timed_out = False
        for i in pbar:
            if deadline is not None and time.perf_counter() >= deadline:
                timed_out = True
                break

            # Sample batch
            key, subkey = jax.random.split(key)
            batch = self.train_dataset.sample_batch(subkey, batch_size)
            
            # Training step
            key, subkey = jax.random.split(key)
            self.model, self.opt_state, loss_dict = self.train_step(
                self.model, self.opt_state, batch, subkey
            )

            # Record losses (convert to Python floats now)
            loss_dict_float = {k: float(v) for k, v in loss_dict.items()}
            non_finite_reason = _non_finite_loss_names(loss_dict_float)
            if non_finite_reason:
                failure_reason = _format_non_finite_reason(
                    "train",
                    loss_dict_float,
                    step=self.step,
                    batch_index=i + 1,
                    n_batches=n_batches,
                )
                return None, timed_out, failure_reason
            for k, v in loss_dict_float.items():
                epoch_losses[k].append(v)
            
            self.step += 1
            
            # Update progress bar
            pbar.set_postfix({"loss": f"{loss_dict_float['total']:.4f}"})

            if deadline is not None and time.perf_counter() >= deadline:
                timed_out = True
                break
        
        if not epoch_losses["total"]:
            raise RuntimeError(
                "No training batches were completed before the time budget expired."
            )

        # Compute mean losses
        mean_losses = {k: float(jnp.mean(jnp.array(v))) for k, v in epoch_losses.items()}
        
        return mean_losses, timed_out, None
    
    def validate(
        self,
        key: PRNGKeyArray,
        n_batches: int = 50,
        epoch: int | None = None,
    ) -> tuple[Optional[Dict[str, float]], Optional[str]]:
        """Validate on validation set.
        
        Args:
            key: PRNG key
            n_batches: Number of batches to validate on
            
        Returns:
            Tuple of (validation losses, failure reason)
        """
        if self.val_dataset is None:
            return {}, None
        
        batch_size = min(
            self.config["training"]["batch_size"],
            self.val_dataset.n_samples,
        )
        n_batches = max(1, min(n_batches, self.val_dataset.n_samples // batch_size))
        max_val_batches = self.config.get("checkpointing", {}).get("max_val_batches")
        if max_val_batches is not None:
            n_batches = max(1, min(n_batches, int(max_val_batches)))
        
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
            loss_dict_float = {k: float(v) for k, v in loss_dict.items()}
            if _non_finite_loss_names(loss_dict_float):
                failure_reason = _format_non_finite_reason(
                    "val",
                    loss_dict_float,
                    step=self.step,
                    batch_index=i + 1,
                    n_batches=n_batches,
                    epoch=epoch,
                )
                return None, failure_reason
            for k, v in loss_dict_float.items():
                val_losses[k].append(v)
        
        # Compute mean losses
        mean_losses = {k: float(jnp.mean(jnp.array(v))) for k, v in val_losses.items()}
        
        return mean_losses, None
    
    def train(
        self,
        n_epochs: int,
        output_dir: str,
        key: PRNGKeyArray,
        time_budget_seconds: Optional[float] = None,
    ):
        """Full training loop.
        
        Args:
            n_epochs: Number of epochs
            output_dir: Output directory for checkpoints
            key: PRNG key
            time_budget_seconds: Optional wall-clock budget for the training loop
        """
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\nStarting training for {n_epochs} epochs...")
        print(f"Output directory: {output_dir}")
        if time_budget_seconds is not None:
            print(f"Wall-clock budget: {time_budget_seconds:.1f} seconds")

        val_every = self.config.get("checkpointing", {}).get("val_every", 5)
        start_time = time.perf_counter()
        deadline = None
        if time_budget_seconds is not None:
            deadline = start_time + time_budget_seconds
        timed_out = False
        epochs_completed = 0
        failure_reason: str | None = None
        non_finite_detected = False
        
        for epoch in range(n_epochs):
            if deadline is not None and time.perf_counter() >= deadline:
                timed_out = True
                break

            print(f"\n{'='*60}")
            print(f"Epoch {epoch+1}/{n_epochs}")
            print(f"{'='*60}")
            
            # Train epoch
            key, subkey = jax.random.split(key)
            train_losses, epoch_timed_out, epoch_failure = self.train_epoch(
                subkey,
                deadline=deadline,
            )

            if epoch_failure is not None:
                failure_reason = epoch_failure
                non_finite_detected = True
                print(f"\nTraining stopped early: {failure_reason}")
                break

            epochs_completed += 1

            print(f"\nTrain losses: {train_losses}")
            
            # Validate
            should_validate = False
            if self.val_dataset is not None:
                should_validate = (
                    (epoch + 1) % val_every == 0
                    or epoch == n_epochs - 1
                    or epoch_timed_out
                )

            if should_validate:
                key, subkey = jax.random.split(key)
                val_losses, val_failure = self.validate(subkey, epoch=epoch + 1)
                if val_failure is not None:
                    failure_reason = val_failure
                    non_finite_detected = True
                    print(f"Validation stopped early: {failure_reason}")
                    break
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
            if should_validate:
                self.history["val_loss"].append(val_losses["total"])
            self.history["step"].append(self.step)

            if epoch_timed_out:
                timed_out = True
                break
        
        # Save final model
        final_path = os.path.join(output_dir, "final_model.eqx")
        self.model.save(final_path)

        training_seconds = time.perf_counter() - start_time
        if deadline is not None and time.perf_counter() >= deadline:
            timed_out = True
        best_val_loss = None
        if self.best_val_loss != float("inf"):
            best_val_loss = float(self.best_val_loss)
        
        print(f"\n{'='*60}")
        print("Training complete!")
        if best_val_loss is not None:
            print(f"Best validation loss: {best_val_loss:.4f}")
        if failure_reason is not None:
            print(f"Failure reason: {failure_reason}")
        print(f"Epochs completed: {epochs_completed}")
        print(f"Training seconds: {training_seconds:.2f}")
        print(f"Timed out: {timed_out}")
        print(f"{'='*60}\n")

        self.last_train_summary = {
            "epochs_completed": epochs_completed,
            "timed_out": timed_out,
            "training_seconds": training_seconds,
            "best_val_loss": best_val_loss,
            "steps_completed": self.step,
            "failure_reason": failure_reason,
            "non_finite_detected": non_finite_detected,
        }
        
        return self.history
