"""Trainer for the shared universal digital twin."""

from __future__ import annotations

import math
import os
import time
from typing import Dict, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, PRNGKeyArray
from tqdm import tqdm

from dte.data.multi_system_dataset import MultiSystemTrajectoryDataset
from dte.models.universal_digital_twin import UniversalDigitalTwin


def _non_finite_loss_names(losses: Dict[str, float]) -> list[str]:
    return [name for name, value in losses.items() if not math.isfinite(float(value))]


def _masked_mean(values: Array, mask: Array) -> Array:
    mask = mask.astype(values.dtype)
    denom = jnp.maximum(jnp.sum(mask), jnp.asarray(1.0, dtype=values.dtype))
    return jnp.sum(values * mask) / denom


def _masked_huber(diff: Array, mask: Array) -> Array:
    loss = jnp.where(
        jnp.abs(diff) < 0.01,
        0.5 * diff ** 2,
        0.01 * jnp.abs(diff) - 0.00005,
    )
    return _masked_mean(loss, mask)


class UniversalTrainer:
    """Trainer for a shared-checkpoint universal model."""

    def __init__(
        self,
        model: UniversalDigitalTwin,
        config: dict,
        train_dataset: MultiSystemTrajectoryDataset,
        val_dataset: Optional[MultiSystemTrajectoryDataset] = None,
    ):
        self.model = model
        self.config = config
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.step = 0

        opt_cfg = config["optimizer"]
        warmup_steps = int(opt_cfg["warmup_steps"])
        total_steps = max(int(opt_cfg["total_steps"]), warmup_steps + 1)
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=float(opt_cfg["peak_lr"]),
            warmup_steps=warmup_steps,
            decay_steps=total_steps,
            end_value=float(opt_cfg["end_lr"]),
        )
        self.optimizer = optax.chain(
            optax.clip_by_global_norm(float(opt_cfg["gradient_clip"])),
            optax.adam(schedule, b1=0.95, b2=0.99),
        )

        self.filter_spec = self.model.trainable_filter_spec()
        trainable, _ = eqx.partition(self.model, self.filter_spec)
        self.opt_state = self.optimizer.init(trainable)

        self.best_val_loss = float("inf")
        self.history = {"train_loss": [], "val_loss": [], "step": []}
        self.last_train_summary: Dict[str, float | int | bool | str | None] = {}

    def _normalize_batch(self, model: UniversalDigitalTwin, batch: Dict[str, Array]) -> Dict[str, Array]:
        system_ids = batch["system_id"]
        state_mask = batch["state_mask"].astype(jnp.float32)
        control_mask = batch["control_mask"].astype(jnp.float32)
        disturbance_mask = batch["disturbance_mask"].astype(jnp.float32)
        param_mask = batch["param_mask"].astype(jnp.float32)
        time_mask = batch["time_mask"].astype(jnp.float32)

        states_norm = model.normalize_states(batch["states"], system_ids) * state_mask[:, None, :]
        controls_norm = (
            model.normalize_controls(batch["controls"], system_ids) * control_mask[:, None, :]
        )
        disturbances_norm = (
            model.normalize_disturbances(batch["disturbances"], system_ids)
            * disturbance_mask[:, None, :]
        )
        params_scaled = model.scale_params(batch["params"], system_ids) * param_mask

        return {
            "states_norm": states_norm,
            "controls_norm": controls_norm,
            "disturbances_norm": disturbances_norm,
            "params_scaled": params_scaled,
            "state_mask": state_mask,
            "control_mask": control_mask,
            "disturbance_mask": disturbance_mask,
            "param_mask": param_mask,
            "time_mask": time_mask,
            "system_id": system_ids,
            "t": batch["t"],
        }

    def compute_loss(
        self,
        model: UniversalDigitalTwin,
        batch: Dict[str, Array],
        key: PRNGKeyArray,
    ) -> tuple[Array, Dict[str, Array]]:
        normalized = self._normalize_batch(model, batch)
        batch_size = normalized["states_norm"].shape[0]
        sample_keys = jax.random.split(key, batch_size)

        def process_one(
            state_traj,
            control_traj,
            disturbance_traj,
            params_scaled,
            ts,
            state_mask,
            control_mask,
            disturbance_mask,
            param_mask,
            system_id,
            sample_key,
        ):
            key_enc, _ = jax.random.split(sample_key)
            z0, z_mean, z_logvar = model.encode(
                state_traj[0],
                params_scaled,
                control_traj[0],
                state_mask,
                control_mask,
                param_mask,
                system_id,
                key=key_enc,
            )
            recon0 = model.decode(
                z_mean,
                params_scaled,
                control_traj[0],
                state_mask,
                control_mask,
                param_mask,
                system_id,
            )
            z_traj = model.rollout_latent(
                ts,
                z0,
                control_traj,
                disturbance_traj,
                params_scaled,
                control_mask,
                disturbance_mask,
                param_mask,
                system_id,
            )
            pred_states = jax.vmap(
                lambda z, u: model.decode(
                    z,
                    params_scaled,
                    u,
                    state_mask,
                    control_mask,
                    param_mask,
                    system_id,
                )
            )(z_traj, control_traj)

            def teacher_forced_step(state_t, u_t, u_tp1, d_t, d_tp1, t_t, t_tp1):
                _, z_mean_t, _ = model.encode(
                    state_t,
                    params_scaled,
                    u_t,
                    state_mask,
                    control_mask,
                    param_mask,
                    system_id,
                    None,
                )
                z_next = model.latent_step(
                    z_mean_t,
                    u_t,
                    u_tp1,
                    d_t,
                    d_tp1,
                    params_scaled,
                    control_mask,
                    disturbance_mask,
                    param_mask,
                    system_id,
                    t_tp1 - t_t,
                )
                return model.decode(
                    z_next,
                    params_scaled,
                    u_tp1,
                    state_mask,
                    control_mask,
                    param_mask,
                    system_id,
                )

            pred_next = jax.vmap(teacher_forced_step)(
                state_traj[:-1],
                control_traj[:-1],
                control_traj[1:],
                disturbance_traj[:-1],
                disturbance_traj[1:],
                ts[:-1],
                ts[1:],
            )

            return recon0, pred_states, pred_next, z_mean, z_logvar

        recon0, pred_states, pred_next, z_means, z_logvars = jax.vmap(process_one)(
            normalized["states_norm"],
            normalized["controls_norm"],
            normalized["disturbances_norm"],
            normalized["params_scaled"],
            normalized["t"],
            normalized["state_mask"],
            normalized["control_mask"],
            normalized["disturbance_mask"],
            normalized["param_mask"],
            normalized["system_id"],
            sample_keys,
        )

        state_time_mask = normalized["time_mask"][:, :, None] * normalized["state_mask"][:, None, :]
        next_state_mask = (
            normalized["time_mask"][:, 1:, None] * normalized["state_mask"][:, None, :]
        )

        loss_recon = _masked_huber(
            recon0 - normalized["states_norm"][:, 0, :],
            normalized["state_mask"],
        )
        loss_traj = _masked_huber(
            pred_states - normalized["states_norm"],
            state_time_mask,
        )
        loss_one_step = _masked_huber(
            pred_next - normalized["states_norm"][:, 1:, :],
            next_state_mask,
        )
        kl = -0.5 * jnp.sum(1 + z_logvars - z_means ** 2 - jnp.exp(z_logvars), axis=-1)
        loss_kl = jnp.mean(kl)

        weights = self.config["loss_weights"]
        total_loss = (
            float(weights.get("reconstruction", 1.0)) * loss_recon
            + float(weights.get("trajectory", 1.0)) * loss_traj
            + float(weights.get("one_step", 0.0)) * loss_one_step
            + float(weights.get("kl", 0.0)) * loss_kl
        )
        loss_dict = {
            "total": total_loss,
            "reconstruction": loss_recon,
            "trajectory": loss_traj,
            "one_step": loss_one_step,
            "kl": loss_kl,
        }
        return total_loss, loss_dict

    @eqx.filter_jit
    def train_step(self, trainable, frozen, opt_state, batch: Dict[str, Array], key: PRNGKeyArray):
        def loss_fn(trainable_model):
            model = eqx.combine(trainable_model, frozen)
            return self.compute_loss(model, batch, key)

        (loss, loss_dict), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(trainable)
        updates, new_opt_state = self.optimizer.update(grads, opt_state, trainable)
        new_trainable = eqx.apply_updates(trainable, updates)
        new_model = eqx.combine(new_trainable, frozen)
        return new_model, new_opt_state, loss_dict

    @eqx.filter_jit
    def eval_step(self, model: UniversalDigitalTwin, batch: Dict[str, Array], key: PRNGKeyArray):
        _, loss_dict = self.compute_loss(model, batch, key)
        return loss_dict

    def _validate_batches(
        self,
        dataset: MultiSystemTrajectoryDataset,
        key: PRNGKeyArray,
        n_batches: int,
        *,
        system_probabilities: Array | None = None,
    ) -> Dict[str, float]:
        batch_size = min(int(self.config["training"]["batch_size"]), dataset.n_samples)
        losses: Dict[str, list] = {"total": [], "reconstruction": [], "trajectory": [], "one_step": [], "kl": []}
        for _ in range(n_batches):
            key, sample_key, loss_key = jax.random.split(key, 3)
            batch = dataset.sample_batch(
                sample_key,
                batch_size=batch_size,
                seq_len=int(self.config["training"]["seq_len"]),
                system_probabilities=system_probabilities,
            )
            loss_dict = self.eval_step(self.model, batch, loss_key)
            loss_dict_float = {name: float(value) for name, value in loss_dict.items()}
            if _non_finite_loss_names(loss_dict_float):
                raise RuntimeError(f"Non-finite validation losses: {loss_dict_float}")
            for name, value in loss_dict_float.items():
                losses.setdefault(name, []).append(value)
        return {name: float(jnp.mean(jnp.asarray(values))) for name, values in losses.items()}

    def evaluate_per_system(self, key: PRNGKeyArray, n_batches: int = 4) -> Dict[str, Dict[str, float]]:
        dataset = self.val_dataset or self.train_dataset
        results = {}
        for idx, name in enumerate(dataset.system_names):
            probs = jnp.zeros(dataset.n_systems, dtype=jnp.float32).at[idx].set(1.0)
            key, subkey = jax.random.split(key)
            results[name] = self._validate_batches(
                dataset,
                subkey,
                n_batches=n_batches,
                system_probabilities=probs,
            )
        return results

    def train(
        self,
        n_epochs: int,
        output_dir: str,
        key: PRNGKeyArray,
        time_budget_seconds: Optional[float] = None,
    ):
        os.makedirs(output_dir, exist_ok=True)
        batch_size = min(int(self.config["training"]["batch_size"]), self.train_dataset.n_samples)
        full_batches = max(1, self.train_dataset.n_samples // batch_size)
        max_batches_per_epoch = self.config["training"].get("max_batches_per_epoch")
        n_batches = full_batches if max_batches_per_epoch is None else max(1, min(full_batches, int(max_batches_per_epoch)))
        val_every = int(self.config.get("checkpointing", {}).get("val_every", 1))
        max_val_batches = int(self.config.get("checkpointing", {}).get("max_val_batches", 4))
        save_every = int(self.config.get("checkpointing", {}).get("save_every", 5))

        start_time = time.perf_counter()
        deadline = None if time_budget_seconds is None else start_time + time_budget_seconds
        timed_out = False
        epochs_completed = 0
        failure_reason = None

        for epoch in range(n_epochs):
            if deadline is not None and time.perf_counter() >= deadline:
                timed_out = True
                break

            train_losses: Dict[str, list] = {"total": [], "reconstruction": [], "trajectory": [], "one_step": [], "kl": []}
            pbar = tqdm(range(n_batches), desc="Universal Training")
            for batch_index in pbar:
                if deadline is not None and time.perf_counter() >= deadline:
                    timed_out = True
                    break
                key, sample_key, step_key = jax.random.split(key, 3)
                batch = self.train_dataset.sample_batch(
                    sample_key,
                    batch_size=batch_size,
                    seq_len=int(self.config["training"]["seq_len"]),
                )
                trainable, frozen = eqx.partition(self.model, self.filter_spec)
                self.model, self.opt_state, loss_dict = self.train_step(
                    trainable,
                    frozen,
                    self.opt_state,
                    batch,
                    step_key,
                )
                loss_dict_float = {name: float(value) for name, value in loss_dict.items()}
                if _non_finite_loss_names(loss_dict_float):
                    failure_reason = f"non_finite_train_loss at epoch={epoch+1}, batch={batch_index+1}: {loss_dict_float}"
                    break
                for name, value in loss_dict_float.items():
                    train_losses.setdefault(name, []).append(value)
                self.step += 1
                pbar.set_postfix({"loss": f"{loss_dict_float['total']:.4f}"})

            if failure_reason is not None:
                break

            if not train_losses["total"]:
                failure_reason = "no training batches completed before timeout"
                break

            mean_train_losses = {
                name: float(jnp.mean(jnp.asarray(values)))
                for name, values in train_losses.items()
            }
            print(f"\nTrain losses: {mean_train_losses}")
            epochs_completed += 1

            should_validate = self.val_dataset is not None and (
                (epoch + 1) % val_every == 0 or epoch == n_epochs - 1 or timed_out
            )
            if should_validate:
                print("Running universal validation...")
                key, val_key = jax.random.split(key)
                val_losses = self._validate_batches(self.val_dataset, val_key, max_val_batches)
                print(f"Val losses: {val_losses}")
                if val_losses["total"] < self.best_val_loss:
                    self.best_val_loss = val_losses["total"]
                    self.model.save(os.path.join(output_dir, "best_model.eqx"))
                    print(f"✓ Saved best universal model (val_loss={self.best_val_loss:.4f})")
                self.history["val_loss"].append(val_losses["total"])

            if (epoch + 1) % save_every == 0:
                self.model.save(os.path.join(output_dir, f"checkpoint_epoch_{epoch+1}.eqx"))

            self.history["train_loss"].append(mean_train_losses["total"])
            self.history["step"].append(self.step)

            if timed_out:
                break

        final_path = os.path.join(output_dir, "final_model.eqx")
        self.model.save(final_path)

        training_seconds = time.perf_counter() - start_time
        if deadline is not None and time.perf_counter() >= deadline:
            timed_out = True

        self.last_train_summary = {
            "best_val_loss": None if self.best_val_loss == float("inf") else float(self.best_val_loss),
            "epochs_completed": epochs_completed,
            "timed_out": timed_out,
            "training_seconds": training_seconds,
            "failure_reason": failure_reason,
            "steps_completed": self.step,
        }
        return self.last_train_summary
