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
from dte.evaluation.universal import normalize_universal_batch
from dte.models.universal_digital_twin import UniversalDigitalTwin
from dte.physics.constraints import bound_penalty, positivity_penalty


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
        filter_spec=None,
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

        self.filter_spec = filter_spec if filter_spec is not None else self.model.trainable_filter_spec()
        trainable, _ = eqx.partition(self.model, self.filter_spec)
        self.opt_state = self.optimizer.init(eqx.filter(trainable, eqx.is_inexact_array))

        self.best_val_loss = float("inf")
        self.history = {"train_loss": [], "val_loss": [], "step": []}
        self.last_train_summary: Dict[str, float | int | bool | str | None] = {}
        self.pretraining_steps = 0
        self.state_lower_bound_table = train_dataset.metadata.state_lower_bound
        self.state_upper_bound_table = train_dataset.metadata.state_upper_bound
        self.state_role_id_table = train_dataset.metadata.state_role_id

    def _normalize_batch(self, model: UniversalDigitalTwin, batch: Dict[str, Array]) -> Dict[str, Array]:
        return normalize_universal_batch(model, batch)

    def _predict_k_step_for_sample(
        self,
        model: UniversalDigitalTwin,
        state_traj: Array,
        control_traj: Array,
        disturbance_traj: Array,
        params_scaled: Array,
        ts: Array,
        state_mask: Array,
        control_mask: Array,
        disturbance_mask: Array,
        param_mask: Array,
        system_id: Array,
        horizon: int,
    ) -> Array:
        window_count = state_traj.shape[0] - horizon
        z_curr = jax.vmap(
            lambda state_t, control_t: model.encode(
                state_t,
                params_scaled,
                control_t,
                state_mask,
                control_mask,
                param_mask,
                system_id,
                None,
            )[1]
        )(
            state_traj[:window_count],
            control_traj[:window_count],
        )
        for step in range(horizon):
            dt_window = ts[step + 1 : step + 1 + window_count] - ts[step : step + window_count]
            z_curr = jax.vmap(
                lambda z_prev, u_t, u_tp1, d_t, d_tp1, dt_t: model.latent_step(
                    z_prev,
                    u_t,
                    u_tp1,
                    d_t,
                    d_tp1,
                    params_scaled,
                    control_mask,
                    disturbance_mask,
                    param_mask,
                    system_id,
                    dt_t,
                )
            )(
                z_curr,
                control_traj[step : step + window_count],
                control_traj[step + 1 : step + 1 + window_count],
                disturbance_traj[step : step + window_count],
                disturbance_traj[step + 1 : step + 1 + window_count],
                dt_window,
            )
        return jax.vmap(
            lambda z_t, control_t: model.decode(
                z_t,
                params_scaled,
                control_t,
                state_mask,
                control_mask,
                param_mask,
                system_id,
            )
        )(
            z_curr,
            control_traj[horizon : horizon + window_count],
        )

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
        k_step_losses: dict[int, Array] = {}
        multi_horizon_cfg = self.config.get("multi_horizon", {})
        horizons = tuple(
            sorted(
                {
                    int(horizon)
                    for horizon in multi_horizon_cfg.get("k_steps", [])
                    if int(horizon) > 1 and int(horizon) < normalized["states_norm"].shape[1]
                }
            )
        )
        for horizon in horizons:
            pred_k_step = jax.vmap(
                lambda state_traj, control_traj, disturbance_traj, params_scaled, ts, state_mask, control_mask, disturbance_mask, param_mask, system_id: self._predict_k_step_for_sample(
                    model,
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
                    horizon,
                )
            )(
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
            )
            k_step_mask = (
                normalized["time_mask"][:, horizon:, None]
                * normalized["state_mask"][:, None, :]
            )
            k_step_losses[horizon] = _masked_huber(
                pred_k_step - normalized["states_norm"][:, horizon:, :],
                k_step_mask,
            )

        if k_step_losses:
            loss_k_step = jnp.mean(jnp.asarray(list(k_step_losses.values())))
        else:
            loss_k_step = jnp.asarray(0.0, dtype=loss_one_step.dtype)

        pred_states_phys = model.denormalize_states(
            pred_states,
            normalized["system_id"],
        )
        lower = self.state_lower_bound_table[normalized["system_id"]][:, None, :]
        upper = self.state_upper_bound_table[normalized["system_id"]][:, None, :]
        loss_state_bounds = bound_penalty(
            pred_states_phys,
            lower=lower,
            upper=upper,
            mask=state_time_mask,
        )
        loss_positivity = positivity_penalty(
            pred_states_phys,
            mask=state_time_mask,
        )
        kl = -0.5 * jnp.sum(1 + z_logvars - z_means ** 2 - jnp.exp(z_logvars), axis=-1)
        loss_kl = jnp.mean(kl)

        weights = self.config["loss_weights"]
        total_loss = (
            float(weights.get("reconstruction", 1.0)) * loss_recon
            + float(weights.get("trajectory", 1.0)) * loss_traj
            + float(weights.get("one_step", 0.0)) * loss_one_step
            + float(weights.get("k_step", 0.0)) * loss_k_step
            + float(weights.get("kl", 0.0)) * loss_kl
            + float(weights.get("state_bounds", 0.0)) * loss_state_bounds
            + float(weights.get("positivity", 0.0)) * loss_positivity
        )
        loss_dict = {
            "total": total_loss,
            "reconstruction": loss_recon,
            "trajectory": loss_traj,
            "one_step": loss_one_step,
            "k_step": loss_k_step,
            "kl": loss_kl,
            "state_bounds": loss_state_bounds,
            "positivity": loss_positivity,
        }
        for horizon, loss_value in k_step_losses.items():
            loss_dict[f"k_step_{horizon}"] = loss_value
        return total_loss, loss_dict

    def compute_group_pretraining_loss(
        self,
        model: UniversalDigitalTwin,
        batch: Dict[str, Array],
        key: PRNGKeyArray,
    ) -> tuple[Array, Dict[str, Array]]:
        normalized = self._normalize_batch(model, batch)
        batch_size = normalized["states_norm"].shape[0]

        state0 = normalized["states_norm"][:, 0, :]
        control0 = normalized["controls_norm"][:, 0, :]
        params_scaled = normalized["params_scaled"]
        state_mask = normalized["state_mask"]
        control_mask = normalized["control_mask"]
        param_mask = normalized["param_mask"]
        system_ids = normalized["system_id"]

        group_masks = model.state_group_mask_table[system_ids] * state_mask[:, None, :]
        group_active = model.state_group_active_table[system_ids]
        masked_logits = jnp.where(
            group_active > 0.0,
            jnp.zeros_like(group_active),
            jnp.full_like(group_active, -1e9),
        )
        selected_group_idx = jax.random.categorical(key, masked_logits, axis=-1)
        selected_group_mask = group_masks[jnp.arange(batch_size), selected_group_idx]

        corrupted_state0 = state0 * (1.0 - selected_group_mask)

        z_mean, z_logvar = jax.vmap(
            lambda state_t, params_t, control_t, state_m, control_m, param_m, system_id: model.encode(
                state_t,
                params_t,
                control_t,
                state_m,
                control_m,
                param_m,
                system_id,
                key=None,
            )[1:]
        )(
            corrupted_state0,
            params_scaled,
            control0,
            state_mask,
            control_mask,
            param_mask,
            system_ids,
        )
        pred_state0 = jax.vmap(
            lambda z_t, params_t, control_t, state_m, control_m, param_m, system_id: model.decode(
                z_t,
                params_t,
                control_t,
                state_m,
                control_m,
                param_m,
                system_id,
            )
        )(
            z_mean,
            params_scaled,
            control0,
            state_mask,
            control_mask,
            param_mask,
            system_ids,
        )

        recon_loss = _masked_huber(pred_state0 - state0, selected_group_mask)
        kl = -0.5 * jnp.sum(1 + z_logvar - z_mean ** 2 - jnp.exp(z_logvar), axis=-1)
        loss_kl = jnp.mean(kl)

        pre_cfg = self.config.get("pretraining", {})
        recon_weight = float(pre_cfg.get("reconstruction_weight", 1.0))
        kl_weight = float(pre_cfg.get("kl_weight", self.config["loss_weights"].get("kl", 0.0)))
        total_loss = recon_weight * recon_loss + kl_weight * loss_kl
        return total_loss, {
            "total": total_loss,
            "masked_reconstruction": recon_loss,
            "kl": loss_kl,
        }

    @eqx.filter_jit
    def train_step(self, trainable, frozen, opt_state, batch: Dict[str, Array], key: PRNGKeyArray):
        def loss_fn(trainable_model):
            model = eqx.combine(trainable_model, frozen)
            return self.compute_loss(model, batch, key)

        (loss, loss_dict), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(trainable)
        updates, new_opt_state = self.optimizer.update(
            eqx.filter(grads, eqx.is_inexact_array),
            opt_state,
            eqx.filter(trainable, eqx.is_inexact_array),
        )
        new_trainable = eqx.apply_updates(trainable, updates)
        new_model = eqx.combine(new_trainable, frozen)
        return new_model, new_opt_state, loss_dict

    @eqx.filter_jit
    def eval_step(self, model: UniversalDigitalTwin, batch: Dict[str, Array], key: PRNGKeyArray):
        _, loss_dict = self.compute_loss(model, batch, key)
        return loss_dict

    @eqx.filter_jit
    def pretrain_step(self, trainable, frozen, opt_state, batch: Dict[str, Array], key: PRNGKeyArray):
        def loss_fn(trainable_model):
            model = eqx.combine(trainable_model, frozen)
            return self.compute_group_pretraining_loss(model, batch, key)

        (loss, loss_dict), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(trainable)
        updates, new_opt_state = self.optimizer.update(
            eqx.filter(grads, eqx.is_inexact_array),
            opt_state,
            eqx.filter(trainable, eqx.is_inexact_array),
        )
        new_trainable = eqx.apply_updates(trainable, updates)
        new_model = eqx.combine(new_trainable, frozen)
        return new_model, new_opt_state, loss_dict

    def _validate_batches(
        self,
        dataset: MultiSystemTrajectoryDataset,
        key: PRNGKeyArray,
        n_batches: int,
        *,
        system_probabilities: Array | None = None,
    ) -> Dict[str, float]:
        batch_size = min(int(self.config["training"]["batch_size"]), dataset.n_samples)
        losses: Dict[str, list] = {
            "total": [],
            "reconstruction": [],
            "trajectory": [],
            "one_step": [],
            "k_step": [],
            "kl": [],
            "state_bounds": [],
            "positivity": [],
        }
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
        pre_cfg = self.config.get("pretraining", {})
        pretraining_enabled = bool(pre_cfg.get("enabled", False))
        pretraining_epochs = int(pre_cfg.get("epochs", 0))
        pretraining_batches = int(pre_cfg.get("batches_per_epoch", 0))

        if pretraining_enabled and pretraining_epochs > 0 and pretraining_batches > 0:
            for pre_epoch in range(pretraining_epochs):
                if deadline is not None and time.perf_counter() >= deadline:
                    timed_out = True
                    break

                pretrain_losses: Dict[str, list] = {
                    "total": [],
                    "masked_reconstruction": [],
                    "kl": [],
                }
                pbar = tqdm(range(pretraining_batches), desc="Grouped Pretraining")
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
                    self.model, self.opt_state, loss_dict = self.pretrain_step(
                        trainable,
                        frozen,
                        self.opt_state,
                        batch,
                        step_key,
                    )
                    loss_dict_float = {name: float(value) for name, value in loss_dict.items()}
                    if _non_finite_loss_names(loss_dict_float):
                        failure_reason = (
                            f"non_finite_pretrain_loss at epoch={pre_epoch+1}, "
                            f"batch={batch_index+1}: {loss_dict_float}"
                        )
                        break
                    for name, value in loss_dict_float.items():
                        pretrain_losses.setdefault(name, []).append(value)
                    self.step += 1
                    self.pretraining_steps += 1
                    pbar.set_postfix({"loss": f"{loss_dict_float['total']:.4f}"})

                if failure_reason is not None:
                    break
                if not pretrain_losses["total"]:
                    failure_reason = "no pretraining batches completed before timeout"
                    break

                mean_pretrain_losses = {
                    name: float(jnp.mean(jnp.asarray(values)))
                    for name, values in pretrain_losses.items()
                }
                print(f"\nGrouped pretraining losses: {mean_pretrain_losses}")

            if failure_reason is not None or timed_out:
                final_path = os.path.join(output_dir, "final_model.eqx")
                self.model.save(final_path)
                training_seconds = time.perf_counter() - start_time
                self.last_train_summary = {
                    "best_val_loss": None if self.best_val_loss == float("inf") else float(self.best_val_loss),
                    "epochs_completed": epochs_completed,
                    "timed_out": timed_out,
                    "training_seconds": training_seconds,
                    "failure_reason": failure_reason,
                    "steps_completed": self.step,
                    "pretraining_enabled": pretraining_enabled,
                    "pretraining_steps": self.pretraining_steps,
                }
                return self.last_train_summary

        for epoch in range(n_epochs):
            if deadline is not None and time.perf_counter() >= deadline:
                timed_out = True
                break

            train_losses: Dict[str, list] = {
                "total": [],
                "reconstruction": [],
                "trajectory": [],
                "one_step": [],
                "k_step": [],
                "kl": [],
                "state_bounds": [],
                "positivity": [],
            }
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
            "pretraining_enabled": pretraining_enabled,
            "pretraining_steps": self.pretraining_steps,
        }
        return self.last_train_summary
