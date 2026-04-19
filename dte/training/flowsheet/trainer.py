"""Trainer for graph-structured flowsheet trajectory models."""

from __future__ import annotations

import math
import time
from typing import Dict, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, PRNGKeyArray
from tqdm import tqdm

from dte.data.datasets.flowsheet_dataset import FlowsheetTrajectoryDataset
from dte.evaluation.flowsheet_metrics import (
    masked_mse,
    plant_balance_proxy_loss,
    rollout_stability_penalty,
    stream_consistency_loss,
    unit_output_consistency_loss,
)
from dte.models.flowsheet.flowsheet_model import FlowsheetModel


def _non_finite_loss_names(losses: Dict[str, float]) -> list[str]:
    return [name for name, value in losses.items() if not math.isfinite(float(value))]


class FlowsheetTrainer:
    """Minimal but runnable trainer for the Phase 3 flowsheet model."""

    def __init__(
        self,
        model: FlowsheetModel,
        config: dict,
        train_dataset: FlowsheetTrajectoryDataset,
        val_dataset: Optional[FlowsheetTrajectoryDataset] = None,
    ):
        self.model = model
        self.config = config
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.step = 0

        optimizer_cfg = config["optimizer"]
        warmup_steps = int(optimizer_cfg["warmup_steps"])
        total_steps = max(int(optimizer_cfg["total_steps"]), warmup_steps + 1)
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=float(optimizer_cfg["peak_lr"]),
            warmup_steps=warmup_steps,
            decay_steps=total_steps,
            end_value=float(optimizer_cfg["end_lr"]),
        )
        self.optimizer = optax.chain(
            optax.clip_by_global_norm(float(optimizer_cfg["gradient_clip"])),
            optax.adam(schedule, b1=0.95, b2=0.99),
        )
        self.filter_spec = jax.tree.map(eqx.is_inexact_array, self.model)
        trainable, _ = eqx.partition(self.model, self.filter_spec)
        self.opt_state = self.optimizer.init(eqx.filter(trainable, eqx.is_inexact_array))

        self.history = {"train_loss": [], "val_loss": [], "step": []}
        self.best_val_loss = float("inf")
        self.last_train_summary: Dict[str, float | int | bool | str | None] = {}

    def compute_loss(
        self,
        model: FlowsheetModel,
        batch: Dict[str, Array],
    ) -> tuple[Array, Dict[str, Array]]:
        batch_size = int(batch["states"].shape[0])

        def rollout_one(
            states: Array,
            controls: Array,
            disturbances: Array,
            params: Array,
            stream_values: Array,
            global_controls: Array,
            global_disturbances: Array,
            ts: Array,
        ):
            return model.rollout(
                states[0],
                controls,
                disturbances,
                params,
                ts,
                global_controls=global_controls,
                global_disturbances=global_disturbances,
                external_stream_sequence=stream_values,
            )

        predicted_states, predicted_streams = jax.vmap(rollout_one)(
            batch["states"],
            batch["controls"],
            batch["disturbances"],
            batch["params"],
            batch["stream_values"],
            batch["global_controls"],
            batch["global_disturbances"],
            batch["t"],
        )

        time_mask = batch["time_mask"].astype(jnp.float32)
        state_mask = model.unit_state_mask_table[None, None, :, :]
        trajectory = masked_mse(
            predicted_states,
            batch["states"],
            time_mask[..., None, None] * state_mask,
        )
        stream = stream_consistency_loss(
            predicted_streams,
            batch["stream_values"],
            time_mask,
            model.stream_var_mask_table,
        )
        unit_consistency = unit_output_consistency_loss(
            predicted_states,
            predicted_streams,
            time_mask,
            model.stream_source_index_table,
            model.stream_source_var_index_table,
            model.stream_var_mask_table,
            model.stream_delay_table,
        )
        plant_balance = plant_balance_proxy_loss(
            predicted_states,
            time_mask,
            model.stream_source_index_table,
            model.stream_target_index_table,
            model.stream_source_var_index_table,
            model.stream_target_var_index_table,
            model.stream_var_mask_table,
            model.stream_delay_table,
        )
        stability = rollout_stability_penalty(
            predicted_states,
            time_mask,
            model.unit_state_mask_table,
        )

        weights = self.config["loss_weights"]
        total = (
            float(weights.get("trajectory", 1.0)) * trajectory
            + float(weights.get("stream_consistency", 1.0)) * stream
            + float(weights.get("unit_consistency", 0.5)) * unit_consistency
            + float(weights.get("plant_balance", 0.25)) * plant_balance
            + float(weights.get("rollout_stability", 0.0)) * stability
        )
        return total, {
            "total": total,
            "trajectory": trajectory,
            "stream_consistency": stream,
            "unit_consistency": unit_consistency,
            "plant_balance": plant_balance,
            "rollout_stability": stability,
            "batch_size": jnp.asarray(batch_size, dtype=jnp.float32),
        }

    @eqx.filter_jit
    def train_step(self, trainable, frozen, opt_state, batch: Dict[str, Array]):
        def loss_fn(trainable_model):
            model = eqx.combine(trainable_model, frozen)
            return self.compute_loss(model, batch)

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
    def eval_step(self, model: FlowsheetModel, batch: Dict[str, Array]):
        _, loss_dict = self.compute_loss(model, batch)
        return loss_dict

    def train_epoch(
        self,
        key: PRNGKeyArray,
        *,
        deadline: float | None = None,
        n_batches: int | None = None,
    ) -> tuple[Optional[Dict[str, float]], bool, Optional[str]]:
        batch_size = min(
            int(self.config["training"]["batch_size"]),
            self.train_dataset.n_samples,
        )
        full_n_batches = max(1, self.train_dataset.n_samples // batch_size)
        if n_batches is None:
            max_batches = self.config["training"].get("max_batches_per_epoch")
            n_batches = full_n_batches if max_batches is None else max(
                1, min(full_n_batches, int(max_batches))
            )
        else:
            n_batches = max(1, min(int(n_batches), full_n_batches))

        epoch_losses: Dict[str, list[float]] = {
            "total": [],
            "trajectory": [],
            "stream_consistency": [],
            "unit_consistency": [],
            "plant_balance": [],
            "rollout_stability": [],
        }
        timed_out = False
        trainable, frozen = eqx.partition(self.model, self.filter_spec)

        pbar = tqdm(range(n_batches), desc="Flowsheet Training")
        for _ in pbar:
            if deadline is not None and time.perf_counter() >= deadline:
                timed_out = True
                break

            key, subkey = jax.random.split(key)
            batch = self.train_dataset.sample_batch(subkey, batch_size)
            self.model, self.opt_state, loss_dict = self.train_step(
                trainable,
                frozen,
                self.opt_state,
                batch,
            )
            trainable, frozen = eqx.partition(self.model, self.filter_spec)

            loss_float = {name: float(value) for name, value in loss_dict.items()}
            bad_losses = _non_finite_loss_names(loss_float)
            if bad_losses:
                return None, timed_out, f"non_finite_train_loss: {bad_losses}"

            for name, value in loss_float.items():
                if name in epoch_losses:
                    epoch_losses[name].append(value)

            self.step += 1
            pbar.set_postfix({"loss": f"{loss_float['total']:.4f}"})

        if not epoch_losses["total"]:
            raise RuntimeError("No flowsheet training batches completed.")

        mean_losses = {
            name: float(jnp.mean(jnp.asarray(values)))
            for name, values in epoch_losses.items()
        }
        return mean_losses, timed_out, None

    def validate(
        self,
        key: PRNGKeyArray,
        *,
        n_batches: int | None = None,
    ) -> tuple[Optional[Dict[str, float]], Optional[str]]:
        if self.val_dataset is None:
            return {}, None

        batch_size = min(
            int(self.config["training"]["batch_size"]),
            self.val_dataset.n_samples,
        )
        full_n_batches = max(1, self.val_dataset.n_samples // batch_size)
        if n_batches is None:
            max_batches = self.config.get("checkpointing", {}).get("max_val_batches")
            n_batches = full_n_batches if max_batches is None else max(
                1, min(full_n_batches, int(max_batches))
            )
        else:
            n_batches = max(1, min(int(n_batches), full_n_batches))

        val_losses: Dict[str, list[float]] = {
            "total": [],
            "trajectory": [],
            "stream_consistency": [],
            "unit_consistency": [],
            "plant_balance": [],
            "rollout_stability": [],
        }
        pbar = tqdm(range(n_batches), desc="Flowsheet Validation")
        for _ in pbar:
            key, subkey = jax.random.split(key)
            batch = self.val_dataset.sample_batch(subkey, batch_size)
            loss_dict = self.eval_step(self.model, batch)
            loss_float = {name: float(value) for name, value in loss_dict.items()}
            bad_losses = _non_finite_loss_names(loss_float)
            if bad_losses:
                return None, f"non_finite_val_loss: {bad_losses}"

            for name, value in loss_float.items():
                if name in val_losses:
                    val_losses[name].append(value)
            pbar.set_postfix({"loss": f"{loss_float['total']:.4f}"})

        mean_losses = {
            name: float(jnp.mean(jnp.asarray(values)))
            for name, values in val_losses.items()
        }
        return mean_losses, None

    def fit(
        self,
        key: PRNGKeyArray,
        *,
        n_epochs: int,
        deadline: float | None = None,
    ) -> dict[str, float | int | bool | str | None]:
        train_history = []
        val_history = []
        timed_out = False
        failure_reason = None

        for epoch in range(int(n_epochs)):
            train_losses, timed_out, failure_reason = self.train_epoch(
                key,
                deadline=deadline,
            )
            if failure_reason is not None:
                break
            train_history.append(train_losses["total"] if train_losses is not None else float("nan"))
            self.history["train_loss"].append(train_losses["total"])
            self.history["step"].append(self.step)

            if self.val_dataset is not None:
                key, val_key = jax.random.split(key)
                val_losses, failure_reason = self.validate(val_key)
                if failure_reason is not None:
                    break
                val_total = float(val_losses["total"])
                val_history.append(val_total)
                self.history["val_loss"].append(val_total)
                self.best_val_loss = min(self.best_val_loss, val_total)
            else:
                val_history.append(float("nan"))

            if timed_out:
                break
            key, _ = jax.random.split(key)

        summary = {
            "epochs_completed": len(train_history),
            "best_val_loss": None if math.isinf(self.best_val_loss) else self.best_val_loss,
            "final_train_loss": train_history[-1] if train_history else None,
            "final_val_loss": val_history[-1] if val_history else None,
            "timed_out": timed_out,
            "failure_reason": failure_reason,
            "step": self.step,
        }
        self.last_train_summary = summary
        return summary
