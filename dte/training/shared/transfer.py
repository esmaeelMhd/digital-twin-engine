"""Multi-unit transfer learning utilities.

Provides:

- :func:`apply_finetune_mask` -- Compatibility no-op; freezing is applied
  via an Equinox ``filter_spec`` inside :class:`FewShotAdapter`.
- :class:`FewShotAdapter` -- Fine-tune a pre-trained model on a small number
  of trajectories from a *new* unit, with optional zero-shot evaluation.
- :func:`zero_shot_eval` -- Evaluate a model on a new unit without any
  gradient updates.

Typical workflow
----------------
::

    from dte.training.shared.transfer import FewShotAdapter, zero_shot_eval

    # 1. Zero-shot baseline on a held-out evaluation split
    zs_metrics = zero_shot_eval(pretrained_model, eval_dataset, n_batches=20)
    print("Zero-shot MSE:", zs_metrics["mse"])

    # 2. Few-shot fine-tune on a disjoint adaptation split
    #    (freeze encoder + SDE, update decoder only)
    adapter = FewShotAdapter(pretrained_model, system_spec)
    finetuned_model = adapter.finetune(
        adapt_dataset,
        n_steps=200,
        batch_size=8,
        seq_len=20,
        part="decoder",
    )

    # 3. Evaluate on the same held-out split used in step 1
    fs_metrics = zero_shot_eval(finetuned_model, eval_dataset, n_batches=20)
    print("Few-shot MSE:", fs_metrics["mse"])
"""

from __future__ import annotations

from typing import Dict, Literal, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jaxtyping import PRNGKeyArray

from dte.models.unit.digital_twin import DigitalTwin
from dte.simulators.base import SystemSpec


# ---------------------------------------------------------------------------
# Parameter selection helpers
# ---------------------------------------------------------------------------

FinetunePartType = Literal["decoder", "encoder", "all"]


def apply_finetune_mask(
    model: DigitalTwin,
    part: FinetunePartType = "decoder",
) -> DigitalTwin:
    """Compatibility no-op.

    Freezing is applied lazily in :class:`FewShotAdapter` (and in
    ``scripts/train.py``) via an Equinox ``filter_spec`` / ``eqx.partition``.
    This function returns ``model`` unchanged.
    """
    del part
    return model


def _build_filter_spec(
    model: DigitalTwin,
    part: FinetunePartType,
):
    """Build an Equinox filter spec that marks trainable leaves as ``True``
    and frozen leaves as ``False``.

    The spec is a PyTree mirroring the model with bool leaves, suitable for
    use with ``eqx.partition(model, spec)``.
    """
    # Start with "train nothing" (all False), then selectively enable
    all_false = jax.tree_util.tree_map(lambda _: False, model)

    if part == "all":
        # Train all array leaves
        return jax.tree_util.tree_map(eqx.is_array, model)

    if part == "decoder":
        # Only decoder arrays are trainable
        return eqx.tree_at(
            lambda m: m.decoder,
            all_false,
            replace=jax.tree_util.tree_map(eqx.is_array, model.decoder),
        )
    elif part == "encoder":
        # Only encoder arrays are trainable
        return eqx.tree_at(
            lambda m: m.encoder,
            all_false,
            replace=jax.tree_util.tree_map(eqx.is_array, model.encoder),
        )
    else:
        return jax.tree_util.tree_map(eqx.is_array, model)


# ---------------------------------------------------------------------------
# Zero-shot evaluation
# ---------------------------------------------------------------------------


def zero_shot_eval(
    model: DigitalTwin,
    dataset,
    n_batches: int = 50,
    batch_size: int = 16,
    seq_len: int = 50,
    key: Optional[PRNGKeyArray] = None,
) -> Dict[str, float]:
    """Evaluate a model on a dataset without any parameter updates.

    Parameters
    ----------
    model:
        Model to evaluate (pre-trained or fine-tuned).
    dataset:
        :class:`~dte.data.datasets.unit_dataset.TrajectoryDataset` for the target unit.
    n_batches:
        Number of random batches to evaluate.
    batch_size:
        Batch size.
    seq_len:
        Sequence length.
    key:
        PRNG key (uses PRNGKey(0) if not supplied).

    Returns
    -------
    dict
        ``{"mse": float, "rmse": float, "norm_mse": float}`` where
        ``norm_mse`` is the MSE normalised by the per-state standard deviation.
    """
    if key is None:
        key = jax.random.PRNGKey(0)

    @eqx.filter_jit
    def _predict(model, batch, key):
        states = batch["states"]
        controls = batch["controls"]
        disturbances = batch["disturbances"]
        params = batch["params"]
        ts = batch["t"]
        bs = states.shape[0]

        def rollout_one(idx, k):
            _, z_mean, _ = model.encode(states[idx, 0], params[idx], controls[idx, 0], k)
            z_traj = model.latent_sde.mean_trajectory(
                ts[idx], z_mean, controls[idx], params[idx], disturbances=disturbances[idx]
            )
            decode_fn = jax.vmap(lambda z, u: model.decode(z, params[idx], u), in_axes=(0, 0))
            return decode_fn(z_traj, controls[idx])

        keys = jax.random.split(key, bs)
        pred = jax.vmap(rollout_one, in_axes=(0, 0))(jnp.arange(bs), keys)
        mse = jnp.mean((pred - states) ** 2)
        return pred, mse

    mse_vals = []
    all_pred, all_true = [], []

    norm_stats = dataset.get_normalization_stats()
    # Infer state_dim from a sample batch
    sample_key = key if key is not None else jax.random.PRNGKey(0)
    _sample = dataset.sample_batch(sample_key, 1, seq_len=1)
    _state_dim = _sample["states"].shape[-1]
    state_std = np.asarray(norm_stats.get("state_std", np.ones(_state_dim)))

    batch_size_eff = min(batch_size, dataset.n_samples)
    n_batches_eff = min(n_batches, dataset.n_samples // batch_size_eff)

    for _ in range(n_batches_eff):
        key, subkey = jax.random.split(key)
        batch = dataset.sample_batch(subkey, batch_size_eff, seq_len=seq_len)
        key, subkey = jax.random.split(key)
        pred, mse = _predict(model, batch, subkey)
        mse_vals.append(float(mse))
        all_pred.append(np.asarray(pred))
        all_true.append(np.asarray(batch["states"]))

    mean_mse = float(np.mean(mse_vals))

    # Normalised MSE
    if all_pred:
        pred_arr = np.concatenate(all_pred, axis=0)
        true_arr = np.concatenate(all_true, axis=0)
        per_state_norm_mse = ((pred_arr - true_arr) ** 2 / (state_std[None, None] ** 2 + 1e-8)).mean()
    else:
        per_state_norm_mse = float("nan")

    return {
        "mse": mean_mse,
        "rmse": float(np.sqrt(mean_mse)),
        "norm_mse": float(per_state_norm_mse),
        "n_batches": n_batches_eff,
    }


# ---------------------------------------------------------------------------
# Few-shot adapter
# ---------------------------------------------------------------------------


class FewShotAdapter:
    """Fine-tune a pre-trained Digital Twin on a small target-unit dataset.

    Parameters
    ----------
    model:
        Pre-trained model.
    system_spec:
        :class:`~dte.simulators.base.SystemSpec` for the target system.
    learning_rate:
        Fine-tuning learning rate (default 3e-4; usually smaller than
        the offline training rate).
    gradient_clip:
        Global gradient-norm clip.
    """

    def __init__(
        self,
        model: DigitalTwin,
        system_spec: SystemSpec,
        learning_rate: float = 3e-4,
        gradient_clip: float = 1.0,
    ):
        self.model = model
        self.system_spec = system_spec
        self.learning_rate = learning_rate
        self.gradient_clip = gradient_clip
        self._history: Dict[str, list] = {"step": [], "loss": []}

    def finetune(
        self,
        dataset,
        n_steps: int = 200,
        batch_size: int = 8,
        seq_len: int = 20,
        part: FinetunePartType = "decoder",
        key: Optional[PRNGKeyArray] = None,
        verbose: bool = True,
    ) -> DigitalTwin:
        """Run few-shot fine-tuning.

        Parameters
        ----------
        dataset:
            :class:`~dte.data.datasets.unit_dataset.TrajectoryDataset` for the new unit.
        n_steps:
            Number of gradient steps.
        batch_size:
            Mini-batch size for each gradient step.
        seq_len:
            Trajectory length to sample from the dataset.
        part:
            ``"decoder"``: freeze encoder+SDE, update decoder only.
            ``"encoder"``: freeze decoder+SDE, update encoder only.
            ``"all"``: update all parameters.
        key:
            PRNG key (uses PRNGKey(0) if not supplied).
        verbose:
            Print loss every 20 steps.

        Returns
        -------
        DigitalTwin
            The fine-tuned model (the original ``self.model`` is updated in-place
            for subsequent calls and also returned).
        """
        if key is None:
            key = jax.random.PRNGKey(0)

        filter_spec = _build_filter_spec(self.model, part)

        optimizer = optax.chain(
            optax.clip_by_global_norm(self.gradient_clip),
            optax.adam(self.learning_rate),
        )
        # Only initialise optimiser state for trainable leaves
        trainable_init, _ = eqx.partition(self.model, filter_spec)
        opt_state = optimizer.init(eqx.filter(trainable_init, eqx.is_array))

        batch_size_eff = min(batch_size, dataset.n_samples)

        @eqx.filter_jit
        def _train_step(model, opt_state, batch, key):
            # Partition model into trainable / frozen using filter_spec
            trainable, frozen = eqx.partition(model, filter_spec)

            def loss_fn(trainable):
                full_model = eqx.combine(trainable, frozen)
                states = batch["states"]
                controls = batch["controls"]
                disturbances = batch["disturbances"]
                params = batch["params"]
                ts = batch["t"]
                bs = states.shape[0]

                def rollout_one(idx, k):
                    _, z_mean, _ = full_model.encode(
                        states[idx, 0], params[idx], controls[idx, 0], k
                    )
                    z_traj = full_model.latent_sde.mean_trajectory(
                        ts[idx], z_mean, controls[idx], params[idx],
                        disturbances=disturbances[idx],
                    )
                    decode_fn = jax.vmap(
                        lambda z, u: full_model.decode(z, params[idx], u), in_axes=(0, 0)
                    )
                    return decode_fn(z_traj, controls[idx])

                keys = jax.random.split(key, bs)
                pred = jax.vmap(rollout_one, in_axes=(0, 0))(jnp.arange(bs), keys)
                return jnp.mean((pred - states) ** 2)

            loss, grads = eqx.filter_value_and_grad(loss_fn)(trainable)
            updates, new_opt_state = optimizer.update(
                eqx.filter(grads, eqx.is_array),
                opt_state,
                eqx.filter(trainable, eqx.is_array),
            )
            new_trainable = eqx.apply_updates(trainable, updates)
            new_model = eqx.combine(new_trainable, frozen)
            return new_model, new_opt_state, loss

        self._history = {"step": [], "loss": []}
        step = 0
        for i in range(n_steps):
            key, subkey = jax.random.split(key)
            batch = dataset.sample_batch(subkey, batch_size_eff, seq_len=seq_len)
            key, subkey = jax.random.split(key)
            self.model, opt_state, loss = _train_step(self.model, opt_state, batch, subkey)
            step += 1
            self._history["step"].append(step)
            self._history["loss"].append(float(loss))
            if verbose and (i + 1) % 20 == 0:
                print(f"  Fine-tune step {i+1}/{n_steps}  loss={float(loss):.6f}")

        return self.model

    @property
    def history(self) -> Dict[str, list]:
        """Fine-tuning loss history."""
        return self._history
