"""Control-sensitivity utilities for simulator and model comparisons."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Array


def finite_difference_control_jacobian(
    step_fn,
    control: Array,
    delta: Array | float,
) -> Array:
    """Return d(next_state) / d(control) via centered finite differences."""
    control = jnp.asarray(control)
    delta = jnp.asarray(delta, dtype=control.dtype)
    if delta.ndim == 0:
        delta = jnp.full_like(control, delta)
    eye = jnp.eye(control.shape[0], dtype=control.dtype)

    def one_column(direction):
        offset = delta * direction
        step = jnp.maximum(jnp.sum(delta * direction), 1e-6)
        return (step_fn(control + offset) - step_fn(control - offset)) / (2.0 * step)

    jacobian_columns = jax.vmap(one_column)(eye)
    return jnp.swapaxes(jacobian_columns, 0, 1)


def sensitivity_mismatch_metrics(
    predicted_jacobian: Array,
    reference_jacobian: Array,
    state_mask: Array | None = None,
    control_mask: Array | None = None,
) -> dict[str, float]:
    """Summarize local-gain mismatch between two Jacobians."""
    pred = jnp.asarray(predicted_jacobian)
    ref = jnp.asarray(reference_jacobian)

    mask = jnp.ones_like(pred)
    if state_mask is not None:
        mask = mask * jnp.asarray(state_mask, dtype=pred.dtype)[:, None]
    if control_mask is not None:
        mask = mask * jnp.asarray(control_mask, dtype=pred.dtype)[None, :]

    diff = (pred - ref) * mask
    denom = jnp.maximum(jnp.sum(mask), jnp.asarray(1.0, dtype=pred.dtype))
    rmse = jnp.sqrt(jnp.sum(diff ** 2) / denom)
    pred_norm = jnp.sqrt(jnp.sum((pred * mask) ** 2))
    ref_norm = jnp.sqrt(jnp.sum((ref * mask) ** 2))
    relative_l2 = jnp.sqrt(jnp.sum(diff ** 2)) / jnp.maximum(ref_norm, 1e-6)
    cosine = jnp.sum((pred * mask) * (ref * mask)) / jnp.maximum(pred_norm * ref_norm, 1e-6)
    return {
        "rmse": float(rmse),
        "relative_l2": float(relative_l2),
        "cosine_similarity": float(cosine),
    }
