"""Metrics and loss helpers for flowsheet graph rollouts."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Array


def masked_mse(values: Array, target: Array, mask: Array) -> Array:
    mask = mask.astype(values.dtype)
    denom = jnp.maximum(jnp.sum(mask), jnp.asarray(1.0, dtype=values.dtype))
    return jnp.sum(((values - target) ** 2) * mask) / denom


def _endpoint_values_from_states(
    states: Array,
    unit_index: Array,
    variable_index: Array,
    variable_mask: Array,
) -> Array:
    """Extract stream endpoint values from unit state tensors.

    Args:
        states: `(..., n_units, max_state_dim)`
        unit_index: `(n_streams,)`, -1 indicates an external source/sink
        variable_index: `(n_streams, max_stream_vars)`
        variable_mask: `(n_streams, max_stream_vars)`
    """

    leading_shape = states.shape[:-2]
    flat_states = states.reshape((-1, states.shape[-2], states.shape[-1]))
    safe_unit_index = jnp.maximum(unit_index, 0)
    safe_variable_index = jnp.maximum(variable_index, 0)

    def extract_one(flat_state: Array) -> Array:
        selected_units = flat_state[safe_unit_index]
        values = jnp.take_along_axis(selected_units, safe_variable_index, axis=-1)
        internal_mask = (unit_index >= 0).astype(values.dtype)[:, None]
        return values * variable_mask * internal_mask

    extracted = jax.vmap(extract_one)(flat_states)
    return extracted.reshape((*leading_shape, unit_index.shape[0], variable_mask.shape[-1]))


def source_stream_values_from_states(
    states: Array,
    stream_source_index: Array,
    stream_source_var_index: Array,
    stream_var_mask: Array,
) -> Array:
    return _endpoint_values_from_states(
        states,
        stream_source_index,
        stream_source_var_index,
        stream_var_mask,
    )


def target_stream_values_from_states(
    states: Array,
    stream_target_index: Array,
    stream_target_var_index: Array,
    stream_var_mask: Array,
) -> Array:
    return _endpoint_values_from_states(
        states,
        stream_target_index,
        stream_target_var_index,
        stream_var_mask,
    )


def stream_consistency_loss(
    predicted_streams: Array,
    true_streams: Array,
    time_mask: Array,
    stream_var_mask: Array,
) -> Array:
    mask = time_mask[..., None, None] * stream_var_mask[None, None, :, :]
    return masked_mse(predicted_streams, true_streams, mask)


def unit_output_consistency_loss(
    predicted_states: Array,
    predicted_streams: Array,
    time_mask: Array,
    stream_source_index: Array,
    stream_source_var_index: Array,
    stream_var_mask: Array,
    stream_delay: Array | None = None,
) -> Array:
    source_values = source_stream_values_from_states(
        predicted_states,
        stream_source_index,
        stream_source_var_index,
        stream_var_mask,
    )
    internal_source_mask = (stream_source_index >= 0).astype(predicted_states.dtype)
    if stream_delay is not None:
        internal_source_mask = internal_source_mask * (stream_delay <= 0.0).astype(
            predicted_states.dtype
        )
    mask = (
        time_mask[..., None, None]
        * stream_var_mask[None, None, :, :]
        * internal_source_mask[None, None, :, None]
    )
    return masked_mse(source_values, predicted_streams, mask)


def plant_balance_proxy_loss(
    predicted_states: Array,
    time_mask: Array,
    stream_source_index: Array,
    stream_target_index: Array,
    stream_source_var_index: Array,
    stream_target_var_index: Array,
    stream_var_mask: Array,
    stream_delay: Array | None = None,
) -> Array:
    source_values = source_stream_values_from_states(
        predicted_states,
        stream_source_index,
        stream_source_var_index,
        stream_var_mask,
    )
    target_values = target_stream_values_from_states(
        predicted_states,
        stream_target_index,
        stream_target_var_index,
        stream_var_mask,
    )
    internal_mask = (
        (stream_source_index >= 0).astype(predicted_states.dtype)
        * (stream_target_index >= 0).astype(predicted_states.dtype)
    )
    if stream_delay is not None:
        internal_mask = internal_mask * (stream_delay <= 0.0).astype(predicted_states.dtype)
    mask = time_mask[..., None, None] * stream_var_mask[None, None, :, :] * internal_mask[None, None, :, None]
    return masked_mse(source_values, target_values, mask)


def rollout_stability_penalty(
    predicted_states: Array,
    time_mask: Array,
    unit_state_mask: Array,
) -> Array:
    if predicted_states.shape[-3] < 3:
        return jnp.asarray(0.0, dtype=predicted_states.dtype)
    second_diff = (
        predicted_states[:, 2:, :, :]
        - 2.0 * predicted_states[:, 1:-1, :, :]
        + predicted_states[:, :-2, :, :]
    )
    mask = (
        time_mask[:, 2:, None, None]
        * unit_state_mask[None, None, :, :]
    )
    return jnp.sum((second_diff ** 2) * mask) / jnp.maximum(
        jnp.sum(mask),
        jnp.asarray(1.0, dtype=predicted_states.dtype),
    )
