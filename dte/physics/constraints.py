"""Reusable physical constraint utilities for Phase 1 training and evaluation."""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array


def _masked_mean(values: Array, mask: Array | None = None) -> Array:
    if mask is None:
        return jnp.mean(values)
    weights = mask.astype(values.dtype)
    denom = jnp.maximum(jnp.sum(weights), jnp.asarray(1.0, dtype=values.dtype))
    return jnp.sum(values * weights) / denom


def positivity_penalty(values: Array, mask: Array | None = None) -> Array:
    """Quadratic penalty on negative values."""
    violation = jnp.maximum(-values, 0.0)
    return _masked_mean(violation ** 2, mask)


def bound_penalty(
    values: Array,
    lower: Array | None = None,
    upper: Array | None = None,
    mask: Array | None = None,
) -> Array:
    """Quadratic penalty on lower/upper bound violations."""
    penalty = jnp.zeros_like(values)
    if lower is not None:
        lower_violation = jnp.maximum(lower - values, 0.0)
        lower_mask = jnp.isfinite(lower).astype(values.dtype)
        penalty = penalty + lower_mask * lower_violation ** 2
    if upper is not None:
        upper_violation = jnp.maximum(values - upper, 0.0)
        upper_mask = jnp.isfinite(upper).astype(values.dtype)
        penalty = penalty + upper_mask * upper_violation ** 2
    return _masked_mean(penalty, mask)


def balance_residual(
    accumulation: Array,
    inflow: Array,
    outflow: Array,
    source: Array | float | None = None,
    sink: Array | float | None = None,
) -> Array:
    """Generic conservation residual: accumulation - inflow + outflow - source + sink."""
    src = 0.0 if source is None else source
    snk = 0.0 if sink is None else sink
    return accumulation - inflow + outflow - src + snk


def mass_balance_residual(
    accumulation: Array,
    inflow: Array,
    outflow: Array,
    generation: Array | float | None = None,
    consumption: Array | float | None = None,
) -> Array:
    """Mass-balance residual."""
    return balance_residual(
        accumulation,
        inflow,
        outflow,
        source=generation,
        sink=consumption,
    )


def energy_balance_residual(
    accumulation: Array,
    inflow: Array,
    outflow: Array,
    heat: Array | float | None = None,
    work: Array | float | None = None,
    reaction: Array | float | None = None,
) -> Array:
    """Energy-balance residual."""
    source = 0.0 if heat is None else heat
    source = source + (0.0 if work is None else work)
    source = source + (0.0 if reaction is None else reaction)
    return balance_residual(accumulation, inflow, outflow, source=source, sink=None)


def monotonicity_penalty(
    input_delta: Array,
    output_delta: Array,
    expected_sign: float = 1.0,
    mask: Array | None = None,
) -> Array:
    """Penalty for violating a known monotonic input-output direction."""
    signed_response = expected_sign * input_delta * output_delta
    violation = jnp.maximum(-signed_response, 0.0)
    return _masked_mean(violation ** 2, mask)
