"""Uncertainty calibration helpers for rollout diagnostics."""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array


def _masked_mean(values: Array, mask: Array | None = None) -> Array:
    if mask is None:
        return jnp.mean(values)
    weights = jnp.asarray(mask, dtype=values.dtype)
    denom = jnp.maximum(jnp.sum(weights), jnp.asarray(1.0, dtype=values.dtype))
    return jnp.sum(values * weights) / denom


def empirical_coverage(
    mean: Array,
    std: Array,
    target: Array,
    sigma: float = 2.0,
    mask: Array | None = None,
) -> float:
    """Fraction of targets that fall within ``sigma`` standard deviations."""
    safe_std = jnp.maximum(std, 1e-6)
    within = jnp.abs(target - mean) <= sigma * safe_std
    return float(_masked_mean(within.astype(mean.dtype), mask))


def gaussian_nll(
    mean: Array,
    std: Array,
    target: Array,
    mask: Array | None = None,
) -> float:
    """Average Gaussian negative log likelihood."""
    safe_std = jnp.maximum(std, 1e-6)
    variance = safe_std ** 2
    nll = 0.5 * jnp.log(2.0 * jnp.pi * variance) + 0.5 * ((target - mean) ** 2) / variance
    return float(_masked_mean(nll, mask))


def variance_collapse_rate(
    std: Array,
    threshold: float = 1e-3,
    mask: Array | None = None,
) -> float:
    """Fraction of predictive standard deviations below a collapse threshold."""
    collapsed = jnp.asarray(std) < threshold
    return float(_masked_mean(collapsed.astype(jnp.float32), mask))


def calibration_gap(
    mean: Array,
    std: Array,
    target: Array,
    mask: Array | None = None,
    sigma_levels: tuple[float, ...] = (1.0, 2.0),
) -> float:
    """Mean absolute gap between empirical and ideal Gaussian coverage."""
    ideal = {1.0: 0.682689492, 2.0: 0.954499736, 3.0: 0.997300204}
    gaps = []
    for sigma in sigma_levels:
        empirical = empirical_coverage(mean, std, target, sigma=sigma, mask=mask)
        gaps.append(abs(empirical - ideal.get(sigma, empirical)))
    return float(sum(gaps) / max(len(gaps), 1))
