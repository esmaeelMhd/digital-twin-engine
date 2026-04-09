"""Tests for reusable physics constraint utilities."""

import jax.numpy as jnp

from dte.physics.constraints import (
    bound_penalty,
    energy_balance_residual,
    mass_balance_residual,
    monotonicity_penalty,
    positivity_penalty,
)


def test_positivity_and_bound_penalties_detect_violations():
    values = jnp.array([-1.0, 0.5, 3.0], dtype=jnp.float32)
    lower = jnp.array([0.0, 0.0, 0.0], dtype=jnp.float32)
    upper = jnp.array([2.0, 2.0, 2.0], dtype=jnp.float32)

    assert float(positivity_penalty(values)) > 0.0
    assert float(bound_penalty(values, lower=lower, upper=upper)) > 0.0


def test_balance_residual_helpers_zero_out_when_balanced():
    assert float(mass_balance_residual(1.0, inflow=2.0, outflow=1.0)) == 0.0
    assert float(energy_balance_residual(4.0, inflow=5.0, outflow=2.0, heat=1.0)) == 0.0


def test_monotonicity_penalty_flags_wrong_direction():
    input_delta = jnp.array([1.0, 1.0], dtype=jnp.float32)
    output_delta = jnp.array([0.5, -0.5], dtype=jnp.float32)
    penalty = monotonicity_penalty(input_delta, output_delta, expected_sign=1.0)

    assert float(penalty) > 0.0
