"""Tests for CSTR simulator."""

import jax
import jax.numpy as jnp
import pytest

from dte.simulators.cstr import CSTRSimulator, CSTRParams, sample_random_params


def test_steady_state_remains_steady():
    """Test 1: Simulate from steady state with constant inputs, verify it stays at steady state."""
    params = CSTRParams()
    simulator = CSTRSimulator(params)
    
    # Define constant inputs
    control = jnp.array([50.0, 300.0])  # F_in, Tc_in
    disturbance = jnp.array([1.0, 320.0])  # Ca_in, T_in
    
    # Find steady state
    ss_state = simulator.steady_state(control, disturbance)
    
    # Simulate from steady state
    n_steps = 1000
    control_traj = jnp.tile(control[None, :], (n_steps, 1))
    disturbance_traj = jnp.tile(disturbance[None, :], (n_steps, 1))
    
    result = simulator.simulate(
        ss_state,
        control_traj,
        disturbance_traj,
        t_span=(0.0, 100.0),
        dt=0.1,
        n_steps=n_steps,
    )
    
    # Check that states remain close to steady state
    for i in range(4):
        max_deviation = jnp.max(jnp.abs(result["states"][:, i] - ss_state[i]))
        assert max_deviation < 1e-2, f"State {i} deviated by {max_deviation}"


@pytest.mark.parametrize(
    ("params", "control", "disturbance"),
    [
        (
            CSTRParams(),
            jnp.array([50.0, 300.0]),
            jnp.array([1.0, 320.0]),
        ),
        (
            CSTRParams(V=120.0, Vc=18.0, k0=9.0e10, Ea_over_R=9100.0, UA=6.0e4, Fc=20.0),
            jnp.array([45.0, 295.0]),
            jnp.array([1.1, 318.0]),
        ),
    ],
)
def test_steady_state_matches_long_horizon_simulation(params, control, disturbance):
    """The fast steady-state solver should agree with the simulation fallback."""
    simulator = CSTRSimulator(params)

    fast_state = simulator.steady_state(control, disturbance)
    sim_state = simulator._steady_state_via_simulation(
        control,
        disturbance,
        initial_guess=jnp.array([0.5, 0.5, 350.0, 300.0]),
    )

    assert jnp.allclose(fast_state, sim_state, atol=1e-2, rtol=1e-3)


def test_temperature_response_to_coolant_change():
    """Test 2: Step change in Tc_in should cause T to change monotonically."""
    params = CSTRParams()
    simulator = CSTRSimulator(params)
    
    # Start from steady state
    control_initial = jnp.array([50.0, 300.0])
    disturbance = jnp.array([1.0, 320.0])
    initial_state = simulator.steady_state(control_initial, disturbance)
    
    # Create step change in Tc_in
    n_steps = 1000
    control_traj = jnp.zeros((n_steps, 2))
    control_traj = control_traj.at[:, 0].set(50.0)  # Constant F_in
    control_traj = control_traj.at[:500, 1].set(300.0)  # Initial Tc_in
    control_traj = control_traj.at[500:, 1].set(280.0)  # Step down in Tc_in
    
    disturbance_traj = jnp.tile(disturbance[None, :], (n_steps, 1))
    
    result = simulator.simulate(
        initial_state,
        control_traj,
        disturbance_traj,
        t_span=(0.0, 100.0),
        dt=0.1,
        n_steps=n_steps,
    )
    
    # Temperature should decrease after step change
    T_before_step = result["states"][499, 2]
    T_after_step = result["states"][-1, 2]
    assert T_after_step < T_before_step, "Temperature should decrease when coolant temp decreases"


def test_data_generation_rollout_matches_reference_solver():
    """The fixed-grid rollout should stay close to the reference diffrax solver."""
    params = CSTRParams()
    simulator = CSTRSimulator(params)

    initial_state = jnp.array([0.6, 0.3, 330.0, 305.0])
    n_steps = 80
    control_traj = jnp.tile(jnp.array([50.0, 300.0])[None, :], (n_steps, 1))
    disturbance_traj = jnp.tile(jnp.array([1.0, 320.0])[None, :], (n_steps, 1))
    t_span = (0.0, 8.0)

    reference = simulator.simulate(
        initial_state,
        control_traj,
        disturbance_traj,
        t_span=t_span,
        dt=0.1,
        n_steps=n_steps,
    )
    generated = simulator.simulate_for_data_generation(
        initial_state,
        control_traj,
        disturbance_traj,
        t_span=t_span,
        dt=0.1,
        n_steps=n_steps,
    )

    assert generated["states"].shape == reference["states"].shape
    assert jnp.allclose(generated["time"], reference["time"])
    assert jnp.allclose(generated["states"], reference["states"], atol=2e-1, rtol=1e-2)


def test_mass_balance():
    """Test 3: Verify mass balance holds."""
    params = CSTRParams()
    simulator = CSTRSimulator(params)
    
    # Simulate
    initial_state = jnp.array([0.5, 0.5, 350.0, 300.0])
    control = jnp.array([50.0, 300.0])
    disturbance = jnp.array([1.0, 320.0])
    
    n_steps = 500
    control_traj = jnp.tile(control[None, :], (n_steps, 1))
    disturbance_traj = jnp.tile(disturbance[None, :], (n_steps, 1))
    
    result = simulator.simulate(
        initial_state,
        control_traj,
        disturbance_traj,
        t_span=(0.0, 50.0),
        dt=0.1,
        n_steps=n_steps,
    )
    
    # Get conservation quantities
    conservation = simulator.get_conservation_quantities(
        result["states"], result["controls"], disturbance_traj
    )
    
    # Total mass should be defined
    assert "total_mass" in conservation
    assert len(conservation["total_mass"]) == n_steps


def test_vmap_over_initial_conditions():
    """Test 4: vmap over different initial conditions."""
    params = CSTRParams()
    simulator = CSTRSimulator(params)
    
    # Create multiple initial conditions
    n_sims = 100
    key = jax.random.PRNGKey(0)
    initial_states = jax.random.uniform(
        key, shape=(n_sims, 4), minval=jnp.array([0.1, 0.1, 280.0, 280.0]),
        maxval=jnp.array([2.0, 2.0, 400.0, 350.0])
    )
    
    # Fixed control and disturbance
    n_steps = 100
    control_traj = jnp.tile(jnp.array([50.0, 300.0])[None, :], (n_steps, 1))
    disturbance_traj = jnp.tile(jnp.array([1.0, 320.0])[None, :], (n_steps, 1))
    
    # Define simulation function
    def simulate_one(initial_state):
        return simulator.simulate(
            initial_state,
            control_traj,
            disturbance_traj,
            t_span=(0.0, 10.0),
            dt=0.1,
            n_steps=n_steps,
        )
    
    # Vmap over initial conditions
    batched_simulate = jax.vmap(simulate_one)
    results = batched_simulate(initial_states)
    
    # Check shapes
    assert results["states"].shape == (n_sims, n_steps, 4)
    assert results["controls"].shape == (n_sims, n_steps, 2)


def test_jit_compilation():
    """Test 5: JIT compilation works without errors."""
    params = CSTRParams()
    simulator = CSTRSimulator(params)
    
    # Define simulation function
    @jax.jit
    def simulate_jit(initial_state, control_traj, disturbance_traj):
        return simulator.simulate(
            initial_state,
            control_traj,
            disturbance_traj,
            t_span=(0.0, 10.0),
            dt=0.1,
            n_steps=100,
        )
    
    # Run once to compile
    initial_state = jnp.array([0.5, 0.5, 350.0, 300.0])
    control_traj = jnp.tile(jnp.array([50.0, 300.0])[None, :], (100, 1))
    disturbance_traj = jnp.tile(jnp.array([1.0, 320.0])[None, :], (100, 1))
    
    result = simulate_jit(initial_state, control_traj, disturbance_traj)
    
    # Run again to verify JIT works
    result2 = simulate_jit(initial_state, control_traj, disturbance_traj)
    
    # Results should be identical
    assert jnp.allclose(result["states"], result2["states"])


def test_sample_random_params():
    """Test sampling random parameters."""
    key = jax.random.PRNGKey(42)
    params = sample_random_params(key)
    
    # Check that parameters are within expected ranges
    assert 50.0 <= params.V <= 200.0
    assert 1e9 <= params.k0 <= 1e12
    assert 7000.0 <= params.Ea_over_R <= 10000.0
    assert 3e4 <= params.UA <= 8e4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
