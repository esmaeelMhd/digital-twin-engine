"""Tests for physics conservation laws and losses."""

import jax
import jax.numpy as jnp
import pytest
import yaml

from dte.simulators.cstr import CSTRSimulator, CSTRParams
from dte.physics.cstr import CSTRPhysicsLoss
from dte.physics.conservation import (
    mass_balance_residual,
    energy_balance_residual,
    total_conservation_metric,
)
from dte.training.shared.losses import LossComputer


def test_mass_balance_on_ground_truth():
    """Test 1: mass_balance_residual returns near-zero for ground-truth trajectories."""
    # Load config
    with open("configs/cstr_default.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # Convert all numeric values to float (YAML may parse some as strings)
    cstr_params = {k: float(v) for k, v in config["cstr"].items()}
    params = CSTRParams(**cstr_params)
    simulator = CSTRSimulator(params)
    
    # Simulate ground truth
    initial_state = jnp.array([0.5, 0.5, 350.0, 300.0])
    control = jnp.array([50.0, 300.0])
    disturbance = jnp.array([1.0, 320.0])
    
    n_steps = 100
    control_traj = jnp.tile(control[None, :], (n_steps, 1))
    disturbance_traj = jnp.tile(disturbance[None, :], (n_steps, 1))
    
    result = simulator.simulate(
        initial_state,
        control_traj,
        disturbance_traj,
        t_span=(0.0, 10.0),
        dt=0.1,
        n_steps=n_steps,
    )
    
    # Compute mass balance residual
    residual = mass_balance_residual(
        result["states"], control_traj, disturbance_traj, params, dt=0.1
    )
    
    # Should be very small (numerical errors only)
    mean_residual = jnp.mean(residual)
    print(f"\nMean mass balance residual: {mean_residual}")
    assert mean_residual < 0.01, f"Mass balance residual too large: {mean_residual}"


def test_energy_balance_on_ground_truth():
    """Test 2: energy_balance_residual returns near-zero for ground-truth trajectories."""
    # Load config
    with open("configs/cstr_default.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # Convert all numeric values to float
    cstr_params = {k: float(v) for k, v in config["cstr"].items()}
    params = CSTRParams(**cstr_params)
    simulator = CSTRSimulator(params)
    
    # Simulate ground truth
    initial_state = jnp.array([0.5, 0.5, 350.0, 300.0])
    control = jnp.array([50.0, 300.0])
    disturbance = jnp.array([1.0, 320.0])
    
    n_steps = 100
    control_traj = jnp.tile(control[None, :], (n_steps, 1))
    disturbance_traj = jnp.tile(disturbance[None, :], (n_steps, 1))
    
    result = simulator.simulate(
        initial_state,
        control_traj,
        disturbance_traj,
        t_span=(0.0, 10.0),
        dt=0.1,
        n_steps=n_steps,
    )
    
    # Compute energy balance residual
    residual = energy_balance_residual(
        result["states"], control_traj, disturbance_traj, params, dt=0.1
    )
    
    # Should be small
    mean_residual = jnp.mean(residual)
    print(f"\nMean energy balance residual: {mean_residual}")
    assert mean_residual < 5.0, f"Energy balance residual too large: {mean_residual}"


def test_reconstruction_loss():
    """Test 3: reconstruction_loss is zero when predicted == true."""
    # Load config
    with open("configs/cstr_default.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    with open("configs/training_default.yaml", "r") as f:
        train_config = yaml.safe_load(f)
    
    params = CSTRParams(**config["cstr"])
    
    # Create dummy normalization stats
    norm_stats = {
        "state_mean": jnp.array([0.5, 0.5, 350.0, 300.0]),
        "state_std": jnp.array([0.3, 0.3, 20.0, 10.0]),
        "control_mean": jnp.array([50.0, 300.0]),
        "control_std": jnp.array([20.0, 10.0]),
        "disturbance_mean": jnp.array([1.0, 320.0]),
        "disturbance_std": jnp.array([0.5, 15.0]),
    }
    
    loss_computer = LossComputer(
        train_config,
        norm_stats,
        physics_loss=CSTRPhysicsLoss(params),
    )
    
    # Create identical predictions
    states = jax.random.normal(jax.random.PRNGKey(0), shape=(32, 50, 4))
    
    loss = loss_computer.reconstruction_loss(states, states)
    
    assert jnp.abs(loss) < 1e-6, f"Reconstruction loss should be zero, got {loss}"


def test_kl_divergence_loss():
    """Test 4: kl_divergence_loss is zero when mean=0, logvar=0."""
    # Load configs
    with open("configs/cstr_default.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    with open("configs/training_default.yaml", "r") as f:
        train_config = yaml.safe_load(f)
    
    params = CSTRParams(**config["cstr"])
    
    # Create dummy normalization stats
    norm_stats = {
        "state_mean": jnp.zeros(4),
        "state_std": jnp.ones(4),
        "control_mean": jnp.zeros(2),
        "control_std": jnp.ones(2),
        "disturbance_mean": jnp.zeros(2),
        "disturbance_std": jnp.ones(2),
    }
    
    loss_computer = LossComputer(
        train_config,
        norm_stats,
        physics_loss=CSTRPhysicsLoss(params),
    )
    
    # Standard normal (mean=0, logvar=0 => var=1)
    # KL(N(0,1) || N(0,1)) = 0
    z_mean = jnp.zeros((32, 16))
    z_logvar = jnp.zeros((32, 16))
    
    loss = loss_computer.kl_divergence_loss(z_mean, z_logvar)
    
    assert jnp.abs(loss) < 1e-6, f"KL divergence should be zero, got {loss}"


def test_kl_annealing():
    """Test KL weight annealing."""
    # Load configs
    with open("configs/cstr_default.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    with open("configs/training_default.yaml", "r") as f:
        train_config = yaml.safe_load(f)
    
    params = CSTRParams(**config["cstr"])
    
    norm_stats = {
        "state_mean": jnp.zeros(4),
        "state_std": jnp.ones(4),
        "control_mean": jnp.zeros(2),
        "control_std": jnp.ones(2),
        "disturbance_mean": jnp.zeros(2),
        "disturbance_std": jnp.ones(2),
    }
    
    loss_computer = LossComputer(
        train_config,
        norm_stats,
        physics_loss=CSTRPhysicsLoss(params),
    )
    
    # Check weights at different steps
    weights_0 = loss_computer.get_loss_weights(step=0)
    weights_mid = loss_computer.get_loss_weights(step=2500)
    weights_end = loss_computer.get_loss_weights(step=5000)
    
    # KL weight should increase
    assert weights_0["kl"] < weights_mid["kl"] < weights_end["kl"]
    print(f"\nKL weights: start={weights_0['kl']}, mid={weights_mid['kl']}, end={weights_end['kl']}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
