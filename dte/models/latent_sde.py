"""Latent Neural SDE for modeling dynamics in latent space."""

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray
import diffrax


class LatentDrift(eqx.Module):
    """Drift function f(z, u, c) for the latent SDE: dz = f*dt + g*dW"""
    
    layers: list
    output_layer: eqx.nn.Linear
    
    def __init__(
        self,
        latent_dim: int = 16,
        control_dim: int = 2,
        param_dim: int = 6,
        hidden_dim: int = 128,
        n_layers: int = 3,
        *,
        key: PRNGKeyArray,
    ):
        """Initialize drift network.
        
        Args:
            latent_dim: Latent dimension
            control_dim: Control dimension
            param_dim: Parameter dimension (conditioning)
            hidden_dim: Hidden layer dimension
            n_layers: Number of hidden layers
            key: PRNG key
        """
        keys = jax.random.split(key, n_layers + 1)
        
        input_dim = latent_dim + control_dim + param_dim
        
        # Hidden layers
        self.layers = []
        for i in range(n_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            self.layers.append(
                eqx.nn.Linear(in_dim, hidden_dim, key=keys[i])
            )
        
        # Output layer
        self.output_layer = eqx.nn.Linear(hidden_dim, latent_dim, key=keys[-1])
    
    def __call__(
        self,
        z: Float[Array, "latent_dim"],
        u: Float[Array, "control_dim"],
        c: Float[Array, "param_dim"],
    ) -> Float[Array, "latent_dim"]:
        """Compute drift.
        
        Args:
            z: Latent state
            u: Control input
            c: Conditioning (parameters)
            
        Returns:
            Drift vector dz/dt
        """
        x = jnp.concatenate([z, u, c])
        
        for layer in self.layers:
            x = layer(x)
            x = jax.nn.silu(x)
        
        return jax.nn.tanh(self.output_layer(x))


class LatentDiffusion(eqx.Module):
    """Diffusion function g(z, u, c) for the latent SDE"""
    
    layers: list
    output_layer: eqx.nn.Linear
    scale: Float[Array, ""]
    
    def __init__(
        self,
        latent_dim: int = 16,
        control_dim: int = 2,
        param_dim: int = 6,
        hidden_dim: int = 64,
        n_layers: int = 2,
        initial_scale: float = 0.1,
        *,
        key: PRNGKeyArray,
    ):
        """Initialize diffusion network.
        
        Args:
            latent_dim: Latent dimension
            control_dim: Control dimension
            param_dim: Parameter dimension (conditioning)
            hidden_dim: Hidden layer dimension
            n_layers: Number of hidden layers
            initial_scale: Initial diffusion scale
            key: PRNG key
        """
        keys = jax.random.split(key, n_layers + 1)
        
        input_dim = latent_dim + control_dim + param_dim
        
        # Hidden layers
        self.layers = []
        for i in range(n_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            self.layers.append(
                eqx.nn.Linear(in_dim, hidden_dim, key=keys[i])
            )
        
        # Output layer (diagonal diffusion)
        self.output_layer = eqx.nn.Linear(hidden_dim, latent_dim, key=keys[-1])
        
        # Learnable scale
        self.scale = jnp.array(initial_scale)
    
    def __call__(
        self,
        z: Float[Array, "latent_dim"],
        u: Float[Array, "control_dim"],
        c: Float[Array, "param_dim"],
    ) -> Float[Array, "latent_dim"]:
        """Compute diffusion coefficients.
        
        Args:
            z: Latent state
            u: Control input
            c: Conditioning (parameters)
            
        Returns:
            Diffusion vector (diagonal elements)
        """
        x = jnp.concatenate([z, u, c])
        
        for layer in self.layers:
            x = layer(x)
            x = jax.nn.silu(x)
        
        # Softplus to ensure positive + scale
        return self.scale * jax.nn.softplus(self.output_layer(x))


class LatentSDE(eqx.Module):
    """Full latent SDE model combining drift and diffusion"""
    
    drift: LatentDrift
    diffusion: LatentDiffusion
    latent_dim: int
    control_dim: int
    param_dim: int
    
    def __init__(
        self,
        latent_dim: int = 16,
        control_dim: int = 2,
        param_dim: int = 6,
        hidden_dim: int = 128,
        drift_layers: int = 3,
        diffusion_layers: int = 2,
        diffusion_hidden_dim: int = 64,
        initial_diffusion_scale: float = 0.1,
        *,
        key: PRNGKeyArray,
    ):
        """Initialize latent SDE.
        
        Args:
            latent_dim: Latent dimension
            control_dim: Control dimension
            param_dim: Parameter dimension
            hidden_dim: Hidden dimension for drift
            drift_layers: Number of drift layers
            diffusion_layers: Number of diffusion layers
            diffusion_hidden_dim: Hidden dimension for diffusion
            initial_diffusion_scale: Initial diffusion scale
            key: PRNG key
        """
        key_drift, key_diff = jax.random.split(key)
        
        self.drift = LatentDrift(
            latent_dim=latent_dim,
            control_dim=control_dim,
            param_dim=param_dim,
            hidden_dim=hidden_dim,
            n_layers=drift_layers,
            key=key_drift,
        )
        
        self.diffusion = LatentDiffusion(
            latent_dim=latent_dim,
            control_dim=control_dim,
            param_dim=param_dim,
            hidden_dim=diffusion_hidden_dim,
            n_layers=diffusion_layers,
            initial_scale=initial_diffusion_scale,
            key=key_diff,
        )
        
        self.latent_dim = latent_dim
        self.control_dim = control_dim
        self.param_dim = param_dim
    
    def __call__(
        self,
        ts: Float[Array, "n_steps"],
        z0: Float[Array, "latent_dim"],
        controls: Float[Array, "n_steps control_dim"],
        params: Float[Array, "param_dim"],
        key: PRNGKeyArray,
    ) -> Float[Array, "n_steps latent_dim"]:
        """Solve the SDE from z0 over timesteps ts.
        
        Args:
            ts: Time points (n_steps,)
            z0: Initial latent state (latent_dim,)
            controls: Control inputs at each time (n_steps, control_dim)
            params: System parameters (param_dim,)
            key: PRNG key for SDE noise
            
        Returns:
            Latent trajectory (n_steps, latent_dim)
        """
        # Create interpolation for control signal
        control_interp = diffrax.LinearInterpolation(ts, controls)
        
        # Create drift wrapper for diffrax
        def drift_fn(t, z, args):
            u = control_interp.evaluate(t)
            return self.drift(z, u, params)
        
        # Create diffusion wrapper for diffrax (diagonal)
        def diffusion_fn(t, z, args):
            u = control_interp.evaluate(t)
            diffusion_diag = self.diffusion(z, u, params)
            return jnp.diag(diffusion_diag)
        
        # Set up Brownian motion
        dt0 = (ts[1] - ts[0]) / 2
        brownian_motion = diffrax.VirtualBrownianTree(
            t0=ts[0],
            t1=ts[-1],
            tol=dt0 / 2,
            shape=(self.latent_dim,),
            key=key,
        )
        
        # Set up terms
        drift_term = diffrax.ODETerm(drift_fn)
        diffusion_term = diffrax.ControlTerm(diffusion_fn, brownian_motion)
        terms = diffrax.MultiTerm(drift_term, diffusion_term)
        
        # Solver
        solver = diffrax.Heun()  # Good for SDEs
        
        # Solve
        saveat = diffrax.SaveAt(ts=ts)
        solution = diffrax.diffeqsolve(
            terms,
            solver,
            t0=ts[0],
            t1=ts[-1],
            dt0=dt0,
            y0=z0,
            saveat=saveat,
            max_steps=len(ts) * 10,
        )
        
        return solution.ys
    
    def sample_trajectories(
        self,
        ts: Float[Array, "n_steps"],
        z0: Float[Array, "latent_dim"],
        controls: Float[Array, "n_steps control_dim"],
        params: Float[Array, "param_dim"],
        key: PRNGKeyArray,
        n_samples: int = 10,
    ) -> Float[Array, "n_samples n_steps latent_dim"]:
        """Sample multiple SDE paths.
        
        Args:
            ts: Time points
            z0: Initial latent state
            controls: Control trajectory
            params: Parameters
            key: PRNG key
            n_samples: Number of samples
            
        Returns:
            Sampled trajectories (n_samples, n_steps, latent_dim)
        """
        keys = jax.random.split(key, n_samples)
        
        def sample_one(k):
            return self(ts, z0, controls, params, k)
        
        return jax.vmap(sample_one)(keys)
    
    def mean_trajectory(
        self,
        ts: Float[Array, "n_steps"],
        z0: Float[Array, "latent_dim"],
        controls: Float[Array, "n_steps control_dim"],
        params: Float[Array, "param_dim"],
    ) -> Float[Array, "n_steps latent_dim"]:
        """Deterministic forward pass using only drift (no noise).
        
        Args:
            ts: Time points
            z0: Initial latent state
            controls: Control trajectory
            params: Parameters
            
        Returns:
            Mean trajectory (n_steps, latent_dim)
        """
        # Create interpolation for control signal
        control_interp = diffrax.LinearInterpolation(ts, controls)
        
        # Create drift wrapper for diffrax
        def drift_fn(t, z, args):
            u = control_interp.evaluate(t)
            return self.drift(z, u, params)
        
        # Set up ODE (no diffusion)
        term = diffrax.ODETerm(drift_fn)
        solver = diffrax.Tsit5()  # Higher-order ODE solver
        
        # Solve
        dt0 = (ts[1] - ts[0]) / 2
        saveat = diffrax.SaveAt(ts=ts)
        solution = diffrax.diffeqsolve(
            term,
            solver,
            t0=ts[0],
            t1=ts[-1],
            dt0=dt0,
            y0=z0,
            saveat=saveat,
            max_steps=len(ts) * 10,
        )
        
        return solution.ys
