"""Latent Neural SDE for modeling dynamics in latent space."""

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray
import diffrax


def _normalize_control(u: Float[Array, "control_dim"]) -> Float[Array, "control_dim"]:
    """Apply light normalization to control inputs when using the CSTR layout."""
    if u.shape[-1] == 2:
        center = jnp.array([55.0, 300.0], dtype=u.dtype)
        scale = jnp.array([0.02, 0.02], dtype=u.dtype)
        return (u - center) * scale
    return u


def _normalize_disturbance(
    d: Float[Array, "disturbance_dim"],
) -> Float[Array, "disturbance_dim"]:
    """Apply light normalization to disturbances when using the CSTR layout."""
    if d.shape[-1] == 2:
        center = jnp.array([1.0, 320.0], dtype=d.dtype)
        scale = jnp.array([1.0, 0.02], dtype=d.dtype)
        return (d - center) * scale
    return d


def _normalize_params(c: Float[Array, "param_dim"]) -> Float[Array, "param_dim"]:
    """Normalize parameter vector before concatenation into the neural networks."""
    return c * 0.1


class LatentDrift(eqx.Module):
    """Drift function f(z, u, c) for the latent SDE: dz = f*dt + g*dW"""
    
    layers: list
    output_layer: eqx.nn.Linear
    
    def __init__(
        self,
        latent_dim: int = 16,
        control_dim: int = 2,
        disturbance_dim: int = 0,
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
        
        input_dim = latent_dim + control_dim + disturbance_dim + param_dim
        
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
        d: Float[Array, "disturbance_dim"],
        c: Float[Array, "param_dim"],
    ) -> Float[Array, "latent_dim"]:
        """Compute drift.
        
        Args:
            z: Latent state
            u: Control input
            d: Disturbance input
            c: Conditioning (parameters)
            
        Returns:
            Drift vector dz/dt
        """
        u_norm = _normalize_control(u)
        d_norm = _normalize_disturbance(d)
        c_norm = _normalize_params(c)
        x = jnp.concatenate([z, u_norm, d_norm, c_norm])
        
        for layer in self.layers:
            x = layer(x)
            x = jax.nn.silu(x)
        
        return 10.0 * jax.nn.tanh(self.output_layer(x) * 0.1)


class LatentDiffusion(eqx.Module):
    """Diffusion function g(z, u, c) for the latent SDE"""
    
    layers: list
    output_layer: eqx.nn.Linear
    scale: Float[Array, ""]
    
    def __init__(
        self,
        latent_dim: int = 16,
        control_dim: int = 2,
        disturbance_dim: int = 0,
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
        
        input_dim = latent_dim + control_dim + disturbance_dim + param_dim
        
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
        d: Float[Array, "disturbance_dim"],
        c: Float[Array, "param_dim"],
    ) -> Float[Array, "latent_dim"]:
        """Compute diffusion coefficients.
        
        Args:
            z: Latent state
            u: Control input
            d: Disturbance input
            c: Conditioning (parameters)
            
        Returns:
            Diffusion vector (diagonal elements)
        """
        u_norm = _normalize_control(u)
        d_norm = _normalize_disturbance(d)
        c_norm = _normalize_params(c)
        x = jnp.concatenate([z, u_norm, d_norm, c_norm])
        
        for layer in self.layers:
            x = layer(x)
            x = jax.nn.silu(x)
        
        # Sigmoid to strictly bound diffusion and prevent variance explosion
        return self.scale * jax.nn.sigmoid(self.output_layer(x))


class LatentSDE(eqx.Module):
    """Full latent SDE model combining drift and diffusion"""
    
    drift: LatentDrift
    diffusion: LatentDiffusion
    latent_dim: int = eqx.field(static=True)
    control_dim: int = eqx.field(static=True)
    disturbance_dim: int = eqx.field(static=True)
    param_dim: int = eqx.field(static=True)
    
    def __init__(
        self,
        latent_dim: int = 16,
        control_dim: int = 2,
        disturbance_dim: int = 0,
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
            disturbance_dim: Disturbance dimension
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
            disturbance_dim=disturbance_dim,
            param_dim=param_dim,
            hidden_dim=hidden_dim,
            n_layers=drift_layers,
            key=key_drift,
        )
        
        self.diffusion = LatentDiffusion(
            latent_dim=latent_dim,
            control_dim=control_dim,
            disturbance_dim=disturbance_dim,
            param_dim=param_dim,
            hidden_dim=diffusion_hidden_dim,
            n_layers=diffusion_layers,
            initial_scale=initial_diffusion_scale,
            key=key_diff,
        )
        
        self.latent_dim = latent_dim
        self.control_dim = control_dim
        self.disturbance_dim = disturbance_dim
        self.param_dim = param_dim

    def _default_disturbances(
        self,
        ts: Float[Array, "n_steps"],
        dtype,
    ) -> Float[Array, "n_steps disturbance_dim"]:
        """Create a nominal disturbance trajectory when none is provided."""
        if self.disturbance_dim == 0:
            return jnp.zeros((ts.shape[0], 0), dtype=dtype)
        if self.disturbance_dim == 2:
            nominal = jnp.array([1.0, 320.0], dtype=dtype)
        else:
            nominal = jnp.zeros((self.disturbance_dim,), dtype=dtype)
        return jnp.tile(nominal[None, :], (ts.shape[0], 1))
    
    def __call__(
        self,
        ts: Float[Array, "n_steps"],
        z0: Float[Array, "latent_dim"],
        controls: Float[Array, "n_steps control_dim"],
        params: Float[Array, "param_dim"],
        key: PRNGKeyArray,
        disturbances: Float[Array, "n_steps disturbance_dim"] | None = None,
    ) -> Float[Array, "n_steps latent_dim"]:
        """Solve the SDE from z0 over timesteps ts.
        
        Args:
            ts: Time points (n_steps,)
            z0: Initial latent state (latent_dim,)
            controls: Control inputs at each time (n_steps, control_dim)
            params: System parameters (param_dim,)
            key: PRNG key for SDE noise
            disturbances: Disturbance inputs at each time (optional)
            
        Returns:
            Latent trajectory (n_steps, latent_dim)
        """
        # Create interpolation for control signal
        control_interp = diffrax.LinearInterpolation(ts, controls)
        if self.disturbance_dim > 0:
            if disturbances is None:
                disturbances = self._default_disturbances(ts, z0.dtype)
            disturbance_interp = diffrax.LinearInterpolation(ts, disturbances)
            disturbance_at_time = disturbance_interp.evaluate
        else:
            disturbance_at_time = lambda t: jnp.zeros((0,), dtype=z0.dtype)
        
        # Create drift wrapper for diffrax
        def drift_fn(t, z, args):
            u = control_interp.evaluate(t)
            d = disturbance_at_time(t)
            return self.drift(z, u, d, params)
        
        # Create diffusion wrapper for diffrax (diagonal)
        def diffusion_fn(t, z, args):
            u = control_interp.evaluate(t)
            d = disturbance_at_time(t)
            diffusion_diag = self.diffusion(z, u, d, params)
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
        disturbances: Float[Array, "n_steps disturbance_dim"] | None = None,
    ) -> Float[Array, "n_samples n_steps latent_dim"]:
        """Sample multiple SDE paths.
        
        Args:
            ts: Time points
            z0: Initial latent state
            controls: Control trajectory
            params: Parameters
            key: PRNG key
            n_samples: Number of samples
            disturbances: Disturbance trajectory (optional)
            
        Returns:
            Sampled trajectories (n_samples, n_steps, latent_dim)
        """
        keys = jax.random.split(key, n_samples)
        
        def sample_one(k):
            return self(ts, z0, controls, params, k, disturbances=disturbances)
        
        return jax.vmap(sample_one)(keys)
    
    def mean_trajectory(
        self,
        ts: Float[Array, "n_steps"],
        z0: Float[Array, "latent_dim"],
        controls: Float[Array, "n_steps control_dim"],
        params: Float[Array, "param_dim"],
        disturbances: Float[Array, "n_steps disturbance_dim"] | None = None,
    ) -> Float[Array, "n_steps latent_dim"]:
        """Deterministic forward pass using only drift (no noise).
        
        Args:
            ts: Time points
            z0: Initial latent state
            controls: Control trajectory
            params: Parameters
            disturbances: Disturbance trajectory (optional)
            
        Returns:
            Mean trajectory (n_steps, latent_dim)
        """
        # Create interpolation for control signal
        control_interp = diffrax.LinearInterpolation(ts, controls)
        if self.disturbance_dim > 0:
            if disturbances is None:
                disturbances = self._default_disturbances(ts, z0.dtype)
            disturbance_interp = diffrax.LinearInterpolation(ts, disturbances)
            disturbance_at_time = disturbance_interp.evaluate
        else:
            disturbance_at_time = lambda t: jnp.zeros((0,), dtype=z0.dtype)
        
        # Create drift wrapper for diffrax
        def drift_fn(t, z, args):
            u = control_interp.evaluate(t)
            d = disturbance_at_time(t)
            return self.drift(z, u, d, params)
        
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
