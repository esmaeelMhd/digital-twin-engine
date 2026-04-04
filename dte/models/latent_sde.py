"""Latent Neural SDE for modeling dynamics in latent space."""

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray
import diffrax


class LatentDrift(eqx.Module):
    """Drift function f(z, u, d, c) for the latent SDE: dz = f*dt + g*dW"""

    expert1_layers: list
    expert1_out: eqx.nn.Linear
    expert2_layers: list
    expert2_out: eqx.nn.Linear
    gate_layer: eqx.nn.Linear
    physics_prior_z: eqx.nn.Linear
    physics_prior_u: eqx.nn.Linear

    # Input normalizations stored as regular (non-static) JAX arrays.
    # They are frozen from gradient updates via eqx.filter in the optimizer.
    control_center: Float[Array, "control_dim"]
    control_scale: Float[Array, "control_dim"]
    disturbance_center: Float[Array, "disturbance_dim"]
    disturbance_scale: Float[Array, "disturbance_dim"]
    param_scale: float = eqx.field(static=True)
    linear_prior_enabled: bool = eqx.field(static=True)

    def __init__(
        self,
        latent_dim: int = 16,
        control_dim: int = 2,
        disturbance_dim: int = 0,
        param_dim: int = 6,
        hidden_dim: int = 128,
        n_layers: int = 3,
        control_center: list | None = None,
        control_scale: list | None = None,
        disturbance_center: list | None = None,
        disturbance_scale: list | None = None,
        param_scale: float = 0.1,
        linear_prior_enabled: bool = True,
        *,
        key: PRNGKeyArray,
    ):
        k1, k2, k3, k4, k5 = jax.random.split(key, 5)
        input_dim = latent_dim + control_dim + disturbance_dim + param_dim

        self.gate_layer = eqx.nn.Linear(input_dim, 1, key=k1)
        self.physics_prior_z = eqx.nn.Linear(latent_dim, latent_dim, key=k4)
        self.physics_prior_u = eqx.nn.Linear(control_dim, latent_dim, key=k5)

        keys1 = jax.random.split(k2, n_layers + 1)
        self.expert1_layers = []
        for i in range(n_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            self.expert1_layers.append(eqx.nn.Linear(in_dim, hidden_dim, key=keys1[i]))
        self.expert1_out = eqx.nn.Linear(hidden_dim + input_dim, latent_dim, key=keys1[-1])

        keys2 = jax.random.split(k3, n_layers + 1)
        self.expert2_layers = []
        for i in range(n_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            self.expert2_layers.append(eqx.nn.Linear(in_dim, hidden_dim, key=keys2[i]))
        self.expert2_out = eqx.nn.Linear(hidden_dim + input_dim, latent_dim, key=keys2[-1])

        self.control_center = jnp.array(
            control_center if control_center is not None else [0.0] * control_dim
        )
        self.control_scale = jnp.array(
            control_scale if control_scale is not None else [1.0] * control_dim
        )
        dist_dim_eff = max(disturbance_dim, 1)
        self.disturbance_center = jnp.array(
            disturbance_center if disturbance_center is not None else [0.0] * dist_dim_eff
        )
        self.disturbance_scale = jnp.array(
            disturbance_scale if disturbance_scale is not None else [1.0] * dist_dim_eff
        )
        self.param_scale = param_scale
        self.linear_prior_enabled = linear_prior_enabled

    def __call__(
        self,
        z: Float[Array, "latent_dim"],
        u: Float[Array, "control_dim"],
        d: Float[Array, "disturbance_dim"],
        c: Float[Array, "param_dim"],
    ) -> Float[Array, "latent_dim"]:
        """Compute drift."""
        u_norm = (u - self.control_center) * self.control_scale
        c_norm = c * self.param_scale

        parts = [z, u_norm]
        if d.shape[-1] > 0:
            d_norm = (d - self.disturbance_center[: d.shape[-1]]) * self.disturbance_scale[: d.shape[-1]]
            parts.append(d_norm)
        parts.append(c_norm)

        x = jnp.concatenate(parts)
        x_in = x

        gate = jax.nn.sigmoid(self.gate_layer(x_in))

        x1 = x
        for i, layer in enumerate(self.expert1_layers):
            h = layer(x1)
            h = jax.nn.gelu(h)
            x1 = x1 + h if i > 0 else h
        out1 = self.expert1_out(jnp.concatenate([x1, x_in]))

        x2 = x
        for i, layer in enumerate(self.expert2_layers):
            h = layer(x2)
            h = jax.nn.gelu(h)
            x2 = x2 + h if i > 0 else h
        out2 = self.expert2_out(jnp.concatenate([x2, x_in]))

        residual_drift = gate * out1 + (1.0 - gate) * out2

        if not self.linear_prior_enabled:
            return residual_drift

        physics_drift = self.physics_prior_z(z) + self.physics_prior_u(u_norm)
        return physics_drift + residual_drift


class LatentDiffusion(eqx.Module):
    """Diffusion function g(z, u, d, c) for the latent SDE."""

    layers: list
    output_layer: eqx.nn.Linear
    correction_layers: list
    correction_output_layer: eqx.nn.Linear
    scale: Float[Array, "latent_dim"]

    control_center: Float[Array, "control_dim"]
    control_scale: Float[Array, "control_dim"]
    disturbance_center: Float[Array, "disturbance_dim"]
    disturbance_scale: Float[Array, "disturbance_dim"]
    param_scale: float = eqx.field(static=True)

    def __init__(
        self,
        latent_dim: int = 16,
        control_dim: int = 2,
        disturbance_dim: int = 0,
        param_dim: int = 6,
        hidden_dim: int = 64,
        n_layers: int = 2,
        initial_scale: float = 0.1,
        control_center: list | None = None,
        control_scale: list | None = None,
        disturbance_center: list | None = None,
        disturbance_scale: list | None = None,
        param_scale: float = 0.1,
        *,
        key: PRNGKeyArray,
    ):
        keys = jax.random.split(key, n_layers * 2 + 2)

        input_dim = latent_dim + control_dim + disturbance_dim + param_dim

        self.layers = []
        self.correction_layers = []
        for i in range(n_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            self.layers.append(eqx.nn.Linear(in_dim, hidden_dim, key=keys[i]))
            self.correction_layers.append(eqx.nn.Linear(in_dim, hidden_dim, key=keys[n_layers + i]))

        self.output_layer = eqx.nn.Linear(hidden_dim + input_dim, latent_dim, key=keys[-2])
        corr_out = eqx.nn.Linear(hidden_dim + input_dim, latent_dim, key=keys[-1])
        self.correction_output_layer = eqx.tree_at(
            lambda l: (l.weight, l.bias),
            corr_out,
            (corr_out.weight * 0.001, corr_out.bias * 0.001)
        )
        self.scale = jnp.ones(latent_dim) * initial_scale

        self.control_center = jnp.array(
            control_center if control_center is not None else [0.0] * control_dim
        )
        self.control_scale = jnp.array(
            control_scale if control_scale is not None else [1.0] * control_dim
        )
        dist_dim_eff = max(disturbance_dim, 1)
        self.disturbance_center = jnp.array(
            disturbance_center if disturbance_center is not None else [0.0] * dist_dim_eff
        )
        self.disturbance_scale = jnp.array(
            disturbance_scale if disturbance_scale is not None else [1.0] * dist_dim_eff
        )
        self.param_scale = param_scale

    def __call__(
        self,
        z: Float[Array, "latent_dim"],
        u: Float[Array, "control_dim"],
        d: Float[Array, "disturbance_dim"],
        c: Float[Array, "param_dim"],
    ) -> Float[Array, "latent_dim"]:
        """Compute diagonal diffusion coefficients."""
        u_norm = (u - self.control_center) * self.control_scale
        c_norm = c * self.param_scale

        parts = [z, u_norm]
        if d.shape[-1] > 0:
            d_norm = (d - self.disturbance_center[: d.shape[-1]]) * self.disturbance_scale[: d.shape[-1]]
            parts.append(d_norm)
        parts.append(c_norm)

        x = jnp.concatenate(parts)
        x_in = x

        x_base = x
        for i, layer in enumerate(self.layers):
            h = layer(x_base)
            h = jax.nn.gelu(h)
            x_base = x_base + h if i > 0 else h

        x_corr = x
        for i, layer in enumerate(self.correction_layers):
            h = layer(x_corr)
            h = jax.nn.gelu(h)
            x_corr = x_corr + h if i > 0 else h

        base_diff = jax.nn.sigmoid(self.output_layer(jnp.concatenate([x_base, x_in])))
        corr_diff = self.correction_output_layer(jnp.concatenate([x_corr, x_in]))

        # Reparameterise diffusion with a floor to prevent uncertainty collapse
        return self.scale * (0.01 + 0.99 * base_diff) * jnp.exp(jnp.tanh(corr_diff))


class LatentSDE(eqx.Module):
    """Full latent SDE combining drift and diffusion."""

    drift: LatentDrift
    diffusion: LatentDiffusion
    latent_dim: int = eqx.field(static=True)
    control_dim: int = eqx.field(static=True)
    disturbance_dim: int = eqx.field(static=True)
    param_dim: int = eqx.field(static=True)

    # Store nominal disturbance for fallback (when no disturbance is provided)
    nominal_disturbance: Float[Array, "disturbance_dim"]

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
        control_center: list | None = None,
        control_scale: list | None = None,
        disturbance_center: list | None = None,
        disturbance_scale: list | None = None,
        param_scale: float = 0.1,
        nominal_disturbance: list | None = None,
        linear_prior_enabled: bool = True,
        *,
        key: PRNGKeyArray,
    ):
        key_drift, key_diff = jax.random.split(key)

        self.drift = LatentDrift(
            latent_dim=latent_dim,
            control_dim=control_dim,
            disturbance_dim=disturbance_dim,
            param_dim=param_dim,
            hidden_dim=hidden_dim,
            n_layers=drift_layers,
            control_center=control_center,
            control_scale=control_scale,
            disturbance_center=disturbance_center,
            disturbance_scale=disturbance_scale,
            param_scale=param_scale,
            linear_prior_enabled=linear_prior_enabled,
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
            control_center=control_center,
            control_scale=control_scale,
            disturbance_center=disturbance_center,
            disturbance_scale=disturbance_scale,
            param_scale=param_scale,
            key=key_diff,
        )

        self.latent_dim = latent_dim
        self.control_dim = control_dim
        self.disturbance_dim = disturbance_dim
        self.param_dim = param_dim

        if nominal_disturbance is not None:
            self.nominal_disturbance = jnp.array(nominal_disturbance)
        elif disturbance_dim > 0:
            self.nominal_disturbance = jnp.zeros(disturbance_dim)
        else:
            self.nominal_disturbance = jnp.zeros(0)

    def _get_disturbances(
        self,
        ts: Float[Array, "n_steps"],
        disturbances: Float[Array, "n_steps disturbance_dim"] | None,
        dtype,
    ) -> Float[Array, "n_steps disturbance_dim"]:
        """Return disturbances array, falling back to nominal if not provided."""
        if self.disturbance_dim == 0:
            return jnp.zeros((ts.shape[0], 0), dtype=dtype)
        if disturbances is None:
            return jnp.tile(self.nominal_disturbance[None, :].astype(dtype), (ts.shape[0], 1))
        return disturbances

    def __call__(
        self,
        ts: Float[Array, "n_steps"],
        z0: Float[Array, "latent_dim"],
        controls: Float[Array, "n_steps control_dim"],
        params: Float[Array, "param_dim"],
        key: PRNGKeyArray,
        disturbances: Float[Array, "n_steps disturbance_dim"] | None = None,
    ) -> Float[Array, "n_steps latent_dim"]:
        """Solve the SDE from z0 over timesteps ts (stochastic path)."""
        disturbances = self._get_disturbances(ts, disturbances, z0.dtype)
        control_interp = diffrax.LinearInterpolation(ts, controls)

        if self.disturbance_dim > 0:
            disturbance_interp = diffrax.LinearInterpolation(ts, disturbances)
            disturbance_at_time = disturbance_interp.evaluate
        else:
            disturbance_at_time = lambda t: jnp.zeros((0,), dtype=z0.dtype)  # noqa: E731

        def drift_fn(t, z, args):
            u = control_interp.evaluate(t)
            d = disturbance_at_time(t)
            return self.drift(z, u, d, params)

        def diffusion_fn(t, z, args):
            u = control_interp.evaluate(t)
            d = disturbance_at_time(t)
            return jnp.diag(self.diffusion(z, u, d, params))

        dt0 = (ts[1] - ts[0]) / 2
        brownian_motion = diffrax.VirtualBrownianTree(
            t0=ts[0],
            t1=ts[-1],
            tol=dt0 / 2,
            shape=(self.latent_dim,),
            key=key,
        )

        drift_term = diffrax.ODETerm(drift_fn)
        diffusion_term = diffrax.ControlTerm(diffusion_fn, brownian_motion)
        terms = diffrax.MultiTerm(drift_term, diffusion_term)
        solver = diffrax.Euler()

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
        """Sample multiple SDE paths."""
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
        """Deterministic forward pass using only drift (no noise)."""
        disturbances = self._get_disturbances(ts, disturbances, z0.dtype)
        control_interp = diffrax.LinearInterpolation(ts, controls)

        if self.disturbance_dim > 0:
            disturbance_interp = diffrax.LinearInterpolation(ts, disturbances)
            disturbance_at_time = disturbance_interp.evaluate
        else:
            disturbance_at_time = lambda t: jnp.zeros((0,), dtype=z0.dtype)  # noqa: E731

        def drift_fn(t, z, args):
            u = control_interp.evaluate(t)
            d = disturbance_at_time(t)
            return self.drift(z, u, d, params)

        term = diffrax.ODETerm(drift_fn)
        solver = diffrax.Tsit5()

        dt0 = ts[1] - ts[0]
        saveat = diffrax.SaveAt(ts=ts)
        stepsize_controller = diffrax.PIDController(rtol=1e-3, atol=1e-4)
        solution = diffrax.diffeqsolve(
            term,
            solver,
            t0=ts[0],
            t1=ts[-1],
            dt0=dt0,
            y0=z0,
            saveat=saveat,
            stepsize_controller=stepsize_controller,
            max_steps=4096,
        )
        return solution.ys
