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
    # Frozen from gradient updates by DigitalTwin.trainable_filter_spec.
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

        # Learned linear prior (not a physics residual): Az + Bu.
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
    cde_layers: list
    cde_out: eqx.nn.Linear | None
    solver_gate_layers: list
    solver_gate_out: eqx.nn.Linear
    correction_layers: list
    correction_out: eqx.nn.Linear
    latent_dim: int = eqx.field(static=True)
    control_dim: int = eqx.field(static=True)
    disturbance_dim: int = eqx.field(static=True)
    param_dim: int = eqx.field(static=True)
    learned_solver_enabled: bool = eqx.field(static=True)
    self_correcting_enabled: bool = eqx.field(static=True)
    self_correcting_weight: float = eqx.field(static=True)
    neural_cde_enabled: bool = eqx.field(static=True)
    path_dim: int = eqx.field(static=True)
    path_representation: str = eqx.field(static=True)

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
        neural_cde_enabled: bool = False,
        learned_solver_enabled: bool = False,
        solver_hidden_dim: int = 64,
        solver_layers: int = 2,
        self_correcting_enabled: bool = False,
        self_correcting_weight: float = 0.05,
        correction_hidden_dim: int = 64,
        correction_layers: int = 2,
        path_representation: str = "delta",
        *,
        key: PRNGKeyArray,
    ):
        key_drift, key_diff, key_cde, key_solver, key_correction = jax.random.split(key, 5)

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
        self.learned_solver_enabled = learned_solver_enabled
        self.self_correcting_enabled = self_correcting_enabled
        self.self_correcting_weight = self_correcting_weight
        self.neural_cde_enabled = neural_cde_enabled
        self.path_dim = 1 + control_dim + disturbance_dim
        self.path_representation = path_representation

        cde_input_dim = latent_dim + control_dim + disturbance_dim + param_dim
        if self.neural_cde_enabled:
            cde_keys = jax.random.split(key_cde, drift_layers + 1)
            self.cde_layers = []
            for i in range(drift_layers):
                in_dim = cde_input_dim if i == 0 else hidden_dim
                self.cde_layers.append(eqx.nn.Linear(in_dim, hidden_dim, key=cde_keys[i]))
            cde_out = eqx.nn.Linear(
                hidden_dim + cde_input_dim,
                latent_dim * self.path_dim,
                key=cde_keys[-1],
            )
            self.cde_out = eqx.tree_at(
                lambda l: (l.weight, l.bias),
                cde_out,
                (cde_out.weight * 0.01, cde_out.bias * 0.01),
            )
        else:
            self.cde_layers = []
            self.cde_out = None

        solver_input_dim = latent_dim + control_dim + disturbance_dim + param_dim + 1
        solver_keys = jax.random.split(key_solver, solver_layers + 1)
        self.solver_gate_layers = []
        for i in range(solver_layers):
            in_dim = solver_input_dim if i == 0 else solver_hidden_dim
            self.solver_gate_layers.append(
                eqx.nn.Linear(in_dim, solver_hidden_dim, key=solver_keys[i])
            )
        self.solver_gate_out = eqx.nn.Linear(
            solver_hidden_dim if solver_layers > 0 else solver_input_dim,
            1,
            key=solver_keys[-1],
        )

        correction_input_dim = (
            latent_dim
            + latent_dim
            + control_dim
            + disturbance_dim
            + param_dim
            + 1
        )
        correction_keys = jax.random.split(key_correction, correction_layers + 1)
        self.correction_layers = []
        for i in range(correction_layers):
            in_dim = correction_input_dim if i == 0 else correction_hidden_dim
            self.correction_layers.append(
                eqx.nn.Linear(in_dim, correction_hidden_dim, key=correction_keys[i])
            )
        corr_out = eqx.nn.Linear(
            correction_hidden_dim if correction_layers > 0 else correction_input_dim,
            latent_dim,
            key=correction_keys[-1],
        )
        self.correction_out = eqx.tree_at(
            lambda l: (l.weight, l.bias),
            corr_out,
            (corr_out.weight * 0.01, corr_out.bias * 0.01),
        )

        if nominal_disturbance is not None:
            self.nominal_disturbance = jnp.array(nominal_disturbance)
        elif disturbance_dim > 0:
            self.nominal_disturbance = jnp.zeros(disturbance_dim)
        else:
            self.nominal_disturbance = jnp.zeros(0)

    def build_path_features(
        self,
        u_t: Float[Array, "control_dim"],
        u_tp1: Float[Array, "control_dim"],
        d_t: Float[Array, "disturbance_dim"],
        d_tp1: Float[Array, "disturbance_dim"],
        step_dt,
        dtype,
    ) -> Float[Array, "path_dim"]:
        """Build path features for the Neural CDE correction."""
        safe_dt = jnp.maximum(step_dt, jnp.asarray(1e-6, dtype=step_dt.dtype))
        if self.path_representation == "delta":
            path_terms = [
                jnp.array([step_dt], dtype=dtype),
                u_tp1 - u_t,
            ]
            if self.disturbance_dim > 0:
                path_terms.append(d_tp1 - d_t)
        else:
            path_terms = [
                jnp.array([1.0], dtype=dtype),
                (u_tp1 - u_t) / safe_dt,
            ]
            if self.disturbance_dim > 0:
                path_terms.append((d_tp1 - d_t) / safe_dt)
        return jnp.concatenate(path_terms)

    def control_path_term(
        self,
        z: Float[Array, "latent_dim"],
        u: Float[Array, "control_dim"],
        d: Float[Array, "disturbance_dim"],
        c: Float[Array, "param_dim"],
        path_derivative: Float[Array, "path_dim"],
    ) -> Float[Array, "latent_dim"]:
        """Neural CDE control-path contribution evaluated on a rollout interval."""
        if not self.neural_cde_enabled or self.cde_out is None:
            return jnp.zeros_like(z)

        u_norm = (u - self.drift.control_center) * self.drift.control_scale
        c_norm = c * self.drift.param_scale
        parts = [z, u_norm]
        if d.shape[-1] > 0:
            d_norm = (
                d - self.drift.disturbance_center[: d.shape[-1]]
            ) * self.drift.disturbance_scale[: d.shape[-1]]
            parts.append(d_norm)
        parts.append(c_norm)
        x = jnp.concatenate(parts)
        x_in = x

        for i, layer in enumerate(self.cde_layers):
            h = jax.nn.gelu(layer(x))
            x = x + h if i > 0 else h

        matrix = self.cde_out(jnp.concatenate([x, x_in])).reshape(
            self.latent_dim, self.path_dim
        )
        return matrix @ path_derivative

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

    def solver_gate(
        self,
        z: Float[Array, "latent_dim"],
        u: Float[Array, "control_dim"],
        d: Float[Array, "disturbance_dim"],
        c: Float[Array, "param_dim"],
        dt,
    ) -> Float[Array, ""]:
        """Blend Euler and Heun updates for the learned deterministic solver."""
        if not self.learned_solver_enabled:
            return jnp.array(1.0, dtype=z.dtype)

        u_norm = (u - self.drift.control_center) * self.drift.control_scale
        c_norm = c * self.drift.param_scale
        parts = [z, u_norm]
        if d.shape[-1] > 0:
            d_norm = (
                d - self.drift.disturbance_center[: d.shape[-1]]
            ) * self.drift.disturbance_scale[: d.shape[-1]]
            parts.append(d_norm)
        parts.append(c_norm)
        parts.append(jnp.array([dt], dtype=z.dtype))
        x = jnp.concatenate(parts)

        for i, layer in enumerate(self.solver_gate_layers):
            h = jax.nn.gelu(layer(x))
            x = x + h if i > 0 else h
        return jax.nn.sigmoid(self.solver_gate_out(x)).reshape(())

    def self_correction(
        self,
        z: Float[Array, "latent_dim"],
        local_error: Float[Array, "latent_dim"],
        u: Float[Array, "control_dim"],
        d: Float[Array, "disturbance_dim"],
        c: Float[Array, "param_dim"],
        dt,
    ) -> Float[Array, "latent_dim"]:
        """Predict a small learned correction from local solver disagreement."""
        if not self.self_correcting_enabled:
            return jnp.zeros_like(z)

        u_norm = (u - self.drift.control_center) * self.drift.control_scale
        c_norm = c * self.drift.param_scale
        parts = [z, local_error, u_norm]
        if d.shape[-1] > 0:
            d_norm = (
                d - self.drift.disturbance_center[: d.shape[-1]]
            ) * self.drift.disturbance_scale[: d.shape[-1]]
            parts.append(d_norm)
        parts.append(c_norm)
        parts.append(jnp.array([dt], dtype=z.dtype))
        x = jnp.concatenate(parts)

        for i, layer in enumerate(self.correction_layers):
            h = jax.nn.gelu(layer(x))
            x = x + h if i > 0 else h
        raw_correction = self.correction_out(x)
        return self.self_correcting_weight * jnp.tanh(raw_correction)

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
        """Noise-free drift-path integration (not the SDE mean E[z_t])."""
        disturbances = self._get_disturbances(ts, disturbances, z0.dtype)
        if self.neural_cde_enabled:
            dt_steps = ts[1:] - ts[:-1]

            def step_fn(z_prev, step_inputs):
                u_t, u_tp1, d_t, d_tp1, step_dt = step_inputs
                u_mid = 0.5 * (u_t + u_tp1)
                d_mid = 0.5 * (d_t + d_tp1)
                path_features = self.build_path_features(
                    u_t, u_tp1, d_t, d_tp1, step_dt, z_prev.dtype
                )

                def total_drift(z_curr):
                    return self.drift(z_curr, u_mid, d_mid, params) + self.control_path_term(
                        z_curr, u_mid, d_mid, params, path_features
                    )

                k1 = total_drift(z_prev)
                z_euler = z_prev + step_dt * k1
                k2 = total_drift(z_euler)
                z_next = z_prev + 0.5 * step_dt * (k1 + k2)
                return z_next, z_next

            _, z_hist = jax.lax.scan(
                step_fn,
                z0,
                (controls[:-1], controls[1:], disturbances[:-1], disturbances[1:], dt_steps),
            )
            return jnp.concatenate([z0[None, :], z_hist], axis=0)

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
