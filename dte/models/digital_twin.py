"""Full Digital Twin model integrating encoder, decoder, latent SDE, and optional simulator prior."""

from typing import Dict, Optional, Tuple
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray
import diffrax

from dte.models.encoder import Encoder
from dte.models.grouped_encoder import GroupedStateEncoder
from dte.models.decoder import Decoder
from dte.models.latent_sde import LatentSDE
from dte.simulators.base import ProcessSimulator
from dte.simulators.registry import get_simulator


class DigitalTwin(eqx.Module):
    """Digital Twin model combining encoder, decoder, and latent SDE."""

    encoder: Encoder | GroupedStateEncoder
    decoder: Decoder
    latent_sde: LatentSDE
    simulator: Optional[ProcessSimulator] = eqx.field(static=True)
    simulator_prior_enabled: bool = eqx.field(static=True)
    simulator_prior_weight: float = eqx.field(static=True)
    residual_prior_weight: float = eqx.field(static=True)

    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        latent_sde: LatentSDE,
        simulator: Optional[ProcessSimulator] = None,
        simulator_prior_enabled: bool = False,
        simulator_prior_weight: float = 1.0,
        residual_prior_weight: float = 1.0,
    ):
        self.encoder = encoder
        self.decoder = decoder
        self.latent_sde = latent_sde
        self.simulator = simulator
        self.simulator_prior_enabled = simulator_prior_enabled
        self.simulator_prior_weight = simulator_prior_weight
        self.residual_prior_weight = residual_prior_weight

    @classmethod
    def from_config(
        cls,
        config: dict,
        key: PRNGKeyArray,
        system_spec,
        system_config: dict | None = None,
    ) -> "DigitalTwin":
        """Create model from config dict.

        Args:
            config: Training configuration dictionary.
            key: PRNG key.
            system_spec: :class:`~dte.simulators.base.SystemSpec` providing
                dimensions, normalization arrays, and decoder constraints.
            system_config: Parsed system YAML used to reconstruct a simulator
                when simulator-prior rollout is enabled.
        """
        key_enc, key_dec, key_sde = jax.random.split(key, 3)

        model_config = config["model"]

        if system_spec is None:
            raise ValueError("DigitalTwin.from_config requires a system_spec.")

        state_dim = system_spec.state_dim
        param_dim = system_spec.param_dim
        control_dim = system_spec.control_dim
        disturbance_dim = system_spec.disturbance_dim
        latent_dim = model_config["latent_dim"]
        hidden_dim = model_config["hidden_dim"]
        n_layers = model_config["n_layers"]

        norm = system_spec.normalization
        state_center = norm.state_center
        state_scale = norm.state_scale
        control_center = norm.control_center
        control_scale = norm.control_scale
        disturbance_center = norm.disturbance_center
        disturbance_scale = norm.disturbance_scale
        param_scale = norm.param_scale
        decoder_constraints = [
            {
                "type": c.type,
                "indices": c.indices,
                "bias": c.bias,
                "low": c.low,
                "high": c.high,
            }
            for c in system_spec.decoder_constraints
        ]
        nominal_disturbance = system_spec.default_nominal_disturbance

        grouped_encoder_cfg = model_config.get("grouped_encoder", {})
        channel_conditioning_cfg = model_config.get("channel_conditioning", {})
        law_conditioning_cfg = model_config.get("law_conditioning", {})
        if bool(grouped_encoder_cfg.get("enabled", False)):
            encoder = GroupedStateEncoder(
                system_spec=system_spec,
                param_dim=param_dim,
                control_dim=control_dim,
                latent_dim=latent_dim,
                hidden_dim=hidden_dim,
                n_layers=n_layers,
                group_token_dim=int(grouped_encoder_cfg.get("group_token_dim", hidden_dim)),
                group_kind_dim=int(grouped_encoder_cfg.get("group_kind_dim", 8)),
                group_encoder_layers=int(grouped_encoder_cfg.get("group_encoder_layers", 2)),
                group_mixer_layers=int(grouped_encoder_cfg.get("group_mixer_layers", 2)),
                channel_conditioning_enabled=bool(
                    channel_conditioning_cfg.get("enabled", False)
                ),
                law_conditioning_enabled=bool(law_conditioning_cfg.get("enabled", False)),
                key=key_enc,
            )
        else:
            encoder = Encoder(
                state_dim=state_dim,
                param_dim=param_dim,
                control_dim=control_dim,
                latent_dim=latent_dim,
                hidden_dim=hidden_dim,
                n_layers=n_layers,
                state_center=state_center,
                state_scale=state_scale,
                control_center=control_center,
                control_scale=control_scale,
                param_scale=param_scale,
                key=key_enc,
            )

        decoder = Decoder(
            latent_dim=latent_dim,
            param_dim=param_dim,
            control_dim=control_dim,
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            constraints=decoder_constraints,
            control_scale=control_scale,
            param_scale=param_scale,
            key=key_dec,
        )

        simulator_prior_cfg = model_config.get("simulator_prior", {})
        simulator_prior_enabled = bool(simulator_prior_cfg.get("enabled", False))
        hard_residual_only = bool(simulator_prior_cfg.get("hard_residual_only", False))
        correction_cfg = model_config.get("self_correcting_policy", {})
        neural_cde_cfg = model_config.get("neural_cde", {})

        latent_sde = LatentSDE(
            latent_dim=latent_dim,
            control_dim=control_dim,
            disturbance_dim=disturbance_dim,
            param_dim=param_dim,
            hidden_dim=hidden_dim,
            drift_layers=model_config.get("drift_layers", 3),
            diffusion_layers=model_config.get("diffusion_layers", 2),
            diffusion_hidden_dim=model_config.get("diffusion_hidden_dim", 64),
            initial_diffusion_scale=model_config.get("initial_diffusion_scale", 0.1),
            control_center=control_center,
            control_scale=control_scale,
            disturbance_center=disturbance_center,
            disturbance_scale=disturbance_scale,
            param_scale=param_scale,
            nominal_disturbance=nominal_disturbance,
            linear_prior_enabled=not (simulator_prior_enabled and hard_residual_only),
            neural_cde_enabled=bool(neural_cde_cfg.get("enabled", False)),
            learned_solver_enabled=bool(model_config.get("learned_solver", {}).get("enabled", False)),
            solver_hidden_dim=int(model_config.get("learned_solver", {}).get("hidden_dim", 64)),
            solver_layers=int(model_config.get("learned_solver", {}).get("n_layers", 2)),
            self_correcting_enabled=bool(correction_cfg.get("enabled", False)),
            self_correcting_weight=float(correction_cfg.get("weight", 0.05)),
            correction_hidden_dim=int(correction_cfg.get("hidden_dim", 64)),
            correction_layers=int(correction_cfg.get("n_layers", 2)),
            path_representation=str(neural_cde_cfg.get("path_representation", "delta")),
            key=key_sde,
        )

        simulator = None
        if simulator_prior_enabled and system_config is not None:
            simulator = get_simulator(system_spec.name, system_config)

        model = cls(
            encoder,
            decoder,
            latent_sde,
            simulator=simulator,
            simulator_prior_enabled=simulator_prior_enabled,
            simulator_prior_weight=float(simulator_prior_cfg.get("weight", 1.0)),
            residual_prior_weight=float(simulator_prior_cfg.get("residual_weight", 1.0)),
        )

        param_count = sum(x.size for x in jax.tree.leaves(eqx.filter(model, eqx.is_array)))
        print(f"Digital Twin initialized with {param_count:,} parameters")

        return model

    def latent_drift(
        self,
        z: Float[Array, "latent_dim"],
        control: Float[Array, "control_dim"],
        disturbance: Float[Array, "disturbance_dim"],
        params: Float[Array, "param_dim"],
        dt,
    ) -> Float[Array, "latent_dim"]:
        """Return drift with an optional simulator-informed physical prior."""

        residual_drift = self.latent_sde.drift(z, control, disturbance, params)

        if not self.simulator_prior_enabled or self.simulator is None:
            return residual_drift

        state = self.decode(z, params, control)
        state_dot = self.simulator.dynamics(0.0, state, control, disturbance)
        safe_dt = jnp.maximum(jnp.asarray(dt, dtype=state.dtype), jnp.asarray(1e-6, dtype=state.dtype))
        prior_state = state + safe_dt * state_dot
        z_prior_mean, _ = self.encoder.encode(prior_state, params, control)
        physical_drift = (z_prior_mean - z) / safe_dt

        return (
            self.simulator_prior_weight * physical_drift
            + self.residual_prior_weight * residual_drift
        )

    def rollout_latent(
        self,
        ts: Float[Array, "n_steps"],
        z0: Float[Array, "latent_dim"],
        controls: Float[Array, "n_steps control_dim"],
        params: Float[Array, "param_dim"],
        *,
        disturbances: Float[Array, "n_steps disturbance_dim"] | None = None,
        key: PRNGKeyArray | None = None,
        stochastic: bool = False,
    ) -> Float[Array, "n_steps latent_dim"]:
        """Roll out latent states with optional simulator-informed drift."""

        disturbances = self.latent_sde._get_disturbances(ts, disturbances, z0.dtype)
        control_interp = diffrax.LinearInterpolation(ts, controls)

        if self.latent_sde.disturbance_dim > 0:
            disturbance_interp = diffrax.LinearInterpolation(ts, disturbances)
            disturbance_at_time = disturbance_interp.evaluate
        else:
            disturbance_at_time = lambda t: jnp.zeros((0,), dtype=z0.dtype)  # noqa: E731

        base_dt = ts[1] - ts[0]

        def drift_fn(t, z, args):
            u = control_interp.evaluate(t)
            d = disturbance_at_time(t)
            return self.latent_drift(z, u, d, params, base_dt)

        saveat = diffrax.SaveAt(ts=ts)

        if stochastic:
            if key is None:
                raise ValueError("rollout_latent(stochastic=True) requires a PRNG key.")

            def diffusion_fn(t, z, args):
                u = control_interp.evaluate(t)
                d = disturbance_at_time(t)
                return jnp.diag(self.latent_sde.diffusion(z, u, d, params))

            dt0 = base_dt / 2.0
            brownian_motion = diffrax.VirtualBrownianTree(
                t0=ts[0],
                t1=ts[-1],
                tol=dt0 / 2.0,
                shape=(self.latent_sde.latent_dim,),
                key=key,
            )
            drift_term = diffrax.ODETerm(drift_fn)
            diffusion_term = diffrax.ControlTerm(diffusion_fn, brownian_motion)
            terms = diffrax.MultiTerm(drift_term, diffusion_term)
            solution = diffrax.diffeqsolve(
                terms,
                diffrax.Euler(),
                t0=ts[0],
                t1=ts[-1],
                dt0=dt0,
                y0=z0,
                saveat=saveat,
                max_steps=len(ts) * 10,
            )
            return solution.ys

        if self.latent_sde.learned_solver_enabled:
            dt_steps = ts[1:] - ts[:-1]

            def step_fn(z_prev, step_inputs):
                u_t, u_tp1, d_t, d_tp1, step_dt = step_inputs
                u_mid = 0.5 * (u_t + u_tp1)
                d_mid = 0.5 * (d_t + d_tp1)
                path_features = self.latent_sde.build_path_features(
                    u_t, u_tp1, d_t, d_tp1, step_dt, z_prev.dtype
                )

                def total_drift(z_curr):
                    base = self.latent_drift(z_curr, u_mid, d_mid, params, step_dt)
                    return base + self.latent_sde.control_path_term(
                        z_curr, u_mid, d_mid, params, path_features
                    )

                k1 = total_drift(z_prev)
                z_euler = z_prev + step_dt * k1
                k2 = total_drift(z_euler)
                z_heun = z_prev + 0.5 * step_dt * (k1 + k2)
                local_error = z_heun - z_euler
                alpha = self.latent_sde.solver_gate(
                    z_prev,
                    u_mid,
                    d_mid,
                    params,
                    step_dt,
                )
                z_next = alpha * z_heun + (1.0 - alpha) * z_euler
                z_next = z_next + self.latent_sde.self_correction(
                    z_prev,
                    local_error,
                    u_mid,
                    d_mid,
                    params,
                    step_dt,
                )
                return z_next, z_next

            _, z_hist = jax.lax.scan(
                step_fn,
                z0,
                (controls[:-1], controls[1:], disturbances[:-1], disturbances[1:], dt_steps),
            )
            return jnp.concatenate([z0[None, :], z_hist], axis=0)

        solution = diffrax.diffeqsolve(
            diffrax.ODETerm(drift_fn),
            diffrax.Tsit5(),
            t0=ts[0],
            t1=ts[-1],
            dt0=base_dt,
            y0=z0,
            saveat=saveat,
            stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-4),
            max_steps=4096,
        )
        return solution.ys

    def encode(
        self,
        state: Float[Array, "state_dim"],
        params: Float[Array, "param_dim"],
        control: Float[Array, "control_dim"],
        key: PRNGKeyArray = None,
    ) -> Tuple[Float[Array, "latent_dim"], Float[Array, "latent_dim"], Float[Array, "latent_dim"]]:
        """Encode physical state to latent -> (z, z_mean, z_logvar)."""
        z_mean, z_logvar = self.encoder.encode(state, params, control)

        if key is not None:
            z_sampled = self.encoder.sample(z_mean, z_logvar, key)
            # Blend deterministic mean and stochastic sample to stabilize rollout
            z = 0.5 * z_mean + 0.5 * z_sampled
        else:
            z = z_mean

        return z, z_mean, z_logvar

    def decode(
        self,
        z: Float[Array, "latent_dim"],
        params: Float[Array, "param_dim"],
        control: Float[Array, "control_dim"],
    ) -> Float[Array, "state_dim"]:
        """Decode latent state to physical state."""
        return self.decoder(z, params, control)

    def predict(
        self,
        initial_state: Float[Array, "state_dim"],
        controls: Float[Array, "n_steps control_dim"],
        disturbances: Float[Array, "n_steps dist_dim"],
        params: Float[Array, "param_dim"],
        ts: Float[Array, "n_steps"],
        key: PRNGKeyArray,
    ) -> Dict[str, Array]:
        """Full prediction pipeline (stochastic SDE rollout)."""
        key_enc, key_sde = jax.random.split(key)

        z0, z_mean, z_logvar = self.encode(initial_state, params, controls[0], key_enc)

        z_trajectory = self.rollout_latent(
            ts,
            z0,
            controls,
            params,
            disturbances=disturbances,
            key=key_sde,
            stochastic=True,
        )

        decode_fn = jax.vmap(
            lambda z, u: self.decode(z, params, u), in_axes=(0, 0)
        )
        predicted_states = decode_fn(z_trajectory, controls)

        return {
            "states": predicted_states,
            "latent": z_trajectory,
            "z_mean": z_mean,
            "z_logvar": z_logvar,
        }

    def predict_ensemble(
        self,
        initial_state: Float[Array, "state_dim"],
        controls: Float[Array, "n_steps control_dim"],
        disturbances: Float[Array, "n_steps dist_dim"],
        params: Float[Array, "param_dim"],
        ts: Float[Array, "n_steps"],
        key: PRNGKeyArray,
        n_samples: int = 20,
    ) -> Dict[str, Array]:
        """Sample multiple trajectories for uncertainty quantification."""
        keys = jax.random.split(key, n_samples)

        def predict_one(k):
            result = self.predict(initial_state, controls, disturbances, params, ts, k)
            return result["states"]

        states_samples = jax.vmap(predict_one)(keys)

        return {
            "states_mean": jnp.mean(states_samples, axis=0),
            "states_std": jnp.std(states_samples, axis=0),
            "states_samples": states_samples,
        }

    def save(self, path: str):
        """Save model using equinox serialization."""
        with open(path, "wb") as f:
            eqx.tree_serialise_leaves(f, self)
        print(f"Model saved to {path}")

    @classmethod
    def load(
        cls,
        path: str,
        config: dict,
        system_spec,
        system_config: dict | None = None,
    ) -> "DigitalTwin":
        """Load model using equinox deserialization."""
        if system_spec is None:
            raise ValueError("DigitalTwin.load requires a system_spec.")
        key = jax.random.PRNGKey(0)
        template = cls.from_config(
            config,
            key,
            system_spec=system_spec,
            system_config=system_config,
        )

        with open(path, "rb") as f:
            model = eqx.tree_deserialise_leaves(f, template)

        print(f"Model loaded from {path}")
        return model

    def get_parameter_count(self) -> Dict[str, int]:
        """Get parameter counts for each submodule."""
        def count_params(module):
            return sum(x.size for x in jax.tree.leaves(eqx.filter(module, eqx.is_array)))

        return {
            "encoder": count_params(self.encoder),
            "decoder": count_params(self.decoder),
            "latent_sde": count_params(self.latent_sde),
            "total": count_params(self),
        }
