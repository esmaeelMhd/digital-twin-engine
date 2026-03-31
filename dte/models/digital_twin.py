"""Full Digital Twin model integrating encoder, decoder, and latent SDE."""

from typing import Dict, Tuple
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from dte.models.encoder import Encoder
from dte.models.decoder import Decoder
from dte.models.latent_sde import LatentSDE


class DigitalTwin(eqx.Module):
    """Digital Twin model combining encoder, decoder, and latent SDE."""

    encoder: Encoder
    decoder: Decoder
    latent_sde: LatentSDE

    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        latent_sde: LatentSDE,
    ):
        self.encoder = encoder
        self.decoder = decoder
        self.latent_sde = latent_sde

    @classmethod
    def from_config(cls, config: dict, key: PRNGKeyArray, system_spec=None) -> "DigitalTwin":
        """Create model from config dict.

        Args:
            config: Training configuration dictionary.
            key: PRNG key.
            system_spec: Optional :class:`~dte.simulators.base.SystemSpec`.
                When provided, normalization arrays and decoder constraints are
                taken from the spec rather than falling back to defaults.
        """
        key_enc, key_dec, key_sde = jax.random.split(key, 3)

        model_config = config["model"]

        # Dimensions: system_spec takes priority over config file values
        if system_spec is not None:
            state_dim = system_spec.state_dim
            param_dim = system_spec.param_dim
            control_dim = system_spec.control_dim
            disturbance_dim = system_spec.disturbance_dim
        else:
            state_dim = model_config.get("state_dim", 4)
            param_dim = model_config.get("param_dim", 6)
            control_dim = model_config.get("control_dim", 2)
            disturbance_dim = model_config.get("disturbance_dim", 2)
        latent_dim = model_config["latent_dim"]
        hidden_dim = model_config["hidden_dim"]
        n_layers = model_config["n_layers"]

        # Pull normalization and constraints from system_spec when available
        if system_spec is not None:
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
        else:
            state_center = None
            state_scale = None
            control_center = None
            control_scale = None
            disturbance_center = None
            disturbance_scale = None
            param_scale = 0.1
            decoder_constraints = None
            nominal_disturbance = None

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
            control_scale=[0.01] * control_dim if control_scale is None else [s * 0.01 / max(s, 1e-8) for s in control_scale],
            param_scale=param_scale,
            key=key_dec,
        )

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
            key=key_sde,
        )

        model = cls(encoder, decoder, latent_sde)

        param_count = sum(x.size for x in jax.tree.leaves(eqx.filter(model, eqx.is_array)))
        print(f"Digital Twin initialized with {param_count:,} parameters")

        return model

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
            # Keep stochasticity while damping early-training latent jumps.
            z = z_mean + 0.1 * (self.encoder.sample(z_mean, z_logvar, key) - z_mean)
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

        z_trajectory = self.latent_sde(
            ts, z0, controls, params, key_sde, disturbances=disturbances
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
    def load(cls, path: str, config: dict, system_spec=None) -> "DigitalTwin":
        """Load model using equinox deserialization."""
        key = jax.random.PRNGKey(0)
        template = cls.from_config(config, key, system_spec=system_spec)

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
