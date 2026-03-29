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
        """Initialize Digital Twin.
        
        Args:
            encoder: Encoder module
            decoder: Decoder module
            latent_sde: Latent SDE module
        """
        self.encoder = encoder
        self.decoder = decoder
        self.latent_sde = latent_sde
    
    @classmethod
    def from_config(cls, config: dict, key: PRNGKeyArray) -> "DigitalTwin":
        """Create model from config dict.
        
        Args:
            config: Configuration dictionary
            key: PRNG key
            
        Returns:
            Initialized Digital Twin model
        """
        key_enc, key_dec, key_sde = jax.random.split(key, 3)
        
        model_config = config["model"]
        
        # Create encoder
        encoder = Encoder(
            state_dim=model_config.get("state_dim", 4),
            param_dim=model_config.get("param_dim", 6),
            control_dim=model_config.get("control_dim", 2),
            latent_dim=model_config["latent_dim"],
            hidden_dim=model_config["hidden_dim"],
            n_layers=model_config["n_layers"],
            key=key_enc,
        )
        
        # Create decoder
        decoder = Decoder(
            latent_dim=model_config["latent_dim"],
            param_dim=model_config.get("param_dim", 6),
            control_dim=model_config.get("control_dim", 2),
            state_dim=model_config.get("state_dim", 4),
            hidden_dim=model_config["hidden_dim"],
            n_layers=model_config["n_layers"],
            key=key_dec,
        )
        
        # Create latent SDE
        latent_sde = LatentSDE(
            latent_dim=model_config["latent_dim"],
            control_dim=model_config.get("control_dim", 2),
            param_dim=model_config.get("param_dim", 6),
            hidden_dim=model_config["hidden_dim"],
            drift_layers=model_config.get("drift_layers", 3),
            diffusion_layers=model_config.get("diffusion_layers", 2),
            diffusion_hidden_dim=model_config.get("diffusion_hidden_dim", 64),
            initial_diffusion_scale=model_config.get("initial_diffusion_scale", 0.1),
            key=key_sde,
        )
        
        model = cls(encoder, decoder, latent_sde)
        
        # Print parameter count
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
        """Encode physical state to latent.
        
        Args:
            state: Physical state
            params: System parameters
            control: Control input
            key: PRNG key (if None, return mean without sampling)
            
        Returns:
            Tuple of (z, z_mean, z_logvar)
        """
        z_mean, z_logvar = self.encoder.encode(state, params, control)
        z_mean = 10.0 * jnp.tanh(z_mean / 10.0)
        z_logvar = 10.0 * jnp.tanh(z_logvar / 10.0)
        
        if key is not None:
            z = self.encoder.sample(z_mean, z_logvar, key)
        else:
            z = z_mean
        
        return z, z_mean, z_logvar
    
    def decode(
        self,
        z: Float[Array, "latent_dim"],
        params: Float[Array, "param_dim"],
        control: Float[Array, "control_dim"],
    ) -> Float[Array, "state_dim"]:
        """Decode latent state to physical state.
        
        Args:
            z: Latent state
            params: System parameters
            control: Control input
            
        Returns:
            Physical state
        """
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
        """Full prediction pipeline.
        
        1. Encode initial state -> z0
        2. Roll out latent SDE -> z_trajectory
        3. Decode all timesteps -> predicted_states
        
        Args:
            initial_state: Initial physical state
            controls: Control trajectory
            disturbances: Disturbance trajectory (not used in latent dynamics but kept for API consistency)
            params: System parameters
            ts: Time points
            key: PRNG key
            
        Returns:
            Dictionary with:
                - states: (n_steps, state_dim) predicted physical states
                - latent: (n_steps, latent_dim) latent trajectory
                - z_mean: (latent_dim,) encoder mean
                - z_logvar: (latent_dim,) encoder log-variance
        """
        key_enc, key_sde = jax.random.split(key)
        
        # Encode initial state
        z0, z_mean, z_logvar = self.encode(
            initial_state, params, controls[0], key_enc
        )
        
        # Roll out latent SDE
        z_trajectory = self.latent_sde(ts, z0, controls, params, key_sde)
        
        # Decode all timesteps
        decode_fn = jax.vmap(
            lambda z, u: self.decode(z, params, u),
            in_axes=(0, 0)
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
        """Sample multiple trajectories from the SDE.
        
        Args:
            initial_state: Initial physical state
            controls: Control trajectory
            disturbances: Disturbance trajectory
            params: System parameters
            ts: Time points
            key: PRNG key
            n_samples: Number of samples
            
        Returns:
            Dictionary with:
                - states_mean: (n_steps, state_dim) mean prediction
                - states_std: (n_steps, state_dim) standard deviation
                - states_samples: (n_samples, n_steps, state_dim) all samples
        """
        keys = jax.random.split(key, n_samples)
        
        # Generate multiple predictions
        def predict_one(k):
            result = self.predict(initial_state, controls, disturbances, params, ts, k)
            return result["states"]
        
        states_samples = jax.vmap(predict_one)(keys)
        
        # Compute statistics
        states_mean = jnp.mean(states_samples, axis=0)
        states_std = jnp.std(states_samples, axis=0)
        
        return {
            "states_mean": states_mean,
            "states_std": states_std,
            "states_samples": states_samples,
        }
    
    def save(self, path: str):
        """Save model using equinox serialization.
        
        Args:
            path: Output file path
        """
        with open(path, "wb") as f:
            eqx.tree_serialise_leaves(f, self)
        print(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path: str, config: dict) -> "DigitalTwin":
        """Load model using equinox deserialization.
        
        Args:
            path: Input file path
            config: Configuration dictionary (needed to reconstruct structure)
            
        Returns:
            Loaded Digital Twin model
        """
        # Create a template model with same structure
        key = jax.random.PRNGKey(0)  # Key doesn't matter for structure
        template = cls.from_config(config, key)
        
        # Load parameters into template
        with open(path, "rb") as f:
            model = eqx.tree_deserialise_leaves(f, template)
        
        print(f"Model loaded from {path}")
        return model
    
    def get_parameter_count(self) -> Dict[str, int]:
        """Get parameter counts for each submodule.
        
        Returns:
            Dictionary with parameter counts
        """
        def count_params(module):
            return sum(x.size for x in jax.tree.leaves(eqx.filter(module, eqx.is_array)))
        
        return {
            "encoder": count_params(self.encoder),
            "decoder": count_params(self.decoder),
            "latent_sde": count_params(self.latent_sde),
            "total": count_params(self),
        }
