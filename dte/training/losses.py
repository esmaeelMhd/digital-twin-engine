"""Loss functions for training the digital twin model."""

from typing import Dict, Tuple
import jax.numpy as jnp
from jaxtyping import Array, Float

from dte.physics.conservation import mass_balance_residual, energy_balance_residual
from dte.simulators.cstr import CSTRParams


class LossComputer:
    """Computes all loss terms for the digital twin model."""
    
    def __init__(self, config: dict, normalization_stats: dict, params: CSTRParams):
        """Initialize loss computer.
        
        Args:
            config: Configuration dictionary with loss weights
            normalization_stats: Data normalization statistics
            params: CSTR parameters for physics losses
        """
        self.config = config
        self.norm_stats = normalization_stats
        self.params = params
        
        # Extract loss weights
        self.w_recon = config["loss_weights"]["reconstruction"]
        self.w_kl = 0.0001
        self.w_traj = config["loss_weights"]["trajectory"]
        self.w_mass = config["loss_weights"]["mass_balance"]
        self.w_energy = config["loss_weights"]["energy_balance"]
    
    def reconstruction_loss(
        self,
        predicted_states: Float[Array, "batch seq_len state_dim"],
        true_states: Float[Array, "batch seq_len state_dim"],
    ) -> Float[Array, ""]:
        """MSE in normalized state space.
        
        Args:
            predicted_states: Predicted states (normalized)
            true_states: True states (normalized)
            
        Returns:
            Scalar loss
        """
        diff = predicted_states - true_states
        mse = jnp.mean(jnp.where(jnp.abs(diff) < 1.0, 0.5 * diff ** 2, jnp.abs(diff) - 0.5))
        return mse
    
    def kl_divergence_loss(
        self,
        z_mean: Float[Array, "batch latent_dim"],
        z_logvar: Float[Array, "batch latent_dim"],
    ) -> Float[Array, ""]:
        """Standard VAE KL divergence.
        
        KL(q(z|x) || p(z)) where p(z) = N(0, I)
        = -0.5 * sum(1 + logvar - mean^2 - exp(logvar))
        
        Args:
            z_mean: Encoder mean
            z_logvar: Encoder log-variance
            
        Returns:
            Scalar loss
        """
        kl = -0.5 * jnp.sum(1 + z_logvar - z_mean**2 - jnp.exp(z_logvar), axis=-1)
        return jnp.mean(kl)
    
    def physics_mass_loss(
        self,
        predicted_states: Float[Array, "batch seq_len state_dim"],
        controls: Float[Array, "batch seq_len control_dim"],
        disturbances: Float[Array, "batch seq_len dist_dim"],
        dt: float,
    ) -> Float[Array, ""]:
        """Mean mass balance residual.
        
        Args:
            predicted_states: Predicted states (DENORMALIZED)
            controls: Control inputs (DENORMALIZED)
            disturbances: Disturbances (DENORMALIZED)
            dt: Time step
            
        Returns:
            Scalar loss
        """
        batch_size = predicted_states.shape[0]
        residuals = []
        
        for i in range(batch_size):
            res = mass_balance_residual(
                predicted_states[i],
                controls[i],
                disturbances[i],
                self.params,
                dt,
            )
            residuals.append(jnp.mean(res))
        
        return jnp.mean(jnp.array(residuals))
    
    def physics_energy_loss(
        self,
        predicted_states: Float[Array, "batch seq_len state_dim"],
        controls: Float[Array, "batch seq_len control_dim"],
        disturbances: Float[Array, "batch seq_len dist_dim"],
        dt: float,
    ) -> Float[Array, ""]:
        """Mean energy balance residual.
        
        Args:
            predicted_states: Predicted states (DENORMALIZED)
            controls: Control inputs (DENORMALIZED)
            disturbances: Disturbances (DENORMALIZED)
            dt: Time step
            
        Returns:
            Scalar loss
        """
        batch_size = predicted_states.shape[0]
        residuals = []
        
        for i in range(batch_size):
            res = energy_balance_residual(
                predicted_states[i],
                controls[i],
                disturbances[i],
                self.params,
                dt,
            )
            residuals.append(jnp.mean(res))
        
        return jnp.mean(jnp.array(residuals))
    
    def trajectory_loss(
        self,
        predicted_trajectory: Float[Array, "batch seq_len state_dim"],
        true_trajectory: Float[Array, "batch seq_len state_dim"],
    ) -> Float[Array, ""]:
        """MSE over full trajectory with linearly increasing weights.
        
        Weight later timesteps higher to encourage accurate long-horizon prediction.
        
        Args:
            predicted_trajectory: Predicted trajectory (normalized)
            true_trajectory: True trajectory (normalized)
            
        Returns:
            Scalar loss
        """
        seq_len = predicted_trajectory.shape[1]
        
        # Uniform weights to prevent over-penalizing inherently noisy later SDE steps
        weights = jnp.ones(seq_len)
        weights = weights / jnp.mean(weights)  # Normalize to mean 1
        
        # Weighted Huber loss
        diff = predicted_trajectory - true_trajectory
        squared_error = jnp.where(jnp.abs(diff) < 0.5, 0.5 * diff ** 2, 0.5 * jnp.abs(diff) - 0.125)
        weighted_mse = jnp.mean(squared_error * weights[None, :, None])
        
        return weighted_mse
    
    def denormalize_states(
        self, states: Float[Array, "... state_dim"]
    ) -> Float[Array, "... state_dim"]:
        """Denormalize states."""
        return states * (self.norm_stats["state_std"] + 1e-8) + self.norm_stats["state_mean"]
    
    def denormalize_controls(
        self, controls: Float[Array, "... control_dim"]
    ) -> Float[Array, "... control_dim"]:
        """Denormalize controls."""
        return controls * (self.norm_stats["control_std"] + 1e-8) + self.norm_stats["control_mean"]
    
    def denormalize_disturbances(
        self, disturbances: Float[Array, "... dist_dim"]
    ) -> Float[Array, "... dist_dim"]:
        """Denormalize disturbances."""
        return disturbances * (self.norm_stats["disturbance_std"] + 1e-8) + self.norm_stats["disturbance_mean"]
    
    def total_loss(
        self,
        model,
        batch: Dict[str, Array],
        key,
        dt: float = 0.1,
    ) -> Tuple[Float[Array, ""], Dict[str, Float[Array, ""]]]:
        """Compute all losses for a batch.
        
        This method will be implemented in the DigitalTwin class
        since it requires the full model pipeline.
        
        Args:
            model: Digital twin model
            batch: Batch of data
            key: PRNG key
            dt: Time step
            
        Returns:
            Tuple of (total_loss, loss_dict)
        """
        raise NotImplementedError("This method is implemented in the DigitalTwin class")
    
    def get_loss_weights(self, step: int = 0) -> Dict[str, float]:
        """Get loss weights with optional KL annealing.
        
        Args:
            step: Current training step
            
        Returns:
            Dictionary of loss weights
        """
        kl_config = self.config.get("kl_annealing", {})
        start_weight = kl_config.get("start_weight", 0.0)
        end_weight = kl_config.get("end_weight", self.w_kl)
        anneal_steps = kl_config.get("anneal_steps", 5000)
        
        # Linear annealing
        if step < anneal_steps:
            kl_weight = start_weight + (end_weight - start_weight) * (step / anneal_steps)
        else:
            kl_weight = end_weight
        
        return {
            "reconstruction": self.w_recon,
            "kl": kl_weight,
            "trajectory": self.w_traj,
            "mass_balance": self.w_mass,
            "energy_balance": self.w_energy,
        }
