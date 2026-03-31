"""Loss functions for training the digital twin model."""

from typing import Dict, List, Optional, Tuple
import jax.numpy as jnp
from jaxtyping import Array, Float

from dte.physics.base import PhysicsLoss, NullPhysicsLoss


class LossComputer:
    """Computes all loss terms for the digital twin model.

    Completely decoupled from any specific process system.  Physics residuals
    are delegated to a :class:`~dte.physics.base.PhysicsLoss` instance that
    is supplied at construction time.
    """

    def __init__(
        self,
        config: dict,
        normalization_stats: dict,
        physics_loss: Optional[PhysicsLoss] = None,
        state_names: Optional[List[str]] = None,
    ):
        """
        Args:
            config: Full training configuration dict.
            normalization_stats: Dataset normalization statistics
                (``state_mean``, ``state_std``, etc.).
            physics_loss: System-specific physics loss implementation.
                Pass ``None`` or omit to disable physics losses.
            state_names: Ordered list of state variable names used for
                per-state loss weighting.  Defaults to generic labels.
        """
        self.config = config
        self.norm_stats = normalization_stats
        self.physics = physics_loss if physics_loss is not None else NullPhysicsLoss()

        # Extract scalar loss weights from config
        lw = config["loss_weights"]
        self.w_recon = lw["reconstruction"]
        self.w_kl = lw.get("kl", 0.0001)
        self.w_traj = lw["trajectory"]
        self.w_one_step = lw.get("one_step", 0.0)

        # Physics weights are keyed by the residual names returned by the
        # physics loss object.  Fall back to zero for unknown residuals.
        phys_weights_cfg = config.get("physics_loss_weights", {})
        # Support both flat loss_weights keys (new) and legacy aliases
        legacy_map = {
            "mass": lw.get("mass_balance", lw.get("mass", 0.0)),
            "species_mass": lw.get("species_mass_balance", lw.get("species_mass", 0.0)),
            "energy": lw.get("energy_balance", lw.get("energy", 0.0)),
        }
        self.physics_weights: Dict[str, float] = {}
        for name in self.physics.residual_names():
            self.physics_weights[name] = phys_weights_cfg.get(
                name, legacy_map.get(name, lw.get(name, 0.0))
            )

        # Per-state loss weights
        if state_names is None:
            state_mean = normalization_stats.get("state_mean")
            if state_mean is None:
                raise ValueError(
                    "normalization_stats must include state_mean when state_names are omitted."
                )
            # Fall back to generic state_0, state_1, ... labels
            n_states = state_mean.shape[-1]
            state_names = [f"state_{i}" for i in range(n_states)]
        self.state_names = state_names

        raw_sw = config.get("state_loss_weights", {})
        if isinstance(raw_sw, dict):
            raw_sw_list = [float(raw_sw.get(s, 1.0)) for s in state_names]
        elif isinstance(raw_sw, list):
            raw_sw_list = [float(v) for v in raw_sw]
        else:
            raw_sw_list = [1.0] * len(state_names)
        self.state_loss_weights = jnp.asarray(raw_sw_list, dtype=jnp.float32)
        self.state_loss_weights = self.state_loss_weights / jnp.mean(self.state_loss_weights)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _huber_loss(diff: Array) -> Array:
        """Elementwise Huber loss with delta=0.01."""
        return jnp.where(
            jnp.abs(diff) < 0.01,
            0.5 * diff ** 2,
            0.01 * jnp.abs(diff) - 0.00005,
        )

    def _weighted_state_loss(self, diff: Array) -> Float[Array, ""]:
        """State-weighted Huber loss in normalized state space."""
        state_weights = self.state_loss_weights.reshape((1,) * (diff.ndim - 1) + (-1,))
        return jnp.mean(self._huber_loss(diff) * state_weights)

    # ------------------------------------------------------------------
    # Individual loss terms
    # ------------------------------------------------------------------

    def reconstruction_loss(
        self,
        predicted_states: Float[Array, "batch state_dim"],
        true_states: Float[Array, "batch state_dim"],
    ) -> Float[Array, ""]:
        """MSE in normalized state space (initial step)."""
        return self._weighted_state_loss(predicted_states - true_states)

    def kl_divergence_loss(
        self,
        z_mean: Float[Array, "batch latent_dim"],
        z_logvar: Float[Array, "batch latent_dim"],
    ) -> Float[Array, ""]:
        """Standard VAE KL divergence KL(q || N(0,I))."""
        kl = -0.5 * jnp.sum(1 + z_logvar - z_mean ** 2 - jnp.exp(z_logvar), axis=-1)
        return jnp.mean(kl)

    def sde_kl_loss(
        self,
        diffusion_trajectory: Float[Array, "batch seq_len latent_dim"],
        dt: float,
    ) -> Float[Array, ""]:
        """Approximate SDE KL term: penalizes diffusion magnitudes.

        This is a tractable surrogate for the Girsanov KL between the learned
        SDE and a unit Wiener process prior.  It encourages the diffusion
        coefficients to stay small unless the data requires stochasticity.

        KL ~ (1 / (2*dt)) * sum_t ||sigma(z_t)||^2 * dt
           = (1/2) * mean_t ||sigma||^2
        """
        # diffusion_trajectory: (batch, seq_len, latent_dim) -- squared coefficients
        return 0.5 * jnp.mean(diffusion_trajectory ** 2)

    def trajectory_loss(
        self,
        predicted_trajectory: Float[Array, "batch seq_len state_dim"],
        true_trajectory: Float[Array, "batch seq_len state_dim"],
    ) -> Float[Array, ""]:
        """Weighted trajectory MSE -- earlier steps weighted higher."""
        seq_len = predicted_trajectory.shape[1]
        weights = jnp.linspace(1.0, 0.1, seq_len)
        weights = weights / jnp.mean(weights)

        diff = predicted_trajectory - true_trajectory
        state_weights = self.state_loss_weights[None, None, :]
        weighted_huber = self._huber_loss(diff) * weights[None, :, None] * state_weights
        return jnp.mean(weighted_huber)

    def one_step_loss(
        self,
        predicted_next: Float[Array, "batch seq_len_minus_one state_dim"],
        true_next: Float[Array, "batch seq_len_minus_one state_dim"],
    ) -> Float[Array, ""]:
        """Teacher-forced one-step prediction loss."""
        return self._weighted_state_loss(predicted_next - true_next)

    def physics_losses(
        self,
        predicted_states: Float[Array, "batch seq_len state_dim"],
        controls: Float[Array, "batch seq_len control_dim"],
        disturbances: Float[Array, "batch seq_len disturbance_dim"],
        dt,
    ) -> Tuple[Float[Array, ""], Dict[str, Float[Array, ""]]]:
        """Compute all physics residuals and their weighted sum.

        Returns:
            total_physics_loss: weighted sum of all residuals.
            residual_dict: raw (unweighted) residuals keyed by name.
        """
        residuals = self.physics.compute_residuals(
            predicted_states, controls, disturbances, dt
        )
        total = sum(
            self.physics_weights.get(k, 0.0) * v for k, v in residuals.items()
        )
        if not residuals:
            total = jnp.array(0.0)
        return total, residuals

    # ------------------------------------------------------------------
    # Denormalization helpers
    # ------------------------------------------------------------------

    def denormalize_states(self, states):
        return states * (self.norm_stats["state_std"] + 1e-8) + self.norm_stats["state_mean"]

    def denormalize_controls(self, controls):
        return controls * (self.norm_stats["control_std"] + 1e-8) + self.norm_stats["control_mean"]

    def denormalize_disturbances(self, disturbances):
        return disturbances * (self.norm_stats["disturbance_std"] + 1e-8) + self.norm_stats["disturbance_mean"]

    # ------------------------------------------------------------------
    # KL annealing
    # ------------------------------------------------------------------

    def get_loss_weights(self, step: int = 0) -> Dict[str, float]:
        """Return current loss weights with KL annealing applied."""
        kl_config = self.config.get("kl_annealing", {})
        start_weight = kl_config.get("start_weight", 0.0)
        end_weight = kl_config.get("end_weight", self.w_kl)
        anneal_steps = kl_config.get("anneal_steps", 5000)

        if step < anneal_steps:
            kl_weight = start_weight + (end_weight - start_weight) * (step / anneal_steps)
        else:
            kl_weight = end_weight

        weights = {
            "reconstruction": self.w_recon,
            "kl": kl_weight,
            "one_step": self.w_one_step,
            "trajectory": self.w_traj,
        }
        weights.update(self.physics_weights)
        return weights
