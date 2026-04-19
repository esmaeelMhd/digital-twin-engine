"""Base protocol for system-specific physics loss implementations."""

from abc import ABC, abstractmethod
from typing import Dict

from jaxtyping import Array, Float


class PhysicsLoss(ABC):
    """Protocol for computing physics-based residual losses.

    Each process system implements this class in its own physics module.
    The :class:`~dte.training.shared.losses.LossComputer` holds a single
    ``PhysicsLoss`` instance and calls :meth:`compute_residuals` without
    needing to know which system it is operating on.
    """

    @abstractmethod
    def compute_residuals(
        self,
        states: Float[Array, "batch seq_len state_dim"],
        controls: Float[Array, "batch seq_len control_dim"],
        disturbances: Float[Array, "batch seq_len disturbance_dim"],
        dt: float,
        params_batch=None,
    ) -> Dict[str, Float[Array, ""]]:
        """Return a dict of named scalar residuals.

        Each value should already be reduced to a scalar (e.g. mean over
        the batch and time dimensions). Typical keys: ``"mass"``,
        ``"energy"``, ``"species_mass"``. ``params_batch`` contains one
        physical-parameter vector per trajectory when the dataset was
        generated with per-trajectory parameter randomization.
        """

    def residual_names(self) -> list:
        """Return the residual names this physics module produces.

        Used by :class:`~dte.training.shared.losses.LossComputer` to build the
        loss weight dictionary dynamically.  Override if your system
        uses non-standard names.
        """
        return []


class NullPhysicsLoss(PhysicsLoss):
    """A no-op physics loss for systems that have no implemented constraints."""

    def compute_residuals(self, states, controls, disturbances, dt, params_batch=None):
        del states, controls, disturbances, dt, params_batch
        return {}

    def residual_names(self):
        return []
