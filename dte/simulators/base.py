"""Base classes and SystemSpec for process system abstraction."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import jax.numpy as jnp
from jaxtyping import Array, Float


@dataclass
class DecoderConstraint:
    """Specification for a physical constraint on decoder outputs.

    Supported types:
        - "softplus": output = softplus(raw + bias), enforces non-negativity
        - "sigmoid_range": output = low + (high - low) * sigmoid(raw), bounded range
        - "none": no constraint, pass through raw value
    """

    type: str
    indices: List[int]
    bias: float = 0.5
    low: float = 0.0
    high: float = 1.0


@dataclass
class NormalizationSpec:
    """Centers and scales used to normalize model inputs.

    All arrays are length-matched to the corresponding dimension.
    """

    state_center: List[float]
    state_scale: List[float]
    control_center: List[float]
    control_scale: List[float]
    disturbance_center: List[float]
    disturbance_scale: List[float]
    param_scale: float = 0.1


@dataclass
class SystemSpec:
    """Complete specification of a process system.

    This is the single object passed through the engine to make every
    component system-agnostic. Create one per system type (CSTR, heat
    exchanger, etc.) and register it in the system registry.
    """

    name: str
    state_dim: int
    control_dim: int
    disturbance_dim: int
    param_dim: int

    state_names: List[str]
    control_names: List[str]
    disturbance_names: List[str]

    decoder_constraints: List[DecoderConstraint]
    normalization: NormalizationSpec

    default_initial_state: List[float]
    default_nominal_disturbance: List[float]

    # Operating range info used by dashboard / MPC scripts
    control_ranges: Dict[str, List[float]] = field(default_factory=dict)
    disturbance_ranges: Dict[str, List[float]] = field(default_factory=dict)

    def state_center_array(self) -> Float[Array, "state_dim"]:
        return jnp.array(self.normalization.state_center)

    def state_scale_array(self) -> Float[Array, "state_dim"]:
        return jnp.array(self.normalization.state_scale)

    def control_center_array(self) -> Float[Array, "control_dim"]:
        return jnp.array(self.normalization.control_center)

    def control_scale_array(self) -> Float[Array, "control_dim"]:
        return jnp.array(self.normalization.control_scale)

    def disturbance_center_array(self) -> Float[Array, "disturbance_dim"]:
        return jnp.array(self.normalization.disturbance_center)

    def disturbance_scale_array(self) -> Float[Array, "disturbance_dim"]:
        return jnp.array(self.normalization.disturbance_scale)

    def default_initial_state_array(self) -> Float[Array, "state_dim"]:
        return jnp.array(self.default_initial_state)

    def default_nominal_disturbance_array(self) -> Float[Array, "disturbance_dim"]:
        return jnp.array(self.default_nominal_disturbance)


class ProcessSimulator(ABC):
    """Abstract base class for all process simulators.

    Every system must implement this interface so that the data generation
    pipeline can operate generically.
    """

    @property
    @abstractmethod
    def spec(self) -> SystemSpec:
        """Return the SystemSpec for this process."""

    @abstractmethod
    def dynamics(
        self,
        t: float,
        state: Float[Array, "state_dim"],
        control: Float[Array, "control_dim"],
        disturbance: Float[Array, "disturbance_dim"],
    ) -> Float[Array, "state_dim"]:
        """Compute state time-derivatives (ODE right-hand side)."""

    @abstractmethod
    def simulate(
        self,
        initial_state: Float[Array, "state_dim"],
        control_trajectory: Float[Array, "n_steps control_dim"],
        disturbance_trajectory: Float[Array, "n_steps disturbance_dim"],
        t_span: Tuple[float, float],
        dt: float = 0.1,
        n_steps: int = 1000,
    ) -> Dict[str, Float[Array, "..."]]:
        """Simulate a trajectory and return a dict with 'time', 'states', 'controls'."""

    @abstractmethod
    def steady_state(
        self,
        control: Float[Array, "control_dim"],
        disturbance: Float[Array, "disturbance_dim"],
        initial_guess: Optional[Float[Array, "state_dim"]] = None,
    ) -> Float[Array, "state_dim"]:
        """Compute the steady state for the given constant inputs."""
