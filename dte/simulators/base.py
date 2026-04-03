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

    def simulate_for_data_generation(
        self,
        initial_state: Float[Array, "state_dim"],
        control_trajectory: Float[Array, "n_steps control_dim"],
        disturbance_trajectory: Float[Array, "n_steps disturbance_dim"],
        t_span: Tuple[float, float],
        dt: float = 0.1,
        n_steps: int = 1000,
        ) -> Dict[str, Float[Array, "..."]]:
        """Optional fast rollout path used by offline data generation.

        Systems can override this with a cheaper fixed-grid or otherwise
        specialized implementation. The default falls back to ``simulate`` so
        future systems pick up the generic interface automatically.
        """
        return self.simulate(
            initial_state,
            control_trajectory,
            disturbance_trajectory,
            t_span,
            dt=dt,
            n_steps=n_steps,
        )

    def simulate_batch_for_data_generation(
        self,
        initial_states: Float[Array, "batch state_dim"],
        control_trajectories: Float[Array, "batch n_steps control_dim"],
        disturbance_trajectories: Float[Array, "batch n_steps disturbance_dim"],
        t_span: Tuple[float, float],
        dt: float = 0.1,
        n_steps: int = 1000,
    ) -> Dict[str, Float[Array, "..."]]:
        """Optional batched fast rollout path for offline data generation.

        Systems can override this with a vectorized implementation. The default
        loops over trajectories and delegates to ``simulate_for_data_generation``.
        """
        results = [
            self.simulate_for_data_generation(
                initial_states[idx],
                control_trajectories[idx],
                disturbance_trajectories[idx],
                t_span,
                dt=dt,
                n_steps=n_steps,
            )
            for idx in range(initial_states.shape[0])
        ]
        return {
            "time": jnp.stack([result["time"] for result in results]),
            "states": jnp.stack([result["states"] for result in results]),
            "controls": jnp.stack([result["controls"] for result in results]),
        }

    @abstractmethod
    def steady_state(
        self,
        control: Float[Array, "control_dim"],
        disturbance: Float[Array, "disturbance_dim"],
        initial_guess: Optional[Float[Array, "state_dim"]] = None,
    ) -> Float[Array, "state_dim"]:
        """Compute the steady state for the given constant inputs."""

    def steady_state_for_data_generation(
        self,
        control: Float[Array, "control_dim"],
        disturbance: Float[Array, "disturbance_dim"],
        initial_guess: Optional[Float[Array, "state_dim"]] = None,
    ) -> Float[Array, "state_dim"]:
        """Optional fast steady-state path used during dataset creation.

        Systems can override this when they have a closed-form solve or a
        cheaper approximation than the general-purpose ``steady_state`` path.
        """
        return self.steady_state(control, disturbance, initial_guess=initial_guess)

    def steady_state_batch_for_data_generation(
        self,
        controls: Float[Array, "batch control_dim"],
        disturbances: Float[Array, "batch disturbance_dim"],
        initial_guesses: Optional[Float[Array, "batch state_dim"]] = None,
    ) -> Float[Array, "batch state_dim"]:
        """Optional batched fast steady-state path for offline data generation.

        Systems can override this with a vectorized implementation. The default
        loops over trajectories and delegates to ``steady_state_for_data_generation``.
        """
        states = []
        for idx in range(controls.shape[0]):
            guess = None if initial_guesses is None else initial_guesses[idx]
            states.append(
                self.steady_state_for_data_generation(
                    controls[idx],
                    disturbances[idx],
                    initial_guess=guess,
                )
            )
        return jnp.stack(states)
