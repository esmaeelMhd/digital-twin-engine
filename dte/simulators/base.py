"""Base classes and unit specifications for process system abstraction."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import jax.numpy as jnp
from jaxtyping import Array, Float

from dte.core.state_schema import (
    ParameterDescriptor,
    SignalChannel,
    StateChannel,
    TopologyPort,
    infer_signal_channels,
    infer_state_channels,
)


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
class StateGroupSpec:
    """Semantic grouping for state channels in shared multi-system models."""

    name: str
    kind: str
    indices: List[int]


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
    state_groups: List[StateGroupSpec] = field(default_factory=list)

    def __post_init__(self):
        if not self.state_groups:
            self.state_groups = [
                StateGroupSpec(
                    name="all_states",
                    kind="generic",
                    indices=list(range(self.state_dim)),
                )
            ]

        if len(self.state_names) != self.state_dim:
            raise ValueError(
                f"{self.name}: expected {self.state_dim} state names, got {len(self.state_names)}."
            )
        if len(self.control_names) != self.control_dim:
            raise ValueError(
                f"{self.name}: expected {self.control_dim} control names, got {len(self.control_names)}."
            )
        if len(self.disturbance_names) != self.disturbance_dim:
            raise ValueError(
                f"{self.name}: expected {self.disturbance_dim} disturbance names, "
                f"got {len(self.disturbance_names)}."
            )

        covered: set[int] = set()
        normalized_groups: List[StateGroupSpec] = []
        for group in self.state_groups:
            indices = [int(idx) for idx in group.indices]
            if not indices:
                raise ValueError(f"{self.name}: state group '{group.name}' cannot be empty.")
            for idx in indices:
                if idx < 0 or idx >= self.state_dim:
                    raise ValueError(
                        f"{self.name}: state group '{group.name}' index {idx} is out of range "
                        f"for state_dim={self.state_dim}."
                    )
                if idx in covered:
                    raise ValueError(
                        f"{self.name}: state index {idx} appears in multiple state groups."
                    )
                covered.add(idx)
            normalized_groups.append(
                StateGroupSpec(name=str(group.name), kind=str(group.kind), indices=indices)
            )

        missing = [idx for idx in range(self.state_dim) if idx not in covered]
        if missing:
            raise ValueError(
                f"{self.name}: state_groups must cover every state index exactly once; "
                f"missing indices: {missing}."
            )

        self.state_groups = normalized_groups

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


@dataclass
class ProcessUnitSpec(SystemSpec):
    """Backward-compatible extension of :class:`SystemSpec`.

    Existing code can continue to treat this like a ``SystemSpec``. Phase 1
    surfaces the richer typed channel metadata through the additional fields.
    """

    state_channels: List[StateChannel] = field(default_factory=list)
    control_channels: List[SignalChannel] = field(default_factory=list)
    disturbance_channels: List[SignalChannel] = field(default_factory=list)
    parameter_descriptors: List[ParameterDescriptor] = field(default_factory=list)
    unit_type: str = "generic_unit"
    family: str = "generic"
    subtype: Optional[str] = None
    law_tags: List[str] = field(default_factory=list)
    topology_ports: List[TopologyPort] = field(default_factory=list)
    constraints_metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        super().__post_init__()

        if not self.state_channels:
            self.state_channels = infer_state_channels(
                self.state_names,
                self.state_groups,
                self.decoder_constraints,
            )
        if not self.control_channels:
            self.control_channels = infer_signal_channels(
                self.control_names,
                self.control_ranges,
            )
        if not self.disturbance_channels:
            self.disturbance_channels = infer_signal_channels(
                self.disturbance_names,
                self.disturbance_ranges,
            )
        if not self.parameter_descriptors:
            self.parameter_descriptors = [
                ParameterDescriptor(name=f"param_{idx}")
                for idx in range(self.param_dim)
            ]

        if len(self.state_channels) != self.state_dim:
            raise ValueError(
                f"{self.name}: expected {self.state_dim} state channels, "
                f"got {len(self.state_channels)}."
            )
        if len(self.control_channels) != self.control_dim:
            raise ValueError(
                f"{self.name}: expected {self.control_dim} control channels, "
                f"got {len(self.control_channels)}."
            )
        if len(self.disturbance_channels) != self.disturbance_dim:
            raise ValueError(
                f"{self.name}: expected {self.disturbance_dim} disturbance channels, "
                f"got {len(self.disturbance_channels)}."
            )
        if len(self.parameter_descriptors) != self.param_dim:
            raise ValueError(
                f"{self.name}: expected {self.param_dim} parameter descriptors, "
                f"got {len(self.parameter_descriptors)}."
            )

        if not self.constraints_metadata:
            self.constraints_metadata = {
                "decoder_constraints": [
                    {
                        "type": constraint.type,
                        "indices": list(constraint.indices),
                        "bias": constraint.bias,
                        "low": constraint.low,
                        "high": constraint.high,
                    }
                    for constraint in self.decoder_constraints
                ]
            }

    def state_lower_bounds(self) -> Float[Array, "state_dim"]:
        return jnp.asarray(
            [
                -jnp.inf if channel.lower_bound is None else channel.lower_bound
                for channel in self.state_channels
            ],
            dtype=jnp.float32,
        )

    def state_upper_bounds(self) -> Float[Array, "state_dim"]:
        return jnp.asarray(
            [
                jnp.inf if channel.upper_bound is None else channel.upper_bound
                for channel in self.state_channels
            ],
            dtype=jnp.float32,
        )

    def state_role_names(self) -> Tuple[str, ...]:
        return tuple(channel.role for channel in self.state_channels)


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

    def simulate_for_data_generation_with_params(
        self,
        initial_state: Float[Array, "state_dim"],
        control_trajectory: Float[Array, "n_steps control_dim"],
        disturbance_trajectory: Float[Array, "n_steps disturbance_dim"],
        params: Float[Array, "param_dim"],
        t_span: Tuple[float, float],
        dt: float = 0.1,
        n_steps: int = 1000,
    ) -> Dict[str, Float[Array, "..."]]:
        """Optional param-aware fast rollout path for offline data generation.

        The default ignores ``params`` and delegates to the bound-simulator path.
        Systems with per-trajectory parameter randomization should override this.
        """
        return self.simulate_for_data_generation(
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
        params_batch: Optional[Float[Array, "batch param_dim"]] = None,
        dt: float = 0.1,
        n_steps: int = 1000,
    ) -> Dict[str, Float[Array, "..."]]:
        """Optional batched fast rollout path for offline data generation.

        Systems can override this with a vectorized implementation. The default
        loops over trajectories and delegates to ``simulate_for_data_generation``.
        """
        results = [
            (
                self.simulate_for_data_generation(
                    initial_states[idx],
                    control_trajectories[idx],
                    disturbance_trajectories[idx],
                    t_span,
                    dt=dt,
                    n_steps=n_steps,
                )
                if params_batch is None
                else self.simulate_for_data_generation_with_params(
                    initial_states[idx],
                    control_trajectories[idx],
                    disturbance_trajectories[idx],
                    params_batch[idx],
                    t_span,
                    dt=dt,
                    n_steps=n_steps,
                )
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

    def steady_state_for_data_generation_with_params(
        self,
        control: Float[Array, "control_dim"],
        disturbance: Float[Array, "disturbance_dim"],
        params: Float[Array, "param_dim"],
        initial_guess: Optional[Float[Array, "state_dim"]] = None,
    ) -> Float[Array, "state_dim"]:
        """Optional param-aware fast steady-state path for data generation.

        The default ignores ``params`` and delegates to the bound-simulator path.
        """
        return self.steady_state_for_data_generation(
            control,
            disturbance,
            initial_guess=initial_guess,
        )

    def steady_state_batch_for_data_generation(
        self,
        controls: Float[Array, "batch control_dim"],
        disturbances: Float[Array, "batch disturbance_dim"],
        params_batch: Optional[Float[Array, "batch param_dim"]] = None,
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
                (
                    self.steady_state_for_data_generation(
                        controls[idx],
                        disturbances[idx],
                        initial_guess=guess,
                    )
                    if params_batch is None
                    else self.steady_state_for_data_generation_with_params(
                        controls[idx],
                        disturbances[idx],
                        params_batch[idx],
                        initial_guess=guess,
                    )
                )
            )
        return jnp.stack(states)

    def sample_data_generation_params(
        self,
        key,
    ) -> Float[Array, "param_dim"]:
        """Return the parameter vector stored in generated datasets.

        Systems with per-trajectory parameter randomization should override this.
        The default returns a vector of ones to match the historic generic path.
        """
        del key
        return jnp.ones(self.spec.param_dim)

    def sample_data_generation_params_batch(
        self,
        keys,
    ) -> Float[Array, "batch param_dim"]:
        """Vectorized parameter sampling hook for offline data generation."""
        return jnp.stack([self.sample_data_generation_params(key) for key in keys], axis=0)

    def format_data_generation_params(
        self,
        params: Float[Array, "param_dim"],
    ) -> Float[Array, "stored_param_dim"]:
        """Transform sampled simulator params into the dataset storage vector.

        Most systems can store the sampled simulator parameters directly. Systems
        that use an internal packed parameterization for fast rollout can
        override this to preserve their historic dataset schema.
        """
        return params

    def format_data_generation_params_batch(
        self,
        params_batch: Float[Array, "batch param_dim"],
    ) -> Float[Array, "batch stored_param_dim"]:
        """Vectorized dataset-parameter formatting hook."""
        return jnp.stack(
            [self.format_data_generation_params(params_batch[idx]) for idx in range(params_batch.shape[0])],
            axis=0,
        )

    def apply_measurement_noise(
        self,
        key,
        states: Float[Array, "n_steps state_dim"],
    ) -> Float[Array, "n_steps state_dim"]:
        """Optional measurement-noise hook for offline data generation."""
        del key
        return states

    def apply_measurement_noise_batch(
        self,
        keys,
        states: Float[Array, "batch n_steps state_dim"],
    ) -> Float[Array, "batch n_steps state_dim"]:
        """Vectorized measurement-noise hook for offline data generation."""
        return jnp.stack(
            [self.apply_measurement_noise(keys[idx], states[idx]) for idx in range(states.shape[0])],
            axis=0,
        )

    def is_valid_trajectory(
        self,
        states: Float[Array, "n_steps state_dim"],
    ) -> bool:
        """Optional trajectory validity hook for offline data generation."""
        return bool(jnp.all(jnp.isfinite(states)))

    def valid_trajectory_mask(
        self,
        states: Float[Array, "batch n_steps state_dim"],
    ) -> Float[Array, "batch"]:
        """Vectorized trajectory-validity hook for offline data generation."""
        return jnp.asarray(
            [self.is_valid_trajectory(states[idx]) for idx in range(states.shape[0])],
            dtype=bool,
        )
