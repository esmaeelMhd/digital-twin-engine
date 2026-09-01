"""Generic control-facing rollout and optimisation interface."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from .foundation_adapter import FoundationModel, rollout_model_ensemble
from dte.evaluation.control_metrics import closed_loop_metrics, summarize_constraint_violations
from dte.simulators.base import ProcessSimulator, ProcessUnitSpec

from .state_correction import StateCorrectionHook


CostHook = Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], float]
ConstraintHook = Callable[[ProcessUnitSpec, np.ndarray, np.ndarray], float | dict[str, float]]


@dataclass
class MPCInterfaceConfig:
    """Configuration for the generic MPC-facing runtime."""

    dt: float = 0.1
    horizon: int = 20
    constraint_penalty: float = 10.0
    rollout_samples: int = 8


def _clip_controls(spec: ProcessUnitSpec, controls: np.ndarray) -> np.ndarray:
    lower = np.asarray([spec.control_ranges[name][0] for name in spec.control_names], dtype=np.float32)
    upper = np.asarray([spec.control_ranges[name][1] for name in spec.control_names], dtype=np.float32)
    return np.clip(np.asarray(controls, dtype=np.float32), lower[None, :], upper[None, :])


def _default_disturbances(spec: ProcessUnitSpec, horizon: int) -> np.ndarray:
    base = np.asarray(spec.default_nominal_disturbance, dtype=np.float32)
    return np.tile(base[None, :], (horizon, 1))


def _simulate_open_loop(
    spec: ProcessUnitSpec,
    simulator: ProcessSimulator,
    initial_state: np.ndarray,
    controls: np.ndarray,
    disturbances: np.ndarray,
    dt: float,
) -> np.ndarray:
    n_steps = controls.shape[0]
    states = np.zeros((n_steps, spec.state_dim), dtype=np.float32)
    states[0] = np.asarray(initial_state, dtype=np.float32)
    for step in range(1, n_steps):
        deriv = simulator.dynamics(
            float((step - 1) * dt),
            states[step - 1],
            controls[step - 1],
            disturbances[step - 1],
        )
        states[step] = states[step - 1] + float(dt) * np.asarray(deriv, dtype=np.float32)
    return states


def _midpoint_control(spec: ProcessUnitSpec) -> np.ndarray:
    return np.asarray(
        [0.5 * sum(spec.control_ranges[name]) for name in spec.control_names],
        dtype=np.float32,
    )


class ProcessMPCInterface:
    """Expose rollout, evaluation, and optimisation hooks for control algorithms."""

    def __init__(
        self,
        spec: ProcessUnitSpec,
        simulator: ProcessSimulator,
        model: FoundationModel | None = None,
        *,
        params: np.ndarray | None = None,
        config: MPCInterfaceConfig | None = None,
        state_correction: StateCorrectionHook | None = None,
    ):
        self.spec = spec
        self.simulator = simulator
        self.model = model
        self.config = config or MPCInterfaceConfig()
        self.params = (
            np.asarray(params, dtype=np.float32)
            if params is not None
            else np.ones(spec.param_dim, dtype=np.float32)
        )
        self.state_correction = state_correction
        self._state_estimate = np.asarray(spec.default_initial_state, dtype=np.float32)
        self._previous_control = _midpoint_control(spec)

    def reset(
        self,
        *,
        initial_state: np.ndarray | None = None,
        params: np.ndarray | None = None,
    ) -> np.ndarray:
        if initial_state is not None:
            self._state_estimate = np.asarray(initial_state, dtype=np.float32)
        else:
            self._state_estimate = np.asarray(self.spec.default_initial_state, dtype=np.float32)
        if params is not None:
            self.params = np.asarray(params, dtype=np.float32)
        self._previous_control = _midpoint_control(self.spec)
        if self.state_correction is not None:
            self.state_correction.reset(self._state_estimate)
        return self._state_estimate.copy()

    def current_state_estimate(self) -> np.ndarray:
        if self.state_correction is not None and self.state_correction.state_estimate is not None:
            self._state_estimate = self.state_correction.state_estimate
        return self._state_estimate.copy()

    def assimilate_measurement(
        self,
        measurement: np.ndarray,
        *,
        control: np.ndarray | None = None,
        measurement_mask: Sequence[bool] | None = None,
        timestamp: float | None = None,
        seed: int = 0,
    ) -> dict[str, np.ndarray | float | None]:
        """Update the current estimate from an observation."""

        control_arr = (
            np.asarray(control, dtype=np.float32)
            if control is not None
            else self._previous_control.copy()
        )
        if control is not None:
            self._previous_control = control_arr.copy()
        if self.state_correction is None:
            self._state_estimate = np.asarray(measurement, dtype=np.float32)
            return {
                "corrected_state": self._state_estimate.copy(),
                "latent_mean": None,
                "latent_logvar": None,
                "timestamp": timestamp,
            }

        result = self.state_correction.correct(
            prior_state=self.current_state_estimate(),
            measurement=np.asarray(measurement, dtype=np.float32),
            control=control_arr,
            params=self.params,
            measurement_mask=measurement_mask,
            timestamp=timestamp,
            seed=seed,
        )
        self._state_estimate = result.corrected_state.copy()
        return {
            "corrected_state": result.corrected_state.copy(),
            "latent_mean": None if result.latent_mean is None else result.latent_mean.copy(),
            "latent_logvar": None if result.latent_logvar is None else result.latent_logvar.copy(),
            "timestamp": result.timestamp,
        }

    def rollout_candidate(
        self,
        controls: np.ndarray,
        *,
        disturbances: np.ndarray | None = None,
        initial_state: np.ndarray | None = None,
        use_model: bool | None = None,
        n_samples: int | None = None,
        seed: int = 0,
    ) -> dict[str, np.ndarray | str]:
        """Roll out one candidate control sequence under the current runtime."""

        controls_arr = _clip_controls(self.spec, np.asarray(controls, dtype=np.float32))
        horizon = controls_arr.shape[0]
        disturbance_arr = (
            np.asarray(disturbances, dtype=np.float32)
            if disturbances is not None
            else _default_disturbances(self.spec, horizon)
        )
        initial = (
            np.asarray(initial_state, dtype=np.float32)
            if initial_state is not None
            else self.current_state_estimate()
        )
        do_model_rollout = bool(self.model is not None) if use_model is None else bool(use_model)

        if do_model_rollout and self.model is not None:
            states, std = rollout_model_ensemble(
                self.model,
                self.spec,
                initial,
                controls_arr,
                disturbance_arr,
                self.params,
                dt=self.config.dt,
                n_samples=max(int(n_samples or self.config.rollout_samples), 1),
                seed=seed,
            )
            return {
                "source": "model",
                "states": states,
                "std": std,
                "controls": controls_arr,
                "disturbances": disturbance_arr,
            }

        states = _simulate_open_loop(
            self.spec,
            self.simulator,
            initial,
            controls_arr,
            disturbance_arr,
            self.config.dt,
        )
        return {
            "source": "simulator",
            "states": states,
            "std": np.zeros_like(states),
            "controls": controls_arr,
            "disturbances": disturbance_arr,
        }

    def evaluate_candidate(
        self,
        controls: np.ndarray,
        *,
        disturbances: np.ndarray | None = None,
        target_state: np.ndarray | None = None,
        cost_hook: CostHook | None = None,
        constraint_hook: ConstraintHook | None = None,
        state_weights: Sequence[float] | None = None,
        control_weights: Sequence[float] | None = None,
        use_model: bool | None = None,
        seed: int = 0,
    ) -> dict[str, np.ndarray | float | str | dict[str, float]]:
        """Evaluate a candidate sequence with optional custom cost/constraint hooks."""

        rollout = self.rollout_candidate(
            controls,
            disturbances=disturbances,
            use_model=use_model,
            seed=seed,
        )
        states = np.asarray(rollout["states"], dtype=np.float32)
        controls_arr = np.asarray(rollout["controls"], dtype=np.float32)
        disturbance_arr = np.asarray(rollout["disturbances"], dtype=np.float32)
        target = (
            np.asarray(target_state, dtype=np.float32)
            if target_state is not None
            else np.asarray(self.spec.default_initial_state, dtype=np.float32)
        )

        metrics = closed_loop_metrics(
            self.spec,
            states,
            controls_arr,
            target,
            state_weights=state_weights,
            control_weights=control_weights,
            previous_control=self._previous_control,
            constraint_penalty=self.config.constraint_penalty,
        )
        objective = float(metrics["total_cost"])
        if cost_hook is not None:
            objective = float(cost_hook(states, controls_arr, target, disturbance_arr))

        extra_constraints: dict[str, float] = {}
        if constraint_hook is not None:
            raw_constraints = constraint_hook(self.spec, states, controls_arr)
            if isinstance(raw_constraints, dict):
                extra_constraints = {str(key): float(value) for key, value in raw_constraints.items()}
                objective += float(extra_constraints.get("penalty", 0.0))
            else:
                objective += float(raw_constraints)
                extra_constraints = {"penalty": float(raw_constraints)}

        return {
            "objective": objective,
            "source": str(rollout["source"]),
            "states": states,
            "controls": controls_arr,
            "disturbances": disturbance_arr,
            "metrics": metrics,
            "constraints": {
                **summarize_constraint_violations(self.spec, states, controls_arr),
                **extra_constraints,
            },
        }

    def optimize_random_shooting(
        self,
        *,
        target_state: np.ndarray,
        disturbances: np.ndarray | None = None,
        horizon: int | None = None,
        n_candidates: int = 32,
        cost_hook: CostHook | None = None,
        constraint_hook: ConstraintHook | None = None,
        state_weights: Sequence[float] | None = None,
        control_weights: Sequence[float] | None = None,
        use_model: bool | None = None,
        seed: int = 0,
    ) -> dict[str, np.ndarray | float | str | dict[str, float]]:
        """Optimise a candidate control sequence with a generic random-shooting search."""

        horizon_len = int(horizon if horizon is not None else self.config.horizon)
        lower = np.asarray([self.spec.control_ranges[name][0] for name in self.spec.control_names], dtype=np.float32)
        upper = np.asarray([self.spec.control_ranges[name][1] for name in self.spec.control_names], dtype=np.float32)
        disturbance_arr = (
            np.asarray(disturbances, dtype=np.float32)
            if disturbances is not None
            else _default_disturbances(self.spec, horizon_len)
        )
        rng = np.random.default_rng(seed)
        best_result: dict[str, np.ndarray | float | str | dict[str, float]] | None = None
        best_objective = math.inf

        for candidate_index in range(max(int(n_candidates), 1)):
            start = rng.uniform(lower, upper).astype(np.float32)
            end = rng.uniform(lower, upper).astype(np.float32)
            controls = np.stack(
                [
                    np.linspace(start[idx], end[idx], horizon_len, dtype=np.float32)
                    for idx in range(self.spec.control_dim)
                ],
                axis=-1,
            )
            result = self.evaluate_candidate(
                controls,
                disturbances=disturbance_arr,
                target_state=target_state,
                cost_hook=cost_hook,
                constraint_hook=constraint_hook,
                state_weights=state_weights,
                control_weights=control_weights,
                use_model=use_model,
                seed=seed + candidate_index,
            )
            objective = float(result["objective"])
            if objective < best_objective:
                best_objective = objective
                best_result = result

        if best_result is None:
            raise RuntimeError("Random-shooting optimisation did not evaluate any candidates.")
        return best_result
