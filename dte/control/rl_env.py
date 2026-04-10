"""Gymnasium-style environment wrapper for process-control experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from dte.evaluation.control_metrics import closed_loop_metrics
from dte.simulators.base import ProcessSimulator, ProcessUnitSpec

from .state_correction import StateCorrectionHook


@dataclass
class BoxSpace:
    """Minimal Box-like space to avoid a hard Gymnasium dependency."""

    low: np.ndarray
    high: np.ndarray
    shape: tuple[int, ...]
    dtype: np.dtype = np.float32

    def sample(self, seed: int | None = None) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return rng.uniform(self.low, self.high).astype(self.dtype)


@dataclass
class ProcessControlEnvConfig:
    """Configuration for the RL environment wrapper."""

    horizon: int = 50
    dt: float = 0.1
    terminate_on_violation: bool = False
    constraint_penalty: float = 10.0


def _state_space(spec: ProcessUnitSpec) -> BoxSpace:
    lower = np.asarray(spec.state_lower_bounds(), dtype=np.float32)
    upper = np.asarray(spec.state_upper_bounds(), dtype=np.float32)
    lower = np.where(np.isfinite(lower), lower, -1.0e6)
    upper = np.where(np.isfinite(upper), upper, 1.0e6)
    return BoxSpace(low=lower, high=upper, shape=(spec.state_dim,))


def _action_space(spec: ProcessUnitSpec) -> BoxSpace:
    lower = np.asarray([spec.control_ranges[name][0] for name in spec.control_names], dtype=np.float32)
    upper = np.asarray([spec.control_ranges[name][1] for name in spec.control_names], dtype=np.float32)
    return BoxSpace(low=lower, high=upper, shape=(spec.control_dim,))


class ProcessControlEnv:
    """A thin process-control environment with Gymnasium-style reset/step."""

    def __init__(
        self,
        spec: ProcessUnitSpec,
        simulator: ProcessSimulator,
        *,
        target_state: np.ndarray | None = None,
        params: np.ndarray | None = None,
        disturbance_schedule: np.ndarray | None = None,
        disturbance_fn: Callable[[int], np.ndarray] | None = None,
        config: ProcessControlEnvConfig | None = None,
        correction_hook: StateCorrectionHook | None = None,
    ):
        self.spec = spec
        self.simulator = simulator
        self.target_state = (
            np.asarray(target_state, dtype=np.float32)
            if target_state is not None
            else np.asarray(spec.default_initial_state, dtype=np.float32)
        )
        self.params = (
            np.asarray(params, dtype=np.float32)
            if params is not None
            else np.ones(spec.param_dim, dtype=np.float32)
        )
        self.disturbance_schedule = (
            np.asarray(disturbance_schedule, dtype=np.float32)
            if disturbance_schedule is not None
            else None
        )
        self.disturbance_fn = disturbance_fn
        self.config = config or ProcessControlEnvConfig()
        self.correction_hook = correction_hook

        self.observation_space = _state_space(spec)
        self.action_space = _action_space(spec)

        self._rng = np.random.default_rng(0)
        self._state = np.asarray(spec.default_initial_state, dtype=np.float32)
        self._step_index = 0
        self._previous_action = 0.5 * (self.action_space.low + self.action_space.high)

    def _disturbance_at(self, step_index: int) -> np.ndarray:
        if self.disturbance_schedule is not None and step_index < self.disturbance_schedule.shape[0]:
            return self.disturbance_schedule[step_index].astype(np.float32)
        if self.disturbance_fn is not None:
            return np.asarray(self.disturbance_fn(step_index), dtype=np.float32)
        return np.asarray(self.spec.default_nominal_disturbance, dtype=np.float32)

    def reset(
        self,
        *,
        seed: int | None = None,
        initial_state: np.ndarray | None = None,
    ) -> tuple[np.ndarray, dict]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._step_index = 0
        self._state = (
            np.asarray(initial_state, dtype=np.float32)
            if initial_state is not None
            else np.asarray(self.spec.default_initial_state, dtype=np.float32)
        )
        self._previous_action = 0.5 * (self.action_space.low + self.action_space.high)
        if self.correction_hook is not None:
            self.correction_hook.reset(self._state)
        info = {
            "disturbance": self._disturbance_at(0).copy(),
            "target_state": self.target_state.copy(),
        }
        return self._state.copy(), info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        action_arr = np.clip(
            np.asarray(action, dtype=np.float32),
            self.action_space.low,
            self.action_space.high,
        )
        disturbance = self._disturbance_at(self._step_index)
        derivative = np.asarray(
            self.simulator.dynamics(
                float(self._step_index * self.config.dt),
                self._state,
                action_arr,
                disturbance,
            ),
            dtype=np.float32,
        )
        next_state = self._state + float(self.config.dt) * derivative
        corrected_state = next_state.copy()
        if self.correction_hook is not None:
            correction = self.correction_hook.correct(
                prior_state=self._state,
                measurement=next_state,
                control=action_arr,
                params=self.params,
                timestamp=float((self._step_index + 1) * self.config.dt),
            )
            corrected_state = correction.corrected_state

        metrics = closed_loop_metrics(
            self.spec,
            corrected_state[None, :],
            action_arr[None, :],
            self.target_state,
            previous_control=self._previous_action,
            constraint_penalty=self.config.constraint_penalty,
        )
        reward = -float(metrics["total_cost"])
        constraint_violation = float(metrics["constraint_penalty"]) > 0.0
        self._state = corrected_state
        self._previous_action = action_arr
        self._step_index += 1

        terminated = bool(self.config.terminate_on_violation and constraint_violation)
        truncated = bool(self._step_index >= self.config.horizon)
        info = {
            "disturbance": disturbance.copy(),
            "metrics": metrics,
            "step_index": self._step_index,
        }
        return corrected_state.copy(), reward, terminated, truncated, info

    def rollout(
        self,
        policy: Callable[[np.ndarray, dict], np.ndarray],
        *,
        seed: int | None = None,
        initial_state: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        """Run one policy rollout for quick controller smoke tests."""

        observation, info = self.reset(seed=seed, initial_state=initial_state)
        observations = [observation]
        rewards: list[float] = []
        actions = []
        terminated = False
        truncated = False

        while not (terminated or truncated):
            action = np.asarray(policy(observation.copy(), info), dtype=np.float32)
            observation, reward, terminated, truncated, info = self.step(action)
            observations.append(observation)
            rewards.append(float(reward))
            actions.append(action)

        return {
            "observations": np.asarray(observations, dtype=np.float32),
            "actions": np.asarray(actions, dtype=np.float32),
            "rewards": np.asarray(rewards, dtype=np.float32),
        }
