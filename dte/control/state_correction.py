"""State correction hooks for control-facing online estimation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np

from .foundation_adapter import FoundationModel, encode_state, predict_one_step
from dte.simulators.base import ProcessUnitSpec


def _measurement_mask(state_dim: int, measurement_mask: Sequence[bool] | None) -> np.ndarray:
    if measurement_mask is None:
        return np.ones(state_dim, dtype=bool)
    mask = np.asarray(measurement_mask, dtype=bool)
    if mask.shape != (state_dim,):
        raise ValueError(f"measurement_mask must have shape ({state_dim},), got {mask.shape}.")
    return mask


def _state_bounds(spec: ProcessUnitSpec) -> tuple[np.ndarray, np.ndarray]:
    lower = np.asarray(spec.state_lower_bounds(), dtype=np.float32)
    upper = np.asarray(spec.state_upper_bounds(), dtype=np.float32)
    return lower, upper


def apply_measurement_assimilation(
    prior_state: np.ndarray,
    measurement: np.ndarray,
    *,
    gain: float,
    measurement_mask: Sequence[bool] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Blend a prior estimate with a new measurement."""

    prior = np.asarray(prior_state, dtype=np.float32)
    observed = np.asarray(measurement, dtype=np.float32)
    mask = _measurement_mask(prior.shape[0], measurement_mask)
    innovation = np.zeros_like(prior)
    innovation[mask] = observed[mask] - prior[mask]
    corrected = prior + float(gain) * innovation
    corrected[~mask] = prior[~mask]
    return corrected, innovation


def exponential_filter_update(
    previous_filtered: np.ndarray | None,
    measurement: np.ndarray,
    *,
    alpha: float,
    measurement_mask: Sequence[bool] | None = None,
) -> np.ndarray:
    """Update a filtered measurement stream with optional partial observations."""

    observed = np.asarray(measurement, dtype=np.float32)
    mask = _measurement_mask(observed.shape[0], measurement_mask)
    if previous_filtered is None:
        filtered = observed.copy()
    else:
        prev = np.asarray(previous_filtered, dtype=np.float32)
        filtered = prev.copy()
        filtered[mask] = (1.0 - float(alpha)) * prev[mask] + float(alpha) * observed[mask]
    filtered[~mask] = observed[~mask]
    return filtered


@dataclass
class StateCorrectionConfig:
    """Configuration for measurement assimilation and latent correction."""

    assimilation_gain: float = 0.65
    filter_alpha: float = 0.35
    clip_to_state_bounds: bool = True
    default_dt: float = 0.1


@dataclass
class CorrectionResult:
    """Return value for one measurement-correction update."""

    corrected_state: np.ndarray
    filtered_measurement: np.ndarray
    innovation: np.ndarray
    latent_mean: np.ndarray | None
    latent_logvar: np.ndarray | None
    timestamp: float | None


class StateCorrectionHook:
    """Maintain corrected physical and latent estimates from measurements."""

    def __init__(
        self,
        spec: ProcessUnitSpec,
        model: FoundationModel | None = None,
        *,
        config: StateCorrectionConfig | None = None,
    ):
        self.spec = spec
        self.model = model
        self.config = config or StateCorrectionConfig()
        self._filtered_measurement: np.ndarray | None = None
        self._state_estimate: np.ndarray | None = None
        self._latent_mean: np.ndarray | None = None
        self._latent_logvar: np.ndarray | None = None
        self._last_timestamp: float | None = None

    @property
    def state_estimate(self) -> np.ndarray | None:
        return None if self._state_estimate is None else self._state_estimate.copy()

    @property
    def latent_mean(self) -> np.ndarray | None:
        return None if self._latent_mean is None else self._latent_mean.copy()

    def reset(self, initial_state: np.ndarray | None = None) -> None:
        self._filtered_measurement = None
        self._state_estimate = (
            None if initial_state is None else np.asarray(initial_state, dtype=np.float32)
        )
        self._latent_mean = None
        self._latent_logvar = None
        self._last_timestamp = None

    def correct(
        self,
        *,
        prior_state: np.ndarray,
        measurement: np.ndarray,
        control: np.ndarray,
        params: np.ndarray | None = None,
        measurement_mask: Sequence[bool] | None = None,
        timestamp: float | None = None,
        seed: int = 0,
    ) -> CorrectionResult:
        """Assimilate one measurement and refresh the latent estimate."""

        filtered = exponential_filter_update(
            self._filtered_measurement,
            measurement,
            alpha=self.config.filter_alpha,
            measurement_mask=measurement_mask,
        )
        corrected, innovation = apply_measurement_assimilation(
            prior_state,
            filtered,
            gain=self.config.assimilation_gain,
            measurement_mask=measurement_mask,
        )
        if self.config.clip_to_state_bounds:
            lower, upper = _state_bounds(self.spec)
            corrected = np.clip(corrected, lower, upper)

        latent_mean = None
        latent_logvar = None
        if self.model is not None:
            params_arr = (
                np.asarray(params, dtype=np.float32)
                if params is not None
                else np.ones(self.spec.param_dim, dtype=np.float32)
            )
            _, z_mean, z_logvar = encode_state(
                self.model,
                self.spec,
                corrected,
                params_arr,
                np.asarray(control, dtype=np.float32),
                seed=seed,
            )
            latent_mean = z_mean
            latent_logvar = z_logvar

        self._filtered_measurement = filtered
        self._state_estimate = corrected
        self._latent_mean = latent_mean
        self._latent_logvar = latent_logvar
        self._last_timestamp = timestamp

        return CorrectionResult(
            corrected_state=corrected.copy(),
            filtered_measurement=filtered.copy(),
            innovation=innovation.copy(),
            latent_mean=None if latent_mean is None else latent_mean.copy(),
            latent_logvar=None if latent_logvar is None else latent_logvar.copy(),
            timestamp=timestamp,
        )

    def predict(
        self,
        *,
        control: np.ndarray,
        disturbance: np.ndarray | None = None,
        params: np.ndarray | None = None,
        dt: float | None = None,
    ) -> np.ndarray | None:
        """Project the current estimate one step forward with the latent drift."""

        if self.model is None or self._state_estimate is None:
            return self.state_estimate

        params_arr = (
            np.asarray(params, dtype=np.float32)
            if params is not None
            else np.ones(self.spec.param_dim, dtype=np.float32)
        )
        control_arr = np.asarray(control, dtype=np.float32)
        disturbance_arr = (
            np.asarray(disturbance, dtype=np.float32)
            if disturbance is not None
            else np.asarray(self.spec.default_nominal_disturbance, dtype=np.float32)
        )
        dt_value = float(dt if dt is not None else self.config.default_dt)

        state_arr, latent_next = predict_one_step(
            self.model,
            self.spec,
            self._latent_mean,
            self._state_estimate,
            control_arr,
            disturbance_arr,
            params_arr,
            dt=dt_value,
            seed=0,
        )
        if self.config.clip_to_state_bounds:
            lower, upper = _state_bounds(self.spec)
            state_arr = np.clip(state_arr, lower, upper)
        self._state_estimate = state_arr
        self._latent_mean = np.asarray(latent_next, dtype=np.float32)
        return state_arr.copy()
