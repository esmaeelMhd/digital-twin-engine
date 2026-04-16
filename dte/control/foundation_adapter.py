"""Shared control/runtime adapters for unit and universal foundation models."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from dte.models.unit.digital_twin import DigitalTwin
from dte.models.universal.digital_twin import UniversalDigitalTwin
from dte.simulators.base import ProcessUnitSpec


FoundationModel = DigitalTwin | UniversalDigitalTwin


@dataclass(frozen=True)
class UniversalControlContext:
    """Precomputed universal-model tensors for one concrete process spec."""

    system_id: int
    system_id_arr: jax.Array
    state_mask: jax.Array
    control_mask: jax.Array
    disturbance_mask: jax.Array
    param_mask: jax.Array


def is_universal_model(model: FoundationModel | None) -> bool:
    """Return whether a runtime model is the shared universal backbone."""

    return isinstance(model, UniversalDigitalTwin)


def resolve_universal_context(
    model: UniversalDigitalTwin,
    spec: ProcessUnitSpec,
) -> UniversalControlContext:
    """Resolve the universal system row that corresponds to a process spec."""

    try:
        system_id = model.system_names.index(spec.name)
    except ValueError as exc:
        raise ValueError(
            f"System '{spec.name}' is not available in the universal checkpoint."
        ) from exc
    system_id_arr = jnp.asarray(system_id, dtype=jnp.int32)
    return UniversalControlContext(
        system_id=system_id,
        system_id_arr=system_id_arr,
        state_mask=model.state_mask_table[system_id_arr],
        control_mask=model.control_mask_table[system_id_arr],
        disturbance_mask=model.disturbance_mask_table[system_id_arr],
        param_mask=model.param_mask_table[system_id_arr],
    )


def _pad_vector(values: np.ndarray, max_dim: int, active_dim: int) -> np.ndarray:
    padded = np.zeros(max_dim, dtype=np.float32)
    active = np.asarray(values, dtype=np.float32).reshape(-1)
    padded[: min(active.shape[0], active_dim)] = active[:active_dim]
    return padded


def _pad_sequence(values: np.ndarray, max_dim: int, active_dim: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    padded = np.zeros((arr.shape[0], max_dim), dtype=np.float32)
    padded[:, :active_dim] = arr[:, :active_dim]
    return padded


def encode_state(
    model: FoundationModel,
    spec: ProcessUnitSpec,
    state: np.ndarray,
    params: np.ndarray,
    control: np.ndarray,
    *,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Encode one physical state for either model family."""

    if isinstance(model, DigitalTwin):
        z, z_mean, z_logvar = model.encode(
            jnp.asarray(state, dtype=jnp.float32),
            jnp.asarray(params, dtype=jnp.float32),
            jnp.asarray(control, dtype=jnp.float32),
            jax.random.PRNGKey(seed),
        )
        return (
            np.asarray(z, dtype=np.float32),
            np.asarray(z_mean, dtype=np.float32),
            np.asarray(z_logvar, dtype=np.float32),
        )

    ctx = resolve_universal_context(model, spec)
    padded_state = _pad_vector(state, model.max_state_dim, spec.state_dim)
    padded_control = _pad_vector(control, model.max_control_dim, spec.control_dim)
    padded_params = _pad_vector(params, model.max_param_dim, spec.param_dim)
    state_norm = model.normalize_states(jnp.asarray(padded_state), ctx.system_id_arr) * ctx.state_mask
    control_norm = (
        model.normalize_controls(jnp.asarray(padded_control), ctx.system_id_arr) * ctx.control_mask
    )
    params_scaled = model.scale_params(jnp.asarray(padded_params), ctx.system_id_arr) * ctx.param_mask
    z, z_mean, z_logvar = model.encode(
        state_norm,
        params_scaled,
        control_norm,
        ctx.state_mask,
        ctx.control_mask,
        ctx.param_mask,
        ctx.system_id_arr,
        jax.random.PRNGKey(seed),
    )
    return (
        np.asarray(z, dtype=np.float32),
        np.asarray(z_mean, dtype=np.float32),
        np.asarray(z_logvar, dtype=np.float32),
    )


def rollout_model_ensemble(
    model: FoundationModel,
    spec: ProcessUnitSpec,
    initial_state: np.ndarray,
    controls: np.ndarray,
    disturbances: np.ndarray,
    params: np.ndarray,
    *,
    dt: float,
    n_samples: int,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Roll out one model-backed candidate and return mean/std in physical units."""

    controls_arr = np.asarray(controls, dtype=np.float32)
    disturbances_arr = np.asarray(disturbances, dtype=np.float32)
    sample_count = max(int(n_samples), 1)

    if isinstance(model, DigitalTwin):
        ts = jnp.linspace(
            0.0,
            max(controls_arr.shape[0] - 1, 0) * float(dt),
            controls_arr.shape[0],
            dtype=jnp.float32,
        )
        if sample_count > 1:
            ensemble = model.predict_ensemble(
                jnp.asarray(initial_state, dtype=jnp.float32),
                jnp.asarray(controls_arr, dtype=jnp.float32),
                jnp.asarray(disturbances_arr, dtype=jnp.float32),
                jnp.asarray(params, dtype=jnp.float32),
                ts,
                jax.random.PRNGKey(seed),
                n_samples=sample_count,
            )
            return (
                np.asarray(ensemble["states_mean"], dtype=np.float32),
                np.asarray(ensemble["states_std"], dtype=np.float32),
            )

        _, z_mean, _ = model.encode(
            jnp.asarray(initial_state, dtype=jnp.float32),
            jnp.asarray(params, dtype=jnp.float32),
            jnp.asarray(controls_arr[0], dtype=jnp.float32),
            jax.random.PRNGKey(seed),
        )
        z_traj = model.rollout_latent(
            ts,
            z_mean,
            jnp.asarray(controls_arr, dtype=jnp.float32),
            jnp.asarray(params, dtype=jnp.float32),
            disturbances=jnp.asarray(disturbances_arr, dtype=jnp.float32),
            stochastic=False,
        )
        decode_fn = jax.vmap(
            lambda z_t, control_t: model.decode(
                z_t,
                jnp.asarray(params, dtype=jnp.float32),
                control_t,
            ),
            in_axes=(0, 0),
        )
        states = np.asarray(
            decode_fn(z_traj, jnp.asarray(controls_arr, dtype=jnp.float32)),
            dtype=np.float32,
        )
        return states, np.zeros_like(states)

    ctx = resolve_universal_context(model, spec)
    n_steps = int(controls_arr.shape[0])
    ts = jnp.linspace(0.0, max(n_steps - 1, 0) * float(dt), n_steps, dtype=jnp.float32)
    padded_state = _pad_vector(initial_state, model.max_state_dim, spec.state_dim)
    padded_controls = _pad_sequence(controls_arr, model.max_control_dim, spec.control_dim)
    padded_disturbances = _pad_sequence(
        disturbances_arr,
        model.max_disturbance_dim,
        spec.disturbance_dim,
    )
    padded_params = _pad_vector(params, model.max_param_dim, spec.param_dim)
    state_norm = model.normalize_states(jnp.asarray(padded_state), ctx.system_id_arr) * ctx.state_mask
    controls_norm = (
        model.normalize_controls(jnp.asarray(padded_controls), ctx.system_id_arr) * ctx.control_mask
    )
    disturbances_norm = (
        model.normalize_disturbances(
            jnp.asarray(padded_disturbances),
            ctx.system_id_arr,
        )
        * ctx.disturbance_mask
    )
    params_scaled = model.scale_params(jnp.asarray(padded_params), ctx.system_id_arr) * ctx.param_mask

    if sample_count == 1:
        sample_keys = [None]
    else:
        sample_keys = list(jax.random.split(jax.random.PRNGKey(seed), sample_count))

    trajectories = []
    for sample_key in sample_keys:
        z0, z_mean, _ = model.encode(
            state_norm,
            params_scaled,
            controls_norm[0],
            ctx.state_mask,
            ctx.control_mask,
            ctx.param_mask,
            ctx.system_id_arr,
            sample_key,
        )
        z_init = z_mean if sample_key is None else z0
        z_traj = model.rollout_latent(
            ts,
            z_init,
            controls_norm,
            disturbances_norm,
            params_scaled,
            ctx.control_mask,
            ctx.disturbance_mask,
            ctx.param_mask,
            ctx.system_id_arr,
        )
        pred_norm = jax.vmap(
            lambda z_t, control_t: model.decode(
                z_t,
                params_scaled,
                control_t,
                ctx.state_mask,
                ctx.control_mask,
                ctx.param_mask,
                ctx.system_id_arr,
            )
        )(z_traj, controls_norm)
        pred_states = model.denormalize_states(pred_norm, ctx.system_id_arr)
        trajectories.append(
            np.asarray(pred_states[:, : spec.state_dim], dtype=np.float32)
        )

    samples = np.asarray(trajectories, dtype=np.float32)
    return np.mean(samples, axis=0), np.std(samples, axis=0)


def predict_one_step(
    model: FoundationModel,
    spec: ProcessUnitSpec,
    latent_mean: np.ndarray | None,
    state_estimate: np.ndarray,
    control: np.ndarray,
    disturbance: np.ndarray,
    params: np.ndarray,
    *,
    dt: float,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Project one physical step forward and return the next state and latent mean."""

    if isinstance(model, DigitalTwin):
        if latent_mean is None:
            _, latent_mean_arr, _ = model.encode(
                jnp.asarray(state_estimate, dtype=jnp.float32),
                jnp.asarray(params, dtype=jnp.float32),
                jnp.asarray(control, dtype=jnp.float32),
                jax.random.PRNGKey(seed),
            )
        else:
            latent_mean_arr = jnp.asarray(latent_mean, dtype=jnp.float32)
        drift = model.latent_sde.drift(
            latent_mean_arr,
            jnp.asarray(control, dtype=jnp.float32),
            jnp.asarray(disturbance, dtype=jnp.float32),
            jnp.asarray(params, dtype=jnp.float32),
        )
        z_next = latent_mean_arr + float(dt) * drift
        next_state = model.decode(
            z_next,
            jnp.asarray(params, dtype=jnp.float32),
            jnp.asarray(control, dtype=jnp.float32),
        )
        return (
            np.asarray(next_state, dtype=np.float32),
            np.asarray(z_next, dtype=np.float32),
        )

    ctx = resolve_universal_context(model, spec)
    if latent_mean is None:
        _, latent_mean_arr, _ = encode_state(
            model,
            spec,
            state_estimate,
            params,
            control,
            seed=seed,
        )
    else:
        latent_mean_arr = np.asarray(latent_mean, dtype=np.float32)

    padded_control = _pad_sequence(
        np.tile(np.asarray(control, dtype=np.float32)[None, :], (2, 1)),
        model.max_control_dim,
        spec.control_dim,
    )
    padded_disturbance = _pad_sequence(
        np.tile(np.asarray(disturbance, dtype=np.float32)[None, :], (2, 1)),
        model.max_disturbance_dim,
        spec.disturbance_dim,
    )
    padded_params = _pad_vector(params, model.max_param_dim, spec.param_dim)
    params_scaled = model.scale_params(jnp.asarray(padded_params), ctx.system_id_arr) * ctx.param_mask
    controls_norm = (
        model.normalize_controls(jnp.asarray(padded_control), ctx.system_id_arr) * ctx.control_mask
    )
    disturbances_norm = (
        model.normalize_disturbances(jnp.asarray(padded_disturbance), ctx.system_id_arr)
        * ctx.disturbance_mask
    )
    ts = jnp.asarray([0.0, float(dt)], dtype=jnp.float32)
    z_traj = model.rollout_latent(
        ts,
        jnp.asarray(latent_mean_arr, dtype=jnp.float32),
        controls_norm,
        disturbances_norm,
        params_scaled,
        ctx.control_mask,
        ctx.disturbance_mask,
        ctx.param_mask,
        ctx.system_id_arr,
    )
    z_next = z_traj[-1]
    pred_norm = model.decode(
        z_next,
        params_scaled,
        controls_norm[-1],
        ctx.state_mask,
        ctx.control_mask,
        ctx.param_mask,
        ctx.system_id_arr,
    )
    pred_state = model.denormalize_states(pred_norm, ctx.system_id_arr)
    return (
        np.asarray(pred_state[: spec.state_dim], dtype=np.float32),
        np.asarray(z_next, dtype=np.float32),
    )
