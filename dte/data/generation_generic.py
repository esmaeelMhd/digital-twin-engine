"""Generic data generation pipeline for any ProcessSimulator.

Provides a drop-in replacement for the CSTR-specific :class:`DataGenerator`
that works with any :class:`~dte.simulators.base.ProcessSimulator` /
:class:`~dte.simulators.base.SystemSpec` combination.

The excitation signal generation, steady-state initialization, and rollout
hooks are all config- and simulator-driven so every registered system can use
the same fast dataset path.
"""

from __future__ import annotations

import os
import time
from functools import partial
from typing import Any, Dict, Optional, Tuple

import h5py
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, PRNGKeyArray
from tqdm import tqdm

from dte.simulators.base import ProcessSimulator, SystemSpec


def _generate_prbs_signal_core(
    key: PRNGKeyArray,
    n_steps: int,
    min_val: float,
    max_val: float,
    switch_prob: float,
) -> Float[Array, "n_steps 1"]:
    """Pure JAX PRBS signal core used by single and batched helpers."""
    keys = jax.random.split(key, n_steps)
    switches = jax.vmap(lambda k: jax.random.bernoulli(k, switch_prob))(keys)
    state = jnp.cumsum(switches) % 2
    signal = jnp.where(state == 0, min_val, max_val)
    return signal[:, None]


def _generate_chirp_signal_core(
    n_steps: int,
    dt: float,
    min_val: float,
    max_val: float,
    min_freq: float,
    max_freq: float,
) -> Float[Array, "n_steps 1"]:
    """Pure JAX chirp signal core used by single and batched helpers."""
    t = jnp.arange(n_steps) * dt
    total_time = jnp.maximum(t[-1], 1.0)
    freq = min_freq + (max_freq - min_freq) * t / total_time
    phase = jnp.cumsum(freq) * dt * 2.0 * jnp.pi
    chirp = jnp.sin(phase)
    signal = min_val + (max_val - min_val) * (chirp + 1.0) / 2.0
    return signal[:, None]


def _generate_multistep_signal_core(
    key: PRNGKeyArray,
    n_steps: int,
    min_val: float,
    max_val: float,
    n_changes: int,
) -> Float[Array, "n_steps 1"]:
    """Pure JAX multistep signal core used by single and batched helpers."""
    key_times, key_values = jax.random.split(key)
    change_times = jnp.sort(
        jax.random.randint(key_times, shape=(n_changes,), minval=0, maxval=n_steps)
    )
    values = jax.random.uniform(
        key_values, shape=(n_changes + 1,), minval=min_val, maxval=max_val
    )
    segment_ids = jnp.searchsorted(change_times, jnp.arange(n_steps), side="right")
    return values[segment_ids][:, None]


@partial(jax.jit, static_argnames=("n_steps", "n_changes", "mixed"))
def _generate_signal_batch_jit(
    keys: PRNGKeyArray,
    n_steps: int,
    dt: float,
    min_val: float,
    max_val: float,
    signal_type_index: int,
    mixed: bool,
    switch_prob: float,
    n_changes: int,
    min_freq: float,
    max_freq: float,
) -> Float[Array, "batch n_steps 1"]:
    """Compiled batched signal generator shared by all systems."""
    if mixed:
        split_keys = jax.vmap(lambda key: jax.random.split(key, 2))(keys)
        choice_keys = split_keys[:, 0]
        signal_keys = split_keys[:, 1]
        signal_indices = jax.vmap(
            lambda choice_key: jax.random.randint(choice_key, shape=(), minval=0, maxval=3)
        )(choice_keys)
    else:
        signal_keys = keys
        signal_indices = jnp.full((keys.shape[0],), signal_type_index, dtype=jnp.int32)

    def generate_one(signal_key, index):
        return jax.lax.switch(
            index,
            (
                lambda k: _generate_prbs_signal_core(k, n_steps, min_val, max_val, switch_prob),
                lambda k: _generate_chirp_signal_core(
                    n_steps, dt, min_val, max_val, min_freq, max_freq
                ),
                lambda k: _generate_multistep_signal_core(
                    k, n_steps, min_val, max_val, n_changes
                ),
            ),
            signal_key,
        )

    return jax.vmap(generate_one)(signal_keys, signal_indices)


class GenericDataGenerator:
    """Data generator that works with any :class:`ProcessSimulator`.

    Generates trajectories by:
    1. Sampling random PRBS / multistep excitation signals for each control
       and disturbance channel.
    2. Finding the initial steady state for each trajectory's nominal inputs.
    3. Simulating the ODE forward from that steady state.
    4. Saving everything to an HDF5 file in the same format expected by
       :class:`~dte.data.dataset.TrajectoryDataset`.
    """

    def __init__(
        self,
        simulator: ProcessSimulator,
        config: dict,
        system_spec: Optional[SystemSpec] = None,
    ):
        self.simulator = simulator
        self.config = config
        self.spec = system_spec or simulator.spec
        self.last_profile: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    # Signal generation helpers
    # ------------------------------------------------------------------

    def _default_signal_policy(self, kind: str) -> Dict[str, Any]:
        """Return the default excitation policy for one channel kind."""
        if kind == "control":
            return {
                "type": "prbs",
                "switch_prob": 0.05,
                "n_changes": 5,
                "min_freq": 0.01,
                "max_freq": 0.1,
            }
        return {
            "type": "multistep",
            "switch_prob": 0.05,
            "n_changes": 5,
            "min_freq": 0.01,
            "max_freq": 0.1,
        }

    def _get_signal_policy(self, kind: str, channel_name: str) -> Dict[str, Any]:
        """Resolve a per-channel excitation policy from config."""
        data_generation_cfg = self.config.get("data_generation", {})
        section_name = "control_signals" if kind == "control" else "disturbance_signals"
        raw_policy = data_generation_cfg.get(section_name, {}).get(channel_name, {})
        policy = dict(self._default_signal_policy(kind))
        if isinstance(raw_policy, dict):
            policy.update(raw_policy)
        policy["type"] = str(policy.get("type", "prbs")).strip().lower()
        policy["switch_prob"] = float(policy.get("switch_prob", 0.05))
        policy["n_changes"] = int(policy.get("n_changes", 5))
        policy["min_freq"] = float(policy.get("min_freq", 0.01))
        policy["max_freq"] = float(policy.get("max_freq", 0.1))
        return policy

    def _generate_signal(
        self,
        key: PRNGKeyArray,
        n_steps: int,
        dt: float,
        min_val: float,
        max_val: float,
        signal_type: str,
        switch_prob: float = 0.05,
        n_changes: int = 5,
        min_freq: float = 0.01,
        max_freq: float = 0.1,
    ) -> Float[Array, "n_steps 1"]:
        """Generate a single channel excitation signal."""
        signal_type_to_index = {
            "prbs": 0,
            "chirp": 1,
            "multistep": 2,
        }
        if signal_type == "mixed":
            choice_key, signal_key = jax.random.split(key)
            signal_index = jax.random.randint(choice_key, shape=(), minval=0, maxval=3)
        else:
            signal_key = key
            signal_index = jnp.asarray(signal_type_to_index[signal_type], dtype=jnp.int32)

        return jax.lax.switch(
            signal_index,
            (
                lambda k: _generate_prbs_signal_core(k, n_steps, min_val, max_val, switch_prob),
                lambda k: _generate_chirp_signal_core(
                    n_steps, dt, min_val, max_val, min_freq, max_freq
                ),
                lambda k: _generate_multistep_signal_core(
                    k, n_steps, min_val, max_val, n_changes
                ),
            ),
            signal_key,
        )

    def _generate_signal_batch(
        self,
        keys: PRNGKeyArray,
        n_steps: int,
        dt: float,
        min_val: float,
        max_val: float,
        signal_type: str,
        switch_prob: float = 0.05,
        n_changes: int = 5,
        min_freq: float = 0.01,
        max_freq: float = 0.1,
    ) -> Float[Array, "batch n_steps 1"]:
        """Generate one channel across a batch of trajectories."""
        signal_type_to_index = {
            "prbs": 0,
            "chirp": 1,
            "multistep": 2,
        }
        mixed = signal_type == "mixed"
        signal_type_index = signal_type_to_index.get(signal_type, 0)
        return _generate_signal_batch_jit(
            keys,
            n_steps,
            dt,
            min_val,
            max_val,
            signal_type_index,
            mixed,
            switch_prob,
            n_changes,
            min_freq,
            max_freq,
        )

    def _generate_control_trajectory(
        self,
        key: PRNGKeyArray,
        n_steps: int,
        dt: float,
    ) -> Float[Array, "n_steps control_dim"]:
        """Generate configured excitation signals for all control channels."""
        ops = self.config.get("operating_ranges", {})
        channels = []

        for name in self.spec.control_names:
            key, subkey = jax.random.split(key)
            lo, hi = ops.get(name, [0.0, 1.0])
            policy = self._get_signal_policy("control", name)
            channels.append(
                self._generate_signal(
                    subkey,
                    n_steps,
                    dt,
                    float(lo),
                    float(hi),
                    policy["type"],
                    switch_prob=policy["switch_prob"],
                    n_changes=policy["n_changes"],
                    min_freq=policy["min_freq"],
                    max_freq=policy["max_freq"],
                )
            )

        return jnp.concatenate(channels, axis=1)

    def _generate_disturbance_trajectory(
        self,
        key: PRNGKeyArray,
        n_steps: int,
        dt: float,
    ) -> Float[Array, "n_steps disturbance_dim"]:
        """Generate configured excitation signals for all disturbance channels."""
        ops = self.config.get("operating_ranges", {})
        channels = []

        for name in self.spec.disturbance_names:
            key, subkey = jax.random.split(key)
            lo, hi = ops.get(name, [0.0, 1.0])
            policy = self._get_signal_policy("disturbance", name)
            channels.append(
                self._generate_signal(
                    subkey,
                    n_steps,
                    dt,
                    float(lo),
                    float(hi),
                    policy["type"],
                    switch_prob=policy["switch_prob"],
                    n_changes=policy["n_changes"],
                    min_freq=policy["min_freq"],
                    max_freq=policy["max_freq"],
                )
            )

        return jnp.concatenate(channels, axis=1)

    def _generate_control_trajectories(
        self,
        keys: PRNGKeyArray,
        n_steps: int,
        dt: float,
    ) -> Float[Array, "batch n_steps control_dim"]:
        """Generate control trajectories for a batch of seeds."""
        ops = self.config.get("operating_ranges", {})
        channels = []
        channel_keys = keys
        for name in self.spec.control_names:
            split_keys = jax.vmap(lambda key: jax.random.split(key, 2))(channel_keys)
            channel_keys = split_keys[:, 0]
            subkeys = split_keys[:, 1]
            lo, hi = ops.get(name, [0.0, 1.0])
            policy = self._get_signal_policy("control", name)
            channels.append(
                self._generate_signal_batch(
                    subkeys,
                    n_steps,
                    dt,
                    float(lo),
                    float(hi),
                    policy["type"],
                    switch_prob=policy["switch_prob"],
                    n_changes=policy["n_changes"],
                    min_freq=policy["min_freq"],
                    max_freq=policy["max_freq"],
                )
            )
        return jnp.concatenate(channels, axis=2)

    def _generate_disturbance_trajectories(
        self,
        keys: PRNGKeyArray,
        n_steps: int,
        dt: float,
    ) -> Float[Array, "batch n_steps disturbance_dim"]:
        """Generate disturbance trajectories for a batch of seeds."""
        ops = self.config.get("operating_ranges", {})
        channels = []
        channel_keys = keys
        for name in self.spec.disturbance_names:
            split_keys = jax.vmap(lambda key: jax.random.split(key, 2))(channel_keys)
            channel_keys = split_keys[:, 0]
            subkeys = split_keys[:, 1]
            lo, hi = ops.get(name, [0.0, 1.0])
            policy = self._get_signal_policy("disturbance", name)
            channels.append(
                self._generate_signal_batch(
                    subkeys,
                    n_steps,
                    dt,
                    float(lo),
                    float(hi),
                    policy["type"],
                    switch_prob=policy["switch_prob"],
                    n_changes=policy["n_changes"],
                    min_freq=policy["min_freq"],
                    max_freq=policy["max_freq"],
                )
            )
        return jnp.concatenate(channels, axis=2)

    # ------------------------------------------------------------------
    # Params sampling
    # ------------------------------------------------------------------

    def _sample_params(self, key: PRNGKeyArray) -> Float[Array, "param_dim"]:
        """Sample one raw simulator parameter vector for offline generation."""
        return self.simulator.sample_data_generation_params(key)

    # ------------------------------------------------------------------
    # Single trajectory generation
    # ------------------------------------------------------------------

    def _generate_trajectory(
        self,
        key: PRNGKeyArray,
        n_steps: int,
        dt: float,
        simulation_mode: str,
    ) -> Optional[Dict[str, Array]]:
        """Generate one trajectory.  Returns None if simulation diverges."""
        key_ctrl, key_dist, key_ic, key_params = jax.random.split(key, 4)

        controls = self._generate_control_trajectory(key_ctrl, n_steps, dt)
        disturbances = self._generate_disturbance_trajectory(key_dist, n_steps, dt)
        params = self._sample_params(key_params)
        stored_params = self.simulator.format_data_generation_params(params)

        # Initial state: use the simulator's data-generation steady-state hook
        # so systems can swap in cheaper closed-form or approximate solvers.
        mean_control = jnp.mean(controls, axis=0)
        mean_disturbance = jnp.mean(disturbances, axis=0)
        try:
            initial_state = self.simulator.steady_state_for_data_generation_with_params(
                mean_control,
                mean_disturbance,
                params,
            )
        except Exception:
            # Fall back to spec default
            initial_state = self.spec.default_initial_state_array()

        if not bool(jnp.all(jnp.isfinite(initial_state))):
            return None

        t_end = n_steps * dt
        ts = jnp.linspace(0.0, t_end, n_steps)

        if simulation_mode == "dataset":
            result = self.simulator.simulate_for_data_generation_with_params(
                initial_state,
                controls,
                disturbances,
                params,
                (0.0, t_end),
                dt=dt,
                n_steps=n_steps,
            )
        elif simulation_mode == "reference":
            result = self.simulator.simulate(
                initial_state,
                controls,
                disturbances,
                (0.0, t_end),
                dt=dt,
                n_steps=n_steps,
            )
        else:
            raise ValueError(f"Unknown simulation_mode: {simulation_mode}")

        noise_start = time.perf_counter()
        states = self.simulator.apply_measurement_noise(key_ic, result["states"])
        noise_seconds = time.perf_counter() - noise_start

        # Reject divergent trajectories
        validation_start = time.perf_counter()
        is_valid = self.simulator.is_valid_trajectory(states)
        validation_seconds = time.perf_counter() - validation_start
        if not is_valid:
            return None

        return {
            "states": states,
            "controls": controls,
            "disturbances": disturbances,
            "params": stored_params,
            "t": ts,
            "measurement_noise_seconds": noise_seconds,
            "validation_seconds": validation_seconds,
        }

    # ------------------------------------------------------------------
    # Dataset generation
    # ------------------------------------------------------------------

    def _generate_trajectories_batched(
        self,
        keys: PRNGKeyArray,
        n_steps: int,
        dt: float,
        simulation_mode: str,
    ) -> Dict[str, Array]:
        """Generate a batch of trajectories using simulator batch hooks."""
        split_keys = jax.vmap(lambda k: jax.random.split(k, 4))(keys)
        key_ctrl = split_keys[:, 0]
        key_dist = split_keys[:, 1]
        key_ic = split_keys[:, 2]
        key_params = split_keys[:, 3]

        controls = self._generate_control_trajectories(key_ctrl, n_steps, dt)
        disturbances = self._generate_disturbance_trajectories(key_dist, n_steps, dt)
        params = self.simulator.sample_data_generation_params_batch(key_params)

        mean_controls = jnp.mean(controls, axis=1)
        mean_disturbances = jnp.mean(disturbances, axis=1)
        default_initial = jnp.tile(self.spec.default_initial_state_array()[None, :], (keys.shape[0], 1))

        steady_state_start = time.perf_counter()
        try:
            initial_states = self.simulator.steady_state_batch_for_data_generation(
                mean_controls,
                mean_disturbances,
                params_batch=params,
                initial_guesses=default_initial,
            )
        except Exception:
            initial_states = default_initial
        steady_state_seconds = time.perf_counter() - steady_state_start

        initial_state_valid_mask = jnp.all(jnp.isfinite(initial_states), axis=1)
        valid_candidate_indices = np.flatnonzero(np.asarray(initial_state_valid_mask))
        candidate_count = int(keys.shape[0])

        if valid_candidate_indices.size == 0:
            return {
                "states": np.zeros((0, n_steps, self.spec.state_dim), dtype=np.float32),
                "controls": np.zeros((0, n_steps, self.spec.control_dim), dtype=np.float32),
                "disturbances": np.zeros((0, n_steps, self.spec.disturbance_dim), dtype=np.float32),
                "params": np.zeros((0, self.spec.param_dim), dtype=np.float32),
                "time": np.zeros((0, n_steps), dtype=np.float32),
                "measurement_noise_seconds": 0.0,
                "steady_state_seconds": steady_state_seconds,
                "candidate_count": candidate_count,
            }

        valid_candidate_indices_jax = jnp.asarray(valid_candidate_indices)
        valid_initial_states = initial_states[valid_candidate_indices_jax]
        valid_controls = controls[valid_candidate_indices_jax]
        valid_disturbances = disturbances[valid_candidate_indices_jax]
        valid_params = params[valid_candidate_indices_jax]
        valid_stored_params = self.simulator.format_data_generation_params_batch(valid_params)
        valid_noise_keys = key_ic[valid_candidate_indices_jax]

        t_end = n_steps * dt
        rollout_chunk_size = min(self.recommend_batch_size(jax.default_backend()), valid_candidate_indices.size)
        state_chunks = []
        time_chunks = []
        control_chunks = []
        disturbance_chunks = []
        param_chunks = []
        noise_seconds = 0.0

        for start in range(0, valid_candidate_indices.size, rollout_chunk_size):
            stop = min(start + rollout_chunk_size, valid_candidate_indices.size)
            chunk_slice = slice(start, stop)
            chunk_initial_states = valid_initial_states[chunk_slice]
            chunk_controls = valid_controls[chunk_slice]
            chunk_disturbances = valid_disturbances[chunk_slice]
            chunk_params = valid_params[chunk_slice]
            chunk_stored_params = valid_stored_params[chunk_slice]
            chunk_noise_keys = valid_noise_keys[chunk_slice]

            if simulation_mode == "dataset":
                chunk_result = self.simulator.simulate_batch_for_data_generation(
                    chunk_initial_states,
                    chunk_controls,
                    chunk_disturbances,
                    (0.0, t_end),
                    params_batch=chunk_params,
                    dt=dt,
                    n_steps=n_steps,
                )
            elif simulation_mode == "reference":
                results = [
                    self.simulator.simulate(
                        chunk_initial_states[idx],
                        chunk_controls[idx],
                        chunk_disturbances[idx],
                        (0.0, t_end),
                        dt=dt,
                        n_steps=n_steps,
                    )
                    for idx in range(chunk_initial_states.shape[0])
                ]
                chunk_result = {
                    "time": jnp.stack([item["time"] for item in results]),
                    "states": jnp.stack([item["states"] for item in results]),
                    "controls": jnp.stack([item["controls"] for item in results]),
                }
            else:
                raise ValueError(f"Unknown simulation_mode: {simulation_mode}")

            noise_start = time.perf_counter()
            chunk_noisy_states = self.simulator.apply_measurement_noise_batch(
                chunk_noise_keys,
                chunk_result["states"],
            )
            noise_seconds += time.perf_counter() - noise_start

            state_chunks.append(np.asarray(chunk_noisy_states))
            time_chunks.append(np.asarray(chunk_result["time"]))
            control_chunks.append(np.asarray(chunk_controls))
            disturbance_chunks.append(np.asarray(chunk_disturbances))
            param_chunks.append(np.asarray(chunk_stored_params))

        return {
            "states": np.concatenate(state_chunks, axis=0),
            "controls": np.concatenate(control_chunks, axis=0),
            "disturbances": np.concatenate(disturbance_chunks, axis=0),
            "params": np.concatenate(param_chunks, axis=0),
            "time": np.concatenate(time_chunks, axis=0),
            "measurement_noise_seconds": noise_seconds,
            "steady_state_seconds": steady_state_seconds,
            "candidate_count": candidate_count,
        }

    def generate_dataset_to_hdf5(
        self,
        key: PRNGKeyArray,
        output_path: str,
        n_trajectories: int = 1000,
        n_steps: int = 500,
        simulation_mode: str = "dataset",
        batch_size: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Generate trajectories and save to HDF5.

        Returns a summary dict compatible with the CSTR DataGenerator summary.
        """
        dt = self.config.get("simulation", {}).get("dt", 0.1)
        batch_size_eff = batch_size or self.recommend_batch_size(jax.default_backend())

        all_states = []
        all_controls = []
        all_disturbances = []
        all_params = []
        all_times = []

        t_start = time.perf_counter()
        invalid = 0
        exceptions = 0
        signal_seconds = 0.0
        steady_state_seconds = 0.0
        rollout_seconds = 0.0
        measurement_noise_seconds = 0.0
        validation_seconds = 0.0

        pbar = tqdm(total=n_trajectories, desc="Generating trajectories")
        generated = 0
        attempts = 0

        while generated < n_trajectories:
            try:
                current_batch_size = min(batch_size_eff, n_trajectories - generated)
                batch_keys = jax.random.split(key, current_batch_size + 1)
                key = batch_keys[0]
                traj_keys = batch_keys[1:]
                attempts += current_batch_size
                if generated == 0 and attempts == current_batch_size:
                    print(
                        "Compiling the first rollout batch. "
                        "The progress bar may stay at 0% until this warmup finishes.",
                        flush=True,
                    )

                if current_batch_size == 1:
                    signal_start = time.perf_counter()
                    traj = self._generate_trajectory(traj_keys[0], n_steps, dt, simulation_mode)
                    elapsed = time.perf_counter() - signal_start
                    rollout_seconds += elapsed
                    if traj is None:
                        invalid += 1
                        continue
                    validation_seconds += float(traj.get("validation_seconds", 0.0))
                    measurement_noise_seconds += float(traj.get("measurement_noise_seconds", 0.0))
                    valid_trajs = [traj]
                else:
                    batch_start = time.perf_counter()
                    batch = self._generate_trajectories_batched(
                        traj_keys,
                        n_steps,
                        dt,
                        simulation_mode,
                    )
                    batch_elapsed = time.perf_counter() - batch_start
                    rollout_seconds += batch_elapsed
                    measurement_noise_seconds += float(batch.get("measurement_noise_seconds", 0.0))
                    steady_state_seconds += float(batch.get("steady_state_seconds", 0.0))

                    validation_start = time.perf_counter()
                    valid_mask = self.simulator.valid_trajectory_mask(batch["states"])
                    validation_seconds += time.perf_counter() - validation_start
                    valid_indices = np.asarray(jnp.where(valid_mask)[0])
                    invalid += int(batch.get("candidate_count", current_batch_size)) - int(valid_mask.sum())
                    valid_trajs = [
                        {
                            "states": np.array(batch["states"][idx]),
                            "controls": np.array(batch["controls"][idx]),
                            "disturbances": np.array(batch["disturbances"][idx]),
                            "params": np.array(batch["params"][idx]),
                            "t": np.array(batch["time"][idx]),
                        }
                        for idx in valid_indices.tolist()
                    ]
            except Exception:
                exceptions += current_batch_size
                continue

            pbar.set_postfix(
                attempts=attempts,
                invalid=invalid,
                valid=generated,
            )
            pbar.refresh()

            for traj in valid_trajs:
                all_states.append(np.array(traj["states"]))
                all_controls.append(np.array(traj["controls"]))
                all_disturbances.append(np.array(traj["disturbances"]))
                all_params.append(np.array(traj["params"]))
                all_times.append(np.array(traj["t"]))
                generated += 1
                pbar.update(1)

        pbar.close()

        states_arr = np.stack(all_states)
        controls_arr = np.stack(all_controls)
        disturbances_arr = np.stack(all_disturbances)
        params_arr = np.stack(all_params)
        times_arr = np.stack(all_times)

        # Compute normalization statistics
        state_mean = states_arr.mean(axis=(0, 1))
        state_std = states_arr.std(axis=(0, 1)) + 1e-8
        control_mean = controls_arr.mean(axis=(0, 1))
        control_std = controls_arr.std(axis=(0, 1)) + 1e-8
        disturbance_mean = disturbances_arr.mean(axis=(0, 1))
        disturbance_std = disturbances_arr.std(axis=(0, 1)) + 1e-8

        with h5py.File(output_path, "w") as f:
            # Use the same key names as the CSTR DataGenerator for compatibility
            # with TrajectoryDataset.load_dataset
            f.create_dataset("states", data=states_arr)
            f.create_dataset("controls", data=controls_arr)
            f.create_dataset("disturbances", data=disturbances_arr)
            f.create_dataset("params", data=params_arr)
            f.create_dataset("time", data=times_arr)

            norm = f.create_group("normalization")
            norm.create_dataset("state_mean", data=state_mean)
            norm.create_dataset("state_std", data=state_std)
            norm.create_dataset("control_mean", data=control_mean)
            norm.create_dataset("control_std", data=control_std)
            norm.create_dataset("disturbance_mean", data=disturbance_mean)
            norm.create_dataset("disturbance_std", data=disturbance_std)

        elapsed = time.perf_counter() - t_start
        self.last_profile = {
            "total_generation_seconds": elapsed,
            "signal_generation_seconds": signal_seconds,
            "steady_state_seconds": steady_state_seconds,
            "rollout_seconds": rollout_seconds,
            "measurement_noise_seconds": measurement_noise_seconds,
            "validation_seconds": validation_seconds,
            "attempts": attempts,
            "invalid_trajectories": invalid,
            "exceptions": exceptions,
            "batch_size": batch_size_eff,
        }

        return {
            "states_shape": states_arr.shape,
            "controls_shape": controls_arr.shape,
            "disturbances_shape": disturbances_arr.shape,
            "params_shape": params_arr.shape,
            "time_shape": times_arr.shape,
            "normalization": {
                "state_mean": state_mean,
                "state_std": state_std,
                "control_mean": control_mean,
                "control_std": control_std,
            },
        }

    def recommend_batch_size(self, backend: str) -> int:
        """Recommend a batch size based on the hardware backend."""
        normalized_backend = backend.lower()
        if normalized_backend in {"gpu", "cuda", "rocm"}:
            return 8
        if normalized_backend == "tpu":
            return 16
        return 4
