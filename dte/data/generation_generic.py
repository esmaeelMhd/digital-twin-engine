"""Generic data generation pipeline for any ProcessSimulator.

Provides a drop-in replacement for the CSTR-specific :class:`DataGenerator`
that works with any :class:`~dte.simulators.base.ProcessSimulator` /
:class:`~dte.simulators.base.SystemSpec` combination.

The PRBS / multistep / chirp excitation signal helpers from
:mod:`dte.data.generation` are reused here.
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, Optional, Tuple

import h5py
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, PRNGKeyArray
from tqdm import tqdm

from dte.simulators.base import ProcessSimulator, SystemSpec


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

    def _generate_control_trajectory(
        self,
        key: PRNGKeyArray,
        n_steps: int,
        dt: float,
    ) -> Float[Array, "n_steps control_dim"]:
        """Generate PRBS-like excitation for all control channels."""
        ops = self.config.get("operating_ranges", {})
        control_dim = self.spec.control_dim
        channels = []

        for i, name in enumerate(self.spec.control_names):
            key, subkey = jax.random.split(key)
            lo, hi = ops.get(name, [0.0, 1.0])
            channels.append(
                self._prbs_channel(subkey, n_steps, float(lo), float(hi))
            )

        return jnp.stack(channels, axis=-1)

    def _generate_disturbance_trajectory(
        self,
        key: PRNGKeyArray,
        n_steps: int,
        dt: float,
    ) -> Float[Array, "n_steps disturbance_dim"]:
        """Generate slowly varying (multistep) disturbances."""
        ops = self.config.get("operating_ranges", {})
        channels = []

        for name in self.spec.disturbance_names:
            key, subkey = jax.random.split(key)
            lo, hi = ops.get(name, [0.0, 1.0])
            channels.append(
                self._multistep_channel(subkey, n_steps, float(lo), float(hi), n_changes=5)
            )

        return jnp.stack(channels, axis=-1)

    def _generate_control_trajectories(
        self,
        keys: PRNGKeyArray,
        n_steps: int,
        dt: float,
    ) -> Float[Array, "batch n_steps control_dim"]:
        """Generate control trajectories for a batch of seeds."""
        return jnp.stack(
            [self._generate_control_trajectory(key, n_steps, dt) for key in keys],
            axis=0,
        )

    def _generate_disturbance_trajectories(
        self,
        keys: PRNGKeyArray,
        n_steps: int,
        dt: float,
    ) -> Float[Array, "batch n_steps disturbance_dim"]:
        """Generate disturbance trajectories for a batch of seeds."""
        return jnp.stack(
            [self._generate_disturbance_trajectory(key, n_steps, dt) for key in keys],
            axis=0,
        )

    @staticmethod
    def _prbs_channel(
        key: PRNGKeyArray,
        n_steps: int,
        lo: float,
        hi: float,
        switch_prob: float = 0.05,
    ) -> Float[Array, "n_steps"]:
        """Generate a pseudo-random binary signal."""
        keys = jax.random.split(key, n_steps)
        switches = jax.vmap(lambda k: jax.random.bernoulli(k, switch_prob))(keys)
        state = jnp.cumsum(switches) % 2
        return jnp.where(state == 0, lo, hi)

    @staticmethod
    def _multistep_channel(
        key: PRNGKeyArray,
        n_steps: int,
        lo: float,
        hi: float,
        n_changes: int = 5,
    ) -> Float[Array, "n_steps"]:
        """Generate a piecewise-constant multistep signal."""
        key_times, key_vals = jax.random.split(key)
        change_times = jnp.sort(
            jax.random.randint(key_times, shape=(n_changes,), minval=0, maxval=n_steps)
        )
        values = jax.random.uniform(key_vals, shape=(n_changes + 1,), minval=lo, maxval=hi)
        segment_ids = jnp.searchsorted(change_times, jnp.arange(n_steps), side="right")
        return values[segment_ids]

    # ------------------------------------------------------------------
    # Params sampling
    # ------------------------------------------------------------------

    def _sample_params(self, key: PRNGKeyArray) -> Float[Array, "param_dim"]:
        """Return normalized parameter vector.

        Uses unit params (all ones) by default; override for parametric
        diversity by subclassing or modifying.
        """
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
            "params": params,
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

        try:
            initial_states = self.simulator.steady_state_batch_for_data_generation(
                mean_controls,
                mean_disturbances,
                params_batch=params,
                initial_guesses=default_initial,
            )
        except Exception:
            initial_states = default_initial

        t_end = n_steps * dt
        if simulation_mode == "dataset":
            result = self.simulator.simulate_batch_for_data_generation(
                initial_states,
                controls,
                disturbances,
                (0.0, t_end),
                params_batch=params,
                dt=dt,
                n_steps=n_steps,
            )
        elif simulation_mode == "reference":
            results = [
                self.simulator.simulate(
                    initial_states[idx],
                    controls[idx],
                    disturbances[idx],
                    (0.0, t_end),
                    dt=dt,
                    n_steps=n_steps,
                )
                for idx in range(keys.shape[0])
            ]
            result = {
                "time": jnp.stack([item["time"] for item in results]),
                "states": jnp.stack([item["states"] for item in results]),
                "controls": jnp.stack([item["controls"] for item in results]),
            }
        else:
            raise ValueError(f"Unknown simulation_mode: {simulation_mode}")

        noise_start = time.perf_counter()
        noisy_states = self.simulator.apply_measurement_noise_batch(key_ic, result["states"])
        noise_seconds = time.perf_counter() - noise_start

        return {
            "states": noisy_states,
            "controls": controls,
            "disturbances": disturbances,
            "params": params,
            "time": result["time"],
            "measurement_noise_seconds": noise_seconds,
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

                    validation_start = time.perf_counter()
                    valid_mask = self.simulator.valid_trajectory_mask(batch["states"])
                    validation_seconds += time.perf_counter() - validation_start
                    valid_indices = np.asarray(jnp.where(valid_mask)[0])
                    invalid += current_batch_size - int(valid_mask.sum())
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
        return 100
