"""Legacy CSTR-only data generation pipeline."""

from functools import partial
from typing import Dict, Tuple
import time
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray
import h5py
from tqdm import tqdm

from dte.simulators.cstr import (
    CSTRSimulator,
    CSTRParams,
    pack_params,
    sample_random_params,
    simulate_data_generation_batch_jit,
    steady_state_batch_with_residuals_jit,
)


def _generate_prbs_signal_core(
    key: PRNGKeyArray,
    n_steps: int,
    min_val: float,
    max_val: float,
    switch_prob: float,
) -> Float[Array, "n_steps 1"]:
    """Pure JAX PRBS signal core used by compiled batch helpers."""
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
    """Pure JAX chirp signal core used by compiled batch helpers."""
    t = jnp.arange(n_steps) * dt
    total_time = jnp.maximum(t[-1], 1.0)
    freq = min_freq + (max_freq - min_freq) * t / total_time
    phase = jnp.cumsum(freq) * dt * 2 * jnp.pi
    chirp = jnp.sin(phase)
    signal = min_val + (max_val - min_val) * (chirp + 1) / 2
    return signal[:, None]


def _generate_multistep_signal_core(
    key: PRNGKeyArray,
    n_steps: int,
    min_val: float,
    max_val: float,
    n_changes: int,
) -> Float[Array, "n_steps 1"]:
    """Pure JAX multistep signal core used by compiled batch helpers."""
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
    """Compiled batched signal generator for control/disturbance trajectories."""
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

class DataGenerator:
    """Generate diverse CSTR trajectories for training."""

    def __init__(self, simulator: CSTRSimulator, config: dict):
        """Initialize data generator.
        
        Args:
            simulator: CSTR simulator instance
            config: Configuration dictionary with operating ranges
        """
        self.simulator = simulator
        self.config = config
        self.operating_ranges = config["operating_ranges"]
        self.dt = config["simulation"]["dt"]
        self.last_profile: Dict[str, float] = {}

    @staticmethod
    def _initialize_profile() -> Dict[str, float]:
        """Create an empty profiling record for a dataset-generation run."""
        return {
            "signal_generation_seconds": 0.0,
            "steady_state_seconds": 0.0,
            "rollout_seconds": 0.0,
            "measurement_noise_seconds": 0.0,
            "validation_seconds": 0.0,
            "exception_seconds": 0.0,
            "total_generation_seconds": 0.0,
            "attempts": 0,
            "successful_trajectories": 0,
            "invalid_trajectories": 0,
            "exceptions": 0,
            "steady_state_fast_successes": 0,
            "steady_state_fallbacks": 0,
            "simulation_mode": "",
        }

    @staticmethod
    def _merge_profile_metrics(target: Dict[str, float], source: Dict[str, float]) -> None:
        """Merge additive profile metrics from a nested generation call."""
        additive_keys = [
            "signal_generation_seconds",
            "steady_state_seconds",
            "rollout_seconds",
            "measurement_noise_seconds",
            "validation_seconds",
            "exception_seconds",
            "invalid_trajectories",
            "exceptions",
            "steady_state_fast_successes",
            "steady_state_fallbacks",
        ]
        for key in additive_keys:
            target[key] += source.get(key, 0)

    @staticmethod
    def recommend_batch_size(backend: str) -> int:
        """Return a conservative default batch size for the current backend.

        These defaults are intended to be safe out of the box. Use the benchmark
        harness to tune them for a specific machine.
        """
        normalized_backend = backend.lower()
        if normalized_backend in {"gpu", "cuda", "rocm"}:
            return 8
        if normalized_backend == "tpu":
            return 16
        return 4

    @staticmethod
    def _initialize_running_stats() -> Dict[str, Array]:
        """Create running sums for online normalization-stat computation."""
        return {
            "state_sum": jnp.zeros((4,)),
            "state_sum_sq": jnp.zeros((4,)),
            "control_sum": jnp.zeros((2,)),
            "control_sum_sq": jnp.zeros((2,)),
            "disturbance_sum": jnp.zeros((2,)),
            "disturbance_sum_sq": jnp.zeros((2,)),
            "param_sum": jnp.zeros((6,)),
            "param_sum_sq": jnp.zeros((6,)),
            "state_count": 0,
            "control_count": 0,
            "disturbance_count": 0,
            "param_count": 0,
        }

    @staticmethod
    def _update_running_stats(running_stats: Dict[str, Array], batch: Dict[str, Array]) -> None:
        """Update online normalization statistics with a new batch."""
        states = batch["states"].reshape(-1, 4)
        controls = batch["controls"].reshape(-1, 2)
        disturbances = batch["disturbances"].reshape(-1, 2)
        params = batch["params"]

        running_stats["state_sum"] = running_stats["state_sum"] + jnp.sum(states, axis=0)
        running_stats["state_sum_sq"] = running_stats["state_sum_sq"] + jnp.sum(states**2, axis=0)
        running_stats["control_sum"] = running_stats["control_sum"] + jnp.sum(controls, axis=0)
        running_stats["control_sum_sq"] = (
            running_stats["control_sum_sq"] + jnp.sum(controls**2, axis=0)
        )
        running_stats["disturbance_sum"] = (
            running_stats["disturbance_sum"] + jnp.sum(disturbances, axis=0)
        )
        running_stats["disturbance_sum_sq"] = (
            running_stats["disturbance_sum_sq"] + jnp.sum(disturbances**2, axis=0)
        )
        running_stats["param_sum"] = running_stats["param_sum"] + jnp.sum(params, axis=0)
        running_stats["param_sum_sq"] = running_stats["param_sum_sq"] + jnp.sum(params**2, axis=0)
        running_stats["state_count"] += int(states.shape[0])
        running_stats["control_count"] += int(controls.shape[0])
        running_stats["disturbance_count"] += int(disturbances.shape[0])
        running_stats["param_count"] += int(params.shape[0])

    @staticmethod
    def _finalize_running_stats(running_stats: Dict[str, Array]) -> Dict[str, Array]:
        """Convert running sums into mean/std normalization statistics."""
        def mean_std(sum_, sum_sq, count):
            mean = sum_ / count
            variance = jnp.maximum(sum_sq / count - mean**2, 0.0)
            return mean, jnp.sqrt(variance)

        state_mean, state_std = mean_std(
            running_stats["state_sum"], running_stats["state_sum_sq"], running_stats["state_count"]
        )
        control_mean, control_std = mean_std(
            running_stats["control_sum"],
            running_stats["control_sum_sq"],
            running_stats["control_count"],
        )
        disturbance_mean, disturbance_std = mean_std(
            running_stats["disturbance_sum"],
            running_stats["disturbance_sum_sq"],
            running_stats["disturbance_count"],
        )
        param_mean, param_std = mean_std(
            running_stats["param_sum"], running_stats["param_sum_sq"], running_stats["param_count"]
        )

        return {
            "state_mean": state_mean,
            "state_std": state_std,
            "control_mean": control_mean,
            "control_std": control_std,
            "disturbance_mean": disturbance_mean,
            "disturbance_std": disturbance_std,
            "param_mean": param_mean,
            "param_std": param_std,
        }

    @staticmethod
    def _generate_prbs_signal_impl(
        key: PRNGKeyArray,
        n_steps: int,
        min_val: float,
        max_val: float,
        switch_prob: float,
    ) -> Float[Array, "n_steps 1"]:
        """JAX-friendly PRBS implementation shared by single and batched generation."""
        return _generate_prbs_signal_core(key, n_steps, min_val, max_val, switch_prob)

    @staticmethod
    def _generate_chirp_signal_impl(
        n_steps: int,
        dt: float,
        min_val: float,
        max_val: float,
        min_freq: float,
        max_freq: float,
    ) -> Float[Array, "n_steps 1"]:
        """JAX-friendly chirp implementation shared by single and batched generation."""
        return _generate_chirp_signal_core(
            n_steps, dt, min_val, max_val, min_freq, max_freq
        )

    @staticmethod
    def _generate_multistep_signal_impl(
        key: PRNGKeyArray,
        n_steps: int,
        min_val: float,
        max_val: float,
        n_changes: int,
    ) -> Float[Array, "n_steps 1"]:
        """JAX-friendly multistep implementation shared by single and batched generation."""
        return _generate_multistep_signal_core(key, n_steps, min_val, max_val, n_changes)

    def _generate_signal(
        self,
        key: PRNGKeyArray,
        n_steps: int,
        dt: float,
        min_val: float,
        max_val: float,
        signal_type: str,
        switch_prob: float = 0.05,
        n_changes: int = 10,
        min_freq: float = 0.01,
        max_freq: float = 0.1,
    ) -> Float[Array, "n_steps 1"]:
        """Generate a single input signal using JAX-friendly helpers."""
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
                lambda k: self._generate_prbs_signal_impl(
                    k, n_steps, min_val, max_val, switch_prob
                ),
                lambda k: self._generate_chirp_signal_impl(
                    n_steps, dt, min_val, max_val, min_freq, max_freq
                ),
                lambda k: self._generate_multistep_signal_impl(
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
        n_changes: int = 10,
        min_freq: float = 0.01,
        max_freq: float = 0.1,
    ) -> Float[Array, "batch n_steps 1"]:
        """Generate a batch of input signals while preserving single-path behavior."""
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

    def generate_prbs_signal(
        self,
        key: PRNGKeyArray,
        n_steps: int,
        dt: float,
        min_val: float,
        max_val: float,
        switch_prob: float = 0.05,
    ) -> Float[Array, "n_steps 1"]:
        """Generate Pseudo-Random Binary Sequence signal.
        
        Args:
            key: PRNG key
            n_steps: Number of time steps
            dt: Time step size
            min_val: Minimum value
            max_val: Maximum value
            switch_prob: Probability of switching at each step
            
        Returns:
            Signal array of shape (n_steps, 1)
        """
        return self._generate_prbs_signal_impl(key, n_steps, min_val, max_val, switch_prob)

    def generate_chirp_signal(
        self,
        key: PRNGKeyArray,
        n_steps: int,
        dt: float,
        min_val: float,
        max_val: float,
        min_freq: float = 0.01,
        max_freq: float = 0.1,
    ) -> Float[Array, "n_steps 1"]:
        """Generate chirp signal with linearly increasing frequency.
        
        Args:
            key: PRNG key
            n_steps: Number of time steps
            dt: Time step size
            min_val: Minimum value
            max_val: Maximum value
            min_freq: Starting frequency
            max_freq: Ending frequency
            
        Returns:
            Signal array of shape (n_steps, 1)
        """
        return self._generate_chirp_signal_impl(
            n_steps, dt, min_val, max_val, min_freq, max_freq
        )

    def generate_multistep_signal(
        self,
        key: PRNGKeyArray,
        n_steps: int,
        dt: float,
        min_val: float,
        max_val: float,
        n_changes: int = 10,
    ) -> Float[Array, "n_steps 1"]:
        """Generate signal with random step changes.
        
        Args:
            key: PRNG key
            n_steps: Number of time steps
            dt: Time step size
            min_val: Minimum value
            max_val: Maximum value
            n_changes: Number of step changes
            
        Returns:
            Signal array of shape (n_steps, 1)
        """
        return self._generate_multistep_signal_impl(
            key, n_steps, min_val, max_val, n_changes
        )

    def generate_control_trajectory(
        self,
        key: PRNGKeyArray,
        n_steps: int,
        dt: float,
        operating_ranges: dict,
        signal_type: str = "mixed",
    ) -> Float[Array, "n_steps 2"]:
        """Generate control trajectory.
        
        Args:
            key: PRNG key
            n_steps: Number of time steps
            dt: Time step size
            operating_ranges: Dictionary with min/max for each control
            signal_type: Type of signal ("prbs", "chirp", "multistep", "mixed")
            
        Returns:
            Control trajectory of shape (n_steps, 2) for [F_in, Tc_in]
        """
        key1, key2 = jax.random.split(key)

        F_min, F_max = operating_ranges["F_in"]
        F_in = self._generate_signal(
            key1,
            n_steps,
            dt,
            F_min,
            F_max,
            signal_type,
            switch_prob=0.05,
            n_changes=5,
        )

        Tc_min, Tc_max = operating_ranges["Tc_in"]
        Tc_in = self._generate_signal(
            key2,
            n_steps,
            dt,
            Tc_min,
            Tc_max,
            signal_type,
            switch_prob=0.05,
            n_changes=5,
        )

        return jnp.concatenate([F_in, Tc_in], axis=1)

    def generate_disturbance_trajectory(
        self,
        key: PRNGKeyArray,
        n_steps: int,
        dt: float,
        operating_ranges: dict,
    ) -> Float[Array, "n_steps 2"]:
        """Generate disturbance trajectory (slower-varying).
        
        Args:
            key: PRNG key
            n_steps: Number of time steps
            dt: Time step size
            operating_ranges: Dictionary with min/max for each disturbance
            
        Returns:
            Disturbance trajectory of shape (n_steps, 2) for [Ca_in, T_in]
        """
        key1, key2 = jax.random.split(key)

        Ca_min, Ca_max = operating_ranges["Ca_in"]
        Ca_in = self._generate_signal(
            key1, n_steps, dt, Ca_min, Ca_max, "prbs", switch_prob=0.01
        )

        T_min, T_max = operating_ranges["T_in"]
        T_in = self._generate_signal(
            key2, n_steps, dt, T_min, T_max, "prbs", switch_prob=0.01
        )

        return jnp.concatenate([Ca_in, T_in], axis=1)

    def _generate_dataset_batched(
        self,
        key: PRNGKeyArray,
        n_trajectories: int,
        n_steps: int,
        batch_size: int,
        profile: Dict[str, float],
        show_progress: bool = True,
    ) -> Dict[str, Array]:
        """Generate dataset using chunked batched steady-state and rollout stages."""
        all_states = []
        all_controls = []
        all_disturbances = []
        all_params = []
        all_times = []

        successful = 0
        attempt = 0
        pbar = tqdm(total=n_trajectories) if show_progress else None
        generation_start = time.perf_counter()

        while successful < n_trajectories:
            remaining = n_trajectories - successful
            current_batch_size = min(batch_size, remaining)
            batch_indices = jnp.arange(attempt, attempt + current_batch_size)
            batch_keys = jax.vmap(lambda i: jax.random.fold_in(key, i))(batch_indices)
            split_keys = jax.vmap(lambda k: jax.random.split(k, 4))(batch_keys)

            key_params = split_keys[:, 0]
            key_ic = split_keys[:, 1]
            key_control = split_keys[:, 2]
            key_dist = split_keys[:, 3]

            signal_start = time.perf_counter()
            params_list = [sample_random_params(k) for k in key_params]
            packed_params = jnp.stack([pack_params(params) for params in params_list])
            param_vecs = jnp.stack(
                [
                    jnp.array(
                        [
                            params.V,
                            params.k0 / 1e10,
                            params.Ea_over_R / 1000.0,
                            params.UA / 1e4,
                            params.Fc,
                            params.Vc,
                        ]
                    )
                    for params in params_list
                ]
            )
            control_keys_1, control_keys_2 = jax.vmap(jax.random.split)(key_control).transpose(1, 0, 2)
            F_min, F_max = self.operating_ranges["F_in"]
            Tc_min, Tc_max = self.operating_ranges["Tc_in"]
            F_in = self._generate_signal_batch(
                control_keys_1,
                n_steps,
                self.dt,
                F_min,
                F_max,
                "mixed",
                switch_prob=0.05,
                n_changes=5,
            )
            Tc_in = self._generate_signal_batch(
                control_keys_2,
                n_steps,
                self.dt,
                Tc_min,
                Tc_max,
                "mixed",
                switch_prob=0.05,
                n_changes=5,
            )
            controls = jnp.concatenate([F_in, Tc_in], axis=2)

            dist_keys_1, dist_keys_2 = jax.vmap(jax.random.split)(key_dist).transpose(1, 0, 2)
            Ca_min, Ca_max = self.operating_ranges["Ca_in"]
            T_min, T_max = self.operating_ranges["T_in"]
            Ca_in = self._generate_signal_batch(
                dist_keys_1,
                n_steps,
                self.dt,
                Ca_min,
                Ca_max,
                "prbs",
                switch_prob=0.01,
            )
            T_in = self._generate_signal_batch(
                dist_keys_2,
                n_steps,
                self.dt,
                T_min,
                T_max,
                "prbs",
                switch_prob=0.01,
            )
            disturbances = jnp.concatenate([Ca_in, T_in], axis=2)
            profile["signal_generation_seconds"] += time.perf_counter() - signal_start

            steady_state_start = time.perf_counter()
            steady_state_control = jnp.array([50.0, 300.0])
            steady_state_disturbance = jnp.array([1.0, 320.0])
            steady_state_controls = jnp.tile(steady_state_control[None, :], (current_batch_size, 1))
            steady_state_disturbances = jnp.tile(
                steady_state_disturbance[None, :], (current_batch_size, 1)
            )
            default_initial_guesses = jnp.tile(
                jnp.array([[0.5, 0.5, 350.0, 300.0]]),
                (current_batch_size, 1),
            )
            alt_temperature_guess = 0.5 * (
                steady_state_disturbance[1] + steady_state_control[1]
            ) + 20.0
            alternative_initial_guesses = jnp.tile(
                jnp.array([[0.5, 0.5, alt_temperature_guess, steady_state_control[1]]]),
                (current_batch_size, 1),
            )
            candidate_states_default, residuals_default = steady_state_batch_with_residuals_jit(
                steady_state_controls,
                steady_state_disturbances,
                default_initial_guesses,
                packed_params,
            )
            candidate_states_alt, residuals_alt = steady_state_batch_with_residuals_jit(
                steady_state_controls,
                steady_state_disturbances,
                alternative_initial_guesses,
                packed_params,
            )
            use_alternative_mask = residuals_alt < residuals_default
            residuals = jnp.where(use_alternative_mask, residuals_alt, residuals_default)
            candidate_states = jnp.where(
                use_alternative_mask[:, None],
                candidate_states_alt,
                candidate_states_default,
            )
            valid_fast_mask = jnp.isfinite(residuals) & (residuals <= 1e-4)
            profile["steady_state_fast_successes"] += int(jnp.sum(valid_fast_mask))
            profile["steady_state_fallbacks"] += current_batch_size - int(jnp.sum(valid_fast_mask))
            steady_states_list = []
            for idx, params in enumerate(params_list):
                if bool(valid_fast_mask[idx]):
                    steady_states_list.append(candidate_states[idx])
                else:
                    steady_states_list.append(
                        CSTRSimulator(params).steady_state(
                            steady_state_control,
                            steady_state_disturbance,
                        )
                    )
            steady_states = jnp.stack(steady_states_list)
            profile["steady_state_seconds"] += time.perf_counter() - steady_state_start

            noise = jax.vmap(
                lambda k: jax.random.normal(k, shape=(4,))
            )(key_ic) * jnp.array([0.1, 0.1, 5.0, 5.0])
            initial_states = steady_states + noise

            rollout_start = time.perf_counter()
            all_times_batch, states_batch = simulate_data_generation_batch_jit(
                initial_states,
                controls,
                disturbances,
                packed_params,
                0.0,
                n_steps * self.dt,
            )
            profile["rollout_seconds"] += time.perf_counter() - rollout_start

            noise_start = time.perf_counter()
            noise_std = jnp.array([0.01, 0.01, 0.5, 0.5])
            measurement_noise = jax.vmap(
                lambda k: jax.random.normal(k, shape=(n_steps, 4))
            )(key_ic) * noise_std[None, None, :]
            noisy_states = states_batch + measurement_noise
            profile["measurement_noise_seconds"] += time.perf_counter() - noise_start

            validation_start = time.perf_counter()
            valid_mask = (
                ~jnp.isnan(noisy_states).any(axis=(1, 2))
                & ~jnp.isinf(noisy_states).any(axis=(1, 2))
                & ~(noisy_states[:, :, 0] < 0).any(axis=1)
                & ~(noisy_states[:, :, 1] < -0.5).any(axis=1)
            )
            profile["validation_seconds"] += time.perf_counter() - validation_start

            valid_indices = jnp.where(valid_mask)[0]
            valid_count = int(valid_mask.sum())
            profile["invalid_trajectories"] += current_batch_size - valid_count

            if valid_count > 0:
                for idx in valid_indices.tolist():
                    all_states.append(noisy_states[idx])
                    all_controls.append(controls[idx])
                    all_disturbances.append(disturbances[idx])
                    all_params.append(param_vecs[idx])
                    all_times.append(all_times_batch[idx])

                successful += valid_count
                profile["successful_trajectories"] = successful
                if pbar is not None:
                    pbar.update(valid_count)

            attempt += current_batch_size
            profile["attempts"] = attempt

            if attempt > 0 and attempt % (batch_size * 5) == 0 and successful < n_trajectories:
                print(
                    f"\nAttempts: {attempt}, successful trajectories: "
                    f"{successful}/{n_trajectories}"
                )

        if pbar is not None:
            pbar.close()
        profile["total_generation_seconds"] = time.perf_counter() - generation_start

        return {
            "time": jnp.stack(all_times),
            "states": jnp.stack(all_states),
            "controls": jnp.stack(all_controls),
            "disturbances": jnp.stack(all_disturbances),
            "params": jnp.stack(all_params),
        }

    def generate_single_trajectory(
        self,
        key: PRNGKeyArray,
        params: CSTRParams = None,
        n_steps: int = 1000,
        simulation_mode: str = "dataset",
        profiling: Dict[str, float] | None = None,
    ) -> Dict[str, Array]:
        """Generate a single trajectory.
        
        Args:
            key: PRNG key
            params: CSTR parameters (if None, sample random)
            n_steps: Number of time steps
            simulation_mode: Rollout path ("dataset" or "reference")
            profiling: Optional profiling accumulator for timing sub-steps
            
        Returns:
            Dictionary with time, states, controls, disturbances, params
        """
        key_params, key_ic, key_control, key_dist = jax.random.split(key, 4)
        
        # Sample parameters if not provided
        if params is None:
            params = sample_random_params(key_params)
            simulator = CSTRSimulator(params)
        else:
            simulator = self.simulator

        # Generate control and disturbance trajectories
        signal_start = time.perf_counter()
        control_traj = self.generate_control_trajectory(
            key_control, n_steps, self.dt, self.operating_ranges, signal_type="mixed"
        )
        disturbance_traj = self.generate_disturbance_trajectory(
            key_dist, n_steps, self.dt, self.operating_ranges
        )
        if profiling is not None:
            profiling["signal_generation_seconds"] += time.perf_counter() - signal_start
        
        # Initial condition near steady state with noise
        ss_control = jnp.array([50.0, 300.0])
        ss_dist = jnp.array([1.0, 320.0])
        steady_state_start = time.perf_counter()
        ss_state = simulator.steady_state(ss_control, ss_dist)
        if profiling is not None:
            profiling["steady_state_seconds"] += time.perf_counter() - steady_state_start
        
        # Add noise to initial condition
        noise = jax.random.normal(key_ic, shape=(4,)) * jnp.array([0.1, 0.1, 5.0, 5.0])
        initial_state = ss_state + noise
        
        # Simulate
        t_span = (0.0, n_steps * self.dt)
        rollout_start = time.perf_counter()
        if simulation_mode == "dataset":
            result = simulator.simulate_for_data_generation(
                initial_state, control_traj, disturbance_traj, t_span, self.dt, n_steps
            )
        elif simulation_mode == "reference":
            result = simulator.simulate(
                initial_state, control_traj, disturbance_traj, t_span, self.dt, n_steps
            )
        else:
            raise ValueError(f"Unknown simulation mode: {simulation_mode}")
        if profiling is not None:
            profiling["rollout_seconds"] += time.perf_counter() - rollout_start
        
        # Add measurement noise
        noise_start = time.perf_counter()
        noise_std = jnp.array([0.01, 0.01, 0.5, 0.5])  # 1% for conc, 0.5K for temp
        state_noise = jax.random.normal(key_ic, shape=result["states"].shape) * noise_std[None, :]
        noisy_states = result["states"] + state_noise
        if profiling is not None:
            profiling["measurement_noise_seconds"] += time.perf_counter() - noise_start
        
        # Extract parameter vector (subset that varies)
        param_vec = jnp.array([
            params.V,
            params.k0 / 1e10,  # Normalize
            params.Ea_over_R / 1000.0,  # Normalize
            params.UA / 1e4,  # Normalize
            params.Fc,
            params.Vc,
        ])
        
        return {
            "time": result["time"],
            "states": noisy_states,
            "controls": control_traj,
            "disturbances": disturbance_traj,
            "params": param_vec,
        }

    def generate_dataset(
        self,
        key: PRNGKeyArray,
        n_trajectories: int = 10000,
        n_steps: int = 1000,
        simulation_mode: str = "dataset",
        batch_size: int = 1,
    ) -> Dict[str, Array]:
        """Generate full dataset.
        
        Args:
            key: PRNG key
            n_trajectories: Number of trajectories
            n_steps: Steps per trajectory
            simulation_mode: Rollout path ("dataset" or "reference")
            batch_size: Number of trajectories to process together in dataset mode
            
        Returns:
            Dictionary of stacked arrays
        """
        print(f"Generating {n_trajectories} trajectories...")

        # Pre-allocate arrays
        profile = self._initialize_profile()
        profile["simulation_mode"] = simulation_mode
        profile["batch_size"] = batch_size

        if simulation_mode == "dataset" and batch_size > 1:
            dataset = self._generate_dataset_batched(
                key,
                n_trajectories=n_trajectories,
                n_steps=n_steps,
                batch_size=batch_size,
                profile=profile,
            )
            self.last_profile = profile
        else:
            all_states = []
            all_controls = []
            all_disturbances = []
            all_params = []
            all_times = []
            
            # Generate trajectories with progress bar
            successful = 0
            attempt = 0
            pbar = tqdm(total=n_trajectories)
            generation_start = time.perf_counter()
        
            while successful < n_trajectories:
                try:
                    traj_key = jax.random.fold_in(key, attempt)
                    traj = self.generate_single_trajectory(
                        traj_key,
                        n_steps=n_steps,
                        simulation_mode=simulation_mode,
                        profiling=profile,
                    )
                    
                    # Check for NaN or invalid values
                    validation_start = time.perf_counter()
                    if (jnp.isnan(traj["states"]).any() or 
                        jnp.isinf(traj["states"]).any() or
                        (traj["states"][:, 0] < 0).any() or  # Ca must be positive
                        (traj["states"][:, 1] < -0.5).any()):  # Cb should be ~positive (small neg ok)
                        profile["validation_seconds"] += time.perf_counter() - validation_start
                        profile["invalid_trajectories"] += 1
                        attempt += 1
                        continue
                    profile["validation_seconds"] += time.perf_counter() - validation_start
                    
                    all_states.append(traj["states"])
                    all_controls.append(traj["controls"])
                    all_disturbances.append(traj["disturbances"])
                    all_params.append(traj["params"])
                    all_times.append(traj["time"])
                    
                    successful += 1
                    profile["successful_trajectories"] = successful
                    pbar.update(1)
                except Exception as e:
                    exception_start = time.perf_counter()
                    print(f"\nError generating trajectory {attempt}: {e}")
                    profile["exceptions"] += 1
                    profile["exception_seconds"] += time.perf_counter() - exception_start
                
                attempt += 1
                profile["attempts"] = attempt

                if attempt > 0 and attempt % 100 == 0 and successful < n_trajectories:
                    print(
                        f"\nAttempts: {attempt}, successful trajectories: "
                        f"{successful}/{n_trajectories}"
                    )
            
            pbar.close()
            profile["attempts"] = attempt
            profile["total_generation_seconds"] = time.perf_counter() - generation_start
            self.last_profile = profile
            
            # Stack arrays
            all_states = jnp.stack(all_states)
            all_controls = jnp.stack(all_controls)
            all_disturbances = jnp.stack(all_disturbances)
            all_params = jnp.stack(all_params)
            all_times = jnp.stack(all_times)

            dataset = {
                "time": all_times,
                "states": all_states,
                "controls": all_controls,
                "disturbances": all_disturbances,
                "params": all_params,
            }
        
        # Compute normalization statistics
        state_mean = jnp.mean(dataset["states"].reshape(-1, 4), axis=0)
        state_std = jnp.std(dataset["states"].reshape(-1, 4), axis=0)
        
        control_mean = jnp.mean(dataset["controls"].reshape(-1, 2), axis=0)
        control_std = jnp.std(dataset["controls"].reshape(-1, 2), axis=0)
        
        disturbance_mean = jnp.mean(dataset["disturbances"].reshape(-1, 2), axis=0)
        disturbance_std = jnp.std(dataset["disturbances"].reshape(-1, 2), axis=0)
        
        param_mean = jnp.mean(dataset["params"], axis=0)
        param_std = jnp.std(dataset["params"], axis=0)
        
        return {
            "time": dataset["time"],
            "states": dataset["states"],
            "controls": dataset["controls"],
            "disturbances": dataset["disturbances"],
            "params": dataset["params"],
            "normalization": {
                "state_mean": state_mean,
                "state_std": state_std,
                "control_mean": control_mean,
                "control_std": control_std,
                "disturbance_mean": disturbance_mean,
                "disturbance_std": disturbance_std,
                "param_mean": param_mean,
                "param_std": param_std,
            }
        }

    def save_dataset(self, dataset: Dict[str, Array], path: str):
        """Save dataset to HDF5 file.
        
        Args:
            dataset: Dataset dictionary
            path: Output file path
        """
        with h5py.File(path, "w") as f:
            # Save arrays
            f.create_dataset("time", data=dataset["time"])
            f.create_dataset("states", data=dataset["states"])
            f.create_dataset("controls", data=dataset["controls"])
            f.create_dataset("disturbances", data=dataset["disturbances"])
            f.create_dataset("params", data=dataset["params"])
            
            # Save normalization stats
            norm_grp = f.create_group("normalization")
            for key, value in dataset["normalization"].items():
                norm_grp.create_dataset(key, data=value)
        
        print(f"Dataset saved to {path}")

    def generate_dataset_to_hdf5(
        self,
        key: PRNGKeyArray,
        path: str,
        n_trajectories: int = 10000,
        n_steps: int = 1000,
        simulation_mode: str = "dataset",
        batch_size: int = 1,
    ) -> Dict[str, Array]:
        """Generate a dataset directly into an HDF5 file using chunked writes.

        Returns a lightweight summary dictionary rather than reloading the full file.
        """
        print(f"Generating {n_trajectories} trajectories...")

        profile = self._initialize_profile()
        profile["simulation_mode"] = simulation_mode
        profile["batch_size"] = batch_size
        running_stats = self._initialize_running_stats()

        with h5py.File(path, "w") as f:
            time_ds = f.create_dataset("time", shape=(n_trajectories, n_steps), dtype="f4")
            states_ds = f.create_dataset("states", shape=(n_trajectories, n_steps, 4), dtype="f4")
            controls_ds = f.create_dataset(
                "controls", shape=(n_trajectories, n_steps, 2), dtype="f4"
            )
            disturbances_ds = f.create_dataset(
                "disturbances", shape=(n_trajectories, n_steps, 2), dtype="f4"
            )
            params_ds = f.create_dataset("params", shape=(n_trajectories, 6), dtype="f4")

            written = 0
            attempt = 0
            pbar = tqdm(total=n_trajectories)
            generation_start = time.perf_counter()

            while written < n_trajectories:
                remaining = n_trajectories - written
                current_batch_size = min(batch_size, remaining)

                if simulation_mode == "dataset" and current_batch_size > 1:
                    batch_profile = self._initialize_profile()
                    batch = self._generate_dataset_batched(
                        key=jax.random.fold_in(key, attempt),
                        n_trajectories=current_batch_size,
                        n_steps=n_steps,
                        batch_size=current_batch_size,
                        profile=batch_profile,
                        show_progress=False,
                    )
                    self._merge_profile_metrics(profile, batch_profile)
                    attempt += int(batch_profile["attempts"])
                else:
                    batch_states = []
                    batch_controls = []
                    batch_disturbances = []
                    batch_params = []
                    batch_times = []

                    while len(batch_states) < current_batch_size:
                        traj_key = jax.random.fold_in(key, attempt)
                        attempt += 1
                        profile["attempts"] = attempt
                        traj = self.generate_single_trajectory(
                            traj_key,
                            n_steps=n_steps,
                            simulation_mode=simulation_mode,
                            profiling=profile,
                        )

                        validation_start = time.perf_counter()
                        valid = not (
                            jnp.isnan(traj["states"]).any()
                            or jnp.isinf(traj["states"]).any()
                            or (traj["states"][:, 0] < 0).any()
                            or (traj["states"][:, 1] < -0.5).any()
                        )
                        profile["validation_seconds"] += time.perf_counter() - validation_start

                        if not valid:
                            profile["invalid_trajectories"] += 1
                            continue

                        batch_states.append(traj["states"])
                        batch_controls.append(traj["controls"])
                        batch_disturbances.append(traj["disturbances"])
                        batch_params.append(traj["params"])
                        batch_times.append(traj["time"])

                    batch = {
                        "time": jnp.stack(batch_times),
                        "states": jnp.stack(batch_states),
                        "controls": jnp.stack(batch_controls),
                        "disturbances": jnp.stack(batch_disturbances),
                        "params": jnp.stack(batch_params),
                    }

                batch_count = int(batch["states"].shape[0])
                end = written + batch_count

                time_ds[written:end] = batch["time"]
                states_ds[written:end] = batch["states"]
                controls_ds[written:end] = batch["controls"]
                disturbances_ds[written:end] = batch["disturbances"]
                params_ds[written:end] = batch["params"]

                self._update_running_stats(running_stats, batch)

                written = end
                profile["successful_trajectories"] = written
                profile["attempts"] = attempt
                pbar.update(batch_count)

            pbar.close()
            profile["total_generation_seconds"] = time.perf_counter() - generation_start

            normalization = self._finalize_running_stats(running_stats)
            norm_grp = f.create_group("normalization")
            for key_name, value in normalization.items():
                norm_grp.create_dataset(key_name, data=value)

        self.last_profile = profile
        print(f"Dataset saved to {path}")
        return {
            "time_shape": (n_trajectories, n_steps),
            "states_shape": (n_trajectories, n_steps, 4),
            "controls_shape": (n_trajectories, n_steps, 2),
            "disturbances_shape": (n_trajectories, n_steps, 2),
            "params_shape": (n_trajectories, 6),
            "normalization": normalization,
        }

    @staticmethod
    def load_dataset(path: str) -> Dict[str, Array]:
        """Load dataset from HDF5 file.
        
        Args:
            path: Input file path
            
        Returns:
            Dataset dictionary
        """
        with h5py.File(path, "r") as f:
            dataset = {
                "time": jnp.array(f["time"]),
                "states": jnp.array(f["states"]),
                "controls": jnp.array(f["controls"]),
                "disturbances": jnp.array(f["disturbances"]),
                "params": jnp.array(f["params"]),
                "normalization": {
                    key: jnp.array(f["normalization"][key])
                    for key in f["normalization"].keys()
                }
            }
        
        print(f"Dataset loaded from {path}")
        return dataset
