"""Data generation pipeline for CSTR trajectories."""

from typing import Dict, Tuple
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray
import h5py
from tqdm import tqdm

from dte.simulators.cstr import CSTRSimulator, CSTRParams, sample_random_params


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
        # Generate random switches
        keys = jax.random.split(key, n_steps)
        switches = jax.vmap(lambda k: jax.random.bernoulli(k, switch_prob))(keys)
        
        # Cumulative XOR to get current state
        state = jnp.cumsum(switches) % 2
        
        # Map 0/1 to min_val/max_val
        signal = jnp.where(state == 0, min_val, max_val)
        
        return signal[:, None]

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
        t = jnp.arange(n_steps) * dt
        
        # Linearly increasing frequency
        freq = min_freq + (max_freq - min_freq) * t / (t[-1] if t[-1] > 0 else 1.0)
        
        # Integrate frequency to get phase
        phase = jnp.cumsum(freq) * dt * 2 * jnp.pi
        
        # Generate chirp
        chirp = jnp.sin(phase)
        
        # Scale to [min_val, max_val]
        signal = min_val + (max_val - min_val) * (chirp + 1) / 2
        
        return signal[:, None]

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
        key_times, key_values = jax.random.split(key)
        
        # Generate random change times
        change_times = jnp.sort(
            jax.random.randint(key_times, shape=(n_changes,), minval=0, maxval=n_steps)
        )
        
        # Generate random values
        values = jax.random.uniform(
            key_values, shape=(n_changes + 1,), minval=min_val, maxval=max_val
        )
        
        # Create signal
        signal = jnp.zeros(n_steps)
        signal = signal.at[:change_times[0] if n_changes > 0 else n_steps].set(values[0])
        
        for i in range(n_changes - 1):
            signal = signal.at[change_times[i]:change_times[i+1]].set(values[i+1])
        
        if n_changes > 0:
            signal = signal.at[change_times[-1]:].set(values[-1])
        
        return signal[:, None]

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
        
        # F_in signal
        F_min, F_max = operating_ranges["F_in"]
        if signal_type == "mixed":
            signal_choice = jax.random.randint(key1, shape=(), minval=0, maxval=3)
            key1, subkey = jax.random.split(key1)
            if signal_choice == 0:
                F_in = self.generate_prbs_signal(subkey, n_steps, dt, F_min, F_max, switch_prob=0.05)
            elif signal_choice == 1:
                F_in = self.generate_chirp_signal(subkey, n_steps, dt, F_min, F_max)
            else:
                F_in = self.generate_multistep_signal(subkey, n_steps, dt, F_min, F_max, n_changes=5)
        elif signal_type == "prbs":
            F_in = self.generate_prbs_signal(key1, n_steps, dt, F_min, F_max)
        elif signal_type == "chirp":
            F_in = self.generate_chirp_signal(key1, n_steps, dt, F_min, F_max)
        else:
            F_in = self.generate_multistep_signal(key1, n_steps, dt, F_min, F_max)
        
        # Tc_in signal
        Tc_min, Tc_max = operating_ranges["Tc_in"]
        if signal_type == "mixed":
            signal_choice = jax.random.randint(key2, shape=(), minval=0, maxval=3)
            key2, subkey = jax.random.split(key2)
            if signal_choice == 0:
                Tc_in = self.generate_prbs_signal(subkey, n_steps, dt, Tc_min, Tc_max, switch_prob=0.05)
            elif signal_choice == 1:
                Tc_in = self.generate_chirp_signal(subkey, n_steps, dt, Tc_min, Tc_max)
            else:
                Tc_in = self.generate_multistep_signal(subkey, n_steps, dt, Tc_min, Tc_max, n_changes=5)
        elif signal_type == "prbs":
            Tc_in = self.generate_prbs_signal(key2, n_steps, dt, Tc_min, Tc_max)
        elif signal_type == "chirp":
            Tc_in = self.generate_chirp_signal(key2, n_steps, dt, Tc_min, Tc_max)
        else:
            Tc_in = self.generate_multistep_signal(key2, n_steps, dt, Tc_min, Tc_max)
        
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
        
        # Ca_in (slower variation)
        Ca_min, Ca_max = operating_ranges["Ca_in"]
        Ca_in = self.generate_prbs_signal(key1, n_steps, dt, Ca_min, Ca_max, switch_prob=0.01)
        
        # T_in (slower variation)
        T_min, T_max = operating_ranges["T_in"]
        T_in = self.generate_prbs_signal(key2, n_steps, dt, T_min, T_max, switch_prob=0.01)
        
        return jnp.concatenate([Ca_in, T_in], axis=1)

    def generate_single_trajectory(
        self,
        key: PRNGKeyArray,
        params: CSTRParams = None,
        n_steps: int = 1000,
    ) -> Dict[str, Array]:
        """Generate a single trajectory.
        
        Args:
            key: PRNG key
            params: CSTR parameters (if None, sample random)
            n_steps: Number of time steps
            
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
        control_traj = self.generate_control_trajectory(
            key_control, n_steps, self.dt, self.operating_ranges, signal_type="mixed"
        )
        disturbance_traj = self.generate_disturbance_trajectory(
            key_dist, n_steps, self.dt, self.operating_ranges
        )
        
        # Initial condition near steady state with noise
        ss_control = jnp.array([50.0, 300.0])
        ss_dist = jnp.array([1.0, 320.0])
        ss_state = simulator.steady_state(ss_control, ss_dist)
        
        # Add noise to initial condition
        noise = jax.random.normal(key_ic, shape=(4,)) * jnp.array([0.1, 0.1, 5.0, 5.0])
        initial_state = ss_state + noise
        
        # Simulate
        t_span = (0.0, n_steps * self.dt)
        result = simulator.simulate(
            initial_state, control_traj, disturbance_traj, t_span, self.dt, n_steps
        )
        
        # Add measurement noise
        noise_std = jnp.array([0.01, 0.01, 0.5, 0.5])  # 1% for conc, 0.5K for temp
        state_noise = jax.random.normal(key_ic, shape=result["states"].shape) * noise_std[None, :]
        noisy_states = result["states"] + state_noise
        
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
    ) -> Dict[str, Array]:
        """Generate full dataset.
        
        Args:
            key: PRNG key
            n_trajectories: Number of trajectories
            n_steps: Steps per trajectory
            
        Returns:
            Dictionary of stacked arrays
        """
        print(f"Generating {n_trajectories} trajectories...")
        
        keys = jax.random.split(key, n_trajectories * 2)  # Extra keys for retries
        
        # Pre-allocate arrays
        all_states = []
        all_controls = []
        all_disturbances = []
        all_params = []
        all_times = []
        
        # Generate trajectories with progress bar
        successful = 0
        attempt = 0
        pbar = tqdm(total=n_trajectories)
        
        while successful < n_trajectories and attempt < n_trajectories * 2:
            try:
                traj = self.generate_single_trajectory(keys[attempt], n_steps=n_steps)
                
                # Check for NaN or invalid values
                if (jnp.isnan(traj["states"]).any() or 
                    jnp.isinf(traj["states"]).any() or
                    (traj["states"][:, 0] < 0).any() or  # Ca must be positive
                    (traj["states"][:, 1] < -0.5).any()):  # Cb should be ~positive (small neg ok)
                    attempt += 1
                    continue
                
                all_states.append(traj["states"])
                all_controls.append(traj["controls"])
                all_disturbances.append(traj["disturbances"])
                all_params.append(traj["params"])
                all_times.append(traj["time"])
                
                successful += 1
                pbar.update(1)
            except Exception as e:
                print(f"\nError generating trajectory {attempt}: {e}")
                pass
            
            attempt += 1
        
        pbar.close()
        
        # Stack arrays
        all_states = jnp.stack(all_states)
        all_controls = jnp.stack(all_controls)
        all_disturbances = jnp.stack(all_disturbances)
        all_params = jnp.stack(all_params)
        all_times = jnp.stack(all_times)
        
        # Compute normalization statistics
        state_mean = jnp.mean(all_states.reshape(-1, 4), axis=0)
        state_std = jnp.std(all_states.reshape(-1, 4), axis=0)
        
        control_mean = jnp.mean(all_controls.reshape(-1, 2), axis=0)
        control_std = jnp.std(all_controls.reshape(-1, 2), axis=0)
        
        disturbance_mean = jnp.mean(all_disturbances.reshape(-1, 2), axis=0)
        disturbance_std = jnp.std(all_disturbances.reshape(-1, 2), axis=0)
        
        param_mean = jnp.mean(all_params, axis=0)
        param_std = jnp.std(all_params, axis=0)
        
        return {
            "time": all_times,
            "states": all_states,
            "controls": all_controls,
            "disturbances": all_disturbances,
            "params": all_params,
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
