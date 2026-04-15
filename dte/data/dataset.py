"""Dataset class for trajectory data."""

from typing import Dict, Union
import numpy as np
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from dte.data.generation import load_dataset


class TrajectoryDataset:
    """JAX-compatible dataset for trajectory data."""

    def __init__(
        self,
        data: Union[str, Dict[str, Array]],
        seq_len: int = 50,
        stride: int = 10,
    ):
        """Initialize trajectory dataset.
        
        Args:
            data: Path to HDF5 file or dictionary with data
            seq_len: Length of subsequences to extract
            stride: Stride between subsequences
        """
        # Load data if path provided
        if isinstance(data, str):
            data = load_dataset(data)
        
        self.data = data
        self.requested_seq_len = seq_len
        self.seq_len = seq_len
        self.stride = stride
        self._ensure_normalization_contract()
        
        # Extract subsequences
        self._extract_subsequences()

    def _ensure_normalization_contract(self) -> None:
        """Backfill optional normalization statistics expected by older helpers.

        Older generated and ingested datasets omitted parameter normalization
        statistics even though ``normalize_params`` and ``denormalize_params``
        expect them. Populate those fields lazily from the stored parameter
        matrix so existing HDF5 files remain loadable.
        """
        normalization = dict(self.data.get("normalization", {}))
        params = np.asarray(self.data["params"])
        param_dim = params.shape[-1]

        if "param_mean" not in normalization:
            normalization["param_mean"] = jnp.asarray(
                params.mean(axis=0),
                dtype=jnp.float32,
            )
        if "param_std" not in normalization:
            normalization["param_std"] = jnp.asarray(
                np.clip(params.std(axis=0), 1e-8, None),
                dtype=jnp.float32,
            )

        if param_dim == 0:
            normalization["param_mean"] = jnp.zeros((0,), dtype=jnp.float32)
            normalization["param_std"] = jnp.ones((0,), dtype=jnp.float32)

        self.data["normalization"] = normalization
        
    def _extract_subsequences(self):
        """Extract subsequences of length seq_len with given stride."""
        states = np.asarray(self.data["states"])
        controls = np.asarray(self.data["controls"])
        disturbances = np.asarray(self.data["disturbances"])
        params = np.asarray(self.data["params"])
        time = np.asarray(self.data["time"])

        n_trajectories, n_steps_per_traj, _ = states.shape
        effective_seq_len = min(self.requested_seq_len, n_steps_per_traj)
        if effective_seq_len < 1:
            raise ValueError("TrajectoryDataset requires at least one timestep per trajectory.")
        self.seq_len = effective_seq_len
        
        # Calculate number of subsequences per trajectory
        n_subseqs_per_traj = (n_steps_per_traj - self.seq_len) // self.stride + 1
        total_subseqs = n_trajectories * n_subseqs_per_traj
        start_indices = range(0, n_steps_per_traj - self.seq_len + 1, self.stride)

        def extract_sequence_windows(array: np.ndarray) -> np.ndarray:
            windows = np.stack(
                [array[:, start_idx:start_idx + self.seq_len] for start_idx in start_indices],
                axis=1,
            )
            return windows.reshape(total_subseqs, self.seq_len, array.shape[-1])

        time_windows = np.stack(
            [time[:, start_idx:start_idx + self.seq_len] for start_idx in start_indices],
            axis=1,
        ).reshape(total_subseqs, self.seq_len)

        self.subsequences = {
            "states": extract_sequence_windows(states),
            "controls": extract_sequence_windows(controls),
            "disturbances": extract_sequence_windows(disturbances),
            "params": np.repeat(params, n_subseqs_per_traj, axis=0),
            "t": time_windows,
        }

        self._n_samples = total_subseqs
        print(f"Extracted {self._n_samples} subsequences of length {self.seq_len}")

    @property
    def n_samples(self) -> int:
        """Number of samples in dataset."""
        return self._n_samples

    @property
    def state_dim(self) -> int:
        """State dimension."""
        return int(self.data["states"].shape[-1])

    @property
    def control_dim(self) -> int:
        """Control dimension."""
        return int(self.data["controls"].shape[-1])

    @property
    def disturbance_dim(self) -> int:
        """Disturbance dimension."""
        return int(self.data["disturbances"].shape[-1])

    @property
    def param_dim(self) -> int:
        """Parameter dimension."""
        return int(self.data["params"].shape[-1])

    def __getitem__(self, idx: int) -> Dict[str, Array]:
        """Get a single sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary with states, controls, disturbances, params, t
        """
        return {
            "states": jnp.asarray(self.subsequences["states"][idx]),
            "controls": jnp.asarray(self.subsequences["controls"][idx]),
            "disturbances": jnp.asarray(self.subsequences["disturbances"][idx]),
            "params": jnp.asarray(self.subsequences["params"][idx]),
            "t": jnp.asarray(self.subsequences["t"][idx]),
        }

    def sample_batch(
        self, key: PRNGKeyArray, batch_size: int, seq_len: int = None
    ) -> Dict[str, Array]:
        """Sample a random batch.

        Args:
            key: PRNG key
            batch_size: Batch size
            seq_len: Optional sequence length override for curriculum training.
                When provided, each sampled subsequence is truncated (or the
                dataset's native seq_len is used if seq_len is longer).

        Returns:
            Dictionary with batched arrays
        """
        # Smoke-size validation splits can be smaller than the requested mixed
        # batch share for a given system. Fall back to replacement only in that
        # case so larger datasets still sample without duplicates by default.
        replace = batch_size > self._n_samples
        indices = np.asarray(
            jax.random.choice(key, self._n_samples, shape=(batch_size,), replace=replace)
        )

        batch = {
            "states": jnp.asarray(self.subsequences["states"][indices]),
            "controls": jnp.asarray(self.subsequences["controls"][indices]),
            "disturbances": jnp.asarray(self.subsequences["disturbances"][indices]),
            "params": jnp.asarray(self.subsequences["params"][indices]),
            "t": jnp.asarray(self.subsequences["t"][indices]),
        }

        if seq_len is not None:
            native_len = batch["states"].shape[1]
            effective_len = min(seq_len, native_len)
            for name in ("states", "controls", "disturbances", "t"):
                batch[name] = batch[name][:, :effective_len]

        return batch

    def get_normalization_stats(self) -> Dict[str, Array]:
        """Get normalization statistics.
        
        Returns:
            Dictionary with mean and std for each variable type
        """
        return self.data["normalization"]

    def normalize_states(self, states: Array) -> Array:
        """Normalize states."""
        stats = self.get_normalization_stats()
        return (states - stats["state_mean"]) / (stats["state_std"] + 1e-8)

    def denormalize_states(self, states: Array) -> Array:
        """Denormalize states."""
        stats = self.get_normalization_stats()
        return states * (stats["state_std"] + 1e-8) + stats["state_mean"]

    def normalize_controls(self, controls: Array) -> Array:
        """Normalize controls."""
        stats = self.get_normalization_stats()
        return (controls - stats["control_mean"]) / (stats["control_std"] + 1e-8)

    def denormalize_controls(self, controls: Array) -> Array:
        """Denormalize controls."""
        stats = self.get_normalization_stats()
        return controls * (stats["control_std"] + 1e-8) + stats["control_mean"]

    def normalize_disturbances(self, disturbances: Array) -> Array:
        """Normalize disturbances."""
        stats = self.get_normalization_stats()
        return (disturbances - stats["disturbance_mean"]) / (stats["disturbance_std"] + 1e-8)

    def denormalize_disturbances(self, disturbances: Array) -> Array:
        """Denormalize disturbances."""
        stats = self.get_normalization_stats()
        return disturbances * (stats["disturbance_std"] + 1e-8) + stats["disturbance_mean"]

    def normalize_params(self, params: Array) -> Array:
        """Normalize params."""
        stats = self.get_normalization_stats()
        return (params - stats["param_mean"]) / (stats["param_std"] + 1e-8)

    def denormalize_params(self, params: Array) -> Array:
        """Denormalize params."""
        stats = self.get_normalization_stats()
        return params * (stats["param_std"] + 1e-8) + stats["param_mean"]

    def split(self, val_fraction: float = 0.2) -> tuple["TrajectoryDataset", "TrajectoryDataset"]:
        """Split dataset into train and validation.
        
        Args:
            val_fraction: Fraction for validation set
            
        Returns:
            Tuple of (train_dataset, val_dataset)
        """
        n_trajectories = self.data["states"].shape[0]
        if n_trajectories < 2:
            raise ValueError("Need at least two trajectories to create a train/validation split.")

        n_val = int(n_trajectories * val_fraction)
        n_val = max(1, min(n_val, n_trajectories - 1))
        n_train = n_trajectories - n_val
        
        # Create new datasets with split data
        train_data = {
            "states": self.data["states"][:n_train],
            "controls": self.data["controls"][:n_train],
            "disturbances": self.data["disturbances"][:n_train],
            "params": self.data["params"][:n_train],
            "time": self.data["time"][:n_train],
            "normalization": self.data["normalization"],
        }
        
        val_data = {
            "states": self.data["states"][n_train:],
            "controls": self.data["controls"][n_train:],
            "disturbances": self.data["disturbances"][n_train:],
            "params": self.data["params"][n_train:],
            "time": self.data["time"][n_train:],
            "normalization": self.data["normalization"],
        }
        
        train_dataset = TrajectoryDataset(train_data, self.seq_len, self.stride)
        val_dataset = TrajectoryDataset(val_data, self.seq_len, self.stride)
        
        return train_dataset, val_dataset
