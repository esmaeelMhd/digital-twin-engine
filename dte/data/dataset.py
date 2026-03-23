"""Dataset class for trajectory data."""

from typing import Dict, Union
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray


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
            from dte.data.generation import DataGenerator
            data = DataGenerator.load_dataset(data)
        
        self.data = data
        self.seq_len = seq_len
        self.stride = stride
        
        # Extract subsequences
        self._extract_subsequences()
        
    def _extract_subsequences(self):
        """Extract subsequences of length seq_len with given stride."""
        n_trajectories, n_steps_per_traj, _ = self.data["states"].shape
        
        # Calculate number of subsequences per trajectory
        n_subseqs_per_traj = (n_steps_per_traj - self.seq_len) // self.stride + 1
        
        # Pre-allocate
        total_subseqs = n_trajectories * n_subseqs_per_traj
        self.subsequences = {
            "states": jnp.zeros((total_subseqs, self.seq_len, 4)),
            "controls": jnp.zeros((total_subseqs, self.seq_len, 2)),
            "disturbances": jnp.zeros((total_subseqs, self.seq_len, 2)),
            "params": jnp.zeros((total_subseqs, 6)),
            "t": jnp.zeros((total_subseqs, self.seq_len)),
        }
        
        idx = 0
        for traj_idx in range(n_trajectories):
            for start_idx in range(0, n_steps_per_traj - self.seq_len + 1, self.stride):
                end_idx = start_idx + self.seq_len
                
                self.subsequences["states"] = self.subsequences["states"].at[idx].set(
                    self.data["states"][traj_idx, start_idx:end_idx]
                )
                self.subsequences["controls"] = self.subsequences["controls"].at[idx].set(
                    self.data["controls"][traj_idx, start_idx:end_idx]
                )
                self.subsequences["disturbances"] = self.subsequences["disturbances"].at[idx].set(
                    self.data["disturbances"][traj_idx, start_idx:end_idx]
                )
                self.subsequences["params"] = self.subsequences["params"].at[idx].set(
                    self.data["params"][traj_idx]
                )
                self.subsequences["t"] = self.subsequences["t"].at[idx].set(
                    self.data["time"][traj_idx, start_idx:end_idx]
                )
                
                idx += 1
        
        self._n_samples = idx
        print(f"Extracted {self._n_samples} subsequences of length {self.seq_len}")

    @property
    def n_samples(self) -> int:
        """Number of samples in dataset."""
        return self._n_samples

    @property
    def state_dim(self) -> int:
        """State dimension."""
        return 4

    @property
    def control_dim(self) -> int:
        """Control dimension."""
        return 2

    @property
    def disturbance_dim(self) -> int:
        """Disturbance dimension."""
        return 2

    @property
    def param_dim(self) -> int:
        """Parameter dimension."""
        return 6

    def __getitem__(self, idx: int) -> Dict[str, Array]:
        """Get a single sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary with states, controls, disturbances, params, t
        """
        return {
            "states": self.subsequences["states"][idx],
            "controls": self.subsequences["controls"][idx],
            "disturbances": self.subsequences["disturbances"][idx],
            "params": self.subsequences["params"][idx],
            "t": self.subsequences["t"][idx],
        }

    def sample_batch(
        self, key: PRNGKeyArray, batch_size: int
    ) -> Dict[str, Array]:
        """Sample a random batch.
        
        Args:
            key: PRNG key
            batch_size: Batch size
            
        Returns:
            Dictionary with batched arrays
        """
        indices = jax.random.choice(
            key, self._n_samples, shape=(batch_size,), replace=False
        )
        
        return {
            "states": self.subsequences["states"][indices],
            "controls": self.subsequences["controls"][indices],
            "disturbances": self.subsequences["disturbances"][indices],
            "params": self.subsequences["params"][indices],
            "t": self.subsequences["t"][indices],
        }

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
