"""Compatibility helpers for dataset generation I/O."""

from typing import Dict

import h5py
import jax.numpy as jnp
from jaxtyping import Array


def load_dataset(path: str) -> Dict[str, Array]:
    """Load a generated HDF5 dataset.

    The generation pipeline now lives in ``dte.data.generation_generic``. This
    module remains only as a stable import path for HDF5 dataset loading.
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
            },
        }

    return dataset
