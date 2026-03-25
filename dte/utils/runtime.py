"""Runtime helpers for CLI entrypoints."""

from __future__ import annotations

import os


def configure_runtime_logging() -> None:
    """Reduce noisy low-level JAX/XLA logs for CLI workflows.

    Set ``DTE_SUPPRESS_XLA_LOGS=0`` to opt out and keep the default backend logs.
    Existing log-level configuration is preserved.
    """

    if os.environ.get("DTE_SUPPRESS_XLA_LOGS", "1") == "0":
        return

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
