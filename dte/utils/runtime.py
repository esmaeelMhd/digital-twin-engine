"""Runtime helpers for CLI entrypoints."""

from __future__ import annotations

import os


_QUIET_RUNTIME_DEFAULTS = {
    "TF_CPP_MIN_LOG_LEVEL": "3",
    "GLOG_minloglevel": "3",
    "ABSL_MIN_LOG_LEVEL": "3",
    "JAX_LOG_COMPILES": "0",
}


def runtime_env_defaults() -> dict[str, str]:
    """Return default environment overrides for quiet CLI/runtime execution.

    Set ``DTE_SUPPRESS_XLA_LOGS=0`` to opt out and keep backend compiler logs.
    Existing environment overrides always win over these defaults.
    """

    if os.environ.get("DTE_SUPPRESS_XLA_LOGS", "1") == "0":
        return {}

    return {
        key: value
        for key, value in _QUIET_RUNTIME_DEFAULTS.items()
        if key not in os.environ
    }


def configure_runtime_logging() -> None:
    """Reduce noisy low-level JAX/XLA logs for CLI workflows.

    Set ``DTE_SUPPRESS_XLA_LOGS=0`` to opt out and keep the default backend logs.
    Existing log-level configuration is preserved.
    """

    for key, value in runtime_env_defaults().items():
        os.environ.setdefault(key, value)
