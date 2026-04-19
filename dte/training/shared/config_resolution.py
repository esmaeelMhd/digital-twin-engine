"""Helpers for canonical single-system training config resolution."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


def resolve_single_system_training_config(
    config: dict[str, Any],
) -> dict[str, Any]:
    """Return the canonical runtime config for single-system training.

    The active repo no longer carries legacy compatibility override modes. The
    YAML is treated as the source of truth and copied before runtime overrides
    such as CLI epoch or batch-size arguments are applied.
    """

    return deepcopy(config)
