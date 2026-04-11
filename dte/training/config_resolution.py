"""Helpers for explicit single-system training config resolution."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


LEGACY_SAFE_OVERRIDES: tuple[tuple[str, ...], Any] = (
    (("model", "initial_diffusion_scale"), 0.0001),
    (("model", "disturbance_dim"), 2),
    (("optimizer", "peak_lr"), 5.0e-4),
    (("optimizer", "gradient_clip"), 0.5),
    (("loss_weights", "kl"), 0.0),
)


def _ensure_path(root: dict[str, Any], path: tuple[str, ...]) -> dict[str, Any]:
    current = root
    for key in path[:-1]:
        current = current.setdefault(key, {})
    return current


def _get_value(root: dict[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = root
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def _set_value(root: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    parent = _ensure_path(root, path)
    parent[path[-1]] = value


def resolve_single_system_training_config(
    config: dict[str, Any],
    *,
    mode: str = "strict",
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Resolve a loaded train config into an explicit runtime config.

    Parameters
    ----------
    config:
        Loaded YAML config.
    mode:
        ``"strict"`` respects the YAML as written.
        ``"legacy_safe"`` reproduces the historic ``scripts/train.py``
        overrides, but does so explicitly and returns an audit trail.
    """
    resolved = deepcopy(config)
    applied: list[dict[str, Any]] = []

    if mode not in {"strict", "legacy_safe"}:
        raise ValueError(f"Unsupported config resolution mode '{mode}'.")

    if mode == "legacy_safe":
        for path, value in LEGACY_SAFE_OVERRIDES:
            old_value = _get_value(resolved, path)
            if old_value != value:
                _set_value(resolved, path, value)
                applied.append(
                    {
                        "path": ".".join(path),
                        "old_value": old_value,
                        "new_value": value,
                    }
                )

    return resolved, applied
