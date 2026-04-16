"""Canonical single-system trainer import path."""

from dte.training.trainer import (
    Trainer,
    _format_non_finite_reason,
    _non_finite_loss_names,
)

__all__ = [
    "Trainer",
    "_format_non_finite_reason",
    "_non_finite_loss_names",
]
