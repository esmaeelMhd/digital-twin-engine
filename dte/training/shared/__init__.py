"""Canonical shared training import paths."""

from dte.training.shared.config_resolution import resolve_single_system_training_config
from dte.training.shared.losses import LossComputer
from dte.training.shared.transfer import (
    FewShotAdapter,
    FinetunePartType,
    build_finetune_filter_spec,
    zero_shot_eval,
)

__all__ = [
    "FewShotAdapter",
    "FinetunePartType",
    "LossComputer",
    "build_finetune_filter_spec",
    "resolve_single_system_training_config",
    "zero_shot_eval",
]
