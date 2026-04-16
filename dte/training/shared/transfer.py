"""Canonical transfer-learning import path."""

from dte.training.transfer import (
    FewShotAdapter,
    FinetunePartType,
    apply_finetune_mask,
    zero_shot_eval,
)

__all__ = [
    "FewShotAdapter",
    "FinetunePartType",
    "apply_finetune_mask",
    "zero_shot_eval",
]
