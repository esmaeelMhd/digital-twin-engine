"""Tests for trainer failure detection helpers."""

from dte.training.trainer import _format_non_finite_reason, _non_finite_loss_names
from scripts.train import _json_safe_float


def test_non_finite_loss_names_reports_nan_and_inf():
    """NaN and Inf losses should be detected explicitly."""

    losses = {
        "total": float("nan"),
        "reconstruction": 1.0,
        "kl": float("inf"),
    }

    assert _non_finite_loss_names(losses) == ["total", "kl"]


def test_format_non_finite_reason_includes_stage_and_location():
    """Failure reasons should include enough context for debugging."""

    losses = {
        "total": float("nan"),
        "trajectory": float("inf"),
    }

    reason = _format_non_finite_reason(
        "val",
        losses,
        step=42,
        batch_index=3,
        n_batches=7,
        epoch=5,
    )

    assert "non_finite_val_loss" in reason
    assert "epoch=5" in reason
    assert "step=42" in reason
    assert "batch=3/7" in reason
    assert "total=nan" in reason
    assert "trajectory=inf" in reason


def test_json_safe_float_drops_non_finite_values():
    """Training summaries should not emit NaN or Inf into JSON."""

    assert _json_safe_float(1.25) == 1.25
    assert _json_safe_float(float("nan")) is None
    assert _json_safe_float(float("inf")) is None
    assert _json_safe_float(None) is None
