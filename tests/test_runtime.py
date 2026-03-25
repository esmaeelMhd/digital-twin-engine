"""Tests for CLI runtime configuration helpers."""

import os

from dte.utils.runtime import configure_runtime_logging


def test_configure_runtime_logging_sets_quiet_default(monkeypatch):
    """CLI scripts should suppress low-level XLA logs by default."""

    monkeypatch.delenv("DTE_SUPPRESS_XLA_LOGS", raising=False)
    monkeypatch.delenv("TF_CPP_MIN_LOG_LEVEL", raising=False)

    configure_runtime_logging()

    assert os.environ["TF_CPP_MIN_LOG_LEVEL"] == "3"


def test_configure_runtime_logging_respects_opt_out(monkeypatch):
    """Users can opt back into backend logs explicitly."""

    monkeypatch.setenv("DTE_SUPPRESS_XLA_LOGS", "0")
    monkeypatch.delenv("TF_CPP_MIN_LOG_LEVEL", raising=False)

    configure_runtime_logging()

    assert "TF_CPP_MIN_LOG_LEVEL" not in os.environ


def test_configure_runtime_logging_preserves_existing_level(monkeypatch):
    """Explicit log-level overrides should win over the default helper."""

    monkeypatch.delenv("DTE_SUPPRESS_XLA_LOGS", raising=False)
    monkeypatch.setenv("TF_CPP_MIN_LOG_LEVEL", "1")

    configure_runtime_logging()

    assert os.environ["TF_CPP_MIN_LOG_LEVEL"] == "1"
