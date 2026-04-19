"""Tests for CLI runtime configuration helpers."""

import os

from dte.utils.runtime import configure_runtime_logging, runtime_env_defaults


def test_configure_runtime_logging_sets_quiet_default(monkeypatch):
    """CLI scripts should suppress low-level XLA logs by default."""

    monkeypatch.delenv("DTE_SUPPRESS_XLA_LOGS", raising=False)
    monkeypatch.delenv("TF_CPP_MIN_LOG_LEVEL", raising=False)
    monkeypatch.delenv("GLOG_minloglevel", raising=False)
    monkeypatch.delenv("ABSL_MIN_LOG_LEVEL", raising=False)
    monkeypatch.delenv("JAX_LOG_COMPILES", raising=False)

    configure_runtime_logging()

    assert os.environ["TF_CPP_MIN_LOG_LEVEL"] == "3"
    assert os.environ["GLOG_minloglevel"] == "3"
    assert os.environ["ABSL_MIN_LOG_LEVEL"] == "3"
    assert os.environ["JAX_LOG_COMPILES"] == "0"


def test_configure_runtime_logging_respects_opt_out(monkeypatch):
    """Users can opt back into backend logs explicitly."""

    monkeypatch.setenv("DTE_SUPPRESS_XLA_LOGS", "0")
    monkeypatch.delenv("TF_CPP_MIN_LOG_LEVEL", raising=False)
    monkeypatch.delenv("GLOG_minloglevel", raising=False)
    monkeypatch.delenv("ABSL_MIN_LOG_LEVEL", raising=False)
    monkeypatch.delenv("JAX_LOG_COMPILES", raising=False)

    configure_runtime_logging()

    assert "TF_CPP_MIN_LOG_LEVEL" not in os.environ
    assert "GLOG_minloglevel" not in os.environ
    assert "ABSL_MIN_LOG_LEVEL" not in os.environ
    assert "JAX_LOG_COMPILES" not in os.environ


def test_configure_runtime_logging_preserves_existing_level(monkeypatch):
    """Explicit log-level overrides should win over the default helper."""

    monkeypatch.delenv("DTE_SUPPRESS_XLA_LOGS", raising=False)
    monkeypatch.setenv("TF_CPP_MIN_LOG_LEVEL", "1")
    monkeypatch.setenv("GLOG_minloglevel", "1")
    monkeypatch.setenv("ABSL_MIN_LOG_LEVEL", "1")
    monkeypatch.setenv("JAX_LOG_COMPILES", "1")

    configure_runtime_logging()

    assert os.environ["TF_CPP_MIN_LOG_LEVEL"] == "1"
    assert os.environ["GLOG_minloglevel"] == "1"
    assert os.environ["ABSL_MIN_LOG_LEVEL"] == "1"
    assert os.environ["JAX_LOG_COMPILES"] == "1"


def test_runtime_env_defaults_only_returns_missing_values(monkeypatch):
    """Subprocess helpers should inherit user overrides and fill in quiet defaults."""

    monkeypatch.delenv("DTE_SUPPRESS_XLA_LOGS", raising=False)
    monkeypatch.setenv("TF_CPP_MIN_LOG_LEVEL", "2")
    monkeypatch.delenv("GLOG_minloglevel", raising=False)
    monkeypatch.delenv("ABSL_MIN_LOG_LEVEL", raising=False)
    monkeypatch.delenv("JAX_LOG_COMPILES", raising=False)

    defaults = runtime_env_defaults()

    assert "TF_CPP_MIN_LOG_LEVEL" not in defaults
    assert defaults["GLOG_minloglevel"] == "3"
    assert defaults["ABSL_MIN_LOG_LEVEL"] == "3"
    assert defaults["JAX_LOG_COMPILES"] == "0"
