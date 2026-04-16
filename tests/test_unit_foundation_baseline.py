"""Tests for the canonical unit-foundation baseline runner."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_unit_foundation_baseline_dry_run_writes_summary(tmp_path: Path):
    workspace_dir = tmp_path / "baseline"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_unit_foundation_baseline.py",
            "--workspace_dir",
            str(workspace_dir),
            "--dry_run",
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    summary_path = workspace_dir / "summary.json"
    assert summary_path.exists()

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["status"] == "dry_run"
    assert [step["name"] for step in summary["steps"]] == [
        "generate_regime_corpus",
        "train_unit_foundation",
        "evaluate_unit_foundation",
        "control_gate",
    ]
