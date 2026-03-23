"""Tests for the autoresearch workflow helpers."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from dte.autoresearch.workflow import (
    RESULTS_COLUMNS,
    append_result_row,
    ensure_results_file,
    load_baseline_state,
    make_run_id,
    metric_improved,
    promote_baseline,
    read_json,
    slugify,
    write_json,
)


def test_slugify_and_run_id_are_stable():
    """Freeform descriptions should become predictable run ids."""

    now = datetime(2026, 3, 23, 14, 5, 7)
    assert slugify("LR x2 + Wider Decoder") == "lr-x2-wider-decoder"
    assert make_run_id("LR x2 + Wider Decoder", now=now) == (
        "20260323-140507-lr-x2-wider-decoder"
    )


def test_metric_improved_supports_min_and_max_modes():
    """Metric comparison should respect directionality."""

    assert metric_improved(0.05, None, mode="min")
    assert metric_improved(0.04, 0.05, mode="min")
    assert not metric_improved(0.05, 0.05, mode="min")
    assert metric_improved(0.91, 0.90, mode="max")
    assert not metric_improved(0.90, 0.91, mode="max")


def test_results_file_gets_header_and_rows(tmp_path: Path):
    """Results logging should create a TSV header and append rows."""

    results_path = tmp_path / "results.tsv"
    ensure_results_file(results_path)
    append_result_row(
        results_path,
        {
            "timestamp": "20260323",
            "run_id": "20260323-140507-baseline",
            "commit": "abc1234",
            "metric_name": "best_val_loss",
            "metric_value": "0.123456",
            "baseline_before": "",
            "training_seconds": "300.00",
            "status": "keep",
            "description": "baseline",
        },
    )

    lines = results_path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "\t".join(RESULTS_COLUMNS)
    assert lines[1].startswith("20260323\t20260323-140507-baseline\tabc1234")


def test_promote_baseline_copies_artifacts_and_metadata(tmp_path: Path):
    """Promoting a run should snapshot its artifacts and metadata."""

    run_dir = tmp_path / "runs" / "20260323-140507-baseline"
    artifacts_dir = run_dir / "artifacts"
    artifacts_dir.mkdir(parents=True)
    (artifacts_dir / "best_model.eqx").write_text("model", encoding="utf-8")
    (run_dir / "train.log").write_text("ok", encoding="utf-8")

    summary = {
        "best_val_loss": 0.123456,
        "training_seconds": 301.2,
    }
    write_json(run_dir / "summary.json", summary)

    baseline_dir = tmp_path / "baseline"
    metadata = promote_baseline(
        run_dir=run_dir,
        baseline_dir=baseline_dir,
        summary=summary,
        metric_name="best_val_loss",
        description="baseline",
        commit="abc1234",
    )

    assert metadata["metric_value"] == 0.123456
    assert (baseline_dir / "artifacts" / "best_model.eqx").read_text(encoding="utf-8") == "model"
    assert read_json(baseline_dir / "summary.json")["best_val_loss"] == 0.123456

    baseline_state = load_baseline_state(baseline_dir, "best_val_loss")
    assert baseline_state is not None
    assert baseline_state.metric_value == 0.123456
    assert baseline_state.commit == "abc1234"
