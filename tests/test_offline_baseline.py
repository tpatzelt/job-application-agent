"""Tests for the --check-baseline regression gate in evals/offline_eval.py.

The gate recomputes the G1 metrics over the real fixture corpus and compares
them against a committed baseline.json. These tests use a stub baseline file
built from the real, currently-computed totals so they stay deterministic
and don't depend on the exact numbers checked into evals/baseline.json.
"""

from __future__ import annotations

import json
from pathlib import Path

from evals.offline_eval import DEFAULT_CORPUS_DIR, check_baseline, compare_to_baseline, main, run


def _write_baseline(path: Path, metrics: dict[str, float | None]) -> None:
    path.write_text(
        json.dumps({"totals": {"records": 0, "kept": 0, "metrics": metrics}}),
        encoding="utf-8",
    )


def test_compare_to_baseline_all_equal_passes() -> None:
    current = {
        "posting_shape_rate": 0.75, "aggregator_drop_rate": 1.0, "location_match_rate": 1.0,
        "staleness_detection_rate": 1.0, "dedup_rate": 0.75,
    }
    ok, rows = compare_to_baseline(current, dict(current))
    assert ok is True
    assert all(not r["regressed"] for r in rows)


def test_compare_to_baseline_flags_regressed_metric_by_name() -> None:
    current = {
        "posting_shape_rate": 0.75, "aggregator_drop_rate": 1.0, "location_match_rate": 1.0,
        "staleness_detection_rate": 1.0, "dedup_rate": 0.75,
    }
    baseline = dict(current)
    baseline["dedup_rate"] = 0.9  # current dropped below this
    ok, rows = compare_to_baseline(current, baseline)
    assert ok is False
    regressed = [r["metric"] for r in rows if r["regressed"]]
    assert regressed == ["dedup_rate"]


def test_compare_to_baseline_within_tolerance_is_not_a_regression() -> None:
    current = {"posting_shape_rate": 0.75}
    baseline = {"posting_shape_rate": 0.75 + 1e-12}
    ok, rows = compare_to_baseline(current, baseline)
    assert ok is True
    assert rows[0]["regressed"] is False


def test_compare_to_baseline_missing_current_metric_is_a_regression() -> None:
    current: dict[str, float | None] = {"posting_shape_rate": None}
    baseline = {"posting_shape_rate": 0.75}
    ok, rows = compare_to_baseline(current, baseline)
    assert ok is False
    assert rows[0]["regressed"] is True


def test_check_baseline_passes_when_baseline_matches_current(tmp_path: Path) -> None:
    current_totals = run(DEFAULT_CORPUS_DIR)["totals"]["metrics"]
    baseline_path = tmp_path / "baseline.json"
    _write_baseline(baseline_path, current_totals)

    ok, message = check_baseline(DEFAULT_CORPUS_DIR, baseline_path)

    assert ok is True
    assert "No metric regressed" in message


def test_check_baseline_fails_and_names_regressed_metric(tmp_path: Path) -> None:
    current_totals = run(DEFAULT_CORPUS_DIR)["totals"]["metrics"]
    inflated = dict(current_totals)
    inflated["posting_shape_rate"] = 1.0  # higher than what the corpus actually yields
    baseline_path = tmp_path / "baseline.json"
    _write_baseline(baseline_path, inflated)

    ok, message = check_baseline(DEFAULT_CORPUS_DIR, baseline_path)

    assert ok is False
    assert "posting_shape_rate" in message


def test_main_check_baseline_exit_code_zero_on_committed_baseline() -> None:
    assert main(["--check-baseline"]) == 0


def test_main_check_baseline_exit_code_one_on_regression(tmp_path: Path) -> None:
    current_totals = run(DEFAULT_CORPUS_DIR)["totals"]["metrics"]
    inflated = dict(current_totals)
    inflated["dedup_rate"] = 1.0
    baseline_path = tmp_path / "baseline.json"
    _write_baseline(baseline_path, inflated)

    assert main(["--check-baseline", "--baseline", str(baseline_path)]) == 1
