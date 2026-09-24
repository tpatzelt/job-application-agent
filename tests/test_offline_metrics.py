"""Tests for the offline G1 metrics (evals/metrics.py) and the
score_page/check_result split in evals/checks.py.

Each metric test is built so a plausible-but-wrong implementation would
disagree with the asserted 0.5 (see inline notes), and every metric is
checked to return None -- not 0.0, not a ZeroDivisionError -- when its
denominator is empty.
"""

from __future__ import annotations

from typing import Any

import pytest

from evals.checks import check_result, score_page
from evals.metrics import (
    aggregator_drop_rate,
    dedup_rate,
    location_match_rate,
    posting_shape_rate,
    staleness_detection_rate,
)


def _r(url: str, **fields: Any) -> dict[str, Any]:
    return {"url": url, **fields}


def test_posting_shape_rate_is_precision_over_kept_records() -> None:
    kept = [_r("k1", kind="posting"), _r("k2", kind="posting"),
            _r("k3", kind="listing"), _r("k4", kind="other")]
    # Extra dropped postings: dividing by len(all) instead of len(kept)
    # would give 8/10 = 0.8, not 0.5.
    all_records = kept + [_r(f"d{i}", kind="posting") for i in range(6)]

    assert posting_shape_rate(kept, all_records) == 0.5


def test_aggregator_drop_rate_requires_aggregator_and_index() -> None:
    all_records = [
        _r("u1", aggregator=True, kind="index"),  # dropped -> counts
        _r("u2", aggregator=True, kind="index"),  # dropped -> counts
        _r("u3", aggregator=True, kind="index"),  # kept -> not counted
        _r("u4", aggregator=True, kind="index"),  # kept -> not counted
        # Dropped but wrong kind/aggregator: ignoring either field would
        # inflate this to 3/5 = 0.6 instead of 0.5.
        _r("u5", aggregator=True, kind="posting"),
        _r("u6", aggregator=False, kind="index"),
    ]
    kept = [_r("u3"), _r("u4")]

    assert aggregator_drop_rate(kept, all_records) == 0.5


def test_location_match_rate_calls_mentions_location_not_the_label() -> None:
    locs = ["Berlin, Germany"]
    # location_ok labels are deliberately wrong: reading the stored label
    # instead of calling mentions_location would score this 1/4 = 0.25.
    kept = [
        _r("a", text="We are hiring in Berlin, Germany.", locations=locs, location_ok=False),
        _r("b", text="Fully remote, no city mentioned.", locations=locs, location_ok=True),
        _r("c", text="Our office is in Munich.", locations=locs, location_ok=False),
        _r("d", text="Join our Berlin based team.", locations=locs, location_ok=False),
    ]

    assert location_match_rate(kept, kept) == 0.5


def test_staleness_detection_rate_over_stale_labelled_records_only() -> None:
    all_records = [
        _r("s1", stale=True), _r("s2", stale=True),  # dropped -> count
        _r("s3", stale=True), _r("s4", stale=True),  # kept -> not counted
        # Non-stale but also dropped: using "all dropped" as the
        # denominator would give 2/3 = 0.667, not 0.5.
        _r("n1", stale=False), _r("n2", stale=False),
    ]
    kept = [_r("s3"), _r("s4"), _r("n2")]

    assert staleness_detection_rate(kept, all_records) == 0.5


def test_dedup_rate_uses_canonical_url_not_exact_match() -> None:
    all_records = [
        # utm_source variant: canonical_url strips it, exact-match wouldn't.
        _r("https://boards.greenhouse.io/acme/jobs/123?utm_source=x",
           duplicate_of="https://boards.greenhouse.io/acme/jobs/123"),
        # /apply suffix variant of the same posting.
        _r("https://jobs.lever.co/acme/abc/apply", duplicate_of="https://jobs.lever.co/acme/abc"),
        # Genuinely different job IDs: must NOT collapse.
        _r("https://boards.greenhouse.io/acme/jobs/999",
           duplicate_of="https://boards.greenhouse.io/acme/jobs/111"),
        _r("https://boards.greenhouse.io/other/jobs/1",
           duplicate_of="https://boards.greenhouse.io/other/jobs/2"),
        # Not labelled as a duplicate: excluded from the denominator.
        _r("https://boards.greenhouse.io/acme/jobs/555", duplicate_of=None),
    ]

    assert dedup_rate([], all_records) == 0.5


@pytest.mark.parametrize(
    "metric,kept,all_records",
    [
        (posting_shape_rate, [], [_r("a", kind="posting")]),
        (aggregator_drop_rate, [], [_r("a", aggregator=False, kind="posting")]),
        (location_match_rate, [], [_r("a", text="Berlin", locations=["Berlin"])]),
        (staleness_detection_rate, [], [_r("a", stale=False)]),
        (dedup_rate, [], [_r("a", duplicate_of=None)]),
    ],
)
def test_metric_is_none_when_denominator_empty(metric, kept, all_records) -> None:
    assert metric(kept, all_records) is None


def test_score_page_is_pure_and_scores_expected_fields() -> None:
    text = "x" * 400 + " Berlin office, apply today."
    result = score_page(
        url="https://boards.greenhouse.io/acme/jobs/123",
        final_url="https://boards.greenhouse.io/acme/jobs/123",
        status=200,
        text=text,
        locations=["Berlin, Germany"],
        industries=["Software"],
    )

    assert result["live"] is True
    assert result["fresh"] is True
    assert result["location_ok"] is True
    assert result["posting"] is True
    assert "error" not in result


def test_check_result_delegates_to_score_page(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeResponse:
        status_code = 200
        url = "https://boards.greenhouse.io/acme/jobs/123"
        text = "<html><body>" + "x" * 400 + " Berlin office</body></html>"

    monkeypatch.setattr("evals.checks.requests.get", lambda *a, **k: FakeResponse())

    result = check_result(
        "https://boards.greenhouse.io/acme/jobs/123", locations=["Berlin, Germany"]
    )

    assert result["error"] is None
    assert result["live"] is True
    assert result["posting"] is True
    assert result["location_ok"] is True


def test_check_result_sets_error_on_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_get(*args: Any, **kwargs: Any) -> Any:
        raise ConnectionError("boom")

    monkeypatch.setattr("evals.checks.requests.get", fake_get)

    result = check_result("https://example.com/gone", locations=[])

    assert result["error"] == "boom"
    assert result["live"] is False
