"""Pure per-metric scoring for the G1 offline result-quality harness.

Each function has the same signature — `(kept_records, all_records)` —
and returns a rate in `[0, 1]`, or `None` when its denominator is empty
(never `0.0`, never a `ZeroDivisionError`, so an empty corpus reads as
"not measured" rather than "failed").

A record is a plain dict merging a corpus entry's ground-truth label
(`kind`, `aggregator`, `stale`, `duplicate_of` — see `evals.corpus`)
with the page's `url`/`text` and the search profile's `locations`.
`locations` is carried per record, rather than passed as a separate
argument, so every metric keeps the same two-argument signature instead
of threading profile state through some functions and not others.

`kept_records` is the subset of `all_records` a triage/dedup pass
decided to keep; `all_records` is the full labelled corpus it ran over.
Records are matched between the two lists by `url`.
"""

from __future__ import annotations

from typing import Any

from src.page_signals import mentions_location
from src.url_heuristics import canonical_url

Record = dict[str, Any]


def _urls(records: list[Record]) -> set[str]:
    return {record["url"] for record in records}


def posting_shape_rate(
    kept_records: list[Record], all_records: list[Record]
) -> float | None:
    """Precision: of the records the pipeline kept, what fraction are
    actually postings by ground-truth label. Scored over kept records
    only, so it can't be gamed by classifying every page as a posting."""
    if not kept_records:
        return None
    postings = sum(1 for record in kept_records if record["kind"] == "posting")
    return postings / len(kept_records)


def aggregator_drop_rate(
    kept_records: list[Record], all_records: list[Record]
) -> float | None:
    """Of the labelled aggregator index pages in the corpus, what fraction
    did triage correctly drop (i.e. not keep)."""
    aggregator_index = [
        record
        for record in all_records
        if record["aggregator"] and record["kind"] == "index"
    ]
    if not aggregator_index:
        return None
    kept_urls = _urls(kept_records)
    dropped = sum(1 for record in aggregator_index if record["url"] not in kept_urls)
    return dropped / len(aggregator_index)


def location_match_rate(
    kept_records: list[Record], all_records: list[Record]
) -> float | None:
    """Of the kept records, what fraction mention one of the profile's
    preferred locations, per `src.page_signals.mentions_location`."""
    if not kept_records:
        return None
    matches = sum(
        1
        for record in kept_records
        if mentions_location(record["text"], record["locations"])
    )
    return matches / len(kept_records)


def staleness_detection_rate(
    kept_records: list[Record], all_records: list[Record]
) -> float | None:
    """Of the labelled-stale records in the corpus, what fraction did
    triage correctly drop."""
    stale = [record for record in all_records if record["stale"]]
    if not stale:
        return None
    kept_urls = _urls(kept_records)
    dropped = sum(1 for record in stale if record["url"] not in kept_urls)
    return dropped / len(stale)


def dedup_rate(
    kept_records: list[Record], all_records: list[Record]
) -> float | None:
    """Of the labelled duplicate records, what fraction collapse onto
    their `duplicate_of` target under `src.url_heuristics.canonical_url`."""
    duplicates = [record for record in all_records if record.get("duplicate_of")]
    if not duplicates:
        return None
    matches = sum(
        1
        for record in duplicates
        if canonical_url(record["url"]) == canonical_url(record["duplicate_of"])
    )
    return matches / len(duplicates)
