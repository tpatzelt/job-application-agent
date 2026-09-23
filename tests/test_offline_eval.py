"""Tests for the offline G1 replay driver (`evals/offline_eval.py`).

`replay_keep` reimplements only the *ordering* of
`Orchestrator._triage_urls` / `_process_url` (dedup, non-job drop,
aggregator-index drop, redirected/dead/stale/no-location drop, and
finally the posting-only gate); each drop reason gets its own case so a
broken gate fails on its own instead of only showing up as a shifted
rate downstream.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from evals.corpus import CorpusRecord, RecordLabel
from evals.offline_eval import METRICS, format_table, locations_for, main, replay_keep, run, score_profile

BERLIN = ["Berlin, Germany"]
POSTING_TEXT = "We are hiring a Project Manager in Berlin, Germany for digital transformation."
STALE_TEXT = POSTING_TEXT + " This position has been filled."
NO_LOCATION_TEXT = "We are hiring a Project Manager for digital transformation in Munich."


def _record(url, text=POSTING_TEXT, *, final_url=None, http_status=200, kind="posting",
            aggregator=False, stale=False, duplicate_of=None) -> CorpusRecord:
    return CorpusRecord(
        url=url, title="Project Manager", final_url=final_url or url,
        http_status=http_status, text=text,
        label=RecordLabel(kind=kind, aggregator=aggregator, stale=stale,
                           location_ok=True, duplicate_of=duplicate_of),
    )


def _record_dict(record: CorpusRecord) -> dict:
    label = record.label
    return {
        "url": record.url, "title": record.title, "final_url": record.final_url,
        "http_status": record.http_status, "text": record.text,
        "label": {"kind": label.kind, "aggregator": label.aggregator, "stale": label.stale,
                   "location_ok": label.location_ok, "duplicate_of": label.duplicate_of},
    }


DROP_CASES = {
    "non_job_url": _record("https://example.com/about-us", kind="other"),
    "aggregator_index": _record(
        "https://www.indeed.com/jobs?q=pm&l=berlin", kind="index", aggregator=True
    ),
    "redirected_off_posting": _record(
        "https://boards.greenhouse.io/acme/jobs/999",
        final_url="https://boards.greenhouse.io/acme?error=true",
    ),
    "dead_http_status": _record("https://boards.greenhouse.io/acme/jobs/404", http_status=404),
    "empty_page": _record("https://boards.greenhouse.io/acme/jobs/555", text=""),
    "stale_posting": _record("https://boards.greenhouse.io/acme/jobs/222", STALE_TEXT, stale=True),
    "no_preferred_location": _record("https://boards.greenhouse.io/acme/jobs/333", NO_LOCATION_TEXT),
    "careers_listing_page": _record("https://acme.example/careers", kind="listing"),
    "non_aggregator_index_page": _record(
        "https://acme.example.com/jobs?search=manager", kind="index"
    ),
}


@pytest.mark.parametrize("record", DROP_CASES.values(), ids=DROP_CASES.keys())
def test_replay_drops(record: CorpusRecord) -> None:
    assert replay_keep([record], BERLIN) == []


def test_replay_keeps_a_clean_matching_posting() -> None:
    record = _record("https://boards.greenhouse.io/acme/jobs/111")
    assert replay_keep([record], BERLIN) == [record]


def test_replay_drops_hub_pages_but_keeps_postings_on_the_same_profile() -> None:
    # A careers page and a company's own INDEX-shaped /jobs?search= page
    # are both hubs: production harvests posting links out of them and
    # never scores the hub itself, so neither can become a result. Both
    # clear every earlier gate (live, fresh, Berlin in the text) -- only
    # the posting gate removes them -- while the ATS posting alongside
    # them survives.
    careers = _record("https://acme.example/careers", kind="listing")
    company_index = _record("https://acme.example.com/jobs?search=manager", kind="index")
    posting = _record("https://boards.greenhouse.io/acme/jobs/111")

    assert replay_keep([careers, company_index, posting], BERLIN) == [posting]


def test_replay_drops_duplicate_by_canonical_url() -> None:
    original = _record("https://boards.greenhouse.io/acme/jobs/111")
    duplicate = _record("https://boards.greenhouse.io/acme/jobs/111?utm_source=linkedin")
    assert replay_keep([original, duplicate], BERLIN) == [original]


def test_replay_location_gate_skipped_when_no_locations_given() -> None:
    record = _record("https://boards.greenhouse.io/acme/jobs/333", NO_LOCATION_TEXT)
    assert replay_keep([record], []) == [record]


def test_locations_for() -> None:
    assert locations_for("project-manager-berlin") == ["Berlin, Germany"]
    assert locations_for("no-such-profile") == []


# Six records designed so a plausible-but-wrong implementation would
# disagree. R1 is a clean posting and survives. R2 collapses onto R1
# under canonicalization. R3 carries a stale marker. R4 is a genuine
# indeed.com search URL and is dropped as an aggregator index. R5 is
# labelled an aggregator index page but its URL shape
# (indeed.com/viewjob) is what the real classifier treats as an
# individual posting, so it clears both the aggregator gate and the
# posting-only gate and survives -- pinning aggregator_drop_rate and
# posting_shape_rate at a real 0.5, not a trivial 1.0/0.0. R6 is a
# non-job URL. Ground-truth labels are deliberately left as they are:
# the metrics score the labels, the gates score the URL shapes, and
# this corpus is where the two disagree.
MIXED_CORPUS = [
    _record("https://boards.greenhouse.io/acme/jobs/111"),
    _record("https://boards.greenhouse.io/acme/jobs/111?utm_source=linkedin",
            duplicate_of="https://boards.greenhouse.io/acme/jobs/111"),
    _record("https://boards.greenhouse.io/beta/jobs/222", STALE_TEXT, stale=True),
    _record("https://www.indeed.com/jobs?q=pm&l=berlin", kind="index", aggregator=True),
    _record("https://www.indeed.com/viewjob?jk=xyz789", kind="index", aggregator=True),
    _record("https://example.com/about-us", kind="other"),
]


def test_score_profile_computes_expected_metric_values() -> None:
    scored = score_profile("project-manager-berlin", MIXED_CORPUS)
    card = scored["scorecard"]

    assert card["records"] == 6
    assert card["kept"] == 2  # only R1 and the mislabelled R5 survive
    assert card["metrics"] == {
        "posting_shape_rate": 0.5,
        "aggregator_drop_rate": 0.5,
        "location_match_rate": 1.0,
        "staleness_detection_rate": 1.0,
        "dedup_rate": 1.0,
    }


def test_score_profile_empty_corpus_has_all_five_metric_keys_as_none() -> None:
    metrics = score_profile("project-manager-berlin", [])["scorecard"]["metrics"]
    assert set(metrics) == set(METRICS)
    assert all(value is None for value in metrics.values())


def _write_corpus(directory: Path, profile: str, records: list[CorpusRecord]) -> None:
    path = directory / f"{profile}.jsonl"
    path.write_text("\n".join(json.dumps(_record_dict(r)) for r in records), encoding="utf-8")


def test_run_and_main_build_totals_and_report(tmp_path: Path) -> None:
    corpus_dir = tmp_path / "fixtures"
    corpus_dir.mkdir()
    _write_corpus(corpus_dir, "project-manager-berlin", MIXED_CORPUS)
    _write_corpus(corpus_dir, "no-such-profile", MIXED_CORPUS[:1])

    report = run(corpus_dir)
    assert {c["profile"] for c in report["profiles"]} == {"project-manager-berlin", "no-such-profile"}
    assert report["totals"]["records"] == 7
    assert report["totals"]["kept"] == 3

    out_dir = tmp_path / "out"
    assert main(["--corpus", str(corpus_dir), "--out", str(out_dir)]) == 0
    written = json.loads((out_dir / "report.json").read_text(encoding="utf-8"))
    assert set(written["totals"]["metrics"]) == set(METRICS)
    assert all(set(card["metrics"]) == set(METRICS) for card in written["profiles"])


def test_format_table_includes_profile_names_and_total_row() -> None:
    scored = score_profile("project-manager-berlin", MIXED_CORPUS)
    card = scored["scorecard"]
    report = {"profiles": [card], "totals": {k: card[k] for k in ("records", "kept", "metrics")}}

    table = format_table(report)

    assert "project-manager-berlin" in table
    assert "TOTAL" in table
    assert all(name in table for name in METRICS)


def test_module_imports_no_network_client() -> None:
    import evals.offline_eval as module

    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])

    assert not imported & {"requests", "httpx", "urllib", "http", "socket", "botasaurus"}
