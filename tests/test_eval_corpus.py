from __future__ import annotations

import json
from pathlib import Path

import pytest

from evals.corpus import VALID_KINDS, CorpusRecord, RecordLabel, load_all, load_corpus
from evals.profiles import PROFILES_BY_NAME

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "evals" / "fixtures"

VALID_RECORD = {
    "url": "https://boards.greenhouse.io/acme/jobs/123",
    "title": "Example Role",
    "final_url": "https://boards.greenhouse.io/acme/jobs/123",
    "http_status": 200,
    "text": "x" * 350,
    "label": {
        "kind": "posting",
        "aggregator": False,
        "stale": False,
        "location_ok": True,
        "duplicate_of": None,
    },
}


def _write_jsonl(path: Path, records: list[dict]) -> Path:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    return path


def test_parses_valid_record(tmp_path: Path) -> None:
    path = _write_jsonl(tmp_path / "profile.jsonl", [VALID_RECORD])

    [parsed] = load_corpus(path)

    assert parsed == CorpusRecord(
        url=VALID_RECORD["url"],
        title=VALID_RECORD["title"],
        final_url=VALID_RECORD["final_url"],
        http_status=VALID_RECORD["http_status"],
        text=VALID_RECORD["text"],
        label=RecordLabel(
            kind="posting",
            aggregator=False,
            stale=False,
            location_ok=True,
            duplicate_of=None,
        ),
    )


def test_skips_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "profile.jsonl"
    path.write_text("\n" + json.dumps(VALID_RECORD) + "\n\n", encoding="utf-8")

    assert len(load_corpus(path)) == 1


def test_parses_duplicate_of_as_string(tmp_path: Path) -> None:
    record = dict(VALID_RECORD, label=dict(VALID_RECORD["label"]))
    record["label"]["duplicate_of"] = "https://boards.greenhouse.io/acme/jobs/999"
    path = _write_jsonl(tmp_path / "profile.jsonl", [record])

    [parsed] = load_corpus(path)

    assert parsed.label.duplicate_of == "https://boards.greenhouse.io/acme/jobs/999"


def _drop(mapping: dict, key: str) -> dict:
    return {k: v for k, v in mapping.items() if k != key}


@pytest.mark.parametrize(
    "record,match",
    [
        (_drop(VALID_RECORD, "title"), "missing keys"),
        (dict(VALID_RECORD, extra_field="surprise"), "unknown keys"),
        (dict(VALID_RECORD, label=_drop(VALID_RECORD["label"], "stale")), "label missing keys"),
        (dict(VALID_RECORD, label=dict(VALID_RECORD["label"], score=42)), "label has unknown keys"),
        (dict(VALID_RECORD, label=dict(VALID_RECORD["label"], kind="job-ish")), "label.kind"),
        (dict(VALID_RECORD, label="posting"), "'label' must be a JSON object"),
    ],
)
def test_rejects_invalid_record_instead_of_defaulting(
    tmp_path: Path, record: dict, match: str
) -> None:
    path = _write_jsonl(tmp_path / "profile.jsonl", [record])

    with pytest.raises(ValueError, match=match):
        load_corpus(path)


def test_load_all_keys_by_profile_stem_and_ignores_other_files(tmp_path: Path) -> None:
    _write_jsonl(tmp_path / "profile-a.jsonl", [VALID_RECORD])
    _write_jsonl(tmp_path / "profile-b.jsonl", [VALID_RECORD])
    (tmp_path / "README.md").write_text("not a corpus", encoding="utf-8")

    loaded = load_all(tmp_path)

    assert set(loaded) == {"profile-a", "profile-b"}
    assert all(len(v) == 1 for v in loaded.values())


# --- Coverage of the committed fixtures -------------------------------------
#
# Pins down what the offline harness (G1) needs to actually exercise the
# deterministic triage/dedup logic: enough records and variety, plus the
# specific edge cases (staleness in two languages, tracking-param/apply
# suffix/scheme-or-www duplicates, no-location pages) so a metric can't
# pass vacuously against a trivial corpus.

EXPECTED_PROFILES = {
    "ml-engineer-berlin",
    "frontend-developer-munich",
    "backend-engineer-remote-germany",
}
ATS_HOSTS = ("greenhouse.io", "lever.co", "personio", "workable.com")
AGGREGATOR_HOSTS = ("indeed.", "stepstone.", "linkedin.com")


@pytest.fixture(scope="module")
def all_records() -> list[CorpusRecord]:
    corpus = load_all(FIXTURES_DIR)
    assert EXPECTED_PROFILES <= set(corpus)
    assert set(corpus) <= set(PROFILES_BY_NAME), "corpus profile with no matching eval profile"
    return [record for records in corpus.values() for record in records]


def test_at_least_18_records_total(all_records: list[CorpusRecord]) -> None:
    assert len(all_records) >= 18


def test_text_length_bounds(all_records: list[CorpusRecord]) -> None:
    for record in all_records:
        assert 300 < len(record.text) < 600, record.url


def test_all_labels_have_valid_kind(all_records: list[CorpusRecord]) -> None:
    assert all(record.label.kind in VALID_KINDS for record in all_records)


def test_at_least_three_stale_records_in_two_languages(
    all_records: list[CorpusRecord],
) -> None:
    stale = [r for r in all_records if r.label.stale]
    assert len(stale) >= 3

    german_markers = ("besetzt", "abgelaufen", "bewerbungsfrist", "stelle")
    english_markers = ("expired", "no longer accepting", "closed")
    assert any(m in r.text.lower() for r in stale for m in german_markers)
    assert any(m in r.text.lower() for r in stale for m in english_markers)


def test_duplicate_pairs_via_tracking_params_and_apply_suffix(
    all_records: list[CorpusRecord],
) -> None:
    urls = {r.url for r in all_records}
    dup_pairs = [r for r in all_records if r.label.duplicate_of is not None]
    for record in dup_pairs:
        assert record.label.duplicate_of in urls, record.url
    assert len(dup_pairs) >= 3

    tracking_markers = ("gh_src=", "utm_source=", "lever-source")
    assert any(m in r.url for r in dup_pairs for m in tracking_markers)
    assert any(r.url.rstrip("/").endswith("/apply") for r in dup_pairs)


def test_at_least_one_scheme_or_www_duplicate_pair(all_records: list[CorpusRecord]) -> None:
    by_url = {r.url: r for r in all_records}

    def differs_by_scheme_or_www(a: str, b: str) -> bool:
        return a.split(":", 1)[0] != b.split(":", 1)[0] or ("://www." in a) != ("://www." in b)

    dup_pairs = (
        (r, by_url[r.label.duplicate_of])
        for r in all_records
        if r.label.duplicate_of in by_url
    )
    assert any(differs_by_scheme_or_www(r.url, other.url) for r, other in dup_pairs)


def test_at_least_two_records_with_no_preferred_location(
    all_records: list[CorpusRecord],
) -> None:
    assert sum(1 for r in all_records if not r.label.location_ok) >= 2


@pytest.mark.parametrize("host", ATS_HOSTS)
def test_covers_employer_ats_host(all_records: list[CorpusRecord], host: str) -> None:
    assert any(host in r.url for r in all_records)


@pytest.mark.parametrize("host", AGGREGATOR_HOSTS)
def test_covers_aggregator_host(all_records: list[CorpusRecord], host: str) -> None:
    assert any(host in r.url for r in all_records)


def test_includes_every_url_kind(all_records: list[CorpusRecord]) -> None:
    assert {"posting", "listing", "index", "other"} <= {r.label.kind for r in all_records}


def test_includes_aggregator_and_non_aggregator_records(
    all_records: list[CorpusRecord],
) -> None:
    assert any(r.label.aggregator for r in all_records)
    assert any(not r.label.aggregator for r in all_records)
