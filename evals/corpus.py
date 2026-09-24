"""Loader for the offline eval corpus: hand-labelled recordings of
search-result pages (one JSONL file per search profile under
`evals/fixtures/`), replayed to test the agent's deterministic
triage/dedup logic without network or LLM calls. Labels are assigned by a
human reading the page, not computed by `src.url_heuristics` /
`src.page_signals`, so replaying them is a real test of that code."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

VALID_KINDS = frozenset({"posting", "listing", "index", "other"})

_LABEL_KEYS = frozenset({"kind", "aggregator", "stale", "location_ok", "duplicate_of"})
_RECORD_KEYS = frozenset({"url", "title", "final_url", "http_status", "text", "label"})


@dataclass(frozen=True)
class RecordLabel:
    """Hand-assigned ground truth for one recorded page."""

    kind: str
    aggregator: bool
    stale: bool
    location_ok: bool
    duplicate_of: str | None


@dataclass(frozen=True)
class CorpusRecord:
    """A recorded page plus its ground-truth label."""

    url: str
    title: str
    final_url: str
    http_status: int
    text: str
    label: RecordLabel


def _fail(source: Path, line_no: int, message: str) -> None:
    raise ValueError(f"{source}:{line_no}: {message}")


def _parse_label(raw: object, *, line_no: int, source: Path) -> RecordLabel:
    if not isinstance(raw, dict):
        _fail(source, line_no, "'label' must be a JSON object")
    keys = set(raw)
    missing = _LABEL_KEYS - keys
    if missing:
        _fail(source, line_no, f"label missing keys: {sorted(missing)}")
    unknown = keys - _LABEL_KEYS
    if unknown:
        _fail(source, line_no, f"label has unknown keys: {sorted(unknown)}")
    kind = raw["kind"]
    if kind not in VALID_KINDS:
        _fail(
            source,
            line_no,
            f"label.kind {kind!r} is not one of {sorted(VALID_KINDS)}",
        )
    duplicate_of = raw["duplicate_of"]
    if duplicate_of is not None and not isinstance(duplicate_of, str):
        _fail(source, line_no, "label.duplicate_of must be a string or null")
    return RecordLabel(
        kind=kind,
        aggregator=bool(raw["aggregator"]),
        stale=bool(raw["stale"]),
        location_ok=bool(raw["location_ok"]),
        duplicate_of=duplicate_of,
    )


def _parse_record(raw: object, *, line_no: int, source: Path) -> CorpusRecord:
    if not isinstance(raw, dict):
        _fail(source, line_no, "record must be a JSON object")
    keys = set(raw)
    missing = _RECORD_KEYS - keys
    if missing:
        _fail(source, line_no, f"record missing keys: {sorted(missing)}")
    unknown = keys - _RECORD_KEYS
    if unknown:
        _fail(source, line_no, f"record has unknown keys: {sorted(unknown)}")
    label = _parse_label(raw["label"], line_no=line_no, source=source)
    return CorpusRecord(
        url=raw["url"],
        title=raw["title"],
        final_url=raw["final_url"],
        http_status=raw["http_status"],
        text=raw["text"],
        label=label,
    )


def load_corpus(path: Path) -> list[CorpusRecord]:
    """Load one profile's labelled corpus from a JSONL file.

    Raises ValueError if any record (or its label) is missing a required
    key, carries an unknown key, or has an invalid `label.kind`.
    """
    records: list[CorpusRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            records.append(
                _parse_record(json.loads(line), line_no=line_no, source=path)
            )
    return records


def load_all(directory: Path) -> dict[str, list[CorpusRecord]]:
    """Load every `*.jsonl` corpus in a directory, keyed by profile name."""
    return {
        path.stem: load_corpus(path) for path in sorted(directory.glob("*.jsonl"))
    }
