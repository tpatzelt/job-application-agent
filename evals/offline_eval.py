"""Offline replay driver for the G1 result-quality harness.

Replays each profile's hand-labelled corpus (`evals/fixtures/*.jsonl`,
via `evals.corpus`) through the same accept/drop *ordering*
`Orchestrator._triage_urls` / `_process_url` apply (dedup by canonical
URL, drop non-job URLs, aggregator index pages, redirected/dead/stale
pages, and pages with no preferred location) by calling the real
`src.url_heuristics` / `src.page_signals` functions rather than copying
their rules. No network or LLM call happens here; what survives replay
is what would be *reported as a result* in production, and the five G1
metrics (`evals.metrics`) are scored over that set.

Only POSTING-classified URLs survive. `Orchestrator._process_url`
harvests posting links out of a careers/board page and returns without
scoring the hub page itself, so a LISTING or INDEX page never becomes a
`JobResult` on the common path. Assumption, recorded because nobody can
be asked: a LISTING page from which *no* posting links can be harvested
is scored directly in production, and this harness counts it as dropped
anyway — corpus records carry no links field, so the harvest branch
cannot be replayed, and "hub pages are never reported" is the
conservative reading. The posting gate runs last, after every other
drop, so each dropped record stays attributable to a named rule and the
other four metrics keep measuring exactly what they measured before.

Usage:
    uv run python -m evals.offline_eval
    uv run python -m evals.offline_eval --corpus evals/fixtures --out evals/runs/offline
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Callable

from src.page_signals import find_stale_marker, mentions_location, redirected_off_posting
from src.url_heuristics import INDEX, OTHER, POSTING, canonical_url, classify_url, is_aggregator_url

from .corpus import CorpusRecord, load_all
from .metrics import (
    Record,
    aggregator_drop_rate,
    dedup_rate,
    location_match_rate,
    posting_shape_rate,
    staleness_detection_rate,
)
from .profiles import PROFILES_BY_NAME

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CORPUS_DIR = ROOT / "evals" / "fixtures"
DEFAULT_OUT_DIR = ROOT / "evals" / "runs" / "offline"
DEFAULT_BASELINE_PATH = ROOT / "evals" / "baseline.json"
BASELINE_TOLERANCE = 1e-9

METRICS: dict[str, Callable[[list[Record], list[Record]], float | None]] = {
    "posting_shape_rate": posting_shape_rate, "aggregator_drop_rate": aggregator_drop_rate,
    "location_match_rate": location_match_rate, "staleness_detection_rate": staleness_detection_rate,
    "dedup_rate": dedup_rate,
}


def locations_for(profile_name: str) -> list[str]:
    """Preferred locations for a corpus profile (empty if its stem names
    no known `evals.profiles` entry)."""
    profile = PROFILES_BY_NAME.get(profile_name)
    return list(profile.locations) if profile is not None else []


def replay_keep(records: list[CorpusRecord], locations: list[str]) -> list[CorpusRecord]:
    """Replay dedup + triage + page gates; return the records that would
    be reported as results (POSTING-shaped pages only — see the module
    docstring)."""
    kept: list[CorpusRecord] = []
    seen: set[str] = set()
    for record in records:
        canonical = canonical_url(record.url)
        if canonical in seen:
            continue
        seen.add(canonical)
        kind = classify_url(canonical)
        if kind == OTHER:
            continue
        if kind == INDEX and is_aggregator_url(canonical):
            continue
        if redirected_off_posting(record.url, record.final_url):
            continue
        if record.http_status is not None and record.http_status >= 400:
            continue
        if not record.text:
            continue
        if find_stale_marker(record.text):
            continue
        if locations and not mentions_location(record.text, locations):
            continue
        if kind != POSTING:
            # A careers/board hub is harvested for posting links, never
            # scored itself, so it never becomes a reported result.
            continue
        kept.append(record)
    return kept


def _to_metric_record(record: CorpusRecord, locations: list[str]) -> Record:
    return {
        "url": record.url, "text": record.text, "locations": locations,
        "kind": record.label.kind, "aggregator": record.label.aggregator,
        "stale": record.label.stale, "duplicate_of": record.label.duplicate_of,
    }


def score_profile(name: str, records: list[CorpusRecord]) -> dict[str, Any]:
    """Replay one profile's corpus and score the five G1 metrics. Returns
    the scorecard plus the metric-shaped record lists (kept, all), so
    callers can combine profiles into a totals row without reloading."""
    locations = locations_for(name)
    kept_dicts = [_to_metric_record(r, locations) for r in replay_keep(records, locations)]
    all_dicts = [_to_metric_record(r, locations) for r in records]
    scorecard = {
        "profile": name, "locations": locations, "records": len(records), "kept": len(kept_dicts),
        "metrics": {n: fn(kept_dicts, all_dicts) for n, fn in METRICS.items()},
    }
    return {"scorecard": scorecard, "kept": kept_dicts, "all": all_dicts}


def run(corpus_dir: Path) -> dict[str, Any]:
    """Replay every profile corpus under `corpus_dir`; one scorecard per
    profile plus a totals row scored over every profile combined."""
    scorecards: list[dict[str, Any]] = []
    all_kept: list[Record] = []
    all_records: list[Record] = []
    for name, records in load_all(corpus_dir).items():
        scored = score_profile(name, records)
        scorecards.append(scored["scorecard"])
        all_kept.extend(scored["kept"])
        all_records.extend(scored["all"])
    totals = {
        "records": len(all_records), "kept": len(all_kept),
        "metrics": {n: fn(all_kept, all_records) for n, fn in METRICS.items()},
    }
    return {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "corpus_dir": str(corpus_dir), "profiles": scorecards, "totals": totals,
    }


def _format_rate(rate: float | None) -> str:
    return "n/a" if rate is None else f"{rate:.3f}"


def format_table(report: dict[str, Any]) -> str:
    metric_names = list(METRICS)
    header = ["profile", "records", "kept", *metric_names]

    def row(label: str, records: int, kept: int, metrics: dict[str, float | None]) -> list[str]:
        return [label, str(records), str(kept), *(_format_rate(metrics[m]) for m in metric_names)]

    rows = [row(c["profile"], c["records"], c["kept"], c["metrics"]) for c in report["profiles"]]
    totals = report["totals"]
    rows.append(row("TOTAL", totals["records"], totals["kept"], totals["metrics"]))
    widths = [max(len(r[i]) for r in (header, *rows)) for i in range(len(header))]
    lines = ["  ".join(cell.ljust(w) for cell, w in zip(r, widths)) for r in (header, *rows)]
    lines.insert(1, "  ".join("-" * w for w in widths))
    return "\n".join(lines)


def compare_to_baseline(
    current: dict[str, float | None],
    baseline: dict[str, float | None],
    tolerance: float = BASELINE_TOLERANCE,
) -> tuple[bool, list[dict[str, Any]]]:
    """Compare each current metric against its baseline value. A metric
    regresses if it is more than `tolerance` below the baseline, or if the
    baseline had a value and the current run no longer does. Returns
    (all_ok, rows) where rows carry before/after/delta for display."""
    rows: list[dict[str, Any]] = []
    all_ok = True
    for name in METRICS:
        base = baseline.get(name)
        cur = current.get(name)
        if base is None:
            regressed = False
            delta: float | None = None
        elif cur is None:
            regressed = True
            delta = None
        else:
            delta = cur - base
            regressed = delta < -tolerance
        if regressed:
            all_ok = False
        rows.append({"metric": name, "baseline": base, "current": cur, "delta": delta, "regressed": regressed})
    return all_ok, rows


def format_baseline_table(rows: list[dict[str, Any]]) -> str:
    header = ["metric", "baseline", "current", "delta"]

    def fmt_delta(delta: float | None) -> str:
        return "n/a" if delta is None else f"{delta:+.3f}"

    def row(r: dict[str, Any]) -> list[str]:
        mark = " !" if r["regressed"] else ""
        return [
            r["metric"] + mark, _format_rate(r["baseline"]), _format_rate(r["current"]), fmt_delta(r["delta"])
        ]

    body = [row(r) for r in rows]
    widths = [max(len(r[i]) for r in (header, *body)) for i in range(len(header))]
    lines = ["  ".join(cell.ljust(w) for cell, w in zip(r, widths)) for r in (header, *body)]
    lines.insert(1, "  ".join("-" * w for w in widths))
    return "\n".join(lines)


def check_baseline(corpus_dir: Path, baseline_path: Path) -> tuple[bool, str]:
    """Recompute the report over `corpus_dir` and compare its totals against
    the committed `baseline_path`. Returns (ok, message) for the caller to
    print and act on."""
    report = run(corpus_dir)
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    ok, rows = compare_to_baseline(report["totals"]["metrics"], baseline["totals"]["metrics"])
    lines = [format_baseline_table(rows)]
    regressed = [r["metric"] for r in rows if r["regressed"]]
    if regressed:
        lines.append("\nRegressed vs baseline: " + ", ".join(regressed))
    else:
        lines.append("\nNo metric regressed vs baseline.")
    return ok, "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Offline, network-free replay of the G1 result-quality harness"
    )
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE_PATH)
    parser.add_argument(
        "--check-baseline", action="store_true",
        help="Recompute metrics and compare against --baseline instead of writing a report; "
        "exit 1 if any metric regressed.",
    )
    args = parser.parse_args(argv)

    if args.check_baseline:
        ok, message = check_baseline(args.corpus, args.baseline)
        print(message)
        return 0 if ok else 1

    report = run(args.corpus)
    print(format_table(report))
    args.out.mkdir(parents=True, exist_ok=True)
    report_path = args.out / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
