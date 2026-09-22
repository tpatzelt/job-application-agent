from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# How many past runs to keep per user; enough to see a trend, small
# enough that the file stays cheap to read on every dashboard poll.
MAX_RUNS_KEPT = 30

# Counters in the order they should be read: what the run saw, then why
# pages were dropped, then why jobs were rejected.
COUNTER_LABELS = {
    "urls_found": "URLs returned by search",
    "already_seen": "already seen in an earlier run",
    "duplicates_skipped": "duplicate links to the same posting",
    "pages_fetched": "pages fetched",
    "skipped_gone": "posting removed (404 or redirected away)",
    "skipped_empty": "no text extracted",
    "skipped_too_short": "too little text",
    "skipped_stale": "closed/filled marker",
    "skipped_no_location": "no preferred location on page",
    "evaluated": "pages scored by the LLM",
    "rejected_location": "rejected: location mismatch",
    "rejected_domain": "rejected: industry mismatch",
    "rejected_low_score": "rejected: score below threshold",
}


@dataclass
class RunReport:
    """What one crawl actually did, so a run that finds nothing can be
    explained without reading the whole log.

    Persisted next to the run's results as `runs.json` and surfaced by the
    dashboard; the counters name every place a URL can drop out of the
    pipeline between search and an accepted job.
    """

    started_at: float
    finished_at: float = 0.0
    counters: dict[str, int] = field(default_factory=dict)
    errors: list[dict[str, str]] = field(default_factory=list)
    queries: list[str] = field(default_factory=list)
    accepted: int = 0
    tool_stats: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def start(cls) -> RunReport:
        return cls(started_at=time.time())

    def count(self, name: str, amount: int = 1) -> None:
        self.counters[name] = self.counters.get(name, 0) + amount

    def record_error(self, kind: str, url: str, exc: Exception) -> None:
        self.count(f"error_{kind}")
        # Keep only the first few of each kind: a broken host can fail on
        # dozens of URLs and the detail adds nothing after the first.
        if sum(1 for item in self.errors if item["kind"] == kind) < 5:
            self.errors.append(
                {"kind": kind, "url": url, "error": f"{type(exc).__name__}: {exc}"[:300]}
            )

    def finish(
        self,
        queries: list[str],
        accepted: int,
        tool_stats: dict[str, Any] | None = None,
    ) -> None:
        self.finished_at = time.time()
        self.queries = queries
        self.accepted = accepted
        self.tool_stats = tool_stats or {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_seconds": round(max(self.finished_at - self.started_at, 0.0), 1),
            "accepted": self.accepted,
            "queries": self.queries,
            "counters": self.counters,
            "errors": self.errors,
            "tool_stats": self.tool_stats,
        }

    def summary_line(self) -> str:
        parts = [f"{self.accepted} accepted", f"{len(self.queries)} queries"]
        parts += [
            f"{value} {COUNTER_LABELS.get(name, name)}"
            for name, value in self.counters.items()
            if value
        ]
        return ", ".join(parts)

    def save(self, path: Path) -> None:
        """Append this run to the rolling history at `path`."""
        history = load_runs(path)
        history.append(self.to_dict())
        history = history[-MAX_RUNS_KEPT:]
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8") as handle:
                json.dump(history, handle, indent=2)
        except OSError as exc:
            logger.warning("Could not write run history to %s: %s", path, exc)


def runs_path_for(results_json: Path) -> Path:
    """Run history file belonging to a results file.

    Kept parallel to the results name so the CLI, each profile, and the
    mock loop keep separate histories in the same directory.
    """
    stem = results_json.stem
    stem = stem.replace("results", "runs") if "results" in stem else f"{stem}_runs"
    return results_json.with_name(f"{stem}.json")


def load_runs(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not read run history %s: %s", path, exc)
        return []
    return [item for item in data if isinstance(item, dict)] if isinstance(data, list) else []
