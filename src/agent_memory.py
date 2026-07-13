from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

MAX_STORED_REFLECTIONS = 10


@dataclass
class QueryStats:
    times_used: int = 0
    urls_found: int = 0
    new_urls: int = 0
    accepted: int = 0
    rejected: int = 0


class AgentMemory:
    """Persistent cross-run memory of what worked and what didn't.

    Tracks per-query effectiveness (URLs found, accepted/rejected jobs),
    per-domain outcomes, and past reflections. Summaries are fed back into
    LLM prompts so the agent avoids repeating dead-end queries and leans
    into strategies that produced accepted jobs.
    """

    def __init__(self) -> None:
        self._queries: dict[str, QueryStats] = {}
        self._domains: dict[str, dict[str, int]] = {}
        self._reflections: list[str] = []
        self._logger = logging.getLogger(self.__class__.__name__)

    @classmethod
    def load(cls, path: Path) -> "AgentMemory":
        memory = cls()
        if not path.exists():
            return memory
        try:
            with path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (json.JSONDecodeError, OSError) as exc:
            logging.getLogger(cls.__name__).warning(
                "Could not load memory from %s: %s", path, exc
            )
            return memory
        for query, stats in data.get("queries", {}).items():
            memory._queries[query] = QueryStats(
                **{k: int(v) for k, v in stats.items() if k in QueryStats.__annotations__}
            )
        memory._domains = {
            domain: {k: int(v) for k, v in counts.items()}
            for domain, counts in data.get("domains", {}).items()
        }
        memory._reflections = [str(item) for item in data.get("reflections", [])]
        return memory

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "queries": {query: asdict(stats) for query, stats in self._queries.items()},
            "domains": self._domains,
            "reflections": self._reflections[-MAX_STORED_REFLECTIONS:],
        }
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    def record_query(self, query: str, urls_found: int, new_urls: int) -> None:
        stats = self._queries.setdefault(query, QueryStats())
        stats.times_used += 1
        stats.urls_found += urls_found
        stats.new_urls += new_urls

    def record_evaluation(self, url: str, accepted: bool, query: str | None = None) -> None:
        domain = urlparse(url).hostname or "unknown"
        counts = self._domains.setdefault(domain, {"accepted": 0, "rejected": 0})
        counts["accepted" if accepted else "rejected"] += 1
        if query is not None and query in self._queries:
            stats = self._queries[query]
            if accepted:
                stats.accepted += 1
            else:
                stats.rejected += 1

    def add_reflection(self, text: str) -> None:
        if text:
            self._reflections.append(text)
            self._reflections = self._reflections[-MAX_STORED_REFLECTIONS:]

    def known_queries(self) -> set[str]:
        return set(self._queries)

    def ineffective_queries(self) -> list[str]:
        return sorted(
            query
            for query, stats in self._queries.items()
            if stats.times_used > 0 and stats.new_urls == 0
        )

    def effective_queries(self) -> list[str]:
        scored = [
            (query, stats.accepted)
            for query, stats in self._queries.items()
            if stats.accepted > 0
        ]
        scored.sort(key=lambda item: item[1], reverse=True)
        return [query for query, _ in scored]

    def productive_domains(self) -> list[str]:
        scored = [
            (domain, counts.get("accepted", 0))
            for domain, counts in self._domains.items()
            if counts.get("accepted", 0) > 0
        ]
        scored.sort(key=lambda item: item[1], reverse=True)
        return [domain for domain, _ in scored]

    def summary_for_prompt(self) -> dict[str, Any]:
        return {
            "effective_queries": self.effective_queries()[:5],
            "ineffective_queries": self.ineffective_queries()[:10],
            "productive_domains": self.productive_domains()[:5],
            "recent_reflections": self._reflections[-3:],
        }
