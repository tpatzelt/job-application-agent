from __future__ import annotations

import csv
import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .config_manager import Config, EffortBudget
from .models import JobResult


class Orchestrator:
    def __init__(
        self,
        config: Config,
        budget: EffortBudget,
        llm_service: Any,
        crawler: Any,
    ) -> None:
        self._config = config
        self._budget = budget
        self._llm_service = llm_service
        self._crawler = crawler
        self._logger = logging.getLogger(self.__class__.__name__)

    def run(
        self,
        cv_text: str,
        preferences: dict[str, Any],
        cache_path: Path,
        results_json: Path,
        results_csv: Path,
    ) -> list[JobResult]:
        seen_urls = self._load_cache(cache_path)
        self._logger.info("Loaded %s cached URLs", len(seen_urls))
        results: list[JobResult] = []
        history: list[dict[str, Any]] = []

        while (
            len(results) < self._config.max_results
            and self._budget.can_call_llm()
            and self._budget.can_search()
        ):
            context = self._build_context(cv_text, preferences, results)
            self._logger.info("Generating queries with %s results so far", len(results))
            queries = self._llm_service.generate_search_queries(
                context, history
            ).queries
            if not queries:
                self._logger.info("No queries returned, stopping")
                break

            self._logger.info("Generated %s queries", len(queries))

            for query in queries[: self._config.max_queries_per_iteration]:
                self._logger.info("Searching with query: %s", query)
                urls = self._crawler.search(query)
                new_urls = [url for url in urls if url not in seen_urls]
                history.append(
                    {"query": query, "urls_found": len(urls), "new": len(new_urls)}
                )
                self._logger.info("Found %s URLs (%s new)", len(urls), len(new_urls))

                for url in new_urls:
                    if len(results) >= self._config.max_results:
                        break
                    if not self._looks_like_listing(url):
                        self._logger.info("Skipping non-job URL: %s", url)
                        seen_urls.add(url)
                        continue
                    self._logger.info("Fetching job page: %s", url)
                    try:
                        job_text = self._crawler.fetch_job_text(url)
                    except Exception as exc:
                        self._logger.warning(
                            "Failed to fetch %s: %s",
                            url,
                            exc,
                        )
                        seen_urls.add(url)
                        continue
                    if not job_text:
                        self._logger.info("Empty content for %s, skipping", url)
                        seen_urls.add(url)
                        continue
                    if len(job_text) < 800:
                        self._logger.info(
                            "Content too short (%s chars) for %s, skipping",
                            len(job_text),
                            url,
                        )
                        seen_urls.add(url)
                        continue
                    self._logger.info("Scoring job page: %s", url)
                    try:
                        evaluation = self._llm_service.evaluate_job(cv_text, job_text)
                    except Exception as exc:
                        self._logger.warning(
                            "Failed to score %s: %s",
                            url,
                            exc,
                        )
                        seen_urls.add(url)
                        continue
                    if evaluation.score >= self._config.min_score:
                        job_result = JobResult(
                            title=self._extract_title(job_text),
                            company=self._extract_company(job_text),
                            url=url,
                            score=evaluation.score,
                            reason=evaluation.reason,
                            status="new",
                        )
                        results.append(job_result)
                        self._logger.info(
                            "Saved job (%s) with score %s",
                            job_result.url,
                            job_result.score,
                        )
                    else:
                        self._logger.info(
                            "Rejected job (%s) with score %s",
                            url,
                            evaluation.score,
                        )
                    seen_urls.add(url)

                if len(results) >= self._config.max_results:
                    break

        self._save_cache(cache_path, seen_urls)
        self._logger.info("Saved cache with %s URLs", len(seen_urls))
        self._write_results(results_json, results_csv, results)
        self._logger.info("Wrote %s results", len(results))
        return results

    def _build_context(
        self,
        cv_text: str,
        preferences: dict[str, Any],
        results: list[JobResult],
    ) -> dict[str, Any]:
        return {
            "cv_summary": cv_text[:1500],
            "preferences": preferences,
            "results": [asdict(item) for item in results],
        }

    def _load_cache(self, cache_path: Path) -> set[str]:
        if not cache_path.exists():
            return set()
        with cache_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return set(data.get("seen_urls", []))

    def _save_cache(self, cache_path: Path, seen_urls: set[str]) -> None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("w", encoding="utf-8") as handle:
            json.dump({"seen_urls": sorted(seen_urls)}, handle, indent=2)

    def _write_results(
        self,
        results_json: Path,
        results_csv: Path,
        results: list[JobResult],
    ) -> None:
        results_json.parent.mkdir(parents=True, exist_ok=True)
        payload = [item.model_dump() for item in results]
        with results_json.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

        with results_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["title", "company", "url", "score", "reason", "status"],
            )
            writer.writeheader()
            writer.writerows(payload)

    def _extract_title(self, job_text: str) -> str:
        words = job_text.split()
        return " ".join(words[:8])

    def _extract_company(self, job_text: str) -> str:
        return "Unknown"

    def _looks_like_listing(self, url: str) -> bool:
        tokens = ["/jobs", "/job", "careers", "apply", "greenhouse", "lever"]
        return any(token in url.lower() for token in tokens)
