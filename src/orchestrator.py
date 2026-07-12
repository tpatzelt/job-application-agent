from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any

from .agent_memory import AgentMemory
from .config_manager import Config, EffortBudget
from .models import JobResult, Reflection, SearchPlan
from .tools import ToolRegistry
from .url_heuristics import INDEX, LISTING, OTHER, POSTING, classify_url


class Orchestrator:
    """Agentic control loop: plan -> act (search/fetch/evaluate) -> reflect.

    The agent first asks the LLM for a search plan, then iterates: generate
    queries (informed by the plan, persistent memory, and the latest
    reflection), execute them through the tool registry, and reflect on the
    outcome to adjust strategy. Persistent memory carries query/domain
    effectiveness across runs so dead-end queries aren't repeated.
    """

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
        self._tools = ToolRegistry()
        if crawler is not None:
            self._tools.register(
                "search", "Search the web for job listing URLs", crawler.search
            )
            self._tools.register(
                "fetch_job_text",
                "Fetch a job page and extract its plain text",
                crawler.fetch_job_text,
            )
        if llm_service is not None:
            self._tools.register(
                "evaluate_job",
                "Score a job description against the CV",
                llm_service.evaluate_job,
            )

    def run(
        self,
        cv_text: str,
        preferences: dict[str, Any],
        cache_path: Path,
        results_json: Path,
        results_csv: Path,
        memory_path: Path | None = None,
    ) -> list[JobResult]:
        seen_urls = self._load_cache(cache_path)
        self._logger.info("Loaded %s cached URLs", len(seen_urls))
        memory_path = memory_path or cache_path.parent / "memory.json"
        memory = AgentMemory.load(memory_path)
        results: list[JobResult] = []
        history: list[dict[str, Any]] = []
        searched_this_run: set[str] = set()

        plan = self._make_plan(cv_text, preferences, memory)
        reflection: Reflection | None = None

        while (
            len(results) < self._config.max_results
            and self._budget.can_call_llm()
            and self._budget.can_search()
        ):
            context = self._build_context(cv_text, preferences, results)
            context["plan"] = plan.model_dump()
            context["memory"] = memory.summary_for_prompt()
            if reflection is not None:
                context["reflection"] = reflection.model_dump()

            self._logger.info("Generating queries with %s results so far", len(results))
            queries = self._llm_service.generate_search_queries(
                context, history
            ).queries
            if not queries:
                self._logger.info("No queries returned, stopping")
                break

            self._logger.info("Generated %s queries", len(queries))
            made_progress = False

            for query in queries[: self._config.max_queries_per_iteration]:
                if query in searched_this_run:
                    self._logger.info("Skipping already-searched query: %s", query)
                    continue
                if query in memory.ineffective_queries():
                    self._logger.info(
                        "Skipping query known to be ineffective: %s", query
                    )
                    continue
                searched_this_run.add(query)
                self._logger.info("Searching with query: %s", query)
                urls = self._tools.invoke("search", query)
                new_urls = [url for url in urls if url not in seen_urls]
                postings, listings, index_pages = self._triage_urls(
                    new_urls, seen_urls
                )
                history.append(
                    {
                        "query": query,
                        "urls_found": len(urls),
                        "new": len(new_urls),
                        "postings": len(postings),
                        "index_pages": len(index_pages),
                    }
                )
                memory.record_query(query, urls_found=len(urls), new_urls=len(new_urls))
                self._logger.info(
                    "Found %s URLs (%s new: %s postings, %s listings, %s index pages)",
                    len(urls),
                    len(new_urls),
                    len(postings),
                    len(listings),
                    len(index_pages),
                )
                if new_urls:
                    made_progress = True

                # Individual postings first, generic careers pages next,
                # board index pages only if capacity remains.
                for url in postings + listings + index_pages:
                    if len(results) >= self._config.max_results:
                        break
                    accepted = self._process_url(
                        url, query, cv_text, seen_urls, results, memory
                    )
                    if accepted:
                        self._logger.info("Accepted job: %s", url)

                if len(results) >= self._config.max_results:
                    break

            if len(results) >= self._config.max_results:
                break
            if not made_progress:
                self._logger.info("No new URLs processed this iteration, stopping")
                break
            if self._budget.can_search() and self._budget.can_call_llm():
                reflection = self._reflect(context, history, memory)

        self._save_cache(cache_path, seen_urls)
        self._logger.info("Saved cache with %s URLs", len(seen_urls))
        memory.save(memory_path)
        self._logger.info("Saved agent memory to %s", memory_path)
        self._write_results(results_json, results_csv, results)
        self._logger.info("Wrote %s results", len(results))
        return results

    def _triage_urls(
        self, urls: list[str], seen_urls: set[str]
    ) -> tuple[list[str], list[str], list[str]]:
        """Split URLs by kind; non-job URLs are marked seen and dropped."""
        postings: list[str] = []
        listings: list[str] = []
        index_pages: list[str] = []
        for url in urls:
            kind = classify_url(url)
            if kind == POSTING:
                postings.append(url)
            elif kind == LISTING:
                listings.append(url)
            elif kind == INDEX:
                self._logger.info("Down-ranking board index page: %s", url)
                index_pages.append(url)
            else:
                self._logger.info("Skipping non-job URL: %s", url)
                seen_urls.add(url)
        return postings, listings, index_pages

    def _process_url(
        self,
        url: str,
        query: str,
        cv_text: str,
        seen_urls: set[str],
        results: list[JobResult],
        memory: AgentMemory,
    ) -> bool:
        """Fetch and score one triaged URL. Returns True if accepted."""
        self._logger.info("Fetching job page: %s", url)
        try:
            job_text = self._tools.invoke("fetch_job_text", url)
        except Exception as exc:
            self._logger.warning("Failed to fetch %s: %s", url, exc)
            seen_urls.add(url)
            return False
        if not job_text:
            self._logger.info("Empty content for %s, skipping", url)
            seen_urls.add(url)
            return False
        if len(job_text) < 800:
            self._logger.info(
                "Content too short (%s chars) for %s, skipping", len(job_text), url
            )
            seen_urls.add(url)
            return False
        self._logger.info("Scoring job page: %s", url)
        try:
            evaluation = self._tools.invoke("evaluate_job", cv_text, job_text)
        except Exception as exc:
            self._logger.warning("Failed to score %s: %s", url, exc)
            seen_urls.add(url)
            return False
        seen_urls.add(url)
        accepted = evaluation.score >= self._config.min_score
        memory.record_evaluation(url, accepted=accepted, query=query)
        if accepted:
            results.append(
                JobResult(
                    title=self._extract_title(job_text),
                    company=self._extract_company(job_text),
                    url=url,
                    score=evaluation.score,
                    reason=evaluation.reason,
                    status="new",
                )
            )
            self._logger.info("Saved job (%s) with score %s", url, evaluation.score)
        else:
            self._logger.info("Rejected job (%s) with score %s", url, evaluation.score)
        return accepted

    def _make_plan(
        self,
        cv_text: str,
        preferences: dict[str, Any],
        memory: AgentMemory,
    ) -> SearchPlan:
        if not self._config.enable_planning:
            return SearchPlan()
        context = self._build_context(cv_text, preferences, [])
        context["memory"] = memory.summary_for_prompt()
        try:
            plan = self._llm_service.plan_search(context)
            self._logger.info(
                "Search plan: roles=%s skills=%s locations=%s",
                plan.target_roles,
                plan.key_skills,
                plan.locations,
            )
            return plan
        except Exception as exc:
            self._logger.warning("Planning failed, continuing without plan: %s", exc)
            return SearchPlan()

    def _reflect(
        self,
        context: dict[str, Any],
        history: list[dict[str, Any]],
        memory: AgentMemory,
    ) -> Reflection:
        if not self._config.enable_reflection:
            return self._heuristic_reflection(history)
        try:
            reflection = self._llm_service.reflect(
                context, history, self._tools.stats()
            )
        except Exception as exc:
            self._logger.warning(
                "LLM reflection failed, using heuristic fallback: %s", exc
            )
            reflection = self._heuristic_reflection(history)
        self._logger.info("Reflection: %s", reflection.assessment)
        memory.add_reflection(reflection.assessment)
        return reflection

    def _heuristic_reflection(self, history: list[dict[str, Any]]) -> Reflection:
        """Deterministic fallback when the LLM can't produce a reflection."""
        effective = [h["query"] for h in history if h.get("new", 0) > 0]
        ineffective = [h["query"] for h in history if h.get("new", 0) == 0]
        adjustments = []
        if ineffective:
            adjustments.append(
                "Avoid repeating queries that returned no new URLs; "
                "vary role keywords, seniority, and locations."
            )
        return Reflection(
            assessment=(
                f"{len(effective)}/{len(history)} queries produced new URLs."
            ),
            effective_queries=effective,
            ineffective_queries=ineffective,
            adjustments=adjustments,
        )

    def _build_context(
        self,
        cv_text: str,
        preferences: dict[str, Any],
        results: list[JobResult],
    ) -> dict[str, Any]:
        return {
            "cv_summary": cv_text[:1500],
            "preferences": preferences,
            "results": [item.model_dump() for item in results],
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
        # Also write a plain text file with one URL per line for easy use.
        results_txt = results_json.with_suffix(".txt")
        with results_txt.open("w", encoding="utf-8") as handle:
            for item in payload:
                url = item.get("url")
                if url:
                    handle.write(f"{url}\n")

    def _extract_title(self, job_text: str) -> str:
        words = job_text.split()
        return " ".join(words[:8])

    def _extract_company(self, job_text: str) -> str:
        return "Unknown"

    def _looks_like_listing(self, url: str) -> bool:
        return classify_url(url) != OTHER
