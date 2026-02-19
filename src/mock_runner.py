from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config_manager import Config, EffortBudget
from .models import JobEvaluation, SearchQueries
from .orchestrator import Orchestrator
from tests.mock_data import (
    MOCK_BRAVE_RESPONSE,
    MOCK_EVALUATION_RESPONSE,
    MOCK_JOB_TEXT,
    MOCK_QUERY_RESPONSE,
)


class MockLLM:
    def __init__(self, budget: EffortBudget):
        self._budget = budget
        self.calls = {"queries": 0, "eval": 0}

    def generate_search_queries(
        self, context: dict[str, Any], history: list[dict[str, Any]]
    ) -> SearchQueries:
        if not self._budget.can_call_llm():
            raise RuntimeError("Effort budget exceeded: LLM calls")
        self._budget.record_llm_call()
        self.calls["queries"] += 1
        return SearchQueries.model_validate(MOCK_QUERY_RESPONSE)

    def evaluate_job(self, cv: str, job_description: str) -> JobEvaluation:
        if not self._budget.can_call_llm():
            raise RuntimeError("Effort budget exceeded: LLM calls")
        self._budget.record_llm_call()
        self.calls["eval"] += 1
        return JobEvaluation.model_validate(MOCK_EVALUATION_RESPONSE)


class MockCrawler:
    def __init__(self, budget: EffortBudget):
        self._budget = budget
        self.search_calls: list[str] = []
        self.fetch_calls: list[str] = []

    def search(self, query: str) -> list[str]:
        if not self._budget.can_search():
            raise RuntimeError("Effort budget exceeded: search iterations")
        self._budget.record_search_iteration()
        self.search_calls.append(query)
        return [item["url"] for item in MOCK_BRAVE_RESPONSE["web"]["results"]]

    def fetch_job_text(self, url: str) -> str:
        self.fetch_calls.append(url)
        return MOCK_JOB_TEXT


@dataclass
class MockRunResult:
    results: list[Any]
    llm_calls: dict[str, int]
    search_calls: list[str]
    fetch_calls: list[str]


def run_mock_loop(root: Path) -> MockRunResult:
    config = Config(
        max_results=3,
        min_score=70,
        results_json="data/mock_results.json",
        results_csv="data/mock_results.csv",
        cache_path="data/mock_cache.json",
        llm_model="mock",
        llm_temperature=0.0,
        llm_max_retries=1,
        llm_min_delay_seconds=0,
        brave_endpoint="mock",
        results_per_query=2,
        request_timeout_seconds=1,
        search_min_delay_seconds=0,
        max_queries_per_iteration=3,
        budget=EffortBudget(max_llm_calls=10, max_search_iterations=2),
    )

    budget = config.budget
    llm = MockLLM(budget)
    crawler = MockCrawler(budget)
    orchestrator = Orchestrator(config, budget, llm, crawler)

    results = orchestrator.run(
        cv_text="I am a Junior Python dev...",
        preferences={"location": "Berlin"},
        cache_path=root / config.cache_path,
        results_json=root / config.results_json,
        results_csv=root / config.results_csv,
    )

    return MockRunResult(
        results=results,
        llm_calls=llm.calls,
        search_calls=crawler.search_calls,
        fetch_calls=crawler.fetch_calls,
    )
