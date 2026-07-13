from typing import Any

import src.crawler_engine as crawler_engine
from src.config_manager import Config, EffortBudget
from src.crawler_engine import CrawlerEngine


def _make_config() -> Config:
    return Config(
        max_results=1,
        min_score=50,
        results_json="data/unused.json",
        results_csv="data/unused.csv",
        cache_path="data/unused_cache.json",
        llm_model="mock",
        llm_temperature=0.0,
        llm_max_retries=1,
        llm_min_delay_seconds=0,
        brave_endpoint="mock",
        results_per_query=2,
        request_timeout_seconds=1,
        search_min_delay_seconds=0,
        max_queries_per_iteration=1,
        budget=EffortBudget(max_llm_calls=5, max_search_iterations=5),
    )


def _make_engine(monkeypatch, responses: list[dict[str, Any]]):
    config = _make_config()
    engine = CrawlerEngine(config, config.budget, brave_api_key=None)
    calls = {"count": 0}

    def fake_search(payload: dict[str, Any]) -> dict[str, Any]:
        index = min(calls["count"], len(responses) - 1)
        calls["count"] += 1
        return responses[index]

    monkeypatch.setattr(crawler_engine, "BRAVE_SEARCH", fake_search)
    monkeypatch.setattr(crawler_engine.time, "sleep", lambda seconds: None)
    return engine, config.budget, calls


GOOD_PAYLOAD = {
    "web": {"results": [{"url": "https://example.com/jobs/1", "title": "Job"}]}
}


def test_transient_error_is_retried(monkeypatch):
    engine, budget, calls = _make_engine(
        monkeypatch, [{"error": "Request failed"}, GOOD_PAYLOAD]
    )
    urls = engine.search("python jobs")
    assert urls == ["https://example.com/jobs/1"]
    assert calls["count"] == 2
    assert budget.search_iterations_used == 1


def test_persistent_error_returns_empty_without_consuming_budget(monkeypatch):
    engine, budget, calls = _make_engine(monkeypatch, [{"error": "Request failed"}])
    urls = engine.search("python jobs")
    assert urls == []
    assert calls["count"] == 3
    assert budget.search_iterations_used == 0


def test_legitimately_empty_results_are_not_retried(monkeypatch):
    engine, budget, calls = _make_engine(monkeypatch, [{"web": {"results": []}}])
    urls = engine.search("obscure query with no hits")
    assert urls == []
    assert calls["count"] == 1
    assert budget.search_iterations_used == 1
