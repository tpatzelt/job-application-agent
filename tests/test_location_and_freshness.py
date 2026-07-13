"""Tests for the location-mismatch rejection, stale-posting skipping, and
Brave freshness/country search parameters."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src import crawler_engine
from src.config_manager import Config, EffortBudget
from src.crawler_engine import CrawlerEngine
from src.llm_service import LLMService
from src.models import JobEvaluation
from src.orchestrator import Orchestrator

from tests.test_agent_loop import (
    LONG_JOB_TEXT,
    ScriptedCrawler,
    ScriptedLLM,
    _make_config,
    _run,
)

POSTING_URL = "https://boards.greenhouse.io/acme/jobs/123"


class LocationAwareLLM(ScriptedLLM):
    """Scores high but flags the job as a location mismatch."""

    def evaluate_job(
        self,
        cv: str,
        job_description: str,
        preferences: dict[str, Any] | None = None,
    ) -> JobEvaluation:
        self._record_call()
        self.eval_preferences = preferences
        return JobEvaluation(
            score=95, reason="great match, wrong city", location_match=False
        )


def test_location_mismatch_rejected_despite_high_score(tmp_path: Path):
    config = _make_config(max_results=1)
    llm = LocationAwareLLM(config.budget, [["python jobs berlin"]])
    crawler = ScriptedCrawler(
        config.budget, {"python jobs berlin": [POSTING_URL]}
    )
    results = _run(Orchestrator(config, config.budget, llm, crawler), tmp_path)

    assert results == []
    assert llm.eval_preferences == {"location": "Berlin"}


def test_preferences_passed_to_evaluation(tmp_path: Path):
    config = _make_config(max_results=1)
    llm = ScriptedLLM(config.budget, [["python jobs berlin"]])
    crawler = ScriptedCrawler(
        config.budget, {"python jobs berlin": [POSTING_URL]}
    )
    results = _run(Orchestrator(config, config.budget, llm, crawler), tmp_path)

    assert len(results) == 1
    assert llm.eval_preferences == {"location": "Berlin"}


def test_stale_posting_skipped_without_scoring(tmp_path: Path):
    stale_text = (
        "Senior Python Developer at Acme. "
        + ("Great role. " * 100)
        + "This job is no longer available."
    )

    class StaleCrawler(ScriptedCrawler):
        def fetch_job_text(
            self, url: str, use_browser_fallback: bool = False
        ) -> str:
            self.fetch_calls.append(url)
            return stale_text

    config = _make_config(max_results=1)
    llm = ScriptedLLM(config.budget, [["python jobs berlin"]])
    crawler = StaleCrawler(
        config.budget, {"python jobs berlin": [POSTING_URL]}
    )
    orchestrator = Orchestrator(config, config.budget, llm, crawler)
    results = _run(orchestrator, tmp_path)

    assert results == []
    assert crawler.fetch_calls == [POSTING_URL]
    # The stale page must never reach the (budget-consuming) evaluator.
    assert orchestrator._tools.get("evaluate_job").calls == 0


def test_country_derived_from_preferences_and_passed_to_search(tmp_path: Path):
    config = _make_config(max_results=1)
    llm = ScriptedLLM(config.budget, [["python jobs berlin"]])
    crawler = ScriptedCrawler(
        config.budget, {"python jobs berlin": [POSTING_URL]}
    )
    orchestrator = Orchestrator(config, config.budget, llm, crawler)
    orchestrator.run(
        cv_text="Python developer CV",
        preferences={"location": "Berlin, Germany", "locations": ["Berlin, Germany"]},
        cache_path=tmp_path / "cache.json",
        results_json=tmp_path / "results.json",
        results_csv=tmp_path / "results.csv",
        memory_path=tmp_path / "memory.json",
    )

    assert crawler.search_countries == ["DE"]


def _search_config(**overrides: Any) -> Config:
    return _make_config(
        budget=EffortBudget(max_llm_calls=5, max_search_iterations=5),
        **overrides,
    )


def test_brave_params_include_freshness_and_country(monkeypatch):
    captured: dict[str, Any] = {}

    def fake_search(payload: dict[str, Any]) -> dict[str, Any]:
        captured.update(payload)
        return {"web": {"results": [{"url": POSTING_URL}]}}

    monkeypatch.setattr(crawler_engine, "BRAVE_SEARCH", fake_search)
    monkeypatch.setattr(crawler_engine.time, "sleep", lambda seconds: None)
    config = _search_config(search_freshness="pm")
    engine = CrawlerEngine(config, config.budget, "key")

    urls = engine.search("ml engineer berlin", country="DE")

    assert urls == [POSTING_URL]
    assert captured["params"]["freshness"] == "pm"
    assert captured["params"]["country"] == "DE"


def test_brave_params_omit_freshness_and_country_when_unset(monkeypatch):
    captured: dict[str, Any] = {}

    def fake_search(payload: dict[str, Any]) -> dict[str, Any]:
        captured.update(payload)
        return {"web": {"results": []}}

    monkeypatch.setattr(crawler_engine, "BRAVE_SEARCH", fake_search)
    monkeypatch.setattr(crawler_engine.time, "sleep", lambda seconds: None)
    config = _search_config(search_freshness="")
    engine = CrawlerEngine(config, config.budget, "key")

    engine.search("ml engineer berlin")

    assert "freshness" not in captured["params"]
    assert "country" not in captured["params"]


def test_evaluation_prompt_contains_location_rules():
    config = _make_config()
    service = LLMService(config, config.budget, api_key=None)
    prompt = service._build_evaluation_prompt(
        "cv text",
        "job text",
        {"location": "Berlin, Germany", "locations": ["Berlin, Germany"]},
    )
    payload = json.loads(prompt)

    assert payload["preferred_locations"] == ["Berlin, Germany"]
    assert "location_match" in payload["output_schema"]
    assert any("location_match" in rule for rule in payload["rules"])


def test_normalize_payload_coerces_location_match_string():
    config = _make_config()
    service = LLMService(config, config.budget, api_key=None)

    payload = service._normalize_payload(
        {"score": 80, "reason": "ok", "location_match": "False"}
    )
    assert payload["location_match"] is False

    payload = service._normalize_payload(
        {"score": 80, "reason": "ok", "location_match": "true"}
    )
    assert payload["location_match"] is True


def test_redirected_off_posting_signal():
    from src.page_signals import redirected_off_posting

    dead = redirected_off_posting(
        "https://boards.greenhouse.io/acme/jobs/123",
        "https://job-boards.greenhouse.io/acme?error=true",
    )
    assert dead

    moved_but_alive = redirected_off_posting(
        "https://boards.greenhouse.io/acme/jobs/123",
        "https://job-boards.greenhouse.io/acme/jobs/123",
    )
    assert not moved_but_alive

    assert not redirected_off_posting(
        "https://boards.greenhouse.io/acme/jobs/123", None
    )
    # Non-posting URLs redirect legitimately all the time.
    assert not redirected_off_posting(
        "https://acme.com/careers", "https://acme.com/en/careers"
    )


def test_fetch_page_drops_redirected_dead_posting(monkeypatch):
    def fake_plain(data):
        return {
            "html": "<html><body><p>" + ("openings " * 300) + "</p></body></html>",
            "final_url": "https://job-boards.greenhouse.io/acme?error=true",
        }

    config = _search_config()
    engine = CrawlerEngine(config, config.budget, brave_api_key=None)
    monkeypatch.setattr(crawler_engine, "FETCH_JOB", fake_plain)

    text, links = engine.fetch_page("https://boards.greenhouse.io/acme/jobs/123")
    assert text == ""
    assert links == []


def test_fetch_page_keeps_posting_that_redirects_to_posting(monkeypatch):
    def fake_plain(data):
        return {
            "html": "<html><body><p>" + ("job details " * 300) + "</p></body></html>",
            "final_url": "https://job-boards.greenhouse.io/acme/jobs/123",
        }

    config = _search_config()
    engine = CrawlerEngine(config, config.budget, brave_api_key=None)
    monkeypatch.setattr(crawler_engine, "FETCH_JOB", fake_plain)

    text, links = engine.fetch_page("https://boards.greenhouse.io/acme/jobs/123")
    assert "job details" in text


def test_page_without_location_mention_skipped_before_scoring(tmp_path):
    no_location_text = (
        "We are hiring a Python developer in Chicago. " + ("Details " * 200)
    )

    class WrongCityCrawler(ScriptedCrawler):
        def fetch_job_text(
            self, url: str, use_browser_fallback: bool = False
        ) -> str:
            self.fetch_calls.append(url)
            return no_location_text

    config = _make_config(max_results=1)
    llm = ScriptedLLM(config.budget, [["python jobs berlin"]])
    crawler = WrongCityCrawler(
        config.budget, {"python jobs berlin": [POSTING_URL]}
    )
    orchestrator = Orchestrator(config, config.budget, llm, crawler)
    results = _run(orchestrator, tmp_path)

    assert results == []
    assert orchestrator._tools.get("evaluate_job").calls == 0


def test_freshness_not_applied_to_site_queries(monkeypatch):
    captured: dict[str, Any] = {}

    def fake_search(payload: dict[str, Any]) -> dict[str, Any]:
        captured.update(payload)
        return {"web": {"results": []}}

    monkeypatch.setattr(crawler_engine, "BRAVE_SEARCH", fake_search)
    monkeypatch.setattr(crawler_engine.time, "sleep", lambda seconds: None)
    config = _search_config(search_freshness="pm")
    engine = CrawlerEngine(config, config.budget, "key")

    engine.search("site:jobs.lever.co ml engineer berlin", country="DE")

    assert "freshness" not in captured["params"]
    assert captured["params"]["country"] == "DE"
