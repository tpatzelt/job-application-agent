from typing import Any

import src.crawler_engine as crawler_engine
from src.config_manager import Config, EffortBudget
from src.crawler_engine import CrawlerEngine

LONG_HTML = "<html><body><p>" + ("job details " * 200) + "</p></body></html>"
SHORT_HTML = "<html><body><p>Loading...</p></body></html>"
BROWSER_HTML = "<html><body><p>" + ("rendered job posting " * 200) + "</p></body></html>"


def _make_config(**overrides: Any) -> Config:
    params: dict[str, Any] = dict(
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
        results_per_query=1,
        request_timeout_seconds=1,
        search_min_delay_seconds=0,
        max_queries_per_iteration=1,
        budget=EffortBudget(max_llm_calls=1, max_search_iterations=1),
    )
    params.update(overrides)
    return Config(**params)


def _make_engine(monkeypatch, plain_html: str, browser_html: str, **config_overrides: Any):
    config = _make_config(**config_overrides)
    engine = CrawlerEngine(config, config.budget, brave_api_key=None)
    calls = {"plain": 0, "browser": 0}

    def fake_plain(data: dict[str, Any]) -> str:
        calls["plain"] += 1
        return plain_html

    def fake_browser(data: dict[str, Any]) -> str:
        calls["browser"] += 1
        if browser_html == "RAISE":
            raise RuntimeError("no chrome available")
        return browser_html

    monkeypatch.setattr(crawler_engine, "FETCH_JOB", fake_plain)
    monkeypatch.setattr(crawler_engine, "FETCH_JOB_BROWSER", fake_browser)
    return engine, calls


def test_long_plain_content_skips_browser(monkeypatch):
    engine, calls = _make_engine(monkeypatch, LONG_HTML, BROWSER_HTML)
    text = engine.fetch_job_text("https://a.com/jobs/1", use_browser_fallback=True)
    assert "job details" in text
    assert calls["browser"] == 0


def test_short_plain_content_triggers_browser(monkeypatch):
    engine, calls = _make_engine(monkeypatch, SHORT_HTML, BROWSER_HTML)
    text = engine.fetch_job_text("https://a.com/jobs/1", use_browser_fallback=True)
    assert "rendered job posting" in text
    assert calls == {"plain": 1, "browser": 1}


def test_browser_failure_returns_plain_text(monkeypatch):
    engine, calls = _make_engine(monkeypatch, SHORT_HTML, "RAISE")
    text = engine.fetch_job_text("https://a.com/jobs/1", use_browser_fallback=True)
    assert text == "Loading..."
    assert calls["browser"] == 1


def test_browser_shorter_result_keeps_plain_text(monkeypatch):
    engine, calls = _make_engine(monkeypatch, SHORT_HTML, "<html><body>x</body></html>")
    text = engine.fetch_job_text("https://a.com/jobs/1", use_browser_fallback=True)
    assert text == "Loading..."
    assert calls["browser"] == 1


def test_no_fallback_flag_skips_browser(monkeypatch):
    engine, calls = _make_engine(monkeypatch, SHORT_HTML, BROWSER_HTML)
    text = engine.fetch_job_text("https://a.com/jobs/1", use_browser_fallback=False)
    assert text == "Loading..."
    assert calls["browser"] == 0


def test_browser_fallback_disabled_by_config(monkeypatch):
    engine, calls = _make_engine(
        monkeypatch, SHORT_HTML, BROWSER_HTML, browser_fallback=False
    )
    text = engine.fetch_job_text("https://a.com/jobs/1", use_browser_fallback=True)
    assert text == "Loading..."
    assert calls["browser"] == 0


def test_empty_plain_content_triggers_browser(monkeypatch):
    engine, calls = _make_engine(monkeypatch, "", BROWSER_HTML)
    text = engine.fetch_job_text("https://a.com/jobs/1", use_browser_fallback=True)
    assert "rendered job posting" in text
    assert calls["browser"] == 1


LINKED_HTML = (
    "<html><body>"
    '<a href="/careers/jobs/123">Job</a>'
    '<a href="https://boards.greenhouse.io/acme/jobs/456#app">Apply</a>'
    '<a href="/careers/jobs/123">Duplicate</a>'
    '<a href="mailto:hr@acme.com">Mail</a>'
    "<p>" + ("job details " * 200) + "</p>"
    "</body></html>"
)


def test_fetch_page_returns_absolute_deduped_links(monkeypatch):
    engine, calls = _make_engine(monkeypatch, LINKED_HTML, BROWSER_HTML)
    text, links = engine.fetch_page("https://acme.com/careers")
    assert "job details" in text
    assert links == [
        "https://acme.com/careers/jobs/123",
        "https://boards.greenhouse.io/acme/jobs/456",
    ]
    assert calls["browser"] == 0


def test_fetch_page_browser_fallback_returns_browser_links(monkeypatch):
    browser_html = (
        '<html><body><a href="/jobs/789">Job</a>'
        "<p>" + ("rendered job posting " * 200) + "</p></body></html>"
    )
    engine, calls = _make_engine(monkeypatch, SHORT_HTML, browser_html)
    text, links = engine.fetch_page(
        "https://acme.com/careers", use_browser_fallback=True
    )
    assert "rendered job posting" in text
    assert links == ["https://acme.com/jobs/789"]
    assert calls["browser"] == 1
