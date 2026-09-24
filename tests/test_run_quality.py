from __future__ import annotations

import json
from pathlib import Path

from src.crawler_engine import PageContent
from src.dashboard import collect_runs
from src.orchestrator import Orchestrator
from src.run_report import RunReport
from tests.test_agent_loop import (
    LONG_JOB_TEXT,
    ScriptedCrawler,
    ScriptedLLM,
    _make_config,
    _run,
)

POSTING = "https://boards.greenhouse.io/acme/jobs/123"


class TitledCrawler(ScriptedCrawler):
    """Crawler whose pages carry an HTML title, like the real engine."""

    class Page(tuple):
        title = "Job Application for Senior Python Developer at Acme"

    def fetch_page(self, url: str, use_browser_fallback: bool = False):
        self.fetch_calls.append(url)
        self.fetch_fallback_flags[url] = use_browser_fallback
        return self.Page((LONG_JOB_TEXT, self._link_map.get(url, [])))


def test_same_posting_behind_tracking_params_is_fetched_once(tmp_path: Path):
    config = _make_config(max_results=5)
    llm = ScriptedLLM(config.budget, [["python jobs berlin"]])
    crawler = ScriptedCrawler(
        config.budget,
        {
            "python jobs berlin": [
                POSTING,
                f"{POSTING}?t=Base10+job+board",
                f"{POSTING}?gh_src=abc#top",
            ]
        },
    )
    results = _run(Orchestrator(config, config.budget, llm, crawler), tmp_path)

    assert crawler.fetch_calls == [POSTING]
    assert [result.url for result in results] == [POSTING]


def test_result_title_and_company_come_from_the_page(tmp_path: Path):
    config = _make_config(max_results=1)
    llm = ScriptedLLM(config.budget, [["python jobs berlin"]])
    crawler = TitledCrawler(config.budget, {"python jobs berlin": [POSTING]})
    results = _run(Orchestrator(config, config.budget, llm, crawler), tmp_path)

    assert results[0].title == "Senior Python Developer"
    assert results[0].company == "Acme"
    assert results[0].found_at > 0


def test_results_from_earlier_runs_are_kept(tmp_path: Path):
    results_json = tmp_path / "results.json"
    results_json.write_text(
        json.dumps(
            [
                {
                    "title": "Older job",
                    "company": "Previously",
                    "url": "https://boards.greenhouse.io/previously/jobs/1",
                    "score": 90,
                    "reason": "found last week",
                    "status": "new",
                }
            ]
        ),
        encoding="utf-8",
    )
    config = _make_config(max_results=1)
    llm = ScriptedLLM(config.budget, [["python jobs berlin"]])
    crawler = ScriptedCrawler(config.budget, {"python jobs berlin": [POSTING]})
    _run(Orchestrator(config, config.budget, llm, crawler), tmp_path)

    stored = json.loads(results_json.read_text(encoding="utf-8"))
    urls = {item["url"] for item in stored}
    assert urls == {POSTING, "https://boards.greenhouse.io/previously/jobs/1"}
    older = next(item for item in stored if item["url"].endswith("/jobs/1"))
    assert older["status"] == "seen"


def test_run_report_records_why_pages_were_dropped(tmp_path: Path):
    short_text = "Too short to score."

    class ShortPageCrawler(ScriptedCrawler):
        def fetch_page(self, url: str, use_browser_fallback: bool = False):
            self.fetch_calls.append(url)
            return short_text, []

    config = _make_config(max_results=1)
    llm = ScriptedLLM(config.budget, [["python jobs berlin"]])
    crawler = ShortPageCrawler(config.budget, {"python jobs berlin": [POSTING]})
    _run(Orchestrator(config, config.budget, llm, crawler), tmp_path)

    runs = json.loads((tmp_path / "runs.json").read_text(encoding="utf-8"))
    assert runs[-1]["counters"]["skipped_too_short"] == 1
    assert runs[-1]["accepted"] == 0
    assert runs[-1]["queries"] == ["python jobs berlin"]


def test_run_report_records_fetch_errors(tmp_path: Path):
    class BrokenCrawler(ScriptedCrawler):
        def fetch_page(self, url: str, use_browser_fallback: bool = False):
            raise RuntimeError("connection reset")

    config = _make_config(max_results=1)
    llm = ScriptedLLM(config.budget, [["python jobs berlin"]])
    crawler = BrokenCrawler(config.budget, {"python jobs berlin": [POSTING]})
    _run(Orchestrator(config, config.budget, llm, crawler), tmp_path)

    runs = json.loads((tmp_path / "runs.json").read_text(encoding="utf-8"))
    error = runs[-1]["errors"][0]
    assert error["kind"] == "fetch_failed"
    assert "connection reset" in error["error"]


def test_run_history_is_capped_and_appended(tmp_path: Path):
    path = tmp_path / "runs.json"
    for index in range(35):
        report = RunReport.start()
        report.count("urls_found", index)
        report.finish(queries=[f"q{index}"], accepted=0)
        report.save(path)

    history = json.loads(path.read_text(encoding="utf-8"))
    assert len(history) == 30
    assert history[-1]["queries"] == ["q34"]


def test_dashboard_lists_runs_for_each_source(tmp_path: Path):
    report = RunReport.start()
    report.count("skipped_stale", 3)
    report.finish(queries=["ml engineer berlin"], accepted=1)
    report.save(tmp_path / "runs.json")
    (tmp_path / "results.json").write_text("[]", encoding="utf-8")

    runs = collect_runs(tmp_path)
    assert runs[0]["source"] == "results"
    assert runs[0]["accepted"] == 1
    labels = {counter["label"] for counter in runs[0]["counters"]}
    assert "closed/filled marker" in labels


def test_health_flags_a_user_that_stopped_being_scanned(tmp_path: Path):
    from src.dashboard import collect_health
    from src.user_store import UserStore

    store = UserStore(tmp_path)
    record = store.load("42")
    record.name = "Tim"
    record.state = "active"
    record.last_scan_at = 1_000_000.0
    store.save(record)

    health = collect_health(tmp_path, now=1_000_000.0 + 40 * 3600)
    assert health["level"] == "error"
    assert any("not been scanned" in issue["message"] for issue in health["issues"])


def test_health_is_quiet_after_a_recent_scan(tmp_path: Path):
    from src.dashboard import collect_health
    from src.user_store import UserStore

    store = UserStore(tmp_path)
    record = store.load("42")
    record.name = "Tim"
    record.state = "active"
    record.last_scan_at = 1_000_000.0
    store.save(record)

    def status() -> dict[str, object]:
        return {"started_at": 1_000_000.0, "queued_or_running": []}

    health = collect_health(tmp_path, status, now=1_000_000.0 + 3600)
    assert health["level"] == "ok"
    assert health["issues"] == []


def test_health_flags_a_streak_of_empty_runs(tmp_path: Path):
    from src.dashboard import collect_health

    (tmp_path / "results.json").write_text("[]", encoding="utf-8")
    for _ in range(3):
        report = RunReport.start()
        report.finish(queries=["ml engineer berlin"], accepted=0)
        report.save(tmp_path / "runs.json")

    def status() -> dict[str, object]:
        return {"started_at": 0.0, "queued_or_running": []}

    health = collect_health(tmp_path, status)
    assert any("accepted no jobs" in issue["message"] for issue in health["issues"])


def test_empty_scan_explains_where_the_pages_went():
    from src.bot_service import _empty_scan_explanation

    report = RunReport.start()
    report.count("pages_fetched", 6)
    report.count("skipped_stale", 4)
    report.count("rejected_low_score", 2)
    message = _empty_scan_explanation(report)
    assert "6 job page(s)" in message
    assert "4 closed/filled marker" in message
    assert "2 rejected: score below threshold" in message


def test_empty_scan_without_any_fetch_says_so():
    from src.bot_service import _empty_scan_explanation

    assert "could not fetch" in _empty_scan_explanation(RunReport.start())


def test_removed_posting_is_reported_as_gone_not_as_an_error(tmp_path: Path):
    class GoneCrawler(ScriptedCrawler):
        def fetch_page(self, url: str, use_browser_fallback: bool = False):
            self.fetch_calls.append(url)
            return PageContent("", [], gone=True)

    config = _make_config(max_results=1)
    llm = ScriptedLLM(config.budget, [["python jobs berlin"]])
    crawler = GoneCrawler(config.budget, {"python jobs berlin": [POSTING]})
    _run(Orchestrator(config, config.budget, llm, crawler), tmp_path)

    counters = json.loads((tmp_path / "runs.json").read_text(encoding="utf-8"))[-1][
        "counters"
    ]
    assert counters["skipped_gone"] == 1
    assert "skipped_empty" not in counters


def test_client_error_is_not_retried_and_marks_the_posting_gone(monkeypatch):
    import src.crawler_engine as crawler_engine

    calls = {"count": 0}

    def fake_fetch(data: dict[str, object]) -> dict[str, object]:
        calls["count"] += 1
        return {"html": "", "final_url": data["url"], "status": 404}

    monkeypatch.setattr(crawler_engine, "FETCH_JOB", fake_fetch)
    config = _make_config(max_results=1)
    engine = crawler_engine.CrawlerEngine(config, config.budget, "key")

    page = engine.fetch_page(POSTING)
    assert page.gone is True
    assert page.text == ""
    assert calls["count"] == 1


def test_empty_scan_says_when_everything_was_already_checked():
    from src.bot_service import _empty_scan_explanation

    report = RunReport.start()
    report.count("urls_found", 40)
    report.count("already_seen", 40)
    message = _empty_scan_explanation(report)
    assert "already checked" in message


def test_query_generation_failure_keeps_the_results_already_found(tmp_path: Path):
    class FlakyLLM(ScriptedLLM):
        def generate_search_queries(self, context, history):
            if self.query_contexts:
                self.query_contexts.append(context)
                raise RuntimeError("LLM returned an empty response")
            return super().generate_search_queries(context, history)

    config = _make_config(max_results=5)
    llm = FlakyLLM(config.budget, [["python jobs berlin"]])
    crawler = ScriptedCrawler(config.budget, {"python jobs berlin": [POSTING]})
    results = _run(Orchestrator(config, config.budget, llm, crawler), tmp_path)

    assert [result.url for result in results] == [POSTING]
    run = json.loads((tmp_path / "runs.json").read_text(encoding="utf-8"))[-1]
    assert run["errors"][0]["kind"] == "query_generation_failed"


def test_empty_llm_content_is_retried_then_reported(monkeypatch):
    from src.config_manager import EffortBudget
    from src.llm_service import LLMService

    calls = {"count": 0}

    def fake_completion(**kwargs):
        calls["count"] += 1
        return {"choices": [{"message": {"content": None}}]}

    monkeypatch.setattr("src.llm_service.completion", fake_completion)
    config = _make_config(max_results=1, llm_max_retries=2)
    service = LLMService(config, EffortBudget(max_llm_calls=5, max_search_iterations=5), "key")

    try:
        service._call_llm("prompt")
    except RuntimeError as exc:
        assert "empty response" in str(exc)
    else:
        raise AssertionError("expected a RuntimeError")
    assert calls["count"] == 2
