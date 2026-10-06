from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import src.crawler_engine as crawler_engine
from src.agent_memory import AgentMemory
from src.crawler_engine import CrawlerEngine
from src.job_meta import extract_job_meta
from src.linkedin_source import (
    linkedin_job_id,
    linkedin_search_url,
    parse_posting,
    parse_search_results,
    posting_title,
)
from src.orchestrator import Orchestrator
from src.url_heuristics import POSTING, canonical_url, classify_url
from tests.test_agent_loop import (
    ScriptedCrawler,
    ScriptedLLM,
    _fixed_rotation,
    _make_config,
    _run,
)

# Trimmed shape of LinkedIn's guest search fragment (seeMoreJobPostings).
SEARCH_HTML = """
<li>
  <div class="base-card job-search-card" data-entity-urn="urn:li:jobPosting:3712345678">
    <a class="base-card__full-link"
       href="https://de.linkedin.com/jobs/view/python-developer-at-acme-3712345678?position=1&amp;pageNum=0&amp;refId=abc&amp;trackingId=xyz">
      Python Developer</a>
  </div>
</li>
<li>
  <div class="base-card job-search-card" data-entity-urn="urn:li:jobPosting:3799999999">
    <a class="base-card__full-link"
       href="https://de.linkedin.com/jobs/view/backend-engineer-at-globex-3799999999?position=2">
      Backend Engineer</a>
    <a href="https://de.linkedin.com/company/globex">Globex</a>
  </div>
</li>
"""

# Trimmed shape of LinkedIn's guest jobPosting fragment.
POSTING_HTML = """
<section class="top-card-layout">
  <h2 class="top-card-layout__title">Senior Python Developer</h2>
  <a class="topcard__org-name-link" href="https://de.linkedin.com/company/acme">
    Acme GmbH </a>
  <span class="topcard__flavor topcard__flavor--bullet">Berlin, Germany</span>
</section>
<div class="show-more-less-html__markup">
  We are hiring a Python developer in Berlin. """ + ("Details " * 150) + """
</div>
"""


def test_job_id_from_every_linkedin_url_shape():
    assert linkedin_job_id("https://www.linkedin.com/jobs/view/3712345678") == "3712345678"
    assert (
        linkedin_job_id(
            "https://de.linkedin.com/jobs/view/python-developer-at-acme-3712345678?refId=1"
        )
        == "3712345678"
    )
    assert (
        linkedin_job_id(
            "https://www.linkedin.com/jobs-guest/jobs/api/jobPosting/3712345678"
        )
        == "3712345678"
    )
    assert (
        linkedin_job_id(
            "https://www.linkedin.com/jobs/search/?currentJobId=3712345678&keywords=x"
        )
        == "3712345678"
    )
    assert linkedin_job_id("https://www.linkedin.com/jobs/search/?keywords=x") == ""
    assert linkedin_job_id("https://boards.greenhouse.io/acme/jobs/3712345678") == ""
    assert linkedin_job_id("https://notlinkedin.com/jobs/view/3712345678") == ""


def test_linkedin_posting_variants_share_one_canonical_url():
    canonical = "https://www.linkedin.com/jobs/view/3712345678"
    for url in (
        canonical,
        "https://de.linkedin.com/jobs/view/python-developer-at-acme-3712345678",
        "https://www.linkedin.com/jobs/view/3712345678/?refId=a&trackingId=b&position=3",
        "http://linkedin.com/jobs/view/3712345678",
    ):
        assert canonical_url(url) == canonical
    assert classify_url(canonical) == POSTING


def test_search_url_maps_brave_freshness_to_linkedin_time_filter():
    params = parse_qs(
        urlparse(linkedin_search_url("product manager", "Berlin", freshness="pm")).query
    )
    assert params["keywords"] == ["product manager"]
    assert params["location"] == ["Berlin"]
    assert params["f_TPR"] == ["r2592000"]
    unfiltered = parse_qs(urlparse(linkedin_search_url("pm", freshness="")).query)
    assert "f_TPR" not in unfiltered
    assert "location" not in unfiltered


def test_parse_search_results_returns_canonical_posting_urls_once():
    assert parse_search_results(SEARCH_HTML) == [
        "https://www.linkedin.com/jobs/view/3712345678",
        "https://www.linkedin.com/jobs/view/3799999999",
    ]
    assert parse_search_results("") == []


def test_posting_title_names_role_and_company_for_job_meta():
    assert parse_posting(POSTING_HTML) == ("Senior Python Developer", "Acme GmbH")
    title = posting_title(POSTING_HTML)
    meta = extract_job_meta("https://www.linkedin.com/jobs/view/3712345678", title)
    assert meta.title == "Senior Python Developer"
    assert meta.company == "Acme GmbH"


def _engine(monkeypatch, fetch):
    config = _make_config()
    engine = CrawlerEngine(config, config.budget, brave_api_key=None)
    monkeypatch.setattr(crawler_engine, "FETCH_JOB", fetch)
    monkeypatch.setattr(crawler_engine.time, "sleep", lambda seconds: None)
    return engine, config.budget


def test_search_linkedin_parses_results_and_uses_one_budget_unit(monkeypatch):
    requested: list[str] = []

    def fetch(data: dict[str, Any]) -> dict[str, Any]:
        requested.append(data["url"])
        return {"html": SEARCH_HTML, "final_url": data["url"], "status": 200}

    engine, budget = _engine(monkeypatch, fetch)
    urls = engine.search_linkedin("Python Developer", "Berlin")

    assert len(urls) == 2
    assert "jobs-guest/jobs/api/seeMoreJobPostings/search" in requested[0]
    assert budget.search_iterations_used == 1
    assert engine.last_search_error is None


def test_throttled_linkedin_search_reports_error_without_using_budget(monkeypatch):
    engine, budget = _engine(
        monkeypatch, lambda data: {"html": "", "final_url": data["url"], "status": 429}
    )
    assert engine.search_linkedin("Python Developer", "Berlin") == []
    assert engine.last_search_error == "LinkedIn: HTTP 429"
    assert budget.search_iterations_used == 0

    def boom(data: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError("connection reset")

    engine, budget = _engine(monkeypatch, boom)
    assert engine.search_linkedin("Python Developer", "Berlin") == []
    assert "connection reset" in (engine.last_search_error or "")
    assert budget.search_iterations_used == 0


def test_linkedin_posting_is_fetched_through_the_guest_endpoint(monkeypatch):
    requested: list[str] = []

    def fetch(data: dict[str, Any]) -> dict[str, Any]:
        requested.append(data["url"])
        return {"html": POSTING_HTML, "final_url": data["url"], "status": 200}

    engine, _ = _engine(monkeypatch, fetch)
    page = engine.fetch_page("https://www.linkedin.com/jobs/view/3712345678")

    assert requested == [
        "https://www.linkedin.com/jobs-guest/jobs/api/jobPosting/3712345678"
    ]
    assert page.title == "Senior Python Developer at Acme GmbH"
    assert "Berlin, Germany" in page.text
    assert len(page.text) > 800


def test_removed_linkedin_posting_is_gone(monkeypatch):
    engine, _ = _engine(
        monkeypatch, lambda data: {"html": "", "final_url": data["url"], "status": 404}
    )
    page = engine.fetch_page("https://www.linkedin.com/jobs/view/3712345678")
    assert page.gone
    assert page.text == ""


class LinkedInCrawler(ScriptedCrawler):
    def __init__(self, budget, url_map, linkedin_urls, linkedin_error=None):
        super().__init__(budget, url_map)
        self._linkedin_urls = linkedin_urls
        self._linkedin_error = linkedin_error
        self.linkedin_calls: list[tuple[str, str]] = []
        self.last_search_error = None

    def search(self, query, country=None, search_lang=None):
        self.last_search_error = None
        return super().search(query, country, search_lang)

    def search_linkedin(self, keywords: str, location: str = "") -> list[str]:
        self.linkedin_calls.append((keywords, location))
        self.last_search_error = self._linkedin_error
        if self._linkedin_error:
            return []
        self._budget.record_search_iteration()
        return self._linkedin_urls


def test_linkedin_search_runs_alongside_brave_queries(tmp_path: Path):
    config = _make_config(max_results=2, max_queries_per_iteration=1)
    llm = ScriptedLLM(config.budget, [["python jobs berlin"]])
    linkedin_posting = "https://www.linkedin.com/jobs/view/3712345678"
    crawler = LinkedInCrawler(
        config.budget,
        {"python jobs berlin": ["https://a.com/jobs/1"]},
        linkedin_urls=[linkedin_posting],
    )
    results = _run(
        _fixed_rotation(Orchestrator(config, config.budget, llm, crawler)), tmp_path
    )

    # Plan roles x plan location; the Brave cap of 1 still leaves the LLM query.
    assert crawler.linkedin_calls == [("Python Developer", "Berlin")]
    assert crawler.search_calls == ["python jobs berlin"]
    assert sorted(r.url for r in results) == [
        "https://a.com/jobs/1",
        linkedin_posting,
    ]


def test_linkedin_search_can_be_disabled(tmp_path: Path):
    config = _make_config(max_results=1, linkedin_search=False)
    llm = ScriptedLLM(config.budget, [["python jobs berlin"]])
    crawler = LinkedInCrawler(
        config.budget,
        {"python jobs berlin": ["https://a.com/jobs/1"]},
        linkedin_urls=["https://www.linkedin.com/jobs/view/3712345678"],
    )
    _run(Orchestrator(config, config.budget, llm, crawler), tmp_path)

    assert crawler.linkedin_calls == []


def test_failed_search_is_not_remembered_as_ineffective(tmp_path: Path):
    config = _make_config(max_results=1)
    llm = ScriptedLLM(config.budget, [["python jobs berlin"]])
    crawler = LinkedInCrawler(
        config.budget,
        {"python jobs berlin": ["https://a.com/jobs/1"]},
        linkedin_urls=[],
        linkedin_error="LinkedIn: HTTP 429",
    )
    _run(_fixed_rotation(Orchestrator(config, config.budget, llm, crawler)), tmp_path)

    assert crawler.linkedin_calls == [("Python Developer", "Berlin")]
    memory = AgentMemory.load(tmp_path / "memory.json")
    assert "linkedin: Python Developer | Berlin" not in memory.ineffective_queries()
