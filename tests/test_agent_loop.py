from pathlib import Path
from typing import Any

from src.agent_memory import AgentMemory
from src.config_manager import Config, EffortBudget
from src.models import JobEvaluation, Reflection, SearchPlan, SearchQueries
from src.orchestrator import Orchestrator

LONG_JOB_TEXT = "We are hiring a Python developer in Berlin. " + ("Details " * 200)


def _make_config(**overrides: Any) -> Config:
    params: dict[str, Any] = dict(
        max_results=5,
        min_score=70,
        results_json="data/unused.json",
        results_csv="data/unused.csv",
        cache_path="data/unused_cache.json",
        llm_model="mock",
        llm_temperature=0.0,
        llm_max_retries=1,
        llm_min_delay_seconds=0,
        brave_endpoint="mock",
        results_per_query=5,
        request_timeout_seconds=1,
        search_min_delay_seconds=0,
        max_queries_per_iteration=5,
        budget=EffortBudget(max_llm_calls=50, max_search_iterations=20),
        # Off by default so scripted-query tests stay deterministic;
        # boost/exclusion tests enable them explicitly.
        ats_query_boost=False,
        company_query_boost=False,
        exclude_aggregator_sites=False,
    )
    params.update(overrides)
    return Config(**params)


class ScriptedLLM:
    """Fake LLM that serves scripted query batches and records contexts."""

    def __init__(
        self,
        budget: EffortBudget,
        query_batches: list[list[str]],
        score: int = 85,
        fail_reflect: bool = False,
        fail_plan: bool = False,
        plan_companies: list[str] | None = None,
    ):
        self._budget = budget
        self._query_batches = query_batches
        self._score = score
        self._fail_reflect = fail_reflect
        self._fail_plan = fail_plan
        self._plan_companies = plan_companies or []
        self.query_contexts: list[dict[str, Any]] = []
        self.plan_calls = 0
        self.reflect_calls = 0

    def _record_call(self) -> None:
        if not self._budget.can_call_llm():
            raise RuntimeError("Effort budget exceeded: LLM calls")
        self._budget.record_llm_call()

    def generate_search_queries(
        self, context: dict[str, Any], history: list[dict[str, Any]]
    ) -> SearchQueries:
        self._record_call()
        self.query_contexts.append(context)
        index = min(len(self.query_contexts) - 1, len(self._query_batches) - 1)
        return SearchQueries(queries=self._query_batches[index])

    def plan_search(self, context: dict[str, Any]) -> SearchPlan:
        self._record_call()
        self.plan_calls += 1
        if self._fail_plan:
            raise RuntimeError("planning exploded")
        return SearchPlan(
            target_roles=["Python Developer"],
            key_skills=["Python"],
            locations=["Berlin"],
            target_companies=self._plan_companies,
            strategy="Focus on employer career sites.",
        )

    def reflect(
        self,
        context: dict[str, Any],
        history: list[dict[str, Any]],
        tool_stats: dict[str, Any],
    ) -> Reflection:
        self._record_call()
        self.reflect_calls += 1
        if self._fail_reflect:
            raise RuntimeError("reflection exploded")
        return Reflection(
            assessment="LLM reflection",
            adjustments=["Try different keywords."],
        )

    def evaluate_job(
        self,
        cv: str,
        job_description: str,
        preferences: dict[str, Any] | None = None,
    ) -> JobEvaluation:
        self._record_call()
        self.eval_preferences = preferences
        return JobEvaluation(score=self._score, reason="scripted")


class ScriptedCrawler:
    """Fake crawler that maps queries to fixed URL lists."""

    def __init__(
        self,
        budget: EffortBudget,
        url_map: dict[str, list[str]],
        link_map: dict[str, list[str]] | None = None,
    ):
        self._budget = budget
        self._url_map = url_map
        self._link_map = link_map or {}
        self.search_calls: list[str] = []
        self.search_countries: list[str | None] = []
        self.search_langs: list[str | None] = []
        self.fetch_calls: list[str] = []
        self.fetch_fallback_flags: dict[str, bool] = {}

    def search(
        self,
        query: str,
        country: str | None = None,
        search_lang: str | None = None,
    ) -> list[str]:
        self._budget.record_search_iteration()
        self.search_calls.append(query)
        self.search_countries.append(country)
        self.search_langs.append(search_lang)
        return self._url_map.get(query, [])

    def fetch_job_text(self, url: str, use_browser_fallback: bool = False) -> str:
        self.fetch_calls.append(url)
        self.fetch_fallback_flags[url] = use_browser_fallback
        return LONG_JOB_TEXT

    def fetch_page(
        self, url: str, use_browser_fallback: bool = False
    ) -> tuple[str, list[str]]:
        self.fetch_calls.append(url)
        self.fetch_fallback_flags[url] = use_browser_fallback
        return LONG_JOB_TEXT, self._link_map.get(url, [])


def _run(orchestrator: Orchestrator, tmp_path: Path) -> list[Any]:
    return orchestrator.run(
        cv_text="Python developer CV",
        preferences={"location": "Berlin"},
        cache_path=tmp_path / "cache.json",
        results_json=tmp_path / "results.json",
        results_csv=tmp_path / "results.csv",
        memory_path=tmp_path / "memory.json",
    )


def test_plan_generated_once_and_passed_to_query_context(tmp_path: Path):
    config = _make_config(max_results=1)
    llm = ScriptedLLM(config.budget, [["python jobs berlin"]])
    crawler = ScriptedCrawler(
        config.budget, {"python jobs berlin": ["https://a.com/jobs/1"]}
    )
    results = _run(Orchestrator(config, config.budget, llm, crawler), tmp_path)

    assert llm.plan_calls == 1
    assert len(results) == 1
    context = llm.query_contexts[0]
    assert context["plan"]["target_roles"] == ["Python Developer"]
    assert "memory" in context


def test_reflection_feeds_next_iteration_context(tmp_path: Path):
    config = _make_config(max_results=3)
    llm = ScriptedLLM(
        config.budget,
        [["query one"], ["query two"], ["query two"]],
    )
    crawler = ScriptedCrawler(
        config.budget,
        {
            "query one": ["https://a.com/jobs/1"],
            "query two": ["https://b.com/jobs/2"],
        },
    )
    _run(Orchestrator(config, config.budget, llm, crawler), tmp_path)

    assert llm.reflect_calls >= 1
    assert "reflection" not in llm.query_contexts[0]
    assert llm.query_contexts[1]["reflection"]["assessment"] == "LLM reflection"


def test_reflection_failure_falls_back_to_heuristic(tmp_path: Path):
    config = _make_config(max_results=3)
    llm = ScriptedLLM(
        config.budget,
        [["query one"], ["query two"], ["query two"]],
        fail_reflect=True,
    )
    crawler = ScriptedCrawler(
        config.budget,
        {
            "query one": ["https://a.com/jobs/1"],
            "query two": ["https://b.com/jobs/2"],
        },
    )
    _run(Orchestrator(config, config.budget, llm, crawler), tmp_path)

    reflection = llm.query_contexts[1]["reflection"]
    assert "queries produced new URLs" in reflection["assessment"]
    assert reflection["effective_queries"] == ["query one"]


def test_planning_failure_is_non_fatal(tmp_path: Path):
    config = _make_config(max_results=1)
    llm = ScriptedLLM(
        config.budget, [["python jobs berlin"]], fail_plan=True
    )
    crawler = ScriptedCrawler(
        config.budget, {"python jobs berlin": ["https://a.com/jobs/1"]}
    )
    results = _run(Orchestrator(config, config.budget, llm, crawler), tmp_path)

    assert len(results) == 1
    assert llm.query_contexts[0]["plan"]["target_roles"] == []


def test_planning_disabled_skips_plan_call(tmp_path: Path):
    config = _make_config(max_results=1, enable_planning=False)
    llm = ScriptedLLM(config.budget, [["python jobs berlin"]])
    crawler = ScriptedCrawler(
        config.budget, {"python jobs berlin": ["https://a.com/jobs/1"]}
    )
    _run(Orchestrator(config, config.budget, llm, crawler), tmp_path)

    assert llm.plan_calls == 0


def test_known_ineffective_queries_are_skipped(tmp_path: Path):
    memory = AgentMemory()
    memory.record_query("dead end query", urls_found=0, new_urls=0)
    memory.save(tmp_path / "memory.json")

    config = _make_config(max_results=1)
    llm = ScriptedLLM(
        config.budget, [["dead end query", "python jobs berlin"]]
    )
    crawler = ScriptedCrawler(
        config.budget,
        {
            "dead end query": ["https://a.com/jobs/should-not-happen"],
            "python jobs berlin": ["https://a.com/jobs/1"],
        },
    )
    _run(Orchestrator(config, config.budget, llm, crawler), tmp_path)

    assert crawler.search_calls == ["python jobs berlin"]


def test_repeated_queries_stop_the_loop(tmp_path: Path):
    config = _make_config(max_results=10)
    llm = ScriptedLLM(config.budget, [["same query"]])
    crawler = ScriptedCrawler(
        config.budget, {"same query": ["https://a.com/jobs/1"]}
    )
    _run(Orchestrator(config, config.budget, llm, crawler), tmp_path)

    # Second iteration regenerates the same query, which is skipped;
    # no progress means the loop terminates instead of spinning.
    assert crawler.search_calls == ["same query"]
    assert len(llm.query_contexts) == 2


def test_memory_persisted_with_query_and_domain_outcomes(tmp_path: Path):
    config = _make_config(max_results=1)
    llm = ScriptedLLM(config.budget, [["python jobs berlin"]])
    crawler = ScriptedCrawler(
        config.budget, {"python jobs berlin": ["https://a.com/jobs/1"]}
    )
    _run(Orchestrator(config, config.budget, llm, crawler), tmp_path)

    memory = AgentMemory.load(tmp_path / "memory.json")
    assert memory.known_queries() == {"python jobs berlin"}
    assert memory.effective_queries() == ["python jobs berlin"]
    assert memory.productive_domains() == ["a.com"]


def test_rejected_jobs_recorded_but_not_returned(tmp_path: Path):
    config = _make_config(max_results=1, min_score=90)
    llm = ScriptedLLM(config.budget, [["python jobs berlin"]], score=50)
    crawler = ScriptedCrawler(
        config.budget, {"python jobs berlin": ["https://a.com/jobs/1"]}
    )
    results = _run(Orchestrator(config, config.budget, llm, crawler), tmp_path)

    assert results == []
    memory = AgentMemory.load(tmp_path / "memory.json")
    assert memory.effective_queries() == []
    assert memory.productive_domains() == []


def test_postings_processed_before_index_pages(tmp_path: Path):
    config = _make_config(max_results=2)
    llm = ScriptedLLM(config.budget, [["q"]])
    crawler = ScriptedCrawler(
        config.budget,
        {
            "q": [
                "https://www.stepstone.de/jobs/project-manager/in-berlin",
                "https://company.com/careers/software-engineer",
                "https://boards.greenhouse.io/acme/jobs/12345",
            ]
        },
    )
    _run(Orchestrator(config, config.budget, llm, crawler), tmp_path)

    # Posting first, then the generic careers page; max_results is hit
    # before the down-ranked board index page is ever fetched.
    assert crawler.fetch_calls == [
        "https://boards.greenhouse.io/acme/jobs/12345",
        "https://company.com/careers/software-engineer",
    ]


def test_browser_fallback_for_postings_and_listings_not_index(tmp_path: Path):
    config = _make_config(max_results=5)
    llm = ScriptedLLM(config.budget, [["q"]])
    posting = "https://boards.greenhouse.io/acme/jobs/12345"
    listing = "https://company.com/careers/software-engineer"
    index = "https://jobboard.example.com/jobs?q=python"
    crawler = ScriptedCrawler(config.budget, {"q": [posting, listing, index]})
    _run(Orchestrator(config, config.budget, llm, crawler), tmp_path)

    assert crawler.fetch_fallback_flags[posting] is True
    assert crawler.fetch_fallback_flags[listing] is True
    assert crawler.fetch_fallback_flags[index] is False


def test_listing_posting_links_harvested_and_scored(tmp_path: Path):
    config = _make_config(max_results=5)
    llm = ScriptedLLM(config.budget, [["q"]])
    listing = "https://company.com/careers/openings"
    posting_a = "https://boards.greenhouse.io/company/jobs/111"
    posting_b = "https://company.com/careers/positions/22222-python-developer"
    crawler = ScriptedCrawler(
        config.budget,
        {"q": [listing]},
        link_map={
            listing: [
                "https://company.com/about",
                posting_a,
                posting_b,
            ]
        },
    )
    results = _run(Orchestrator(config, config.budget, llm, crawler), tmp_path)

    # The hub page itself is never scored; its posting links are.
    assert sorted(r.url for r in results) == sorted([posting_a, posting_b])
    assert crawler.fetch_calls == [listing, posting_a, posting_b]


def test_listing_without_posting_links_scored_directly(tmp_path: Path):
    config = _make_config(max_results=1)
    llm = ScriptedLLM(config.budget, [["q"]])
    listing = "https://company.com/careers/software-engineer"
    crawler = ScriptedCrawler(
        config.budget,
        {"q": [listing]},
        link_map={listing: ["https://company.com/about"]},
    )
    results = _run(Orchestrator(config, config.budget, llm, crawler), tmp_path)

    assert [r.url for r in results] == [listing]


def test_harvest_prefers_employer_links_and_respects_cap(tmp_path: Path):
    config = _make_config(max_results=5, max_harvest_links=2)
    llm = ScriptedLLM(config.budget, [["q"]])
    listing = "https://company.com/careers/openings"
    aggregator_posting = "https://www.linkedin.com/jobs/view/3712345678"
    posting_a = "https://boards.greenhouse.io/company/jobs/111"
    posting_b = "https://boards.greenhouse.io/company/jobs/222"
    crawler = ScriptedCrawler(
        config.budget,
        {"q": [listing]},
        link_map={listing: [aggregator_posting, posting_a, posting_b]},
    )
    _run(Orchestrator(config, config.budget, llm, crawler), tmp_path)

    # Cap of 2: both employer postings make it, the aggregator mirror does not.
    assert crawler.fetch_calls == [listing, posting_a, posting_b]


def test_company_queries_injected_from_plan(tmp_path: Path):
    config = _make_config(max_results=1, company_query_boost=True)
    llm = ScriptedLLM(
        config.budget,
        [["python jobs berlin"]],
        plan_companies=["Acme GmbH", "Globex"],
    )
    crawler = ScriptedCrawler(
        config.budget, {"python jobs berlin": ["https://a.com/jobs/1"]}
    )
    _run(Orchestrator(config, config.budget, llm, crawler), tmp_path)

    assert crawler.search_calls[0] == '"Acme GmbH" careers Python Developer'
    assert crawler.search_calls[1] == "python jobs berlin"


def test_aggregator_index_pages_dropped_when_excluded(tmp_path: Path):
    config = _make_config(max_results=5, exclude_aggregator_sites=True)
    llm = ScriptedLLM(config.budget, [["q"]])
    aggregator_index = "https://www.linkedin.com/jobs/search/?keywords=python"
    generic_index = "https://jobboard.example.com/jobs?q=python"
    posting = "https://boards.greenhouse.io/acme/jobs/12345"
    crawler = ScriptedCrawler(
        config.budget, {"q": [aggregator_index, generic_index, posting]}
    )
    _run(Orchestrator(config, config.budget, llm, crawler), tmp_path)

    # The LinkedIn search page is never fetched; the generic index still is.
    assert crawler.fetch_calls == [posting, generic_index]


def test_aggregator_postings_processed_after_employer_urls(tmp_path: Path):
    config = _make_config(max_results=5)
    llm = ScriptedLLM(config.budget, [["q"]])
    aggregator_posting = "https://www.linkedin.com/jobs/view/3712345678"
    listing = "https://company.com/careers/software-engineer"
    posting = "https://boards.greenhouse.io/acme/jobs/12345"
    crawler = ScriptedCrawler(
        config.budget, {"q": [aggregator_posting, listing, posting]}
    )
    _run(Orchestrator(config, config.budget, llm, crawler), tmp_path)

    assert crawler.fetch_calls == [posting, listing, aggregator_posting]


def test_ats_queries_injected_before_llm_queries(tmp_path: Path):
    config = _make_config(max_results=1, ats_query_boost=True)
    llm = ScriptedLLM(config.budget, [["python jobs berlin"]])
    crawler = ScriptedCrawler(
        config.budget, {"python jobs berlin": ["https://a.com/jobs/1"]}
    )
    _run(Orchestrator(config, config.budget, llm, crawler), tmp_path)

    # ScriptedLLM's plan targets "Python Developer" in "Berlin".
    assert crawler.search_calls[0] == (
        "site:boards.greenhouse.io Python Developer Berlin"
    )
    assert crawler.search_calls[1] == "site:jobs.lever.co Python Developer Berlin"
    assert crawler.search_calls[2] == "python jobs berlin"


def test_ats_queries_rotate_across_iterations(tmp_path: Path):
    config = _make_config(max_results=5, ats_query_boost=True)
    llm = ScriptedLLM(config.budget, [["query one"], ["query two"], ["query two"]])
    crawler = ScriptedCrawler(
        config.budget,
        {
            "query one": ["https://a.com/jobs/1"],
            "query two": ["https://b.com/jobs/2"],
        },
    )
    _run(Orchestrator(config, config.budget, llm, crawler), tmp_path)

    site_queries = [q for q in crawler.search_calls if q.startswith("site:")]
    # Two per iteration, no repeats: the rotation moves on to new hosts.
    assert len(site_queries) == len(set(site_queries))
    assert len(site_queries) >= 4


def test_ats_boost_disabled_produces_no_site_queries(tmp_path: Path):
    config = _make_config(max_results=1)
    llm = ScriptedLLM(config.budget, [["python jobs berlin"]])
    crawler = ScriptedCrawler(
        config.budget, {"python jobs berlin": ["https://a.com/jobs/1"]}
    )
    _run(Orchestrator(config, config.budget, llm, crawler), tmp_path)

    assert crawler.search_calls == ["python jobs berlin"]


def test_ats_queries_empty_without_plan_roles(tmp_path: Path):
    config = _make_config(max_results=1, ats_query_boost=True, enable_planning=False)
    llm = ScriptedLLM(config.budget, [["python jobs berlin"]])
    crawler = ScriptedCrawler(
        config.budget, {"python jobs berlin": ["https://a.com/jobs/1"]}
    )
    _run(Orchestrator(config, config.budget, llm, crawler), tmp_path)

    # Planning disabled -> empty plan -> no roles to build site: queries from.
    assert crawler.search_calls == ["python jobs berlin"]


class RecordingNotifier:
    def __init__(self, fail: bool = False):
        self._fail = fail
        self.notified: list[list[Any]] = []

    def notify_results(self, results: list[Any]) -> bool:
        if self._fail:
            raise RuntimeError("telegram exploded")
        self.notified.append(results)
        return True


def test_notifier_receives_accepted_results(tmp_path: Path):
    config = _make_config(max_results=1)
    llm = ScriptedLLM(config.budget, [["python jobs berlin"]])
    crawler = ScriptedCrawler(
        config.budget, {"python jobs berlin": ["https://a.com/jobs/1"]}
    )
    notifier = RecordingNotifier()
    results = _run(
        Orchestrator(config, config.budget, llm, crawler, notifier=notifier),
        tmp_path,
    )

    assert len(results) == 1
    assert notifier.notified == [results]


def test_notifier_failure_does_not_break_run(tmp_path: Path):
    config = _make_config(max_results=1)
    llm = ScriptedLLM(config.budget, [["python jobs berlin"]])
    crawler = ScriptedCrawler(
        config.budget, {"python jobs berlin": ["https://a.com/jobs/1"]}
    )
    notifier = RecordingNotifier(fail=True)
    results = _run(
        Orchestrator(config, config.budget, llm, crawler, notifier=notifier),
        tmp_path,
    )

    assert len(results) == 1


def test_max_results_stops_loop(tmp_path: Path):
    config = _make_config(max_results=2)
    llm = ScriptedLLM(config.budget, [["q1", "q2", "q3"]])
    crawler = ScriptedCrawler(
        config.budget,
        {
            "q1": ["https://a.com/jobs/1", "https://a.com/jobs/2"],
            "q2": ["https://b.com/jobs/3"],
            "q3": ["https://c.com/jobs/4"],
        },
    )
    results = _run(Orchestrator(config, config.budget, llm, crawler), tmp_path)

    assert len(results) == 2
    assert crawler.search_calls == ["q1"]
