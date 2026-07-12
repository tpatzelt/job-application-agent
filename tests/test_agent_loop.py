from pathlib import Path
from typing import Any

from src.agent_memory import AgentMemory
from src.config_manager import Config, EffortBudget
from src.models import JobEvaluation, Reflection, SearchPlan, SearchQueries
from src.orchestrator import Orchestrator

LONG_JOB_TEXT = "We are hiring a Python developer. " + ("Details " * 200)


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
    ):
        self._budget = budget
        self._query_batches = query_batches
        self._score = score
        self._fail_reflect = fail_reflect
        self._fail_plan = fail_plan
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
            strategy="Focus on job boards.",
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

    def evaluate_job(self, cv: str, job_description: str) -> JobEvaluation:
        self._record_call()
        return JobEvaluation(score=self._score, reason="scripted")


class ScriptedCrawler:
    """Fake crawler that maps queries to fixed URL lists."""

    def __init__(self, budget: EffortBudget, url_map: dict[str, list[str]]):
        self._budget = budget
        self._url_map = url_map
        self.search_calls: list[str] = []
        self.fetch_calls: list[str] = []

    def search(self, query: str) -> list[str]:
        self._budget.record_search_iteration()
        self.search_calls.append(query)
        return self._url_map.get(query, [])

    def fetch_job_text(self, url: str) -> str:
        self.fetch_calls.append(url)
        return LONG_JOB_TEXT


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
