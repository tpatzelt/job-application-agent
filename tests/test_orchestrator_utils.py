from src.orchestrator import Orchestrator
from src.config_manager import Config, EffortBudget


def _make_orch():
    cfg = Config(
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
    return Orchestrator(cfg, cfg.budget, llm_service=None, crawler=None)


def test_looks_like_listing_positive():
    orch = _make_orch()
    urls = [
        "https://company.com/careers/software-engineer",
        "https://jobs.example.com/job/1234",
        "https://boards.greenhouse.io/company/jobs/5678",
    ]
    for u in urls:
        assert orch._looks_like_listing(u)


def test_looks_like_listing_negative():
    orch = _make_orch()
    urls = [
        "https://blog.company.com/article/how-we-hire",
        "https://example.com/about",
        "https://example.com/contact",
    ]
    for u in urls:
        assert not orch._looks_like_listing(u)
