import json

from src.config_manager import Config, EffortBudget
from src.llm_service import LLMService


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
        results_per_query=1,
        request_timeout_seconds=1,
        search_min_delay_seconds=0,
        max_queries_per_iteration=1,
        budget=EffortBudget(max_llm_calls=10, max_search_iterations=10),
    )


def _service_with_response(monkeypatch, response_text: str) -> LLMService:
    config = _make_config()
    service = LLMService(config, config.budget, api_key=None)
    monkeypatch.setattr(service, "_call_llm", lambda prompt: response_text)
    return service


def test_plan_search_parses_valid_json(monkeypatch):
    response = json.dumps(
        {
            "target_roles": ["ML Engineer"],
            "key_skills": ["Python", "PyTorch"],
            "locations": ["Remote"],
            "strategy": "Target ML-specific job boards.",
        }
    )
    service = _service_with_response(monkeypatch, response)
    plan = service.plan_search({"cv_summary": "ML background"})
    assert plan.target_roles == ["ML Engineer"]
    assert plan.strategy == "Target ML-specific job boards."


def test_plan_search_handles_fenced_json(monkeypatch):
    response = '```json\n{"target_roles": ["Data Engineer"], "strategy": "s"}\n```'
    service = _service_with_response(monkeypatch, response)
    plan = service.plan_search({})
    assert plan.target_roles == ["Data Engineer"]
    # Missing keys fall back to model defaults.
    assert plan.locations == []


def test_plan_search_invalid_payload_returns_default(monkeypatch):
    response = json.dumps({"target_roles": "not-a-list-of-strings", "strategy": 12.5})
    service = _service_with_response(monkeypatch, response)
    plan = service.plan_search({})
    assert plan.target_roles == []
    assert plan.strategy == ""


def test_reflect_parses_valid_json(monkeypatch):
    response = json.dumps(
        {
            "assessment": "Queries too broad.",
            "effective_queries": ["python berlin"],
            "ineffective_queries": ["jobs"],
            "adjustments": ["Add seniority terms."],
        }
    )
    service = _service_with_response(monkeypatch, response)
    reflection = service.reflect({}, [], {})
    assert reflection.assessment == "Queries too broad."
    assert reflection.adjustments == ["Add seniority terms."]


def test_prompts_include_agent_context():
    config = _make_config()
    service = LLMService(config, config.budget, api_key=None)

    plan_prompt = json.loads(
        service._build_plan_prompt({"cv_summary": "python dev"})
    )
    assert plan_prompt["context"]["cv_summary"] == "python dev"
    assert set(plan_prompt["output_schema"]) == {
        "target_roles",
        "key_skills",
        "locations",
        "target_companies",
        "strategy",
    }

    reflection_prompt = json.loads(
        service._build_reflection_prompt(
            {"plan": {}},
            [{"query": "q", "urls_found": 1, "new": 0}],
            {"search": {"calls": 1, "errors": 0}},
        )
    )
    assert reflection_prompt["history"][0]["query"] == "q"
    assert reflection_prompt["tool_stats"]["search"]["calls"] == 1
    assert set(reflection_prompt["output_schema"]) == {
        "assessment",
        "effective_queries",
        "ineffective_queries",
        "adjustments",
    }


def test_plan_and_reflect_consume_budget(monkeypatch):
    config = _make_config()
    service = LLMService(config, config.budget, api_key=None)

    def fake_completion(**kwargs):
        return {"choices": [{"message": {"content": "{}"}}]}

    monkeypatch.setattr("src.llm_service.completion", fake_completion)
    service.plan_search({})
    service.reflect({}, [], {})
    assert config.budget.llm_calls_used == 2
