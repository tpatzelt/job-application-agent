import json

from pathlib import Path

from src.llm_service import LLMService
from src.config_manager import Config, EffortBudget


def _make_config():
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


def test_extract_json_object_and_normalize():
    svc = LLMService(_make_config(), _make_config().budget, api_key=None)
    text = 'Here is the answer:\n```json\n{"score": "85.3", "reason": {"why": "match"}}\n```'
    extracted = svc._extract_json_object(text)
    assert extracted is not None
    payload = json.loads(extracted)
    normalized = svc._normalize_payload(payload)
    assert isinstance(normalized, dict)
    assert normalized["score"] == 85
    assert isinstance(normalized["reason"], str)
    assert '"why": "match"' in normalized["reason"]
