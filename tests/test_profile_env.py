from pathlib import Path

import pytest

from src.config_manager import llm_api_key, load_config


def test_load_profile_env(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    # ensure loading base config works
    cfg = load_config(root)
    assert cfg.max_results >= 1


def test_budget_env_override(monkeypatch: pytest.MonkeyPatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.setenv("JOB_CRAWLER_MAX_LLM_CALLS", "3")
    monkeypatch.setenv("JOB_CRAWLER_MAX_SEARCH_ITERATIONS", "1")
    cfg = load_config(root)
    assert cfg.budget.max_llm_calls == 3
    assert cfg.budget.max_search_iterations == 1


def test_budget_env_override_absent_uses_pyproject(monkeypatch: pytest.MonkeyPatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.delenv("JOB_CRAWLER_MAX_LLM_CALLS", raising=False)
    monkeypatch.delenv("JOB_CRAWLER_MAX_SEARCH_ITERATIONS", raising=False)
    cfg = load_config(root)
    # falls back to pyproject.toml [tool.job_crawler.budget]
    assert cfg.budget.max_llm_calls == 40
    assert cfg.budget.max_search_iterations == 8


def test_budget_env_override_invalid_raises(monkeypatch: pytest.MonkeyPatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.setenv("JOB_CRAWLER_MAX_LLM_CALLS", "notanumber")
    with pytest.raises(RuntimeError):
        load_config(root)


def test_llm_model_env_override(monkeypatch: pytest.MonkeyPatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.setenv("JOB_CRAWLER_LLM_MODEL", "openrouter/openrouter/free")
    assert load_config(root).llm_model == "openrouter/openrouter/free"


def test_llm_api_key_follows_model_provider(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-key")
    assert llm_api_key("gemini/gemma-4-26b-a4b-it") == "gemini-key"
    assert llm_api_key("openrouter/openrouter/free") == "openrouter-key"
    assert llm_api_key("mock") is None
