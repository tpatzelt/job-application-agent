from pathlib import Path

import pytest

from src.config_manager import load_config


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
