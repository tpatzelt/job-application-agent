from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tomllib


@dataclass
class EffortBudget:
    max_llm_calls: int
    max_search_iterations: int
    llm_calls_used: int = 0
    search_iterations_used: int = 0

    def can_call_llm(self) -> bool:
        return self.llm_calls_used < self.max_llm_calls

    def can_search(self) -> bool:
        return self.search_iterations_used < self.max_search_iterations

    def record_llm_call(self) -> None:
        self.llm_calls_used += 1

    def record_search_iteration(self) -> None:
        self.search_iterations_used += 1


@dataclass
class Config:
    max_results: int
    min_score: int
    results_json: str
    results_csv: str
    cache_path: str
    llm_model: str
    llm_temperature: float
    llm_max_retries: int
    llm_min_delay_seconds: int
    brave_endpoint: str
    results_per_query: int
    request_timeout_seconds: int
    search_min_delay_seconds: int
    max_queries_per_iteration: int
    budget: EffortBudget


def _load_pyproject_config(
    root: Path, config_path: Path | None = None, profile: str | None = None
) -> dict[str, Any]:
    pyproject_path = config_path or root / "pyproject.toml"
    if not pyproject_path.exists():
        return {}
    with pyproject_path.open("rb") as handle:
        data = tomllib.load(handle)
    base = data.get("tool", {}).get("job_crawler", {})
    if profile:
        profiles = data.get("tool", {}).get("job_crawler", {}).get("profiles", {})
        prof = profiles.get(profile, {})
        merged = dict(base)
        for k, v in prof.items():
            if isinstance(v, dict) and isinstance(merged.get(k), dict):
                merged[k] = {**merged.get(k, {}), **v}
            else:
                merged[k] = v
        return merged
    return base


def _get_env_var(name: str, required: bool = False) -> str | None:
    value = os.getenv(name)
    if required and not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def load_user_profile(root: Path) -> str:
    profile_path = root / "user_profile.txt"
    if not profile_path.exists():
        raise FileNotFoundError("user_profile.txt not found")
    return profile_path.read_text(encoding="utf-8")


def load_preferences(root: Path) -> dict[str, Any]:
    preferences_path = root / "preferences.json"
    if not preferences_path.exists():
        raise FileNotFoundError("preferences.json not found")
    with preferences_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_config(
    root: Path, config_path: Path | None = None, profile: str | None = None
) -> Config:
    data = _load_pyproject_config(root, config_path, profile)

    budget_data = data.get("budget", {})
    output_data = data.get("output", {})
    llm_data = data.get("llm", {})
    search_data = data.get("search", {})

    budget = EffortBudget(
        max_llm_calls=int(budget_data.get("max_llm_calls", 25)),
        max_search_iterations=int(budget_data.get("max_search_iterations", 5)),
    )

    return Config(
        max_results=int(data.get("max_results", 5)),
        min_score=int(data.get("min_score", 70)),
        results_json=str(output_data.get("results_json", "data/results.json")),
        results_csv=str(output_data.get("results_csv", "data/results.csv")),
        cache_path=str(output_data.get("cache_path", "data/cache.json")),
        llm_model=str(llm_data.get("model", "openrouter/free")),
        llm_temperature=float(llm_data.get("temperature", 0.2)),
        llm_max_retries=int(llm_data.get("max_retries", 3)),
        llm_min_delay_seconds=int(llm_data.get("min_delay_seconds", 1)),
        brave_endpoint=str(
            search_data.get(
                "brave_endpoint",
                "https://api.search.brave.com/res/v1/web/search",
            )
        ),
        results_per_query=int(search_data.get("results_per_query", 10)),
        request_timeout_seconds=int(search_data.get("request_timeout_seconds", 30)),
        search_min_delay_seconds=int(search_data.get("min_delay_seconds", 1)),
        max_queries_per_iteration=int(data.get("max_queries_per_iteration", 10)),
        budget=budget,
    )


def load_api_keys() -> dict[str, str]:
    keys: dict[str, str] = {}
    brave_key = _get_env_var("BRAVE_API_KEY")
    openrouter_key = _get_env_var("OPENROUTER_API_KEY")
    if brave_key:
        keys["brave"] = brave_key
    if openrouter_key:
        keys["openrouter"] = openrouter_key
    return keys
