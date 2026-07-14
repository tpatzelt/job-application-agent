"""Tests for the domain/industry-mismatch rejection: a matching job title
in the wrong industry (e.g. software PM vs food PM) must not be accepted."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from evals.checks import mentions_domain, summarize_checks
from src.llm_service import LLMService
from src.models import JobEvaluation, SearchPlan
from src.orchestrator import Orchestrator

from tests.test_agent_loop import (
    ScriptedCrawler,
    ScriptedLLM,
    _make_config,
    _run,
)

POSTING_URL = "https://boards.greenhouse.io/acme/jobs/123"


class DomainAwareLLM(ScriptedLLM):
    """Scores high but flags the job as an industry mismatch."""

    def evaluate_job(
        self,
        cv: str,
        job_description: str,
        preferences: dict[str, Any] | None = None,
    ) -> JobEvaluation:
        self._record_call()
        self.eval_preferences = preferences
        return JobEvaluation(
            score=95,
            reason="same title, but software PM vs food PM",
            domain_match=False,
        )


def test_domain_mismatch_rejected_despite_high_score(tmp_path: Path):
    config = _make_config(max_results=1)
    llm = DomainAwareLLM(config.budget, [["python jobs berlin"]])
    crawler = ScriptedCrawler(
        config.budget, {"python jobs berlin": [POSTING_URL]}
    )
    results = _run(Orchestrator(config, config.budget, llm, crawler), tmp_path)

    assert results == []


def test_evaluation_prompt_contains_domain_rules():
    config = _make_config()
    service = LLMService(config, config.budget, api_key=None)
    prompt = service._build_evaluation_prompt(
        "cv text",
        "job text",
        {
            "location": "Berlin, Germany",
            "locations": ["Berlin, Germany"],
            "job_titles": ["Product Manager Food"],
            "job_description_keywords": ["food", "FMCG"],
            "industries": ["food & beverage"],
        },
    )
    payload = json.loads(prompt)

    assert payload["preferred_job_titles"] == ["Product Manager Food"]
    assert payload["preferred_keywords"] == ["food", "FMCG"]
    assert payload["industries"] == ["food & beverage"]
    assert "domain_match" in payload["output_schema"]
    assert any("domain_match" in rule for rule in payload["rules"])


def test_normalize_payload_coerces_domain_match_string():
    config = _make_config()
    service = LLMService(config, config.budget, api_key=None)

    payload = service._normalize_payload(
        {"score": 80, "reason": "ok", "domain_match": "False"}
    )
    assert payload["domain_match"] is False

    payload = service._normalize_payload(
        {"score": 80, "reason": "ok", "domain_match": "true"}
    )
    assert payload["domain_match"] is True


def test_intake_prompt_extracts_industries():
    config = _make_config()
    service = LLMService(config, config.budget, api_key=None)
    prompt = service._build_intake_prompt("cv", "motivation", "prefs", [])
    payload = json.loads(prompt)

    assert "industries" in payload["output_schema"]
    assert any("industries" in rule for rule in payload["rules"])


def test_ats_queries_carry_industry_term():
    config = _make_config()
    llm = ScriptedLLM(config.budget, [["q"]])
    crawler = ScriptedCrawler(config.budget, {})
    orchestrator = Orchestrator(config, config.budget, llm, crawler)

    plan = SearchPlan(target_roles=["Product Manager"], locations=["Berlin"])
    queries = orchestrator._ats_queries(
        plan, {"industries": ["food & beverage"]}, set()
    )
    assert queries
    assert all("food & beverage" in query for query in queries)

    # Roles that already name the industry aren't doubled up.
    plan = SearchPlan(target_roles=["Product Manager Food"], locations=["Berlin"])
    queries = orchestrator._ats_queries(plan, {"industries": ["Food"]}, set())
    assert queries
    assert all(query.count("Food") == 1 for query in queries)


def test_mentions_domain_matches_terms_and_words():
    text = "We build software for the Food industry and retail."
    assert mentions_domain(text, ["food & beverage"])
    assert mentions_domain(text, ["retail"])
    assert not mentions_domain(text, ["FMCG"])
    assert not mentions_domain("", ["food"])


def test_summarize_checks_rates_domain_only_where_stated():
    base = {
        "live": True,
        "fresh": True,
        "location_ok": True,
        "posting": True,
        "aggregator": False,
    }
    checks = [
        {**base, "domain_ok": True},
        {**base, "domain_ok": False},
        {**base, "domain_ok": None},
    ]
    summary = summarize_checks(checks)
    assert summary["domain_rate"] == 0.5
    # domain_ok False disqualifies a result; None (no industries) does not.
    assert summary["good_results"] == 2

    summary = summarize_checks([{**base, "domain_ok": None}])
    assert summary["domain_rate"] is None
