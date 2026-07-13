"""Integration eval as a test: runs the real agent (Brave + LLM) for the
eval profiles and asserts minimum result quality. Opt-in because it needs
API keys, network, and several minutes per profile:

    RUN_INTEGRATION_EVAL=1 uv run pytest tests/test_integration_eval.py -s

By default it runs a single profile to keep the feedback loop fast; set
RUN_INTEGRATION_EVAL=all to run all five.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

_ENABLED = os.getenv("RUN_INTEGRATION_EVAL", "")
_HAS_KEYS = bool(os.getenv("BRAVE_API_KEY")) and bool(
    os.getenv("OPENROUTER_API_KEY")
)

pytestmark = pytest.mark.skipif(
    not (_ENABLED and _HAS_KEYS),
    reason="set RUN_INTEGRATION_EVAL=1 (and API keys in .env) to run",
)


def test_eval_profiles_yield_fresh_local_postings(tmp_path):
    from evals.run_eval import run_eval

    names = None if _ENABLED == "all" else ["ml-engineer-berlin"]
    report = run_eval(
        tag="pytest",
        profile_names=names,
        runs_dir=tmp_path,
        max_results=3,
        max_llm_calls=20,
        max_searches=6,
    )

    for card in report["profiles"]:
        assert "error" not in card, f"profile {card['profile']} crashed"
        metrics = card["metrics"]
        assert metrics["results"] > 0, (
            f"profile {card['profile']} found no acceptable jobs"
        )
        # The two known failure modes: dead/expired links and wrong location.
        assert metrics["fresh_rate"] >= 0.5, (
            f"profile {card['profile']}: too many dead/expired links: "
            f"{card['results']}"
        )
        assert metrics["location_rate"] >= 0.5, (
            f"profile {card['profile']}: too many location mismatches: "
            f"{card['results']}"
        )
