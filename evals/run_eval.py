"""Integration eval: run the real agent for each search profile and score
the accepted results (liveness, freshness, location match, posting shape).

Usage:
    uv run python -m evals.run_eval --tag baseline
    uv run python -m evals.run_eval --tag fixed --profiles ml-engineer-berlin

Needs BRAVE_API_KEY and OPENROUTER_API_KEY in .env. Each profile runs in an
isolated directory under evals/runs/<tag>/<profile>/ (fresh cache/memory so
runs are comparable), and a combined report.json / report.md is written to
evals/runs/<tag>/.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from src.config_manager import EffortBudget, load_api_keys, load_config
from src.crawler_engine import CrawlerEngine
from src.llm_service import LLMService
from src.orchestrator import Orchestrator

from .checks import check_result, summarize_checks
from .profiles import PROFILES, PROFILES_BY_NAME, EvalProfile

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RUNS_DIR = ROOT / "evals" / "runs"

logger = logging.getLogger("evals")

REPORT_COLUMNS = (
    "results",
    "good_results",
    "good_rate",
    "fresh_rate",
    "location_rate",
    "posting_rate",
    "live_rate",
)


def run_profile(
    profile: EvalProfile,
    out_dir: Path,
    keys: dict[str, str],
    max_results: int,
    max_llm_calls: int,
    max_searches: int,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    budget = EffortBudget(
        max_llm_calls=max_llm_calls, max_search_iterations=max_searches
    )
    config = dataclasses.replace(
        load_config(ROOT),
        budget=budget,
        max_results=max_results,
        # No Chrome on the eval machine; keep both eval runs identical so
        # baseline/after comparisons aren't skewed by fetch capability.
        browser_fallback=False,
        telegram_notifications=False,
    )
    llm = LLMService(config, budget, keys.get("openrouter"))
    crawler = CrawlerEngine(config, budget, keys.get("brave"))
    orchestrator = Orchestrator(config, budget, llm, crawler)

    started = time.time()
    results = orchestrator.run(
        cv_text=profile.cv_text,
        preferences=profile.preferences,
        cache_path=out_dir / "cache.json",
        results_json=out_dir / "results.json",
        results_csv=out_dir / "results.csv",
        memory_path=out_dir / "memory.json",
    )
    duration = round(time.time() - started, 1)

    logger.info(
        "[%s] crawl done: %s results in %ss, checking result quality",
        profile.name,
        len(results),
        duration,
    )
    checks = [check_result(item.url, profile.locations) for item in results]
    scorecard: dict[str, Any] = {
        "profile": profile.name,
        "locations": profile.locations,
        "duration_seconds": duration,
        "llm_calls_used": budget.llm_calls_used,
        "searches_used": budget.search_iterations_used,
        "metrics": summarize_checks(checks),
        "results": [
            {**item.model_dump(), "checks": check}
            for item, check in zip(results, checks)
        ],
    }
    (out_dir / "scorecard.json").write_text(
        json.dumps(scorecard, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return scorecard


def write_report(run_dir: Path, scorecards: list[dict[str, Any]]) -> str:
    report = {
        "run_dir": str(run_dir),
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "profiles": scorecards,
        "totals": _totals(scorecards),
    }
    (run_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    lines = [
        f"# Eval report — {run_dir.name}",
        "",
        f"Generated: {report['generated_at']}",
        "",
        "| profile | " + " | ".join(REPORT_COLUMNS) + " |",
        "|---" * (len(REPORT_COLUMNS) + 1) + "|",
    ]
    for card in scorecards:
        metrics = card["metrics"]
        cells = [str(metrics.get(col)) for col in REPORT_COLUMNS]
        lines.append(f"| {card['profile']} | " + " | ".join(cells) + " |")
    totals = report["totals"]
    lines.append(
        "| **all** | "
        + " | ".join(str(totals.get(col)) for col in REPORT_COLUMNS)
        + " |"
    )
    lines.append("")
    for card in scorecards:
        lines.append(f"## {card['profile']}")
        lines.append("")
        for item in card["results"]:
            checks = item["checks"]
            flags = ", ".join(
                key
                for key in ("live", "fresh", "location_ok", "posting")
                if checks.get(key)
            ) or "none"
            stale = (
                f" — stale: \"{checks['stale_marker']}\""
                if checks.get("stale_marker")
                else ""
            )
            lines.append(
                f"- [{item['score']}] {item['url']} "
                f"(ok: {flags}; http {checks.get('http_status')}{stale})"
            )
        if not card["results"]:
            lines.append("- no accepted results")
        lines.append("")
    text = "\n".join(lines)
    (run_dir / "report.md").write_text(text, encoding="utf-8")
    return text


def _totals(scorecards: list[dict[str, Any]]) -> dict[str, Any]:
    all_checks = [
        item["checks"] for card in scorecards for item in card["results"]
    ]
    return summarize_checks(all_checks)


def run_eval(
    tag: str,
    profile_names: list[str] | None = None,
    runs_dir: Path = DEFAULT_RUNS_DIR,
    max_results: int = 5,
    max_llm_calls: int = 30,
    max_searches: int = 8,
) -> dict[str, Any]:
    load_dotenv(ROOT / ".env")
    keys = load_api_keys()
    if "brave" not in keys or "openrouter" not in keys:
        raise RuntimeError(
            "BRAVE_API_KEY and OPENROUTER_API_KEY are required for the eval"
        )
    profiles = (
        [PROFILES_BY_NAME[name] for name in profile_names]
        if profile_names
        else PROFILES
    )
    run_dir = runs_dir / tag
    run_dir.mkdir(parents=True, exist_ok=True)

    scorecards = []
    for profile in profiles:
        logger.info("=== Running eval profile: %s ===", profile.name)
        try:
            scorecards.append(
                run_profile(
                    profile,
                    run_dir / profile.name,
                    keys,
                    max_results=max_results,
                    max_llm_calls=max_llm_calls,
                    max_searches=max_searches,
                )
            )
        except Exception:
            logger.exception("Profile %s failed", profile.name)
            scorecards.append(
                {
                    "profile": profile.name,
                    "locations": profile.locations,
                    "error": "run failed, see log",
                    "metrics": summarize_checks([]),
                    "results": [],
                }
            )
    write_report(run_dir, scorecards)
    return {"run_dir": str(run_dir), "profiles": scorecards}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True, help="run name, e.g. baseline")
    parser.add_argument(
        "--profiles",
        help="comma-separated profile names (default: all)",
        default=None,
    )
    parser.add_argument("--max-results", type=int, default=5)
    parser.add_argument("--max-llm-calls", type=int, default=30)
    parser.add_argument("--max-searches", type=int, default=8)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    names = args.profiles.split(",") if args.profiles else None
    report = run_eval(
        tag=args.tag,
        profile_names=names,
        max_results=args.max_results,
        max_llm_calls=args.max_llm_calls,
        max_searches=args.max_searches,
    )
    print(json.dumps(_totals(report["profiles"]), indent=2))


if __name__ == "__main__":
    main()
