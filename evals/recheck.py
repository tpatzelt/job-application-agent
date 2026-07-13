"""Re-score an existing eval run's result URLs with the current checker
(no crawling — only the per-URL quality checks and the report are redone).

Usage:
    uv run python -m evals.recheck --tag baseline
"""

from __future__ import annotations

import argparse
import json
import logging

from .checks import check_result, summarize_checks
from .profiles import PROFILES_BY_NAME
from .run_eval import DEFAULT_RUNS_DIR, write_report


def recheck(tag: str) -> None:
    run_dir = DEFAULT_RUNS_DIR / tag
    scorecards = []
    for card_path in sorted(run_dir.glob("*/scorecard.json")):
        card = json.loads(card_path.read_text(encoding="utf-8"))
        profile = PROFILES_BY_NAME[card["profile"]]
        for item in card["results"]:
            item["checks"] = check_result(item["url"], profile.locations)
        card["metrics"] = summarize_checks([i["checks"] for i in card["results"]])
        card_path.write_text(
            json.dumps(card, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        scorecards.append(card)
    write_report(run_dir, scorecards)
    for card in scorecards:
        print(card["profile"], json.dumps(card["metrics"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)
    recheck(args.tag)


if __name__ == "__main__":
    main()
