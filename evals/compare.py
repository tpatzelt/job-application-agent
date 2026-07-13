"""Compare two eval runs side by side.

Usage:
    uv run python -m evals.compare --before baseline --after fixed
"""

from __future__ import annotations

import argparse
import json

from .run_eval import DEFAULT_RUNS_DIR, REPORT_COLUMNS


def _load(tag: str) -> dict:
    return json.loads(
        (DEFAULT_RUNS_DIR / tag / "report.json").read_text(encoding="utf-8")
    )


def compare(before_tag: str, after_tag: str) -> str:
    before = _load(before_tag)
    after = _load(after_tag)
    before_by_name = {c["profile"]: c for c in before["profiles"]}

    lines = [
        f"# Eval comparison: {before_tag} -> {after_tag}",
        "",
        "| profile | metric | " + before_tag + " | " + after_tag + " |",
        "|---|---|---|---|",
    ]
    for card in after["profiles"]:
        old = before_by_name.get(card["profile"], {}).get("metrics", {})
        new = card["metrics"]
        for col in REPORT_COLUMNS:
            lines.append(
                f"| {card['profile']} | {col} | {old.get(col)} | {new.get(col)} |"
            )
    lines.append("")
    lines.append("## Totals")
    lines.append("")
    lines.append("| metric | " + before_tag + " | " + after_tag + " |")
    lines.append("|---|---|---|")
    for col in REPORT_COLUMNS:
        lines.append(
            f"| {col} | {before['totals'].get(col)} | {after['totals'].get(col)} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", required=True)
    parser.add_argument("--after", required=True)
    args = parser.parse_args()
    print(compare(args.before, args.after))


if __name__ == "__main__":
    main()
