"""Result-quality checks for the integration eval.

Each accepted result URL is independently refetched and scored on the
dimensions the agent is supposed to get right:

- live:        the page still resolves and has real content
- fresh:       live and no closed/expired marker on the page
- location_ok: the page mentions one of the preferred locations
- posting:     the URL is shaped like an individual job posting
"""

from __future__ import annotations

import logging
from typing import Any

import requests

from src.crawler_engine import extract_visible_text
from src.page_signals import (
    find_stale_marker,
    mentions_location,
    redirected_off_posting,
)
from src.url_heuristics import POSTING, classify_url, is_aggregator_url

logger = logging.getLogger(__name__)

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
)
MIN_LIVE_TEXT_CHARS = 300


def check_result(
    url: str, locations: list[str], timeout: int = 25
) -> dict[str, Any]:
    status: int | None = None
    text = ""
    error: str | None = None
    final_url: str | None = None
    try:
        response = requests.get(
            url,
            headers={"User-Agent": USER_AGENT, "Accept-Language": "en,de"},
            timeout=timeout,
            allow_redirects=True,
        )
        status = response.status_code
        final_url = str(response.url)
        text = extract_visible_text(response.text)
    except Exception as exc:
        error = str(exc)
        logger.warning("Eval refetch failed for %s: %s", url, exc)

    stale_marker = find_stale_marker(text) if text else None
    # Dead ATS postings often 302 to the company's board page with HTTP
    # 200 — the redirect is the only sign the posting is gone.
    redirected = redirected_off_posting(url, final_url)
    live = status is not None and status < 400 and len(text) >= MIN_LIVE_TEXT_CHARS
    return {
        "url": url,
        "final_url": final_url,
        "http_status": status,
        "error": error,
        "text_chars": len(text),
        "live": live,
        "stale_marker": stale_marker,
        "redirected_off_posting": redirected,
        "fresh": live and stale_marker is None and not redirected,
        "location_ok": bool(text) and mentions_location(text, locations),
        "posting": classify_url(url) == POSTING,
        "aggregator": is_aggregator_url(url),
    }


def summarize_checks(checks: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(checks)

    def rate(key: str) -> float | None:
        if total == 0:
            return None
        return round(sum(1 for c in checks if c[key]) / total, 3)

    good = [
        c for c in checks if c["fresh"] and c["location_ok"] and c["posting"]
    ]
    return {
        "results": total,
        "live_rate": rate("live"),
        "fresh_rate": rate("fresh"),
        "location_rate": rate("location_ok"),
        "posting_rate": rate("posting"),
        "aggregator_rate": rate("aggregator"),
        "good_results": len(good),
        "good_rate": round(len(good) / total, 3) if total else None,
    }
