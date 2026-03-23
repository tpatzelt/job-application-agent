from __future__ import annotations

import logging
import time
import threading
from typing import Any, Callable, cast
from urllib.parse import urlparse

from botasaurus.request import Request, request
from botasaurus.soupify import soupify

from .config_manager import Config, EffortBudget


@request(max_retry=3)
def brave_search_task(req: Request, data: dict[str, Any]) -> dict[str, Any]:
    try:
        response = req.get(
            data["endpoint"],
            headers=data.get("headers"),
            params=data.get("params"),
            timeout=data.get("timeout"),
        )
        response.raise_for_status()
    except Exception as exc:  # catch network / HTTP errors and return empty payload
        # Attempt to surface HTTP response details (status/text) when available
        resp = getattr(exc, "response", None)
        logger = logging.getLogger("CrawlerEngine")
        if resp is not None:
            # Truncate body to avoid huge logs but keep enough for debugging
            body = getattr(resp, "text", "") or ""
            logger.warning(
                "Brave API request failed: %s %s — %s",
                getattr(resp, "status_code", "?"),
                getattr(resp, "reason", ""),
                (body[:500] + "...") if len(body) > 500 else body,
            )
            if getattr(resp, "status_code", None) == 422:
                logger.warning(
                    "Brave returned 422 Unprocessable Content — check BRAVE_API_KEY, endpoint, and request params"
                )
        else:
            logger.warning("Brave API request failed: %s", exc)
        return {"web": {"results": []}}

    try:
        j = response.json()
    except Exception:
        return {"web": {"results": []}}

    if not isinstance(j, dict):
        # ensure we always return a dict for callers
        return {"web": {"results": []}}
    return j


@request(max_retry=3)
def fetch_job_task(req: Request, data: dict[str, Any]) -> str:
    response = req.get(data["url"], timeout=data["timeout"])
    response.raise_for_status()
    return response.text


BraveSearchCallable = Callable[[dict[str, Any]], dict[str, Any]]
FetchJobCallable = Callable[[dict[str, Any]], str]
BRAVE_SEARCH = cast(BraveSearchCallable, brave_search_task)
FETCH_JOB = cast(FetchJobCallable, fetch_job_task)


class CrawlerEngine:
    def __init__(self, config: Config, budget: EffortBudget, brave_api_key: str | None):
        self._config = config
        self._budget = budget
        self._brave_api_key = brave_api_key
        self._logger = logging.getLogger(self.__class__.__name__)
        # Rate limiting for Brave Search: ensure at most 1 request per 1.5s
        self._brave_lock = threading.Lock()
        self._last_brave_search = 0.0

    def search(self, query: str) -> list[str]:
        if not self._budget.can_search():
            # Don't raise here; let the orchestrator stop iterating gracefully.
            self._logger.warning(
                "Effort budget exhausted: skipping search for query: %s", query
            )
            return []

        headers = {"Accept": "application/json"}
        if self._brave_api_key:
            headers["X-Subscription-Token"] = self._brave_api_key
        # Brave Search expects Cache-Control header to be exactly 'no-cache'
        # (see API validation errors). Ensure we send the value they require.
        headers["Cache-Control"] = "no-cache"
        # Pragmatically ask intermediaries not to serve cached responses
        headers.setdefault("Pragma", "no-cache")

        self._logger.info("Brave search: %s", query)
        time.sleep(self._config.search_min_delay_seconds)
        payload = self._run_brave_search_with_backoff(
            {
                "endpoint": self._config.brave_endpoint,
                "headers": headers,
                "params": {"q": query, "count": self._config.results_per_query},
                "timeout": self._config.request_timeout_seconds,
            }
        )
        # only count a search iteration if we actually received results
        if payload:
            self._budget.record_search_iteration()
        if not payload:
            self._logger.warning("Empty Brave search payload for query: %s", query)
            return []
        web_results = payload.get("web", {}).get("results", [])

        def _is_video_item(item: dict[str, Any]) -> bool:
            # Brave may return video results (YouTube, Vimeo, etc.) either via
            # result metadata or simply by URL. Try multiple heuristics.
            # 1) explicit type/format fields
            t = item.get("type") or item.get("format") or item.get("content_type")
            if isinstance(t, str) and "video" in t.lower():
                return True
            # 2) URL-based detection
            url = item.get("url")
            if not url:
                return False
            hostname = urlparse(url).hostname or ""
            video_hosts = (
                "youtube.com",
                "youtu.be",
                "vimeo.com",
                "dailymotion.com",
                "tiktok.com",
            )
            if any(h in hostname for h in video_hosts):
                return True
            # 3) path hints
            path = urlparse(url).path.lower()
            if "/watch" in path or "/video" in path:
                return True
            return False

        urls = [
            item.get("url")
            for item in web_results
            if item.get("url") and not _is_video_item(item)
        ]
        return urls

    def fetch_job_text(self, url: str) -> str:
        self._logger.info("Fetching URL via botasaurus: %s", url)
        html = FETCH_JOB({"url": url, "timeout": self._config.request_timeout_seconds})
        if not html:
            return ""
        soup = soupify(html)

        for tag in soup(["script", "style", "noscript", "header", "footer", "nav"]):
            tag.decompose()

        text = soup.get_text(separator=" ", strip=True)
        return " ".join(text.split())

    def _run_brave_search_with_backoff(self, payload: dict[str, Any]) -> dict[str, Any]:
        delay = 1
        for attempt in range(1, 4):
            try:
                # Enforce a minimum interval between Brave API requests (1.5s).
                with self._brave_lock:
                    now = time.monotonic()
                    min_interval = 1.5
                    elapsed = now - self._last_brave_search
                    if elapsed < min_interval:
                        to_sleep = min_interval - elapsed
                        self._logger.debug(
                            "Sleeping %.3fs to respect Brave API rate limit",
                            to_sleep,
                        )
                        time.sleep(to_sleep)
                    # record the time we are about to start the request
                    self._last_brave_search = time.monotonic()

                result = BRAVE_SEARCH(payload)
                if result:
                    return result
                self._logger.warning("Brave search returned empty payload")
            except Exception as exc:
                self._logger.warning(
                    "Brave search failed on attempt %s/3: %s",
                    attempt,
                    exc,
                )
                if "429" in str(exc) and attempt < 3:
                    time.sleep(delay)
                    delay *= 2
                    continue
                if attempt < 3:
                    time.sleep(1)
                    continue
                return {"web": {"results": []}}
        return {"web": {"results": []}}
