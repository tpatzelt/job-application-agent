from __future__ import annotations

import logging
import time
from typing import Any, Callable, cast

from botasaurus.request import Request, request
from botasaurus.soupify import soupify

from .config_manager import Config, EffortBudget


@request(max_retry=3)
def brave_search_task(req: Request, data: dict[str, Any]) -> dict[str, Any]:
    response = req.get(
        data["endpoint"],
        headers=data["headers"],
        params=data["params"],
        timeout=data["timeout"],
    )
    response.raise_for_status()
    return response.json()


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

    def search(self, query: str) -> list[str]:
        if not self._budget.can_search():
            raise RuntimeError("Effort budget exceeded: search iterations")

        headers = {"Accept": "application/json"}
        if self._brave_api_key:
            headers["X-Subscription-Token"] = self._brave_api_key

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
        urls = [item.get("url") for item in web_results if item.get("url")]
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
