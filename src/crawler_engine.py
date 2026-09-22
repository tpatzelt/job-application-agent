from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any, cast
from urllib.parse import urldefrag, urljoin, urlparse

from botasaurus.browser import Driver, browser
from botasaurus.request import Request, request
from botasaurus.soupify import soupify

from .config_manager import Config, EffortBudget
from .page_signals import redirected_off_posting


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
        # Signal the error so the caller can retry; an empty result set
        # here would be indistinguishable from a legitimately empty search.
        return {"error": str(exc)}

    try:
        j = response.json()
    except Exception as exc:
        return {"error": f"Invalid JSON from Brave API: {exc}"}

    if not isinstance(j, dict):
        # ensure we always return a dict for callers
        return {"web": {"results": []}}
    return j


@request(max_retry=3)
def fetch_job_task(req: Request, data: dict[str, Any]) -> dict[str, Any]:
    response = req.get(data["url"], timeout=data["timeout"])
    status = response.status_code
    # A removed posting answers 404/410 and a blocking board answers 403.
    # Raising would make botasaurus retry the same request three times for
    # an answer that will not change; report it instead.
    if 400 <= status < 500 and status != 429:
        return {"html": "", "final_url": str(response.url), "status": status}
    response.raise_for_status()
    # final_url exposes silent redirects: dead ATS job IDs 302 to the
    # company's board page instead of returning 404.
    return {"html": response.text, "final_url": str(response.url), "status": status}


def extract_visible_text(html: str | None) -> str:
    if not html:
        return ""
    return _visible_text(soupify(html))


def _visible_text(soup: Any) -> str:
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav"]):
        tag.decompose()
    text = soup.get_text(separator=" ", strip=True)
    return " ".join(text.split())


def extract_page_title(html: str | None) -> str:
    """The document <title>, falling back to the first <h1>.

    ATS postings name the role there, which is the only reliable source
    for a readable result title.
    """
    if not html:
        return ""
    return _page_title(soupify(html))


def _page_title(soup: Any) -> str:
    if soup.title and soup.title.string:
        title = " ".join(str(soup.title.string).split())
        if title:
            return title
    heading = soup.find("h1")
    if heading:
        return " ".join(heading.get_text(separator=" ", strip=True).split())
    return ""


def extract_links(html: str | None, base_url: str) -> list[str]:
    """Absolute, de-duplicated http(s) hrefs in document order."""
    if not html:
        return []
    return _links(soupify(html), base_url)


def _links(soup: Any, base_url: str) -> list[str]:
    links: list[str] = []
    seen: set[str] = set()
    for anchor in soup.find_all("a", href=True):
        href = urldefrag(urljoin(base_url, anchor["href"].strip())).url
        if not href.startswith(("http://", "https://")):
            continue
        if href in seen:
            continue
        seen.add(href)
        links.append(href)
    return links


# --no-sandbox: Ubuntu 23.10+ restricts unprivileged user namespaces, which
# crashes non-packaged Chrome builds; the scraper only visits public pages.
@browser(
    headless=True,
    reuse_driver=True,
    max_retry=2,
    close_on_crash=True,
    block_images=True,
    output=None,
    add_arguments=["--no-sandbox", "--disable-gpu"],
)
def fetch_job_browser_task(driver: Driver, data: dict[str, Any]) -> dict[str, Any]:
    # Real browser fetch for JS-rendered pages (Workday-style ATS) and
    # boards that 403 plain HTTP clients. ATS pages often render the job
    # description via XHR well after page load, so poll until the visible
    # text reaches the threshold instead of sleeping a fixed interval.
    driver.get(data["url"])
    min_chars = int(data.get("min_text_chars", 800))
    html = ""
    for _ in range(6):
        driver.sleep(2)
        html = driver.page_html
        if len(extract_visible_text(html)) >= min_chars:
            break
    return {"html": html, "final_url": driver.current_url}


@dataclass(frozen=True)
class PageContent:
    """A fetched page. Unpacks as (text, links) so callers that only want
    those two keep working; `title` feeds readable result titles."""

    text: str
    links: list[str]
    title: str = ""
    # True when the URL answered "this posting no longer exists" (404/410,
    # or a redirect off the posting), as opposed to simply yielding no text.
    gone: bool = False

    def __iter__(self) -> Iterator[Any]:
        return iter((self.text, self.links))


def page_content(html: str | None, base_url: str) -> PageContent:
    """Title, links, and visible text from one parse of the document.

    Order matters: the text pass strips nav/header/footer, so links and
    title are read before it.
    """
    if not html:
        return PageContent("", [])
    soup = soupify(html)
    title = _page_title(soup)
    links = _links(soup, base_url)
    return PageContent(_visible_text(soup), links, title)


# Statuses that mean the posting itself is gone, not that the fetch failed.
GONE_STATUSES = (404, 410)

BraveSearchCallable = Callable[[dict[str, Any]], dict[str, Any]]
# Fetch tasks return {"html": str, "final_url": str}; plain strings are
# also accepted (test fakes, degraded browser results).
FetchJobCallable = Callable[[dict[str, Any]], Any]


def _unwrap_fetch(payload: Any) -> tuple[str, str | None, int | None]:
    if isinstance(payload, dict):
        return payload.get("html") or "", payload.get("final_url"), payload.get("status")
    return payload or "", None, None
BRAVE_SEARCH = cast(BraveSearchCallable, brave_search_task)
FETCH_JOB = cast(FetchJobCallable, fetch_job_task)
FETCH_JOB_BROWSER = cast(FetchJobCallable, fetch_job_browser_task)


class CrawlerEngine:
    def __init__(self, config: Config, budget: EffortBudget, brave_api_key: str | None):
        self._config = config
        self._budget = budget
        self._brave_api_key = brave_api_key
        self._logger = logging.getLogger(self.__class__.__name__)
        # Rate limiting for Brave Search: ensure at most 1 request per 1.5s
        self._brave_lock = threading.Lock()
        self._last_brave_search = 0.0

    def search(
        self,
        query: str,
        country: str | None = None,
        search_lang: str | None = None,
    ) -> list[str]:
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
        params: dict[str, Any] = {
            "q": query,
            "count": self._config.results_per_query,
        }
        # Recently indexed pages only: old postings are usually expired.
        # This applies to site:-scoped ATS queries too — without it Brave
        # returns mostly postings that 404 or redirect away at fetch time —
        # but because it starves their narrow result sets, an empty result
        # is retried unfiltered below.
        if self._config.search_freshness:
            params["freshness"] = self._config.search_freshness
        # Scope results to the user's country so localized boards and
        # employer pages outrank same-language pages from elsewhere.
        if country:
            params["country"] = country
        # Restrict results to the user's preferred posting language.
        if search_lang:
            params["search_lang"] = search_lang
        request_payload = {
            "endpoint": self._config.brave_endpoint,
            "headers": headers,
            "params": params,
            "timeout": self._config.request_timeout_seconds,
        }
        payload = self._run_brave_search_with_backoff(request_payload)
        if not payload or "error" in payload:
            # Transport-level failure: don't consume search budget for it.
            self._logger.warning(
                "Brave search failed for query %r: %s",
                query,
                (payload or {}).get("error", "empty payload"),
            )
            return []
        self._budget.record_search_iteration()
        web_results = payload.get("web", {}).get("results", [])
        if not web_results and "freshness" in params:
            # Narrow queries (site:-scoped ones especially) can have nothing
            # indexed inside the freshness window; ask again without it
            # rather than losing the query.
            self._logger.info(
                "No fresh results for %r, retrying without the freshness filter",
                query,
            )
            unfiltered = {**params}
            unfiltered.pop("freshness")
            payload = self._run_brave_search_with_backoff(
                {**request_payload, "params": unfiltered}
            )
            web_results = (payload or {}).get("web", {}).get("results", [])

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

    def fetch_job_text(self, url: str, use_browser_fallback: bool = False) -> str:
        return self.fetch_page(url, use_browser_fallback=use_browser_fallback).text

    def fetch_page(
        self, url: str, use_browser_fallback: bool = False
    ) -> PageContent:
        """Fetch a page and return its text, outbound links, and title."""
        self._logger.info("Fetching URL via botasaurus: %s", url)
        payload = FETCH_JOB(
            {"url": url, "timeout": self._config.request_timeout_seconds}
        )
        html, final_url, status = _unwrap_fetch(payload)
        if status in GONE_STATUSES:
            self._logger.info("Posting gone: %s returned HTTP %s", url, status)
            return PageContent("", [], gone=True)
        if redirected_off_posting(url, final_url):
            self._logger.info(
                "Posting gone: %s redirected to %s, skipping", url, final_url
            )
            return PageContent("", [], gone=True)
        content = page_content(html, url)
        text = content.text
        if (
            use_browser_fallback
            and self._config.browser_fallback
            and len(text) < self._config.min_job_text_chars
        ):
            self._logger.info(
                "Plain fetch got only %s chars for %s, trying browser fallback",
                len(text),
                url,
            )
            browser_html, browser_final_url, _ = _unwrap_fetch(
                self._fetch_html_with_browser(url)
            )
            if redirected_off_posting(url, browser_final_url):
                self._logger.info(
                    "Posting gone: %s redirected to %s in browser, skipping",
                    url,
                    browser_final_url,
                )
                return PageContent("", [], gone=True)
            browser_content = page_content(browser_html, url)
            if len(browser_content.text) > len(text):
                self._logger.info(
                    "Browser fallback recovered %s chars for %s",
                    len(browser_content.text),
                    url,
                )
                if browser_content.title:
                    return browser_content
                return PageContent(
                    browser_content.text, browser_content.links, content.title
                )
        return content

    def _fetch_html_with_browser(self, url: str) -> Any:
        try:
            return FETCH_JOB_BROWSER(
                {"url": url, "min_text_chars": self._config.min_job_text_chars}
            )
        except Exception as exc:
            self._logger.warning("Browser fetch failed for %s: %s", url, exc)
            return ""

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
                if result and "error" not in result:
                    return result
                error = (result or {}).get("error", "empty payload")
                self._logger.warning(
                    "Brave search attempt %s/3 failed: %s", attempt, error
                )
                if attempt < 3:
                    time.sleep(delay)
                    delay *= 2
                    continue
                return result or {"error": "empty payload"}
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
