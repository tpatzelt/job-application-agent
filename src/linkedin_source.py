from __future__ import annotations

import re
from typing import Any
from urllib.parse import parse_qs, urlencode, urlparse

# LinkedIn's public "guest" job endpoints: the same HTML fragments its
# logged-out job search page loads, so they need no account or API key.
# /jobs/view/<id> itself usually answers a guest with an auth wall, which
# is why postings are fetched through the jobPosting endpoint instead.
LINKEDIN_SEARCH_ENDPOINT = (
    "https://www.linkedin.com/jobs-guest/jobs/api/seeMoreJobPostings/search"
)
LINKEDIN_POSTING_ENDPOINT = "https://www.linkedin.com/jobs-guest/jobs/api/jobPosting/"
LINKEDIN_POSTING_URL = "https://www.linkedin.com/jobs/view/"

# Brave freshness values -> LinkedIn's "time posted" filter (seconds).
_TIME_WINDOWS = {
    "pd": "r86400",
    "pw": "r604800",
    "pm": "r2592000",
    "py": "r31536000",
}

_VIEW_ID_RE = re.compile(r"/jobs/view/(?:[^/]*?-)?(\d{6,})/?$")
_URN_ID_RE = re.compile(r"urn:li:jobPosting:(\d+)")


def is_linkedin_host(host: str) -> bool:
    host = host.lower()
    return host == "linkedin.com" or host.endswith(".linkedin.com")


def linkedin_job_id(url: str) -> str:
    """The numeric posting id in a LinkedIn job URL, or "" for other URLs.

    Covers /jobs/view/<id>, /jobs/view/<slug>-<id> (what search cards link
    to, on any country subdomain), the guest jobPosting endpoint, and
    search pages that pin one posting via ?currentJobId=<id>.
    """
    try:
        parsed = urlparse(url)
    except ValueError:
        return ""
    if not is_linkedin_host(parsed.hostname or ""):
        return ""
    path = (parsed.path or "").rstrip("/")
    match = _VIEW_ID_RE.search(path)
    if match:
        return match.group(1)
    if path.startswith("/jobs-guest/jobs/api/jobPosting/"):
        job_id = path.rsplit("/", 1)[-1]
        return job_id if job_id.isdigit() else ""
    current = parse_qs(parsed.query).get("currentJobId", [""])[0]
    return current if current.isdigit() else ""


def linkedin_posting_url(job_id: str) -> str:
    return f"{LINKEDIN_POSTING_URL}{job_id}"


def linkedin_search_url(
    keywords: str, location: str = "", freshness: str = "", start: int = 0
) -> str:
    """Guest search URL for `keywords` in `location`.

    `freshness` takes the Brave values (pd/pw/pm/py) so one setting scopes
    both sources to recently posted jobs.
    """
    params: dict[str, str] = {"keywords": keywords}
    if location:
        params["location"] = location
    window = _TIME_WINDOWS.get(freshness)
    if window:
        params["f_TPR"] = window
    params["start"] = str(start)
    return f"{LINKEDIN_SEARCH_ENDPOINT}?{urlencode(params)}"


def _soup(html: str) -> Any:
    # Imported here so url_heuristics (which uses the URL helpers above)
    # stays free of the scraping stack.
    from botasaurus.soupify import soupify

    return soupify(html)


def parse_search_results(html: str | None) -> list[str]:
    """Posting URLs (canonical /jobs/view/<id>) from a guest search page."""
    if not html:
        return []
    soup = _soup(html)
    ids: list[str] = []
    for card in soup.select("[data-entity-urn]"):
        match = _URN_ID_RE.search(card.get("data-entity-urn", ""))
        if match:
            ids.append(match.group(1))
    for anchor in soup.find_all("a", href=True):
        job_id = linkedin_job_id(anchor["href"])
        if job_id:
            ids.append(job_id)
    return [linkedin_posting_url(job_id) for job_id in dict.fromkeys(ids)]


def _first_text(soup: Any, selectors: tuple[str, ...]) -> str:
    for selector in selectors:
        node = soup.select_one(selector)
        if node is not None:
            text = " ".join(node.get_text(separator=" ", strip=True).split())
            if text:
                return text
    return ""


def parse_posting(html: str | None) -> tuple[str, str]:
    """(role, company) from a guest jobPosting fragment; either may be ""."""
    if not html:
        return "", ""
    soup = _soup(html)
    role = _first_text(
        soup, ("h2.top-card-layout__title", "h1.top-card-layout__title", "h2", "h1")
    )
    company = _first_text(
        soup, ("a.topcard__org-name-link", "span.topcard__flavor", ".topcard__org-name")
    )
    return role, company


def posting_title(html: str | None) -> str:
    """A "<role> at <company>" title that `job_meta` splits into both halves."""
    role, company = parse_posting(html)
    if role and company:
        return f"{role} at {company}"
    return role
