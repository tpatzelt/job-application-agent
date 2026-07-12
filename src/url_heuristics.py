from __future__ import annotations

import re
from urllib.parse import parse_qs, urlparse

# URL kinds, from most to least valuable for the agent.
POSTING = "posting"  # a single job posting (ATS page, job-ID URL)
LISTING = "listing"  # job-related page, likely a careers/jobs page
INDEX = "index"  # a job board search/list page aggregating many openings
OTHER = "other"  # not job-related

_UUID_RE = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.IGNORECASE
)
_DIGIT_RUN_RE = re.compile(r"\d{5,}")

JOB_TOKENS = ("/jobs", "/job", "careers", "apply", "greenhouse", "lever")

# Job boards/aggregators whose non-posting pages are search/list indexes.
AGGREGATOR_HOSTS = (
    "glassdoor.",
    "stepstone.",
    "indeed.",
    "linkedin.com",
    "xing.com",
    "monster.",
    "ziprecruiter.com",
    "jooble.org",
    "adzuna.",
    "kimeta.de",
    "jobrapido.com",
    "kununu.com",
    "devjobs.de",
)

SEARCH_QUERY_PARAMS = {"q", "query", "search", "keywords", "keyword", "k", "what", "where"}


def classify_url(url: str) -> str:
    """Classify a URL as POSTING, LISTING, INDEX, or OTHER.

    Heuristic only: ATS hosts and job-ID-shaped paths signal a single
    posting; known aggregator hosts and search-style URLs signal an index
    page; generic job tokens without either signal a careers/jobs page.
    """
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    path = parsed.path or ""
    parts = [part for part in path.split("/") if part]

    ats_kind = _ats_kind(host, parts, path)
    if ats_kind is not None:
        return ats_kind

    aggregator_kind = _aggregator_kind(host, path)
    if aggregator_kind is not None:
        return aggregator_kind

    if not any(token in url.lower() for token in JOB_TOKENS):
        return OTHER

    params = set(parse_qs(parsed.query))
    if params & SEARCH_QUERY_PARAMS or "srch" in path.lower() or "search" in parts:
        return INDEX
    if _has_posting_id(parts):
        return POSTING
    return LISTING


def _ats_kind(host: str, parts: list[str], path: str) -> str | None:
    if host.endswith("greenhouse.io"):
        # boards.greenhouse.io/<company>/jobs/<id>
        if "jobs" in parts and parts and parts[-1].isdigit():
            return POSTING
        return LISTING
    if host.endswith("lever.co"):
        # jobs.lever.co/<company>/<uuid>
        return POSTING if len(parts) >= 2 else LISTING
    if host.endswith("myworkdayjobs.com"):
        return POSTING if "/job/" in path.lower() else LISTING
    if host.endswith("smartrecruiters.com"):
        # jobs.smartrecruiters.com/<Company>/<id>-<slug>
        if len(parts) >= 2 and _DIGIT_RUN_RE.search(parts[-1]):
            return POSTING
        return LISTING
    if host.endswith("recruitee.com"):
        # <company>.recruitee.com/o/<slug>
        return POSTING if parts[:1] == ["o"] else LISTING
    if host.endswith("ashbyhq.com"):
        return POSTING if len(parts) >= 2 else LISTING
    if host.endswith("join.com"):
        # join.com/companies/<company>/<id>-<slug>
        if len(parts) >= 3 and parts[0] == "companies":
            return POSTING
        return LISTING
    if host.endswith("workable.com"):
        # jobs.workable.com/view/<id>/<slug> or apply.workable.com/<company>/j/<id>
        if "view" in parts or "j" in parts:
            return POSTING
        return LISTING
    if "personio" in host:
        return POSTING if any(part.isdigit() for part in parts) else LISTING
    return None


def _aggregator_kind(host: str, path: str) -> str | None:
    lower_path = path.lower()
    if "linkedin.com" in host:
        if "/jobs/view/" in lower_path:
            return POSTING
        return INDEX if "/jobs" in lower_path else OTHER
    if "indeed." in host:
        return POSTING if "viewjob" in lower_path else INDEX
    if "glassdoor." in host:
        return POSTING if "/job-listing/" in lower_path else INDEX
    if "stepstone." in host:
        # Individual postings look like /stellenangebote--<slug>--<id>
        return POSTING if "stellenangebote--" in lower_path else INDEX
    if any(aggregator in host for aggregator in AGGREGATOR_HOSTS):
        return INDEX
    return None


def _has_posting_id(parts: list[str]) -> bool:
    for part in parts:
        if _UUID_RE.search(part):
            return True
        if part.isdigit() and len(part) >= 3:
            return True
        if _DIGIT_RUN_RE.search(part):
            return True
    return False
