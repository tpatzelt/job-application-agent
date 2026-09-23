from __future__ import annotations

import re
from urllib.parse import parse_qs, parse_qsl, urlencode, urlparse, urlunparse

# URL kinds, from most to least valuable for the agent.
POSTING = "posting"  # a single job posting (ATS page, job-ID URL)
LISTING = "listing"  # job-related page, likely a careers/jobs page
INDEX = "index"  # a job board search/list page aggregating many openings
OTHER = "other"  # not job-related

_UUID_RE = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.IGNORECASE
)
_DIGIT_RUN_RE = re.compile(r"\d{5,}")
# A bare four-digit segment in 1900-2100 is a calendar year (a blog archive or a
# dated careers page), not a job id.
_YEAR_RE = re.compile(r"19\d{2}|20\d{2}|2100")
# A bare digit segment right after one of these is a page number, not a job id.
_PAGE_SEGMENTS = ("page", "seite")

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
    "stellenanzeigen.de",
    "stellenmarkt.de",
    "meinestadt.de",
    "jobware.de",
    "absolventa.de",
    "jobvector.de",
    "talent.com",
    "careerjet.",
    "neuvoo.",
    "simplyhired.",
    "whatjobs.com",
    "jobsora.",
)

# Query parameters that only identify the traffic source, never the job.
# Postings reached through different referrers (Base10/Accel job boards,
# LinkedIn, a newsletter) are the same posting and must dedupe to one URL.
TRACKING_PARAMS = {
    "t",
    "src",
    "source",
    "ref",
    "referrer",
    "trk",
    "trackingid",
    "gh_src",
    "lever-origin",
    "lever-source",
    "lever-source[]",
    "utm_source",
    "utm_medium",
    "utm_campaign",
    "utm_term",
    "utm_content",
}
# Path suffixes that address the application form of a posting, not a
# different posting.
_APPLY_SUFFIXES = ("apply", "application")


def canonical_url(url: str) -> str:
    """Collapse the variants of one posting URL into a single key.

    Drops the fragment and tracking parameters, lowercases the host,
    removes an /apply suffix and a trailing slash. Used for dedup (cache,
    triage, harvesting, results) so the same job isn't fetched, scored,
    and reported several times under different referral links.
    """
    if not url:
        return url
    try:
        parsed = urlparse(url)
    except ValueError:
        return url
    if not parsed.scheme:
        return url
    host = (parsed.hostname or "").lower()
    if parsed.port:
        host = f"{host}:{parsed.port}"
    path = parsed.path or ""
    parts = [part for part in path.split("/") if part]
    if len(parts) > 1 and parts[-1].lower() in _APPLY_SUFFIXES:
        parts = parts[:-1]
    path = "/" + "/".join(parts) if parts else ""
    kept = [
        (key, value)
        for key, value in parse_qsl(parsed.query, keep_blank_values=True)
        if key.lower() not in TRACKING_PARAMS
        and not key.lower().startswith("utm_")
    ]
    query = urlencode(kept)
    return urlunparse((parsed.scheme.lower(), host, path, "", query, ""))


SEARCH_QUERY_PARAMS = {"q", "query", "search", "keywords", "keyword", "k", "what", "where"}


def is_aggregator_url(url: str) -> bool:
    """True if the URL lives on a known job board/aggregator host."""
    host = (urlparse(url).hostname or "").lower()
    return any(aggregator in host for aggregator in AGGREGATOR_HOSTS)


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

    ats_kind = _ats_kind(host, parts, path, parsed.query)
    if ats_kind is not None:
        return ats_kind

    aggregator_kind = _aggregator_kind(host, parts, path, parsed.query)
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


def _ats_kind(host: str, parts: list[str], path: str, query: str = "") -> str | None:
    if host.endswith("greenhouse.io"):
        # boards.greenhouse.io/<company>/jobs/<id>
        if "jobs" in parts and parts and parts[-1].isdigit():
            return POSTING
        # Embedded application form: boards.greenhouse.io/embed/job_app?token=<id>
        if "embed" in parts and "token" in parse_qs(query):
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
    if host.endswith("bamboohr.com"):
        # <company>.bamboohr.com/careers/<id>; /careers and /careers/list are roots
        if _follows(parts, "careers") and parts[-1].isdigit():
            return POSTING
        return LISTING if _is_customer_board(host, "bamboohr.com") else None
    if host.endswith("teamtailor.com"):
        # <company>.teamtailor.com/jobs/<id>-<slug>; /jobs and /<lang>/jobs are roots
        if _follows(parts, "jobs"):
            return POSTING
        return LISTING if _is_customer_board(host, "teamtailor.com") else None
    if host.endswith("softgarden.io") or host.endswith("softgarden.de"):
        # <company>.softgarden.io/job/<id>; the board root is /<lang>/vacancies
        if _follows(parts, "job"):
            return POSTING
        return LISTING if "vacancies" in parts else None
    if host.endswith("jobvite.com"):
        # jobs.jobvite.com/<company>/job/<id>; /<company> and /<company>/jobs are roots
        if _follows(parts, "job"):
            return POSTING
        return LISTING if _is_customer_board(host, "jobvite.com") else None
    if host.endswith("breezy.hr"):
        # <company>.breezy.hr/p/<id>-<slug>; the board root is /
        if _follows(parts, "p"):
            return POSTING
        return LISTING if _is_customer_board(host, "breezy.hr") else None
    if host.endswith("applytojob.com"):
        # <company>.applytojob.com/apply/<id>/<slug>; /apply and /apply/jobs are roots
        if _follows(parts, "apply") and len(parts) >= 3:
            return POSTING
        return LISTING if _is_customer_board(host, "applytojob.com") else None
    if host.endswith("icims.com"):
        # careers-<company>.icims.com/jobs/<id>/<slug>/job; /jobs/intro is the root
        if _follows(parts, "jobs") and parts[parts.index("jobs") + 1].isdigit():
            return POSTING
        if "search" in parts:
            return INDEX
        return LISTING if _is_customer_board(host, "icims.com") else None
    if host.endswith("taleo.net"):
        # <company>.taleo.net/careersection/<id>/jobdetail.ftl?job=<id>; the board
        # roots are jobsearch.ftl/joblist.ftl, whose short numeric careersection id
        # would otherwise read as a job id.
        lower_path = path.lower()
        if "jobdetail" in lower_path:
            return POSTING
        if "jobsearch" in lower_path or "joblist" in lower_path:
            return LISTING
        return None
    if host.endswith("pinpointhq.com"):
        # <company>.pinpointhq.com/en/jobs/<id>; the board root is /
        if _follows(parts, "jobs") and parts[-1].isdigit():
            return POSTING
        return LISTING if _is_customer_board(host, "pinpointhq.com") else None
    if host.endswith("hire.withgoogle.com"):
        # hire.withgoogle.com/public/jobs/<company>/view/<id>
        if "view" in parts:
            return POSTING
        return LISTING if "jobs" in parts else None
    return None


# Subdomains of an ATS vendor's own domain that serve the vendor's marketing or
# help site rather than a customer's job board.
_VENDOR_SITE_PREFIXES = ("", "www", "help", "support", "blog", "pages", "docs", "newsroom")


def _is_customer_board(host: str, domain: str) -> bool:
    """True if `host` is a customer's board on `domain`, not the vendor's own site."""
    return host[: -len(domain)].strip(".") not in _VENDOR_SITE_PREFIXES


def _follows(parts: list[str], segment: str) -> bool:
    """True if `segment` appears in the path with another segment after it."""
    return segment in parts and parts.index(segment) < len(parts) - 1


def _aggregator_kind(host: str, parts: list[str], path: str, query: str = "") -> str | None:
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
    if "stellenanzeigen.de" in host:
        # Postings: /job/<slug>-<id>/ or /job/detail/<id>/ (singular "job").
        # Listings: /jobs/<role-or-city>/ (plural "jobs").
        if parts[:1] == ["job"] and _DIGIT_RUN_RE.search(parts[-1]):
            return POSTING
        return INDEX
    if "stellenmarkt.de" in host:
        # Postings: /anzeige<id>.html at the root.
        if len(parts) == 1 and re.match(r"anzeige\d+\.html$", parts[0]):
            return POSTING
        return INDEX
    if "meinestadt.de" in host:
        # Postings: /<location>/jk/<id> or /<location>/jkl/<id>.
        if len(parts) >= 3 and parts[-2] in ("jk", "jkl") and _DIGIT_RUN_RE.search(parts[-1]):
            return POSTING
        return INDEX
    if "jobware.de" in host:
        # Postings: /job/detail/<slug>.<id>.html (singular "job").
        # Listings: /jobs/<role-or-city> (plural "jobs").
        if parts[:1] == ["job"] and _DIGIT_RUN_RE.search(parts[-1]):
            return POSTING
        return INDEX
    if "absolventa.de" in host:
        # Postings: /stellenangebote/<id>-b-<slug>.
        if len(parts) >= 2 and parts[0] == "stellenangebote" and _DIGIT_RUN_RE.match(parts[1]):
            return POSTING
        return INDEX
    if "jobvector.de" in host:
        # Postings end in a numeric job id; category/search pages don't.
        if parts and _DIGIT_RUN_RE.search(parts[-1]):
            return POSTING
        return INDEX
    if "talent.com" in host:
        # Postings: /view?id=<id>.
        if parts[:1] == ["view"] and "id" in parse_qs(query):
            return POSTING
        return INDEX
    if "neuvoo." in host:
        # Postings: /job.php?id=<id>.
        if path.rstrip("/").lower() == "/job.php" and "id" in parse_qs(query):
            return POSTING
        return INDEX
    if "simplyhired." in host:
        # Postings: /job/<token> (singular "job"); search is /search.
        if parts[:1] == ["job"]:
            return POSTING
        return INDEX
    if "whatjobs.com" in host:
        # Postings: /jobs?id=<id>; category/search pages don't carry id.
        if parts[:1] == ["jobs"] and "id" in parse_qs(query):
            return POSTING
        return INDEX
    if "jobsora." in host:
        # Postings: /job-<id>; search/listing pages use "jobs-...".
        if parts and re.match(r"job-\d+$", parts[0]):
            return POSTING
        return INDEX
    if any(aggregator in host for aggregator in AGGREGATOR_HOSTS):
        return INDEX
    return None


def _has_posting_id(parts: list[str]) -> bool:
    for index, part in enumerate(parts):
        if _UUID_RE.search(part):
            return True
        if part.isdigit() and len(part) <= 4:
            # A short pure-digit segment is a job id only if it is neither a
            # calendar year (/careers/2024/…) nor a pagination index
            # (/jobs/page/3). Skipping the rest of the loop body for a rejected
            # segment cannot lose a job id: the only branch below that a pure-digit
            # segment could match is _DIGIT_RUN_RE, which is \d{5,} and so never
            # matches a segment of at most four characters.
            if len(part) < 3 or _YEAR_RE.fullmatch(part):
                continue
            if index and parts[index - 1].lower() in _PAGE_SEGMENTS:
                continue
            return True
        if _DIGIT_RUN_RE.search(part):
            return True
    return False
