from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import urlparse

UNKNOWN_COMPANY = "Unknown"

# Boilerplate an ATS puts around the role in the HTML <title>.
_TITLE_PREFIXES = (
    "job application for ",
    "apply for ",
    "apply to ",
    "job posting: ",
    "job: ",
    "stellenangebot: ",
    "karriere: ",
)
# Segments of a "<role> - <company> - <site>" title that carry no role.
_TITLE_NOISE = {
    "greenhouse",
    "lever",
    "personio",
    "workable",
    "smartrecruiters",
    "ashby",
    "ashbyhq",
    "recruitee",
    "join",
    "workday",
    "jobs",
    "job",
    "careers",
    "career",
    "jobs at",
    "current openings",
    "stellenangebote",
    "karriere",
    "home",
}
_TITLE_SEPARATORS = re.compile(r"\s+[|·–—]\s+|\s+-\s+")
# "<role> at <company>" — greenhouse's title format names both.
_AT_SPLIT = re.compile(r"\s+(?:at|bei|@)\s+", re.IGNORECASE)
_WHITESPACE = re.compile(r"\s+")

MAX_TITLE_CHARS = 120

# Words that mark a title segment as the role rather than the employer or
# the ATS. ATS titles put the two halves in either order ("Qonto - Senior
# Backend Engineer" vs "Frontend Developer - IT-Layercom").
_ROLE_WORDS = (
    "engineer",
    "developer",
    "manager",
    "scientist",
    "analyst",
    "designer",
    "architect",
    "consultant",
    "specialist",
    "lead",
    "head of",
    "director",
    "intern",
    "internship",
    "praktikum",
    "werkstudent",
    "entwickler",
    "ingenieur",
    "berater",
    "leiter",
    "researcher",
    "administrator",
    "officer",
    "assistant",
    "coordinator",
    "recruiter",
    "product owner",
)


@dataclass(frozen=True)
class JobMeta:
    title: str
    company: str


def extract_job_meta(url: str, page_title: str = "", page_text: str = "") -> JobMeta:
    """Derive a readable role title and employer name for a posting.

    The HTML <title> of an ATS posting names the role (and usually the
    employer); the URL names the employer on every ATS host. Falling back
    to the first words of the page text is a last resort — that is what
    used to produce titles like "Jobs at Contentful Current openings".
    """
    url_company = company_from_url(url)
    title, title_company = _split_title(page_title, url_company)
    company = url_company or title_company or UNKNOWN_COMPANY
    text_title = _title_from_text(page_text)
    if not title or (not looks_like_role(title) and looks_like_role(text_title)):
        title = text_title or title
    title = _drop_company_suffix(title, company)
    return JobMeta(title=title[:MAX_TITLE_CHARS] or "Untitled posting", company=company)


def company_from_url(url: str) -> str:
    """Employer name as encoded in an ATS URL, or "" for other hosts."""
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    parts = [part for part in (parsed.path or "").split("/") if part]
    slug = ""
    if host.endswith(("greenhouse.io", "lever.co")):
        # boards.greenhouse.io/<company>/jobs/<id>, jobs.lever.co/<company>/<id>
        slug = parts[0] if parts else ""
    elif host.endswith(("smartrecruiters.com", "ashbyhq.com")):
        slug = parts[0] if parts else ""
    elif host.endswith("join.com"):
        # join.com/companies/<company>/<id>-<slug>
        slug = parts[1] if len(parts) >= 2 and parts[0] == "companies" else ""
    elif "personio" in host or host.endswith("recruitee.com"):
        # <company>.jobs.personio.de, <company>.recruitee.com
        slug = host.split(".")[0]
    elif host.endswith("myworkdayjobs.com"):
        slug = host.split(".")[0]
    elif host.endswith("workable.com"):
        slug = parts[0] if parts else ""
    if slug in ("", "embed", "www", "jobs", "boards", "apply", "job-boards"):
        return ""
    return _prettify_slug(slug)


def _prettify_slug(slug: str) -> str:
    name = re.sub(r"[-_]+", " ", slug).strip()
    # ATS hosts add a numeric suffix when a company slug is taken
    # ("xapo61", "cscgeneration 2"); keep it only if nothing else is left.
    trimmed = re.sub(r"\s*\d+$", "", name)
    if len(trimmed) >= 3:
        name = trimmed
    if not name:
        return ""
    return name.title() if name.islower() else name


def _split_title(page_title: str, url_company: str = "") -> tuple[str, str]:
    """(role, company) from an HTML <title>; either half may be empty."""
    raw = _WHITESPACE.sub(" ", page_title or "").strip()
    if not raw:
        return "", ""
    lowered = raw.lower()
    for prefix in _TITLE_PREFIXES:
        if lowered.startswith(prefix):
            raw = raw[len(prefix) :].strip()
            break
    company = ""
    at_parts = _AT_SPLIT.split(raw)
    if len(at_parts) == 2 and at_parts[0].strip() and at_parts[1].strip():
        raw, company = at_parts[0].strip(), at_parts[1].strip()
    segments = [seg.strip() for seg in _TITLE_SEPARATORS.split(raw) if seg.strip()]
    kept = [
        seg
        for seg in segments
        if not _is_noise(seg) and seg.strip().lower() != url_company.lower()
    ]
    if not kept:
        return "", company
    roles = [seg for seg in kept if looks_like_role(seg)]
    if roles:
        role = roles[0]
        others = [seg for seg in kept if seg != role]
        if not company and others:
            company = others[-1]
        return role, _strip_noise_words(company)
    if not company and len(kept) > 1:
        company = kept[-1]
        kept = kept[:-1]
    return kept[0], _strip_noise_words(company)


def looks_like_role(segment: str) -> bool:
    lowered = segment.lower()
    return any(word in lowered for word in _ROLE_WORDS)


def _is_noise(segment: str) -> bool:
    cleaned = segment.strip().strip("-–—|").lower()
    return cleaned in _TITLE_NOISE or not cleaned


def _strip_noise_words(company: str) -> str:
    for separator in ("|", "·", "–", "—"):
        company = company.split(separator)[0]
    company = re.sub(
        r"\b(careers?|jobs?|stellenangebote|karriere)\b", "", company, flags=re.IGNORECASE
    )
    return _WHITESPACE.sub(" ", company).strip(" -–—|") or ""


def _title_from_text(page_text: str) -> str:
    words = (page_text or "").split()
    return " ".join(words[:10])


def _drop_company_suffix(title: str, company: str) -> str:
    """Trim a trailing employer name the <title> repeated."""
    if not company or company == UNKNOWN_COMPANY:
        return title.strip()
    pattern = re.compile(
        rf"\s*[-–—|]?\s*(?:at\s+|bei\s+)?{re.escape(company)}\s*$", re.IGNORECASE
    )
    return pattern.sub("", title).strip(" -–—|") or title.strip()
