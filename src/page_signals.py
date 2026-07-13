from __future__ import annotations

import re

from .url_heuristics import POSTING, classify_url

# Phrases (lowercase) that job boards and ATS pages show once a posting has
# been closed, filled, or removed. Matching is done on whitespace-normalized
# lowercased page text.
STALE_PHRASES: tuple[str, ...] = (
    # English
    "no longer accepting applications",
    "job is no longer available",
    "position is no longer available",
    "position has been filled",
    "vacancy has been filled",
    "job posting has expired",
    "posting has expired",
    "this job has expired",
    "job offer is no longer available",
    "this vacancy is closed",
    "applications for this job are closed",
    "applications are now closed",
    "this job has been closed",
    "position is closed",
    "sorry, this job was removed",
    "job you requested is no longer available",
    "job you are looking for is no longer available",
    "this opportunity is no longer open",
    "job not found",
    "posting not found",
    "job does not exist",
    # German
    "stelle ist nicht mehr verfügbar",
    "stelle ist leider nicht mehr verfügbar",
    "stelle wurde bereits besetzt",
    "stelle ist bereits besetzt",
    "stellenangebot ist nicht mehr verfügbar",
    "stellenanzeige ist nicht mehr verfügbar",
    "stellenanzeige ist abgelaufen",
    "bewerbungsfrist ist abgelaufen",
    "diese stelle wurde geschlossen",
)

# City/country names as they appear in local-language postings, keyed by the
# lowercase English name used in preferences.
LOCATION_ALIASES: dict[str, tuple[str, ...]] = {
    "germany": ("germany", "deutschland"),
    "munich": ("munich", "münchen", "muenchen"),
    "cologne": ("cologne", "köln", "koeln"),
    "nuremberg": ("nuremberg", "nürnberg", "nuernberg"),
    "vienna": ("vienna", "wien"),
    "zurich": ("zurich", "zürich", "zuerich"),
    "austria": ("austria", "österreich", "oesterreich"),
    "switzerland": ("switzerland", "schweiz"),
    "netherlands": ("netherlands", "the netherlands", "nederland"),
    "the netherlands": ("netherlands", "the netherlands", "nederland"),
}

# ISO 3166-1 alpha-2 codes for the Brave Search `country` parameter, keyed
# by lowercase country names as they appear in preference locations.
COUNTRY_CODES: dict[str, str] = {
    "germany": "DE",
    "deutschland": "DE",
    "austria": "AT",
    "österreich": "AT",
    "switzerland": "CH",
    "schweiz": "CH",
    "netherlands": "NL",
    "the netherlands": "NL",
    "nederland": "NL",
    "belgium": "BE",
    "france": "FR",
    "spain": "ES",
    "italy": "IT",
    "poland": "PL",
    "denmark": "DK",
    "sweden": "SE",
    "norway": "NO",
    "finland": "FI",
    "united kingdom": "GB",
    "uk": "GB",
    "england": "GB",
    "ireland": "IE",
    "portugal": "PT",
    "united states": "US",
    "usa": "US",
    "canada": "CA",
}

REMOTE_MARKERS: tuple[str, ...] = (
    "remote",
    "home office",
    "homeoffice",
    "home-office",
    "work from home",
    "fully distributed",
)


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


def find_stale_marker(text: str) -> str | None:
    """Return the closed/expired phrase found in the page text, if any."""
    lowered = _normalize(text)
    for phrase in STALE_PHRASES:
        if phrase in lowered:
            return phrase
    return None


def location_terms(locations: list[str]) -> list[str]:
    """Expand preference locations ("Berlin, Germany") into the lowercase
    terms a matching page could contain, including local-language names."""
    terms: list[str] = []
    for location in locations:
        for part in re.split(r"[,/]", str(location)):
            part = part.strip().lower()
            if not part or part == "remote":
                continue
            for term in LOCATION_ALIASES.get(part, (part,)):
                if term not in terms:
                    terms.append(term)
    return terms


def redirected_off_posting(requested_url: str, final_url: str | None) -> bool:
    """True when a posting-shaped URL landed on a page that is not a single
    posting anymore. ATS boards (Greenhouse et al.) redirect dead job IDs to
    the company's generic board page with HTTP 200 — often the only sign the
    posting is gone (Greenhouse appends error=true)."""
    if not final_url or final_url == requested_url:
        return False
    if classify_url(requested_url) != POSTING:
        return False
    if "error=true" in final_url.lower():
        return True
    return classify_url(final_url) != POSTING


def country_code_for(locations: list[str]) -> str | None:
    """First ISO country code found in the preference locations, for the
    Brave Search `country` parameter."""
    for location in locations:
        for part in re.split(r"[,/]", str(location)):
            code = COUNTRY_CODES.get(part.strip().lower())
            if code:
                return code
    return None


def wants_remote(locations: list[str]) -> bool:
    return any("remote" in str(location).lower() for location in locations)


def mentions_location(text: str, locations: list[str]) -> bool:
    """True when the page text mentions any preferred location (or a remote
    marker, when the preferences ask for remote). Empty preferences match."""
    if not locations:
        return True
    lowered = _normalize(text)
    if wants_remote(locations) and any(m in lowered for m in REMOTE_MARKERS):
        return True
    return any(term in lowered for term in location_terms(locations))
