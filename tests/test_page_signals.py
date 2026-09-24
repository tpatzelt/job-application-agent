from __future__ import annotations

import json
from pathlib import Path

from src.page_signals import (
    find_landing_marker,
    find_stale_marker,
    location_terms,
    mentions_location,
    wants_remote,
)


def test_find_stale_marker_english():
    text = (
        "Senior Engineer at Acme.  This job is  no longer available. "
        "Browse similar openings below."
    )
    assert find_stale_marker(text) == "job is no longer available"


def test_find_stale_marker_german():
    text = "Diese Stelle ist leider nicht mehr verfügbar. Zur Jobsuche."
    assert find_stale_marker(text) is not None


def test_find_stale_marker_clean_page():
    text = (
        "Senior Machine Learning Engineer (m/f/d) Berlin. Apply now! "
        "We are looking for an experienced engineer to join our team."
    )
    assert find_stale_marker(text) is None


def test_find_stale_marker_position_closed():
    text = "Update: this position has been closed. Please check our other openings."
    assert find_stale_marker(text) == "this position has been closed"


def test_find_stale_marker_role_no_longer_open():
    text = "Unfortunately, this role is no longer open for applications."
    assert find_stale_marker(text) == "this role is no longer open"


def test_find_stale_marker_job_selected_no_longer_available():
    text = "Sorry, the job you selected is no longer available on this board."
    assert find_stale_marker(text) == "the job you selected is no longer available"


def test_find_stale_marker_german_not_advertised_anymore():
    text = "Leider ist diese Stelle ist nicht mehr ausgeschrieben."
    assert find_stale_marker(text) == "diese stelle ist nicht mehr ausgeschrieben"


def test_find_stale_marker_german_position_unavailable():
    text = "Diese Position ist nicht mehr verfügbar. Vielen Dank für Ihr Interesse."
    assert find_stale_marker(text) == "diese position ist nicht mehr verfügbar"


def test_find_stale_marker_german_listing_removed():
    text = "Die Stellenanzeige wurde entfernt, da die Position besetzt ist."
    assert find_stale_marker(text) == "die stellenanzeige wurde entfernt"


def test_find_stale_marker_german_listing_inactive():
    text = "Diese Anzeige ist nicht mehr aktiv."
    assert find_stale_marker(text) == "diese anzeige ist nicht mehr aktiv"


def test_find_stale_marker_ignores_application_deadline():
    text = (
        "Senior Backend Engineer (m/f/d) Berlin. Apply now! "
        "Application deadline: October 15th. We look forward to your application."
    )
    assert find_stale_marker(text) is None


def test_location_terms_expand_aliases():
    terms = location_terms(["Munich, Germany"])
    assert "münchen" in terms
    assert "deutschland" in terms
    assert "munich" in terms


def test_mentions_location_local_language():
    text = "Standort: München, Deutschland. Vollzeit."
    assert mentions_location(text, ["Munich, Germany"])


def test_mentions_location_mismatch():
    text = "Location: London, United Kingdom. Hybrid, 3 days on-site."
    assert not mentions_location(text, ["Berlin, Germany"])


def test_mentions_location_remote_preference():
    assert wants_remote(["Remote", "Germany"])
    text = "This is a fully remote position within the EU."
    assert mentions_location(text, ["Remote", "Germany"])


def test_mentions_location_empty_preferences_match():
    assert mentions_location("anything", [])


def test_remote_marker_ignored_without_remote_preference():
    text = "Remote position based anywhere in the US."
    assert not mentions_location(text, ["Berlin, Germany"])


def test_find_landing_marker_whatjobs_search_page():
    text = (
        "Jobs in Deutschland? WhatJobs Deutschland. Job gesucht? Unternehmen, "
        "Lebenslauf registrieren, Arbeitgeber, Anzeige schalten, Einloggen. "
        "Für Bewerber: Jobsuche, Unternehmen suchen, Jobs durchsuchen. Suche nach "
        "Berufsbezeichnung oder Rolle und nach Standort, Stadt, Bundesland oder "
        "Postleitzahl. Remote arbeiten? Suchvorschläge und letzte Suchanfragen. "
        "Für Unternehmen: Anzeige schalten, Personalvermittler, Multiposting und ATS."
    )
    assert find_landing_marker(text) is not None


def test_find_landing_marker_requires_two_phrases():
    # Only one landing phrase present ("post a job with us"); a normal
    # posting's footer can plausibly contain a single such phrase, so this
    # alone must not be enough evidence.
    text = (
        "Senior Backend Engineer (m/f/d) Berlin. We are hiring! "
        "Interested employers can post a job with us on our partner board. "
        "Apply now by sending your CV and cover letter."
    )
    assert find_landing_marker(text) is None


def test_find_landing_marker_ignores_incidental_search_mention():
    text = (
        "Senior Backend Engineer (m/f/d) Berlin. You will build and maintain our "
        "internal search infrastructure and help the team search for performance "
        "improvements across the stack. Apply now by sending your CV."
    )
    assert find_landing_marker(text) is None


def test_find_landing_marker_none_for_fixture_postings():
    fixtures_dir = Path(__file__).resolve().parent.parent / "evals" / "fixtures"
    checked = 0
    for fixture_path in sorted(fixtures_dir.glob("*.jsonl")):
        for line in fixture_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("label", {}).get("kind") != "posting":
                continue
            checked += 1
            assert find_landing_marker(record["text"]) is None, record["url"]
    assert checked > 0
