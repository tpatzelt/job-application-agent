from __future__ import annotations

from src.page_signals import (
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
