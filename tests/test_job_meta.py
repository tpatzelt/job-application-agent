from __future__ import annotations

from src.job_meta import extract_job_meta
from src.url_heuristics import canonical_url


def test_greenhouse_title_yields_role_and_company():
    meta = extract_job_meta(
        "https://boards.greenhouse.io/contentful/jobs/6594753",
        "Job Application for Senior Machine Learning Engineer at Contentful",
        "Jobs at Contentful Current openings at Contentful",
    )
    assert meta.title == "Senior Machine Learning Engineer"
    assert meta.company == "Contentful"


def test_company_first_title_still_finds_the_role():
    meta = extract_job_meta(
        "https://jobs.lever.co/qonto/560f5d61",
        "Qonto - Senior Backend Engineer",
    )
    assert meta.title == "Senior Backend Engineer"
    assert meta.company == "Qonto"


def test_company_comes_from_personio_subdomain():
    meta = extract_job_meta(
        "https://envelio.jobs.personio.de/job/1768728?language=en",
        "Senior Backend Engineer (m/f/d) | envelio Jobs",
    )
    assert meta.title == "Senior Backend Engineer (m/f/d)"
    assert meta.company == "Envelio"


def test_falls_back_to_page_text_when_title_is_boilerplate():
    meta = extract_job_meta(
        "https://acme.com/careers/123",
        "Careers | Acme",
        "Senior Data Engineer Berlin We are looking for a data engineer",
    )
    assert meta.title.startswith("Senior Data Engineer")


def test_unknown_company_when_url_and_title_say_nothing():
    meta = extract_job_meta("https://acme.com/jobs/7", "", "Some text about a role")
    assert meta.company == "Unknown"
    assert meta.title


def test_tracking_parameters_collapse_to_one_url():
    plain = "https://boards.greenhouse.io/contentful/jobs/6594753"
    assert canonical_url(f"{plain}?t=Base10+job+board") == plain
    assert canonical_url(f"{plain}?gh_src=abc123#top") == plain
    assert canonical_url(plain) == plain


def test_apply_suffix_and_trailing_slash_collapse():
    base = "https://jobs.lever.co/qonto/560f5d61"
    assert canonical_url(f"{base}/apply?lever-origin=applied") == base
    assert canonical_url(f"{base}/") == base


def test_meaningful_parameters_survive():
    url = "https://envelio.jobs.personio.de/job/1768728?language=en"
    assert canonical_url(url) == url


def test_non_urls_pass_through_unchanged():
    assert canonical_url("") == ""
    assert canonical_url("not a url") == "not a url"


def test_greenhouse_embedded_application_counts_as_a_posting():
    from src.url_heuristics import POSTING, classify_url

    url = "https://boards.greenhouse.io/embed/job_app?token=4668107008"
    assert classify_url(url) == POSTING


def test_numeric_ats_slug_suffix_is_dropped_from_the_company():
    meta = extract_job_meta("https://boards.greenhouse.io/xapo61/jobs/5301291003", "")
    assert meta.company == "Xapo"
