from src.url_heuristics import (
    INDEX,
    LISTING,
    OTHER,
    POSTING,
    classify_url,
    is_aggregator_url,
)


def test_ats_postings():
    urls = [
        "https://boards.greenhouse.io/acme/jobs/5678",
        "https://jobs.lever.co/acme/8f6f8a2e-1c2d-4e5f-9a0b-1c2d3e4f5a6b",
        "https://acme.wd3.myworkdayjobs.com/en-US/careers/job/Berlin/Project-Manager_R12345",
        "https://jobs.smartrecruiters.com/Acme/743999912345678-project-manager",
        "https://acme.recruitee.com/o/project-manager-berlin",
        "https://jobs.ashbyhq.com/acme/1b2c3d4e-5f60-7a8b-9c0d-1e2f3a4b5c6d",
        "https://join.com/companies/acme/14237456-project-manager",
        "https://jobs.workable.com/view/abc123/project-manager",
        "https://acme-gmbh.jobs.personio.de/job/1471234",
    ]
    for url in urls:
        assert classify_url(url) == POSTING, url


def test_ats_root_pages_are_listings_not_postings():
    urls = [
        "https://boards.greenhouse.io/acme",
        "https://jobs.lever.co/acme",
        "https://acme.recruitee.com/",
    ]
    for url in urls:
        assert classify_url(url) == LISTING, url


def test_aggregator_index_pages():
    # Shapes taken from real crawler runs.
    urls = [
        "https://www.glassdoor.com/Job/berlin-digital-project-manager-jobs-SRCH_IL.0,6_IC2622109_KO7,30.htm",
        "https://www.stepstone.de/jobs/digital-project-manager/in-berlin",
        "https://www.glassdoor.de/Job/digital-transformation-project-manager-jobs-SRCH_KO0,38.htm",
        "https://en.devjobs.de/jobs/digital-project-manager",
        "https://www.stepstone.de/jobs/project-manager-transformation",
        "https://www.indeed.com/jobs?q=project+manager&l=Berlin",
        "https://www.linkedin.com/jobs/search/?keywords=project%20manager",
    ]
    for url in urls:
        assert classify_url(url) == INDEX, url


def test_aggregator_posting_pages():
    urls = [
        "https://www.linkedin.com/jobs/view/3712345678",
        "https://www.indeed.com/viewjob?jk=abcdef1234567890",
        "https://www.glassdoor.com/job-listing/project-manager-acme-JV_IC2622109_KO0,15.htm",
        "https://www.stepstone.de/stellenangebote--Project-Manager-Berlin-Acme--12345678-inline.html",
    ]
    for url in urls:
        assert classify_url(url) == POSTING, url


def test_generic_urls_with_job_id_are_postings():
    urls = [
        "https://jobs.example.com/job/1234",
        "https://company.com/careers/positions/98765-senior-project-manager",
    ]
    for url in urls:
        assert classify_url(url) == POSTING, url


def test_generic_careers_pages_are_listings():
    urls = [
        "https://company.com/careers/software-engineer",
        "https://company.com/jobs/openings",
    ]
    for url in urls:
        assert classify_url(url) == LISTING, url


def test_generic_search_urls_are_index():
    urls = [
        "https://jobboard.example.com/jobs?q=project+manager",
        "https://jobs.example.com/search/project-manager",
    ]
    for url in urls:
        assert classify_url(url) == INDEX, url


def test_is_aggregator_url():
    assert is_aggregator_url("https://www.linkedin.com/jobs/view/3712345678")
    assert is_aggregator_url("https://www.stepstone.de/stellenangebote--x--1.html")
    assert not is_aggregator_url("https://boards.greenhouse.io/acme/jobs/5678")
    assert not is_aggregator_url("https://company.com/careers/software-engineer")


def test_non_job_urls_are_other():
    urls = [
        "https://blog.company.com/article/how-we-hire",
        "https://example.com/about",
        "https://example.com/contact",
    ]
    for url in urls:
        assert classify_url(url) == OTHER, url
