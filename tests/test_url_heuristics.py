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
        # Vendors whose board roots used to be judged on digit shape alone
        # (shapes taken from live Brave Search results, see T-0028).
        "https://euna.bamboohr.com/careers/list",
        "https://everymatrix.teamtailor.com/jobs",
        "https://kompan.teamtailor.com/en/jobs",
        "https://schleswiger-werkstaetten.softgarden.io/de/vacancies",
        "https://jobs.jobvite.com/lhhcareers/jobs",
        "https://taskforcetalent.breezy.hr/",
        "https://phoenix.applytojob.com/apply",
        "https://nro.applytojob.com/apply/jobs/",
        "https://careers-brookings.icims.com/jobs/intro",
        "https://chicago.taleo.net/careersection/100/jobsearch.ftl?lang=en",
        "https://oecd.taleo.net/careersection/ext/joblist.ftl",
        "https://trilongroup.pinpointhq.com/",
        "https://hire.withgoogle.com/public/jobs/touchlab",
    ]
    for url in urls:
        assert classify_url(url) == LISTING, url


def test_ats_vendor_postings():
    # One posting per vendor _ats_kind previously had no rule for; shapes
    # taken from live Brave Search results (see T-0028).
    urls = [
        "https://northtide.bamboohr.com/careers/184",
        "https://footasylum.teamtailor.com/jobs/8429917-seasonal-sales-assistant",
        "https://koelnmesse.softgarden.io/job/3067410?l=de",
        "https://jobs.jobvite.com/egnyte/job/oYKBAfwD",
        "https://freeeup.breezy.hr/p/b8d4f5495eb6-remote-data-entry-specialist",
        "https://partnersforpublicgood.applytojob.com/apply/R8EkJ3qeGX/Research-Fellowship",
        "https://careers-dickblick.icims.com/jobs/5139/retail-human-resources-advisor/job",
        "https://ey.taleo.net/careersection/2/jobdetail.ftl?job=1234567",
        "https://northtide.pinpointhq.com/en/jobs/183726",
        "https://hire.withgoogle.com/public/jobs/telenetworkcom/view/P_AAAAAAEAAHqG0TqHBEvx-U",
    ]
    for url in urls:
        assert classify_url(url) == POSTING, url


def test_ats_vendor_search_pages_are_index():
    assert classify_url("https://careers-northtide.icims.com/jobs/search?ss=1") == INDEX


def test_ats_vendor_own_site_is_not_a_job_board():
    # The vendor's own marketing/help pages share the ATS domain but are not
    # a customer's board, so they must not be promoted to LISTING/POSTING.
    urls = [
        "https://www.jobvite.com/",
        "https://breezy.hr/attract",
        "https://help.breezy.hr/en/articles/5376777-embedding-jobs-on-your-website",
        "https://allyouneedfresh.softgarden.io/de/data-security",
    ]
    for url in urls:
        assert classify_url(url) == OTHER, url


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


# Shapes below taken from live Brave Search results for each host (see
# T-0006 commit message), not guessed, since the corpus fixtures under
# evals/fixtures/ don't cover these hosts.
def test_widened_aggregator_hosts_are_flagged():
    urls = [
        "https://www.stellenanzeigen.de/jobs/software-developer/",
        "https://www.stellenmarkt.de/stellenangebote-in-dortmund",
        "https://jobs.meinestadt.de/berlin",
        "https://www.jobware.de/jobs/muenchen",
        "https://www.absolventa.de/jobs",
        "https://www.jobvector.de/jobs-stellenangebote/datenbanken-data-science/",
        "https://www.talent.com/jobs/l-madison-wi",
        "https://www.careerjet.de/technical-manager-jobs.html",
        "https://neuvoo.de/jobs/N%C3%A4her-jobs",
        "https://www.simplyhired.com/search?l=mount+vernon%2C+in",
        "https://www.whatjobs.com/jobs/all",
        "https://my.jobsora.com/jobs-search",
    ]
    for url in urls:
        assert is_aggregator_url(url), url


def test_widened_aggregator_index_pages():
    urls = [
        "https://www.stellenanzeigen.de/jobs/software-developer/",
        "https://www.stellenanzeigen.de/jobs/wiesbaden/",
        "https://www.stellenmarkt.de/stellenangebote-in-dortmund",
        "https://www.stellenmarkt.de/stellenangebote-softwareentwickler",
        "https://jobs.meinestadt.de/berlin",
        "https://www.jobware.de/jobs/muenchen",
        "https://www.absolventa.de/jobs",
        "https://www.absolventa.de/werkstudentenjobs/job/user-interface-design/stadt/regensburg",
        "https://www.jobvector.de/jobs-stellenangebote/datenbanken-data-science/",
        "https://www.jobvector.de/jobs/data+scientist/deutschland/",
        "https://www.talent.com/jobs/l-madison-wi",
        "https://www.careerjet.de/technical-manager-jobs.html",
        "https://neuvoo.de/jobs/N%C3%A4her-jobs",
        "https://www.simplyhired.com/search?l=mount+vernon%2C+in",
        "https://www.whatjobs.com/jobs/all",
        "https://www.whatjobs.com/jobs/homelines-sales-associate",
        "https://my.jobsora.com/jobs-search",
    ]
    for url in urls:
        assert classify_url(url) == INDEX, url


def test_widened_aggregator_posting_pages():
    urls = [
        "https://www.stellenanzeigen.de/job/strategischer-materialmanager-m-w-d-remscheid-sde-95006/",
        "https://www.stellenanzeigen.de/job/detail/20260410-16180582/",
        "https://www.stellenmarkt.de/anzeige25861448.html",
        "https://jobs.meinestadt.de/deutschland/jk/0-15711-96171",
        "https://www.jobware.de/job/detail/sps-programmierer-softwareentwickler-m-w-d.756748905.html",
        "https://www.absolventa.de/stellenangebote/6588713-b-medizintechniker-m-w-d",
        "https://www.jobvector.de/jobs-stellenangebote/datenbanken-data-science/"
        "business-intelligence-data-warehouse/"
        "data-scientist-data-warehouse-business-intelligence-213504/",
        "https://www.talent.com/view?id=636250678852789108",
        "https://neuvoo.de/job.php?id=uegi3ng6vr&lang=de",
        "https://www.simplyhired.com/job/ugMB5E7EsDREVVd2i7cXApdp6ZPp6XrkQB5fVdRPmTetQj3Oe7EZSA",
        "https://www.whatjobs.com/jobs?id=2047327613",
        "https://us.jobsora.com/job-52226315573",
    ]
    for url in urls:
        assert classify_url(url) == POSTING, url


def test_non_job_urls_are_other():
    urls = [
        "https://blog.company.com/article/how-we-hire",
        "https://example.com/about",
        "https://example.com/contact",
    ]
    for url in urls:
        assert classify_url(url) == OTHER, url


def test_year_segments_are_not_posting_ids():
    """A dated careers/blog path is a hub page, not a single posting."""
    urls = [
        "https://civicstack.example/careers/2024/",
        "https://civicstack.example/careers/2024/rueckblick-digitalprojekte",
        "https://example.com/jobs/1999/review",
        "https://example.com/jobs/2100/outlook",
    ]
    for url in urls:
        assert classify_url(url) == LISTING, url


def test_page_number_segments_are_not_posting_ids():
    """/jobs/page/<n> paginates a board, it does not identify a job."""
    urls = [
        "https://civicstack.example/jobs/page/2",
        "https://civicstack.example/jobs/page/117",
        "https://example.de/karriere/jobs/seite/3",
        "https://example.de/karriere/jobs/Seite/1024",
    ]
    for url in urls:
        assert classify_url(url) in (LISTING, INDEX), url


def test_numeric_posting_ids_still_classify_as_postings():
    """Job ids keep matching: 5+ digit runs, four digits outside the year range, UUIDs."""
    urls = [
        "https://careers.example.com/jobs/4821",  # four digits, not a year
        "https://careers.example.com/jobs/123",
        "https://careers.example.com/jobs/98765",
        "https://careers.example.com/jobs/page/98765-senior-engineer",  # not a bare page number
        "https://careers.example.com/jobs/2024-digital-lead-77310",
        "https://careers.example.com/jobs/8f6f8a2e-1c2d-4e5f-9a0b-1c2d3e4f5a6b",
    ]
    for url in urls:
        assert classify_url(url) == POSTING, url
