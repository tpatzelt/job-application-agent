"""Canonicalisation of host, scheme and query order.

The agent keys its seen-URL cache, triage set and merged results on
``canonical_url``. Before scheme/host/query-order normalisation, one posting
reached over http, over ``www.`` and with its parameters in a different order
was three separate keys, so the same job got fetched, scored and notified
more than once.
"""

from __future__ import annotations

from src.url_heuristics import canonical_url


def test_http_and_https_collapse_to_one_key():
    assert canonical_url("http://jobs.lever.co/brightloop/1f2e3d4c") == canonical_url(
        "https://jobs.lever.co/brightloop/1f2e3d4c"
    )
    assert canonical_url("http://jobs.lever.co/brightloop/1f2e3d4c").startswith("https://")


def test_www_prefix_collapses_in_front_of_a_qualified_host():
    plain = "https://boards.greenhouse.io/contentful/jobs/6594753"
    assert canonical_url("https://www.boards.greenhouse.io/contentful/jobs/6594753") == plain


def test_www_and_scheme_collapse_together():
    """The exact labelled duplicate pair the eval corpus still failed on."""
    uuid = "8c2b1a90-4f3e-4c21-9a77-2b1d5e6f7a88"
    assert canonical_url(f"https://www.jobs.lever.co/brightloop/{uuid}") == canonical_url(
        f"http://jobs.lever.co/brightloop/{uuid}"
    )


def test_query_parameter_order_does_not_matter():
    assert canonical_url("https://acme.com/jobs/1?a=1&b=2") == canonical_url(
        "https://acme.com/jobs/1?b=2&a=1"
    )
    assert canonical_url("https://acme.com/jobs/1?b=2&a=1") == "https://acme.com/jobs/1?a=1&b=2"


def test_query_order_normalises_after_tracking_params_are_dropped():
    plain = "https://envelio.jobs.personio.de/job/1768728?language=en"
    assert canonical_url(f"{plain}&utm_source=newsletter") == plain
    assert (
        canonical_url("https://envelio.jobs.personio.de/job/1768728?utm_source=x&language=en")
        == plain
    )


def test_www_in_front_of_an_apex_host_is_kept():
    """canonical_url is also the URL the crawler fetches.

    ``www.linkedin.com`` is LinkedIn's own canonical hostname; rewriting it to
    the apex would change what gets fetched, so the prefix is only dropped
    when it mirrors an already-qualified host.
    """
    for url in (
        "https://www.linkedin.com/jobs/view/3712345678",
        "https://www.jobs/1",
    ):
        assert canonical_url(url) == url


def test_default_port_is_dropped_but_a_real_port_is_kept():
    assert canonical_url("https://acme.com:443/jobs/1") == "https://acme.com/jobs/1"
    assert canonical_url("http://acme.com:80/jobs/1") == "https://acme.com/jobs/1"
    assert canonical_url("http://acme.com:8080/jobs/1") == "https://acme.com:8080/jobs/1"


def test_existing_collapsing_still_applies_under_the_new_scheme_rules():
    base = "https://jobs.lever.co/qonto/560f5d61"
    assert canonical_url("http://www.jobs.lever.co/qonto/560f5d61/apply?lever-origin=applied") == base
    assert canonical_url(f"{base}/#top") == base


def test_non_urls_and_schemeless_urls_still_pass_through_unchanged():
    assert canonical_url("") == ""
    assert canonical_url("not a url") == "not a url"
    assert canonical_url("www.acme.com/jobs/1") == "www.acme.com/jobs/1"
