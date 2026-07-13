from __future__ import annotations

import requests

from src.models import JobResult
from src.notifier import MAX_MESSAGE_CHARS, TelegramNotifier


def _result(title: str = "Software Engineer", url: str = "https://x.io/1") -> JobResult:
    return JobResult(
        title=title,
        company="Unknown",
        url=url,
        score=85,
        reason="good match",
        status="new",
    )


class _Response:
    def __init__(self, ok: bool = True, status_code: int = 200, text: str = "") -> None:
        self.ok = ok
        self.status_code = status_code
        self.text = text


def test_notify_sends_single_message_with_all_jobs(monkeypatch):
    sent = []

    def fake_post(url, json, timeout):
        sent.append((url, json))
        return _Response()

    monkeypatch.setattr(requests, "post", fake_post)
    notifier = TelegramNotifier("token123", "chat456")

    assert notifier.notify_results([_result(), _result(url="https://x.io/2")])

    assert len(sent) == 1
    url, payload = sent[0]
    assert "bottoken123" in url
    assert payload["chat_id"] == "chat456"
    assert "2 new job(s)" in payload["text"]
    assert "https://x.io/1" in payload["text"]
    assert "https://x.io/2" in payload["text"]
    assert "Score: 85" in payload["text"]


def test_notify_skips_send_when_no_results(monkeypatch):
    def fake_post(url, json, timeout):
        raise AssertionError("should not send for empty results")

    monkeypatch.setattr(requests, "post", fake_post)
    notifier = TelegramNotifier("token", "chat")

    assert notifier.notify_results([])


def test_notify_chunks_long_result_lists(monkeypatch):
    sent = []

    def fake_post(url, json, timeout):
        sent.append(json["text"])
        return _Response()

    monkeypatch.setattr(requests, "post", fake_post)
    notifier = TelegramNotifier("token", "chat")
    results = [
        _result(title="T" * 500, url=f"https://x.io/{i}") for i in range(20)
    ]

    assert notifier.notify_results(results)

    assert len(sent) > 1
    assert all(len(text) <= MAX_MESSAGE_CHARS for text in sent)
    combined = "\n".join(sent)
    for i in range(20):
        assert f"https://x.io/{i}" in combined


def test_notify_returns_false_on_request_error(monkeypatch):
    def fake_post(url, json, timeout):
        raise requests.ConnectionError("network down")

    monkeypatch.setattr(requests, "post", fake_post)
    notifier = TelegramNotifier("token", "chat")

    assert notifier.notify_results([_result()]) is False


def test_notify_returns_false_on_api_error(monkeypatch):
    monkeypatch.setattr(
        requests,
        "post",
        lambda url, json, timeout: _Response(ok=False, status_code=401, text="nope"),
    )
    notifier = TelegramNotifier("token", "chat")

    assert notifier.notify_results([_result()]) is False
