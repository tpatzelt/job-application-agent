from __future__ import annotations

import json

import requests

from src.models import JobResult
from src.notifier import TelegramNotifier


def _result(
    title: str = "Software Engineer",
    url: str = "https://x.io/1",
    reason: str = "matches your Python skills",
) -> JobResult:
    return JobResult(
        title=title,
        company="Unknown",
        url=url,
        score=85,
        reason=reason,
        status="new",
    )


class _Response:
    def __init__(self, ok: bool = True, status_code: int = 200, text: str = "") -> None:
        self.ok = ok
        self.status_code = status_code
        self.text = text


def test_ledger_prevents_resend_across_two_runs(tmp_path, monkeypatch):
    sent = []

    def fake_post(url, json, timeout):
        sent.append(json["text"])
        return _Response()

    monkeypatch.setattr(requests, "post", fake_post)
    ledger_path = tmp_path / "notified.json"

    first = TelegramNotifier("token", "chat", ledger_path=ledger_path)
    assert first.notify_results([_result()])
    assert len(sent) == 1
    assert "Why: matches your Python skills" in sent[0]

    second = TelegramNotifier("token", "chat", ledger_path=ledger_path)
    assert second.notify_results([_result()])

    # No second send: the posting was already recorded as delivered.
    assert len(sent) == 1


def test_failed_send_is_not_recorded_and_is_retried(tmp_path, monkeypatch):
    responses = [_Response(ok=False, status_code=500, text="boom"), _Response()]
    calls = []

    def fake_post(url, json, timeout):
        calls.append(json["text"])
        return responses.pop(0)

    monkeypatch.setattr(requests, "post", fake_post)
    ledger_path = tmp_path / "notified.json"

    first = TelegramNotifier("token", "chat", ledger_path=ledger_path)
    assert first.notify_results([_result()]) is False
    assert len(calls) == 1

    second = TelegramNotifier("token", "chat", ledger_path=ledger_path)
    assert second.notify_results([_result()]) is True
    assert len(calls) == 2


def test_corrupt_ledger_degrades_to_empty(tmp_path, monkeypatch):
    ledger_path = tmp_path / "notified.json"
    ledger_path.write_text("not valid json{{{")

    sent = []

    def fake_post(url, json, timeout):
        sent.append(json["text"])
        return _Response()

    monkeypatch.setattr(requests, "post", fake_post)
    notifier = TelegramNotifier("token", "chat", ledger_path=ledger_path)

    assert notifier.notify_results([_result()])
    assert len(sent) == 1


def test_duplicate_posting_in_one_call_is_sent_once(tmp_path, monkeypatch):
    sent = []

    def fake_post(url, json, timeout):
        sent.append(json["text"])
        return _Response()

    monkeypatch.setattr(requests, "post", fake_post)
    ledger_path = tmp_path / "notified.json"

    notifier = TelegramNotifier("token", "chat", ledger_path=ledger_path)
    results = [
        _result(url="https://jobs.lever.co/a/1"),
        _result(url="http://www.jobs.lever.co/a/1"),
    ]
    assert notifier.notify_results(results)

    assert len(sent) == 1
    message = sent[0]
    assert "1 new job(s) found:" in message
    assert message.count("https://jobs.lever.co/a/1") == 1
    assert "www.jobs.lever.co" not in message


def test_ledger_path_none_preserves_prior_behaviour(monkeypatch):
    sent = []

    def fake_post(url, json, timeout):
        sent.append(json["text"])
        return _Response()

    monkeypatch.setattr(requests, "post", fake_post)
    notifier = TelegramNotifier("token", "chat")

    assert notifier.notify_results([_result()])
    assert notifier.notify_results([_result()])

    # Without a ledger, the same posting is sent again every call.
    assert len(sent) == 2


def test_unwritable_ledger_still_delivers_and_does_not_resend_in_same_notifier(
    tmp_path, monkeypatch, caplog
):
    sent = []

    def fake_post(url, json, timeout):
        sent.append(json["text"])
        return _Response()

    monkeypatch.setattr(requests, "post", fake_post)
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory")
    ledger_path = blocker / "notified.json"

    notifier = TelegramNotifier("token", "chat", ledger_path=ledger_path)

    with caplog.at_level("WARNING"):
        assert notifier.notify_results([_result()]) is True
    assert len(sent) == 1
    assert "Why: matches your Python skills" in sent[0]
    assert "Could not write notification ledger" in caplog.text

    assert notifier.notify_results([_result()]) is True
    assert len(sent) == 1


def test_ledger_records_canonical_url_not_raw_url(tmp_path, monkeypatch):
    sent = []

    def fake_post(url, json, timeout):
        sent.append(json["text"])
        return _Response()

    monkeypatch.setattr(requests, "post", fake_post)
    ledger_path = tmp_path / "notified.json"

    notifier = TelegramNotifier("token", "chat", ledger_path=ledger_path)
    assert notifier.notify_results(
        [_result(url="https://x.io/job/1?utm_source=newsletter")]
    )
    assert len(sent) == 1

    saved = json.loads(ledger_path.read_text())
    assert saved == ["https://x.io/job/1"]

    second = TelegramNotifier("token", "chat", ledger_path=ledger_path)
    assert second.notify_results([_result(url="https://x.io/job/1")])
    assert len(sent) == 1
