from __future__ import annotations

from src.telegram_api import MAX_MESSAGE_CHARS, TelegramClient, parse_update


def test_parse_text_update() -> None:
    update = {
        "update_id": 5,
        "message": {
            "chat": {"id": 42},
            "from": {"first_name": "Tim"},
            "text": "/start",
        },
    }
    message = parse_update(update)
    assert message is not None
    assert message.chat_id == "42"
    assert message.text == "/start"
    assert message.from_name == "Tim"
    assert message.document is None


def test_parse_document_update_uses_caption() -> None:
    update = {
        "update_id": 6,
        "message": {
            "chat": {"id": 42},
            "caption": "my cv",
            "document": {
                "file_id": "abc",
                "file_name": "cv.pdf",
                "mime_type": "application/pdf",
                "file_size": 1000,
            },
        },
    }
    message = parse_update(update)
    assert message is not None
    assert message.document is not None
    assert message.document.file_name == "cv.pdf"
    assert message.text == "my cv"


def test_parse_ignores_non_message_updates() -> None:
    assert parse_update({"update_id": 7}) is None
    assert parse_update({"update_id": 8, "edited_message": {}}) is None


def test_get_updates_sends_poll_timeout_param(monkeypatch) -> None:
    captured = {}

    class FakeResponse:
        def json(self):
            return {"ok": True, "result": []}

    def fake_post(url, json=None, timeout=None):
        captured["url"] = url
        captured["json"] = json
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr("src.telegram_api.requests.post", fake_post)
    client = TelegramClient("token", timeout_seconds=10)
    result = client.get_updates(offset=3, poll_timeout=50)
    assert result == []
    assert captured["json"]["timeout"] == 50
    assert captured["json"]["offset"] == 3
    # HTTP timeout must exceed the long-poll duration.
    assert captured["timeout"] == 60


def test_split_message_respects_limit() -> None:
    text = "line\n" * 2000  # 10000 chars
    chunks = TelegramClient._split_message(text)
    assert all(len(chunk) <= 4096 for chunk in chunks)
    assert "".join(chunk.replace("\n", "") for chunk in chunks).count("line") == 2000


def test_split_message_just_below_limit_is_single_chunk() -> None:
    text = "a" * (MAX_MESSAGE_CHARS - 1)
    assert TelegramClient._split_message(text) == [text]


def test_split_message_exactly_at_limit_is_single_chunk() -> None:
    text = "a" * MAX_MESSAGE_CHARS
    assert TelegramClient._split_message(text) == [text]


def test_split_message_multiline_reconstructs_exactly() -> None:
    # Single newlines only (no blank lines), so every chunk boundary lands
    # exactly on a "\n" and joining with "\n" recovers the original text.
    text = "line\n" * 2000  # well over the limit
    chunks = TelegramClient._split_message(text)
    assert len(chunks) > 1
    assert all(len(chunk) <= MAX_MESSAGE_CHARS for chunk in chunks)
    assert "\n".join(chunks) == text


def test_split_message_no_newline_reconstructs_exactly() -> None:
    text = "x" * (MAX_MESSAGE_CHARS * 2 + 500)
    chunks = TelegramClient._split_message(text)
    assert len(chunks) > 1
    assert all(len(chunk) <= MAX_MESSAGE_CHARS for chunk in chunks)
    assert "".join(chunks) == text


def test_send_message_chunks_match_split_message(monkeypatch) -> None:
    captured: list[dict] = []

    class FakeResponse:
        def json(self):
            return {"ok": True, "result": {}}

    def fake_post(url, json=None, timeout=None):
        captured.append(json)
        return FakeResponse()

    monkeypatch.setattr("src.telegram_api.requests.post", fake_post)
    text = "line\n" * 2000
    expected_chunks = TelegramClient._split_message(text)
    assert len(expected_chunks) > 1

    TelegramClient("token").send_message("1", text)

    assert [call["text"] for call in captured] == expected_chunks
    assert all(call["disable_web_page_preview"] is True for call in captured)
