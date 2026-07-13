from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import requests

TELEGRAM_API_BASE = "https://api.telegram.org"
# Telegram rejects messages over 4096 characters.
MAX_MESSAGE_CHARS = 4096
# Bot API caps file downloads at 20MB; stay below that.
MAX_DOWNLOAD_BYTES = 15 * 1024 * 1024


class TelegramError(RuntimeError):
    """Raised when the Telegram Bot API returns an error response."""


@dataclass
class IncomingDocument:
    file_id: str
    file_name: str
    mime_type: str
    file_size: int


@dataclass
class IncomingMessage:
    chat_id: str
    text: str = ""
    document: IncomingDocument | None = None
    from_name: str = ""
    raw: dict[str, Any] = field(default_factory=dict)


class TelegramClient:
    """Thin requests-based wrapper around the Telegram Bot API.

    Used by the bot service for long polling and file downloads; the
    existing TelegramNotifier stays in charge of result notifications.
    """

    def __init__(self, bot_token: str, timeout_seconds: int = 30) -> None:
        self._token = bot_token
        self._timeout = timeout_seconds
        self._logger = logging.getLogger(self.__class__.__name__)

    def _call(
        self, method: str, *, http_timeout: int | None = None, **params: Any
    ) -> Any:
        url = f"{TELEGRAM_API_BASE}/bot{self._token}/{method}"
        response = requests.post(
            url, json=params, timeout=http_timeout or self._timeout
        )
        data = response.json()
        if not data.get("ok"):
            raise TelegramError(
                f"Telegram API {method} failed: {data.get('description')}"
            )
        return data["result"]

    def get_me(self) -> dict[str, Any]:
        return self._call("getMe")

    def get_updates(
        self, offset: int | None = None, poll_timeout: int = 50
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {
            "timeout": poll_timeout,
            "allowed_updates": ["message"],
        }
        if offset is not None:
            params["offset"] = offset
        # Long poll: the HTTP timeout must exceed the poll timeout.
        return self._call(
            "getUpdates", http_timeout=poll_timeout + self._timeout, **params
        )

    def send_message(self, chat_id: str, text: str) -> None:
        for chunk in self._split_message(text):
            self._call(
                "sendMessage",
                chat_id=chat_id,
                text=chunk,
                disable_web_page_preview=True,
            )

    def download_document(self, document: IncomingDocument) -> bytes:
        if document.file_size and document.file_size > MAX_DOWNLOAD_BYTES:
            raise TelegramError(
                f"File too large ({document.file_size} bytes); "
                f"limit is {MAX_DOWNLOAD_BYTES}"
            )
        info = self._call("getFile", file_id=document.file_id)
        file_path = info.get("file_path")
        if not file_path:
            raise TelegramError("Telegram getFile returned no file_path")
        url = f"{TELEGRAM_API_BASE}/file/bot{self._token}/{file_path}"
        response = requests.get(url, timeout=self._timeout)
        response.raise_for_status()
        return response.content

    @staticmethod
    def _split_message(text: str) -> list[str]:
        if len(text) <= MAX_MESSAGE_CHARS:
            return [text]
        chunks: list[str] = []
        remaining = text
        while remaining:
            chunk = remaining[:MAX_MESSAGE_CHARS]
            split_at = chunk.rfind("\n") if len(remaining) > MAX_MESSAGE_CHARS else -1
            if split_at > MAX_MESSAGE_CHARS // 2:
                chunk = chunk[:split_at]
            chunks.append(chunk)
            remaining = remaining[len(chunk):].lstrip("\n")
        return chunks


def parse_update(update: dict[str, Any]) -> IncomingMessage | None:
    """Convert a raw getUpdates entry into an IncomingMessage, or None
    for update types the bot doesn't handle (edits, channel posts, ...)."""
    message = update.get("message")
    if not isinstance(message, dict):
        return None
    chat = message.get("chat") or {}
    chat_id = chat.get("id")
    if chat_id is None:
        return None
    document = None
    doc_data = message.get("document")
    if isinstance(doc_data, dict) and doc_data.get("file_id"):
        document = IncomingDocument(
            file_id=str(doc_data["file_id"]),
            file_name=str(doc_data.get("file_name", "document")),
            mime_type=str(doc_data.get("mime_type", "")),
            file_size=int(doc_data.get("file_size", 0)),
        )
    sender = message.get("from") or {}
    return IncomingMessage(
        chat_id=str(chat_id),
        text=str(message.get("text") or message.get("caption") or ""),
        document=document,
        from_name=str(sender.get("first_name", "")),
        raw=update,
    )
