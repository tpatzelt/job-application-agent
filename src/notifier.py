from __future__ import annotations

import logging

import requests

from .models import JobResult

TELEGRAM_API_TEMPLATE = "https://api.telegram.org/bot{token}/sendMessage"
# Telegram rejects messages over 4096 characters.
MAX_MESSAGE_CHARS = 4096


class TelegramNotifier:
    """Sends newly found job results to a Telegram chat via the Bot API.

    Delivery failures are logged and reported via the return value but never
    raised, so a Telegram outage can't break a crawl run.
    """

    def __init__(
        self, bot_token: str, chat_id: str, timeout_seconds: int = 15
    ) -> None:
        self._url = TELEGRAM_API_TEMPLATE.format(token=bot_token)
        self._chat_id = chat_id
        self._timeout = timeout_seconds
        self._logger = logging.getLogger(self.__class__.__name__)

    def notify_results(self, results: list[JobResult]) -> bool:
        if not results:
            self._logger.info("No new jobs found, skipping Telegram notification")
            return True
        header = f"\U0001f4bc {len(results)} new job(s) found:"
        entries = [
            self._format_result(index, result)
            for index, result in enumerate(results, start=1)
        ]
        sent_all = True
        for message in self._build_messages(header, entries):
            sent_all = self._send(message) and sent_all
        return sent_all

    def _format_result(self, index: int, result: JobResult) -> str:
        company = f" @ {result.company}" if result.company != "Unknown" else ""
        return (
            f"{index}. {result.title}{company}\n"
            f"Score: {result.score}\n"
            f"{result.url}"
        )

    def _build_messages(self, header: str, entries: list[str]) -> list[str]:
        """Pack the header and entries into as few messages as fit the limit."""
        messages: list[str] = []
        current = header
        for entry in entries:
            entry = entry[:MAX_MESSAGE_CHARS]
            candidate = f"{current}\n\n{entry}"
            if len(candidate) > MAX_MESSAGE_CHARS:
                messages.append(current)
                current = entry
            else:
                current = candidate
        messages.append(current)
        return messages

    def _send(self, message: str) -> bool:
        payload = {
            "chat_id": self._chat_id,
            "text": message,
            "disable_web_page_preview": True,
        }
        try:
            response = requests.post(self._url, json=payload, timeout=self._timeout)
        except requests.RequestException as exc:
            self._logger.warning("Telegram notification failed: %s", exc)
            return False
        if not response.ok:
            self._logger.warning(
                "Telegram API returned %s: %s", response.status_code, response.text
            )
            return False
        return True
