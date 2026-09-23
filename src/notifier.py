from __future__ import annotations

import json
import logging
from pathlib import Path

import requests

from .models import JobResult
from .url_heuristics import canonical_url

TELEGRAM_API_TEMPLATE = "https://api.telegram.org/bot{token}/sendMessage"
# Telegram rejects messages over 4096 characters.
MAX_MESSAGE_CHARS = 4096
# Cap a single result's reason so one verbose LLM explanation can't hog a
# whole message and crowd out the other results in the same batch.
MAX_REASON_CHARS = 300


class TelegramNotifier:
    """Sends newly found job results to a Telegram chat via the Bot API.

    Delivery failures are logged and reported via the return value but never
    raised, so a Telegram outage can't break a crawl run.

    When ``ledger_path`` is given, a JSON file there records the canonical
    URL of every result whose message actually sent, and results already in
    it are skipped on later calls (across process restarts, since the
    ledger is read from disk on construction). A URL is only recorded once
    the message carrying it sends successfully, so a Telegram outage leaves
    it unrecorded and eligible for retry. ``ledger_path=None`` (the default)
    preserves prior behaviour: no dedup, nothing written to disk.
    """

    def __init__(
        self,
        bot_token: str,
        chat_id: str,
        timeout_seconds: int = 15,
        ledger_path: Path | None = None,
    ) -> None:
        self._url = TELEGRAM_API_TEMPLATE.format(token=bot_token)
        self._chat_id = chat_id
        self._timeout = timeout_seconds
        self._logger = logging.getLogger(self.__class__.__name__)
        self._ledger_path = ledger_path
        self._notified = self._load_ledger()

    def _load_ledger(self) -> set[str]:
        if self._ledger_path is None or not self._ledger_path.exists():
            return set()
        try:
            data = json.loads(self._ledger_path.read_text())
            if not isinstance(data, list):
                raise ValueError("notification ledger must be a JSON list")
            return {str(entry) for entry in data}
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            self._logger.warning(
                "Could not read notification ledger %s, treating as empty: %s",
                self._ledger_path,
                exc,
            )
            return set()

    def _save_ledger(self) -> None:
        if self._ledger_path is None:
            return
        try:
            self._ledger_path.parent.mkdir(parents=True, exist_ok=True)
            self._ledger_path.write_text(json.dumps(sorted(self._notified)))
        except OSError as exc:
            self._logger.warning(
                "Could not write notification ledger %s: %s", self._ledger_path, exc
            )

    def notify_results(self, results: list[JobResult]) -> bool:
        pending = []
        seen_in_call: set[str] = set()
        for result in results:
            canonical = canonical_url(result.url)
            if canonical in self._notified or canonical in seen_in_call:
                continue
            seen_in_call.add(canonical)
            pending.append(result)
        if not pending:
            if results:
                self._logger.info(
                    "%s job(s) already notified, skipping Telegram notification",
                    len(results),
                )
            else:
                self._logger.info("No new jobs found, skipping Telegram notification")
            return True
        header = f"\U0001f4bc {len(pending)} new job(s) found:"
        entries = [
            (result, self._format_result(index, result))
            for index, result in enumerate(pending, start=1)
        ]
        sent_all = True
        for message, batch in self._build_messages(header, entries):
            if self._send(message):
                self._mark_notified(batch)
            else:
                sent_all = False
        return sent_all

    def _mark_notified(self, batch: list[JobResult]) -> None:
        if not batch or self._ledger_path is None:
            return
        self._notified.update(canonical_url(result.url) for result in batch)
        self._save_ledger()

    def _format_result(self, index: int, result: JobResult) -> str:
        company = f" @ {result.company}" if result.company != "Unknown" else ""
        lines = [f"{index}. {result.title}{company}", f"Score: {result.score}"]
        reason = self._truncate_reason(result.reason)
        if not reason:
            reason = f"scored {result.score}; the evaluator gave no explanation"
        lines.append(f"Why: {reason}")
        lines.append(result.url)
        return "\n".join(lines)

    def _truncate_reason(self, reason: str) -> str:
        reason = (reason or "").strip()
        if len(reason) <= MAX_REASON_CHARS:
            return reason
        return reason[: MAX_REASON_CHARS - 1].rstrip() + "…"

    def _build_messages(
        self, header: str, entries: list[tuple[JobResult, str]]
    ) -> list[tuple[str, list[JobResult]]]:
        """Pack the header and entries into as few messages as fit the limit.

        Each returned message is paired with the results it carries, so a
        result can only be marked notified once the message that carries it
        has actually sent.
        """
        messages: list[tuple[str, list[JobResult]]] = []
        current = header
        current_batch: list[JobResult] = []
        for result, entry in entries:
            entry = entry[:MAX_MESSAGE_CHARS]
            candidate = f"{current}\n\n{entry}"
            if len(candidate) > MAX_MESSAGE_CHARS:
                messages.append((current, current_batch))
                current = entry
                current_batch = [result]
            else:
                current = candidate
                current_batch.append(result)
        messages.append((current, current_batch))
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
