from __future__ import annotations

import dataclasses
import json
import logging
import os
import queue
import threading
import time
from pathlib import Path

from dotenv import load_dotenv

from .config_manager import Config, EffortBudget, load_api_keys, load_config
from .crawler_engine import CrawlerEngine
from .dashboard import DashboardServer
from .intake import IntakeManager
from .llm_service import LLMService
from .logging_setup import configure_logging
from .models import IntakeExtraction
from .notifier import TelegramNotifier
from .orchestrator import Orchestrator
from .telegram_api import IncomingMessage, TelegramClient, parse_update
from .user_store import STATE_ACTIVE, UserStore

_OFFSET_FILE = "bot_offset.json"


class BotService:
    """Always-on multi-user Telegram bot.

    Main thread long-polls Telegram and drives the intake conversation;
    a scheduler thread enqueues users whose scan interval has elapsed;
    a single worker thread runs crawls sequentially (one at a time keeps
    the shared Brave/LLM rate limits and botasaurus tasks well-behaved).
    """

    def __init__(
        self,
        root: Path,
        config: Config,
        bot_token: str,
        brave_key: str | None,
        openrouter_key: str | None,
    ) -> None:
        self._root = root
        self._config = config
        self._bot_token = bot_token
        self._brave_key = brave_key
        self._openrouter_key = openrouter_key
        self._logger = logging.getLogger(self.__class__.__name__)
        self._telegram = TelegramClient(bot_token, config.request_timeout_seconds)
        self._store = UserStore(root / "data")
        self._intake = IntakeManager(
            self._store,
            self._telegram,
            self._extract_profile,
            scan_interval_hours=config.bot_scan_interval_hours,
        )
        self._scan_queue: queue.Queue[str] = queue.Queue()
        self._queued_or_running: set[str] = set()
        self._queue_lock = threading.Lock()
        self._stop = threading.Event()
        self._started_at = time.time()
        self._current_scan: str | None = None

    # ------------------------------------------------------------------
    # Main loops

    def run(self) -> None:
        me = self._telegram.get_me()
        self._logger.info("Bot @%s online", me.get("username"))
        if self._config.dashboard_enabled:
            try:
                DashboardServer(
                    self._root / "data",
                    self._config.dashboard_host,
                    self._config.dashboard_port,
                    status_provider=self._dashboard_status,
                ).start()
            except OSError as exc:
                self._logger.warning("Dashboard failed to start: %s", exc)
        threading.Thread(
            target=self._scan_worker, name="scan-worker", daemon=True
        ).start()
        threading.Thread(
            target=self._scheduler, name="scan-scheduler", daemon=True
        ).start()
        offset = self._load_offset()
        while not self._stop.is_set():
            try:
                updates = self._telegram.get_updates(
                    offset=offset,
                    poll_timeout=self._config.bot_poll_timeout_seconds,
                )
            except Exception as exc:
                self._logger.warning("Polling failed, retrying: %s", exc)
                time.sleep(5)
                continue
            for update in updates:
                offset = max(offset or 0, int(update.get("update_id", 0)) + 1)
                self._save_offset(offset)
                message = parse_update(update)
                if message is None:
                    continue
                try:
                    self._dispatch(message)
                except Exception as exc:
                    self._logger.exception("Error handling message: %s", exc)
                    self._safe_send(
                        message.chat_id,
                        "⚠️ Something went wrong handling that message. "
                        "Please try again.",
                    )

    def _dispatch(self, message: IncomingMessage) -> None:
        text = message.text.strip()
        if text.startswith("/run"):
            self._handle_run_command(message.chat_id)
            return
        reply = self._intake.handle_message(message)
        if reply:
            self._telegram.send_message(message.chat_id, reply)

    def _handle_run_command(self, chat_id: str) -> None:
        record = self._store.load(chat_id)
        if record.state != STATE_ACTIVE:
            self._safe_send(
                chat_id,
                "Please finish setup first - send /start to continue.",
            )
            return
        if self._enqueue_scan(chat_id):
            self._safe_send(
                chat_id,
                "\U0001f50d Scanning for jobs now - I'll message you with "
                "anything I find. This can take a few minutes.",
            )
        else:
            self._safe_send(chat_id, "A scan is already queued or running for you.")

    def _scheduler(self) -> None:
        interval_seconds = self._config.bot_scan_interval_hours * 3600
        while not self._stop.is_set():
            try:
                now = time.time()
                for chat_id in self._store.list_chat_ids():
                    record = self._store.load(chat_id)
                    if record.state != STATE_ACTIVE:
                        continue
                    if now - record.last_scan_at >= interval_seconds:
                        if self._enqueue_scan(chat_id):
                            self._logger.info(
                                "Scheduled scan for user %s", chat_id
                            )
            except Exception as exc:
                self._logger.exception("Scheduler error: %s", exc)
            self._stop.wait(60)

    def _scan_worker(self) -> None:
        while not self._stop.is_set():
            try:
                chat_id = self._scan_queue.get(timeout=5)
            except queue.Empty:
                continue
            try:
                self._current_scan = chat_id
                self._run_scan(chat_id)
            except Exception as exc:
                self._logger.exception("Scan failed for %s: %s", chat_id, exc)
                self._safe_send(
                    chat_id,
                    "⚠️ The job scan hit an error. I'll try again at the "
                    "next scheduled run.",
                )
            finally:
                self._current_scan = None
                with self._queue_lock:
                    self._queued_or_running.discard(chat_id)

    def _enqueue_scan(self, chat_id: str) -> bool:
        with self._queue_lock:
            if chat_id in self._queued_or_running:
                return False
            self._queued_or_running.add(chat_id)
        self._scan_queue.put(chat_id)
        return True

    # ------------------------------------------------------------------
    # Per-user crawl

    def _run_scan(self, chat_id: str) -> None:
        record = self._store.load(chat_id)
        cv_text = self._store.load_document(chat_id, "cv")
        if not cv_text or not record.preferences:
            self._logger.info("User %s has no profile yet, skipping scan", chat_id)
            return
        self._logger.info("Starting scan for user %s", chat_id)
        # Mark the scan attempt up front so a crashing crawl doesn't make
        # the scheduler re-enqueue the same user every minute.
        record.last_scan_at = time.time()
        self._store.save(record)

        budget = EffortBudget(
            max_llm_calls=self._config.budget.max_llm_calls,
            max_search_iterations=self._config.budget.max_search_iterations,
        )
        config = dataclasses.replace(self._config, budget=budget)
        llm = LLMService(config, budget, self._openrouter_key)
        crawler = CrawlerEngine(config, budget, self._brave_key)
        notifier = TelegramNotifier(
            self._bot_token, chat_id, config.request_timeout_seconds
        )
        orchestrator = Orchestrator(config, budget, llm, crawler, notifier=notifier)
        paths = self._store.crawl_paths(chat_id)
        results = orchestrator.run(
            cv_text=cv_text,
            preferences=record.preferences,
            cache_path=paths["cache_path"],
            results_json=paths["results_json"],
            results_csv=paths["results_csv"],
            memory_path=paths["memory_path"],
        )
        self._logger.info(
            "Scan for user %s finished with %s results", chat_id, len(results)
        )
        if not results:
            self._safe_send(
                chat_id,
                "\U0001f50d Scan finished - no new matching jobs this time. "
                "I'll keep looking.",
            )

    def _extract_profile(
        self,
        cv_text: str,
        motivation_text: str,
        job_prefs_text: str,
        answers: list[dict[str, str]],
    ) -> IntakeExtraction:
        budget = EffortBudget(
            max_llm_calls=self._config.bot_intake_max_llm_calls,
            max_search_iterations=0,
        )
        config = dataclasses.replace(self._config, budget=budget)
        llm = LLMService(config, budget, self._openrouter_key)
        return llm.extract_search_profile(
            cv_text, motivation_text, job_prefs_text, answers
        )

    # ------------------------------------------------------------------
    # Helpers

    def _dashboard_status(self) -> dict[str, object]:
        with self._queue_lock:
            queued = sorted(self._queued_or_running)
        return {
            "service": "bot",
            "started_at": self._started_at,
            "queued_or_running": queued,
            "running": self._current_scan,
        }

    def _safe_send(self, chat_id: str, text: str) -> None:
        try:
            self._telegram.send_message(chat_id, text)
        except Exception as exc:
            self._logger.warning("Failed to message %s: %s", chat_id, exc)

    def _offset_path(self) -> Path:
        return self._root / "data" / _OFFSET_FILE

    def _load_offset(self) -> int | None:
        path = self._offset_path()
        if not path.exists():
            return None
        try:
            with path.open("r", encoding="utf-8") as handle:
                return int(json.load(handle).get("offset"))
        except (ValueError, TypeError, json.JSONDecodeError):
            return None

    def _save_offset(self, offset: int) -> None:
        path = self._offset_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump({"offset": offset}, handle)


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    configure_logging(root, "bot")
    load_dotenv()
    profile = os.getenv("JOB_CRAWLER_PROFILE")
    config = load_config(root, profile=profile)
    keys = load_api_keys()
    bot_token = keys.get("telegram_bot_token")
    if not bot_token:
        raise SystemExit(
            "TELEGRAM_BOT_TOKEN is required to run the bot service. "
            "Create a bot via @BotFather and set it in .env."
        )
    service = BotService(
        root,
        config,
        bot_token,
        brave_key=keys.get("brave"),
        openrouter_key=keys.get("openrouter"),
    )
    service.run()


if __name__ == "__main__":
    main()
