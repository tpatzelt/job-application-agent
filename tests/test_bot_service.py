from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import requests

from src.bot_service import BotService, _format_scan_error
from src.config_manager import Config, EffortBudget
from src.telegram_api import MAX_MESSAGE_CHARS


def _make_config(**overrides: Any) -> Config:
    params: dict[str, Any] = dict(
        max_results=5,
        min_score=70,
        results_json="data/unused.json",
        results_csv="data/unused.csv",
        cache_path="data/unused_cache.json",
        llm_model="mock",
        llm_temperature=0.0,
        llm_max_retries=1,
        llm_min_delay_seconds=0,
        brave_endpoint="mock",
        results_per_query=5,
        request_timeout_seconds=1,
        search_min_delay_seconds=0,
        max_queries_per_iteration=5,
        budget=EffortBudget(max_llm_calls=50, max_search_iterations=20),
    )
    params.update(overrides)
    return Config(**params)


def _service(tmp_path: Path, **cfg: Any) -> BotService:
    return BotService(
        root=tmp_path,
        config=_make_config(**cfg),
        bot_token="test-token",
        brave_key=None,
        openrouter_key=None,
    )


def _epoch(tz: str, year: int, month: int, day: int, hour: int) -> float:
    return datetime(year, month, day, hour, tzinfo=ZoneInfo(tz)).timestamp()


def test_todays_scan_time_uses_configured_hour_and_tz(tmp_path: Path) -> None:
    svc = _service(tmp_path, bot_scan_hour=7, bot_scan_timezone="Europe/Berlin")
    now = _epoch("Europe/Berlin", 2026, 1, 15, 9)  # 09:00 local
    assert svc._todays_scan_time(now) == _epoch("Europe/Berlin", 2026, 1, 15, 7)


def test_due_after_scan_hour_when_not_yet_scanned_today(tmp_path: Path) -> None:
    svc = _service(tmp_path, bot_scan_hour=7, bot_scan_timezone="UTC")
    now = _epoch("UTC", 2026, 1, 15, 8)  # past 07:00
    yesterday = _epoch("UTC", 2026, 1, 14, 7)
    assert svc._is_due(yesterday, now) is True


def test_not_due_before_scan_hour(tmp_path: Path) -> None:
    svc = _service(tmp_path, bot_scan_hour=7, bot_scan_timezone="UTC")
    now = _epoch("UTC", 2026, 1, 15, 6)  # before 07:00
    yesterday = _epoch("UTC", 2026, 1, 14, 7)
    assert svc._is_due(yesterday, now) is False


def test_not_due_when_already_scanned_since_todays_hour(tmp_path: Path) -> None:
    svc = _service(tmp_path, bot_scan_hour=7, bot_scan_timezone="UTC")
    now = _epoch("UTC", 2026, 1, 15, 10)
    scanned_today = _epoch("UTC", 2026, 1, 15, 8)  # after 07:00 already
    assert svc._is_due(scanned_today, now) is False


def test_format_scan_error_names_exception_and_cause() -> None:
    exc = ValueError("Brave API returned 503")
    message = _format_scan_error(exc)
    assert "ValueError" in message
    assert "Brave API returned 503" in message
    assert "next scheduled run" in message
    assert "/run" in message


def test_format_scan_error_scrubs_bot_token_and_query_string() -> None:
    exc = requests.exceptions.RequestException(
        "HTTPSConnectionPool: Max retries exceeded with url: "
        "https://api.telegram.org/bot123456789:AA_supersecrettoken/sendMessage"
        "?chat_id=42 (Caused by NewConnectionError)"
    )
    message = _format_scan_error(exc)
    assert "supersecrettoken" not in message
    assert "123456789" not in message
    assert "chat_id=42" not in message
    assert "RequestException" in message


def test_format_scan_error_scrubs_brave_key_query_param() -> None:
    exc = requests.exceptions.RequestException(
        "GET https://api.search.brave.com/res/v1/web/search"
        "?key=BRAVE_SECRET_VALUE&q=python+jobs failed: 401 Unauthorized"
    )
    message = _format_scan_error(exc)
    assert "BRAVE_SECRET_VALUE" not in message


def test_format_scan_error_truncates_long_cause_within_telegram_limit() -> None:
    exc = RuntimeError("x" * 5000)
    message = _format_scan_error(exc)
    assert len(message) < MAX_MESSAGE_CHARS
    assert "RuntimeError" in message


def test_scan_worker_sends_error_naming_failure(tmp_path: Path) -> None:
    svc = _service(tmp_path)
    sent: list[tuple[str, str]] = []
    svc._telegram.send_message = lambda chat_id, text: sent.append((chat_id, text))

    def boom(chat_id: str) -> None:
        raise ValueError("Brave API returned 503")

    svc._run_scan = boom
    svc._scan_queue.put("42")

    calls = {"n": 0}

    def is_set() -> bool:
        calls["n"] += 1
        return calls["n"] > 1

    svc._stop.is_set = is_set

    svc._scan_worker()

    assert len(sent) == 1
    chat_id, text = sent[0]
    assert chat_id == "42"
    assert "ValueError" in text
    assert "Brave API returned 503" in text


def test_unknown_timezone_falls_back_to_utc(tmp_path: Path) -> None:
    svc = _service(tmp_path, bot_scan_hour=7, bot_scan_timezone="Not/AZone")
    assert svc._scan_tz == ZoneInfo("UTC")
