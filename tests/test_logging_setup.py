from __future__ import annotations

import logging
from logging.handlers import TimedRotatingFileHandler

from src.logging_setup import configure_logging


def _file_handlers() -> list[TimedRotatingFileHandler]:
    return [
        h
        for h in logging.getLogger().handlers
        if isinstance(h, TimedRotatingFileHandler)
    ]


def teardown_function() -> None:
    for handler in _file_handlers():
        handler.close()
    logging.getLogger().handlers.clear()


def test_creates_rotating_file_handler(tmp_path, monkeypatch):
    monkeypatch.delenv("LOG_DIR", raising=False)
    monkeypatch.setenv("LOG_RETENTION_DAYS", "7")

    configure_logging(tmp_path, "crawler")

    handlers = _file_handlers()
    assert len(handlers) == 1
    handler = handlers[0]
    assert handler.baseFilename == str(tmp_path / "data" / "logs" / "crawler.log")
    assert handler.when == "MIDNIGHT"
    assert handler.backupCount == 7

    logging.getLogger("test").info("hello rotation")
    handler.flush()
    assert "hello rotation" in (tmp_path / "data" / "logs" / "crawler.log").read_text()


def test_log_dir_override_and_bad_retention(tmp_path, monkeypatch):
    override = tmp_path / "elsewhere"
    monkeypatch.setenv("LOG_DIR", str(override))
    monkeypatch.setenv("LOG_RETENTION_DAYS", "not-a-number")

    configure_logging(tmp_path, "bot")

    handlers = _file_handlers()
    assert len(handlers) == 1
    assert handlers[0].baseFilename == str(override / "bot.log")
    assert handlers[0].backupCount == 30


def test_unwritable_log_dir_falls_back_to_console(tmp_path, monkeypatch):
    blocker = tmp_path / "blocked"
    blocker.write_text("not a directory")
    monkeypatch.setenv("LOG_DIR", str(blocker / "logs"))

    configure_logging(tmp_path, "crawler")

    assert not _file_handlers()
    assert logging.getLogger().handlers
