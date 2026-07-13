from __future__ import annotations

import logging
import os
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
DEFAULT_RETENTION_DAYS = 30


def configure_logging(root: Path, log_name: str) -> None:
    """Set up console logging plus a daily-rotating log file.

    Files land in ``data/logs/<log_name>.log`` (override with ``LOG_DIR``),
    rotate at midnight UTC, and old files are kept for
    ``LOG_RETENTION_DAYS`` days (default 30). ``LOG_LEVEL`` controls
    verbosity for both handlers.
    """
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    handlers: list[logging.Handler] = [logging.StreamHandler()]

    log_dir = Path(os.getenv("LOG_DIR") or root / "data" / "logs")
    try:
        retention = int(os.getenv("LOG_RETENTION_DAYS", str(DEFAULT_RETENTION_DAYS)))
    except ValueError:
        retention = DEFAULT_RETENTION_DAYS
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        handlers.append(
            TimedRotatingFileHandler(
                log_dir / f"{log_name}.log",
                when="midnight",
                backupCount=retention,
                encoding="utf-8",
                utc=True,
            )
        )
    except OSError as exc:
        logging.getLogger(__name__).warning(
            "Could not set up file logging in %s (%s); console only", log_dir, exc
        )

    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format=LOG_FORMAT,
        handlers=handlers,
        force=True,
    )
