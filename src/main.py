from __future__ import annotations

from pathlib import Path
import os

import logging

from dotenv import load_dotenv

from .config_manager import (
    load_api_keys,
    load_config,
    load_preferences,
    load_user_profile,
)
from .crawler_engine import CrawlerEngine
from .llm_service import LLMService
from .orchestrator import Orchestrator


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    # Allow overriding log level and profile via environment variables
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    load_dotenv()
    # Optional: set JOB_CRAWLER_PROFILE to select a config profile from pyproject.toml
    profile = os.getenv("JOB_CRAWLER_PROFILE")
    if profile:
        logging.getLogger().info("Loading profile: %s", profile)
    config = load_config(root, profile=profile)
    keys = load_api_keys()
    cv_text = load_user_profile(root)
    preferences = load_preferences(root)

    llm = LLMService(config, config.budget, keys.get("openrouter"))
    crawler = CrawlerEngine(config, config.budget, keys.get("brave"))
    orchestrator = Orchestrator(config, config.budget, llm, crawler)

    orchestrator.run(
        cv_text=cv_text,
        preferences=preferences,
        cache_path=root / config.cache_path,
        results_json=root / config.results_json,
        results_csv=root / config.results_csv,
    )


if __name__ == "__main__":
    main()
