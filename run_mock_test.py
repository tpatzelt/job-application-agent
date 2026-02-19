from __future__ import annotations

import logging
from pathlib import Path

from src.mock_runner import run_mock_loop


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    root = Path(__file__).resolve().parent
    cache_path = root / "data" / "mock_cache.json"
    if cache_path.exists():
        cache_path.unlink()
    result = run_mock_loop(root)

    if len(result.results) != 2:
        raise AssertionError("Expected 2 results from mock loop")
    if result.llm_calls["queries"] != 1:
        raise AssertionError("Expected one query generation call")
    if result.llm_calls["eval"] != 2:
        raise AssertionError("Expected two evaluation calls")
    if len(result.search_calls) != 2:
        raise AssertionError("Expected two search calls")
    if len(result.fetch_calls) != 2:
        raise AssertionError("Expected two fetch calls")

    print("Mock test passed")


if __name__ == "__main__":
    main()
