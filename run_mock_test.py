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
    for name in ("mock_cache.json", "mock_memory.json"):
        stale = root / "data" / name
        if stale.exists():
            stale.unlink()
    result = run_mock_loop(root)

    if len(result.results) != 2:
        raise AssertionError("Expected 2 results from mock loop")
    if result.llm_calls["plan"] != 1:
        raise AssertionError("Expected one planning call")
    if result.llm_calls["queries"] != 2:
        raise AssertionError("Expected two query generation calls")
    if result.llm_calls["eval"] != 2:
        raise AssertionError("Expected two evaluation calls")
    if result.llm_calls["reflect"] != 1:
        raise AssertionError("Expected one reflection call")
    if len(result.search_calls) != 2:
        raise AssertionError("Expected two search calls")
    if len(result.fetch_calls) != 2:
        raise AssertionError("Expected two fetch calls")

    memory_path = root / "data" / "mock_memory.json"
    if not memory_path.exists():
        raise AssertionError("Expected agent memory to be persisted")

    print("Mock test passed")


if __name__ == "__main__":
    main()
