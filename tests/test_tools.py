import pytest

from src.tools import ToolRegistry


def test_invoke_records_calls_and_returns_value():
    registry = ToolRegistry()
    registry.register("double", "Double a number", lambda x: x * 2)

    assert registry.invoke("double", 21) == 42
    assert registry.invoke("double", 1) == 2
    stats = registry.stats()
    assert stats["double"]["calls"] == 2
    assert stats["double"]["errors"] == 0


def test_invoke_records_errors_and_reraises():
    registry = ToolRegistry()

    def boom() -> None:
        raise ValueError("kaboom")

    registry.register("boom", "Always fails", boom)
    with pytest.raises(ValueError):
        registry.invoke("boom")
    stats = registry.stats()
    assert stats["boom"]["calls"] == 1
    assert stats["boom"]["errors"] == 1
    assert "kaboom" in stats["boom"]["last_error"]


def test_unknown_tool_raises_key_error():
    registry = ToolRegistry()
    with pytest.raises(KeyError):
        registry.invoke("missing")
