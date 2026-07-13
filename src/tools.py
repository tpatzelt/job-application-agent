from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class Tool:
    name: str
    description: str
    fn: Callable[..., Any]
    calls: int = 0
    errors: int = 0
    last_error: str | None = None


class ToolRegistry:
    """Named tools with invocation telemetry.

    The orchestrator routes every action (search, fetch, evaluate) through
    this registry so the reflection step can see how tools performed
    (call counts, error counts) and adapt strategy accordingly.
    """

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}
        self._logger = logging.getLogger(self.__class__.__name__)

    def register(self, name: str, description: str, fn: Callable[..., Any]) -> None:
        self._tools[name] = Tool(name=name, description=description, fn=fn)

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise KeyError(f"Unknown tool: {name}")
        return self._tools[name]

    def invoke(self, name: str, *args: Any, **kwargs: Any) -> Any:
        tool = self.get(name)
        tool.calls += 1
        try:
            return tool.fn(*args, **kwargs)
        except Exception as exc:
            tool.errors += 1
            tool.last_error = str(exc)
            raise

    def stats(self) -> dict[str, dict[str, Any]]:
        return {
            tool.name: {
                "calls": tool.calls,
                "errors": tool.errors,
                "last_error": tool.last_error,
            }
            for tool in self._tools.values()
        }
