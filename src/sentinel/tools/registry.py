"""Tool registry. Code executes tools; models only ever see their (redacted) results.

Phase 3 registers read-only tools only. Risk levels and approval rules for
side-effecting tools arrive with the guard stage in Phase 4.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class Risk(StrEnum):
    READ_ONLY = "read_only"
    REVERSIBLE = "reversible"
    DESTRUCTIVE = "destructive"


ToolFn = Callable[..., Awaitable[dict[str, Any]]]


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    risk: Risk
    fn: ToolFn


class ToolNotFoundError(KeyError):
    pass


class ToolRegistry:
    def __init__(self, tools: list[Tool] | None = None) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools or []:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"tool {tool.name!r} already registered")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError:
            raise ToolNotFoundError(name) from None

    def names(self) -> list[str]:
        return sorted(self._tools)
