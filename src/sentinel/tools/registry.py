"""Tool registry. Code executes tools; models only ever see their (redacted) results.

Risk levels (the policy in policies/tools.yaml decides what each level may do):
- read_only:   no side effects (lookups). Runs without a guard.
- write:       changes state (e.g. refunds). Runs only if deterministic policy
               allows it AND Jev's guard is confident it is safe.
- destructive: irreversible (e.g. closing an account). Never runs without a human.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError


class Risk(StrEnum):
    READ_ONLY = "read_only"
    WRITE = "write"
    DESTRUCTIVE = "destructive"


class ToolError(Exception):
    """A tool ran but could not do what was asked (e.g. charge not found)."""


class NoArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")


ToolFn = Callable[..., Awaitable[dict[str, Any]]]


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    risk: Risk
    fn: ToolFn
    """Called as fn(customer_id, **args). The customer reference always comes from
    the work item, never from a model."""
    args_model: type[BaseModel] = NoArgs
    proposable: bool = False
    """Whether the drafting model may propose this tool (lookups run in enrich)."""


class ToolResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    tool: str
    args: dict[str, Any]
    ok: bool
    data: dict[str, Any] | None = None
    error: str | None = None
    latency_ms: float


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

    def proposable(self) -> list[Tool]:
        return [t for t in self._tools.values() if t.proposable]

    async def execute(self, name: str, customer_id: str, args: Mapping[str, Any]) -> ToolResult:
        """Run a tool. Never raises for tool failures: they come back as ok=False."""
        started = time.perf_counter()

        def result(**kw: Any) -> ToolResult:
            return ToolResult(
                tool=name, args=dict(args), latency_ms=(time.perf_counter() - started) * 1000, **kw
            )

        tool = self.get(name)
        try:
            parsed = tool.args_model.model_validate(dict(args))
            data = await tool.fn(customer_id, **parsed.model_dump())
        except ValidationError as exc:
            fields = ", ".join(".".join(map(str, e["loc"])) for e in exc.errors())
            return result(ok=False, error=f"invalid arguments ({fields})")
        except ToolError as exc:
            return result(ok=False, error=str(exc))
        except Exception as exc:  # a crashing tool is a failed tool, never a success
            return result(ok=False, error=f"{type(exc).__name__}")
        return result(ok=True, data=data)
