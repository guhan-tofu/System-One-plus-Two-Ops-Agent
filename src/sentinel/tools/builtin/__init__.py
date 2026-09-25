"""Built-in sample tools (mock data)."""

from __future__ import annotations

from sentinel.tools.builtin.actions import CLOSE_ACCOUNT, ISSUE_REFUND
from sentinel.tools.builtin.lookups import LOOKUP_CUSTOMER
from sentinel.tools.registry import ToolRegistry


def default_registry() -> ToolRegistry:
    return ToolRegistry([LOOKUP_CUSTOMER, ISSUE_REFUND, CLOSE_ACCOUNT])
