"""policies/tools.yaml rules: every branch (allow / guard / approval / block)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from sentinel.policy.engine import ToolPolicy, ToolRule
from sentinel.tools.builtin import default_registry
from sentinel.tools.registry import Risk, Tool, ToolRegistry

REPO_POLICY = Path(__file__).parents[2] / "policies" / "tools.yaml"


@pytest.fixture
def policy() -> ToolPolicy:
    return ToolPolicy.load(REPO_POLICY)


def refund(**overrides: Any) -> dict[str, Any]:
    return {"charge_id": "ch_1", "amount": 49.0, "currency": "GBP", "reason": "dup", **overrides}


def test_read_only_runs_without_guard(policy: ToolPolicy) -> None:
    assert policy.check("lookup_customer", {}, has_customer=True).action == "allow"


@pytest.mark.parametrize("amount", [0.01, 49.0, 100.0])
def test_refund_within_limit_needs_guard(policy: ToolPolicy, amount: float) -> None:
    assert policy.check("issue_refund", refund(amount=amount), has_customer=True).action == "guard"


@pytest.mark.parametrize(
    ("args", "reason"),
    [
        (refund(amount=100.01), "auto limit"),
        (refund(amount=5000), "auto limit"),
        (refund(currency="EUR"), "currency"),
    ],
)
def test_refund_over_rules_needs_approval(
    policy: ToolPolicy, args: dict[str, Any], reason: str
) -> None:
    verdict = policy.check("issue_refund", args, has_customer=True)
    assert verdict.action == "approval"
    assert reason in verdict.reasons[0]


@pytest.mark.parametrize("args", [{}, {"reason": "customer asked"}, {"amount": 1}])
def test_destructive_always_needs_approval(policy: ToolPolicy, args: dict[str, Any]) -> None:
    verdict = policy.check("close_account", args, has_customer=True)
    assert verdict.action == "approval"
    assert "destructive" in verdict.reasons[0]


@pytest.mark.parametrize(
    ("tool", "args", "has_customer"),
    [
        ("drop_database", {}, True),
        ("issue_refund", refund(amount=-5), True),
        ("issue_refund", refund(amount="49"), True),
        ("issue_refund", refund(), False),
    ],
)
def test_blocked(policy: ToolPolicy, tool: str, args: dict[str, Any], has_customer: bool) -> None:
    assert policy.check(tool, args, has_customer=has_customer).action == "block"


def test_repo_policy_matches_default_registry(policy: ToolPolicy) -> None:
    policy.check_registry(default_registry())


async def _noop(customer_id: str) -> dict[str, Any]:
    return {}


def test_registry_tool_without_rule_rejected(policy: ToolPolicy) -> None:
    with pytest.raises(ValueError, match="no rule"):
        policy.check_registry(ToolRegistry([Tool("mystery", "x", Risk.READ_ONLY, _noop)]))


def test_registry_risk_mismatch_rejected() -> None:
    policy = ToolPolicy(tools={"close_account": ToolRule(risk=Risk.WRITE)})
    with pytest.raises(ValueError, match="risk"):
        policy.check_registry(ToolRegistry([Tool("close_account", "x", Risk.DESTRUCTIVE, _noop)]))
