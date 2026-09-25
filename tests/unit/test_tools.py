from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from sentinel.tools.builtin import default_registry
from sentinel.tools.builtin.actions import CLOSED_ACCOUNTS, REFUNDS, reset_mock_state
from sentinel.tools.registry import Risk, Tool, ToolRegistry


@pytest.fixture(autouse=True)
def _clean() -> Iterator[None]:
    reset_mock_state()
    yield
    reset_mock_state()


def refund(**overrides: Any) -> dict[str, Any]:
    return {"charge_id": "ch_502", "amount": 49.0, "currency": "GBP", "reason": "dup", **overrides}


async def test_refund_succeeds_once() -> None:
    tools = default_registry()
    first = await tools.execute("issue_refund", "cus_1001", refund())
    assert first.ok and first.data is not None
    assert first.data["status"] == "succeeded" and first.data["amount"] == "49.00 GBP"
    assert "ch_502" in REFUNDS

    second = await tools.execute("issue_refund", "cus_1001", refund())
    assert not second.ok and "already refunded" in (second.error or "")


@pytest.mark.parametrize(
    ("customer", "args", "error"),
    [
        ("cus_9999", refund(), "customer not found"),
        ("cus_1001", refund(charge_id="ch_601"), "not found for this customer"),
        ("cus_1001", refund(amount=50.0), "between 0 and 49.00"),
        ("cus_1001", refund(amount=0), "between 0 and 49.00"),
        ("cus_1001", refund(currency="EUR"), "does not match"),
        ("cus_1001", {"charge_id": "ch_502"}, "invalid arguments"),
        ("cus_1001", refund(extra="x"), "invalid arguments"),
    ],
)
async def test_refund_failures_are_results_not_exceptions(
    customer: str, args: dict[str, Any], error: str
) -> None:
    result = await default_registry().execute("issue_refund", customer, args)
    assert not result.ok and error in (result.error or "")
    assert REFUNDS == {}


async def test_crashing_tool_is_a_failure() -> None:
    async def boom(customer_id: str) -> dict[str, Any]:
        raise RuntimeError("secret internals")

    tools = ToolRegistry([Tool("boom", "x", Risk.READ_ONLY, boom)])
    result = await tools.execute("boom", "cus_1001", {})
    assert not result.ok and result.error == "RuntimeError"


async def test_close_account() -> None:
    result = await default_registry().execute("close_account", "cus_1001", {"reason": "asked"})
    assert result.ok
    assert sorted(CLOSED_ACCOUNTS) == ["cus_1001"]


def test_only_actions_are_proposable() -> None:
    assert [t.name for t in default_registry().proposable()] == ["issue_refund", "close_account"]
