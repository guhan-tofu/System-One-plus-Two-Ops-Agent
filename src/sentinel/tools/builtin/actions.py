"""Side-effecting sample tools against the in-memory mock store. All data is fake."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from itertools import count
from typing import Any

from pydantic import BaseModel, ConfigDict

from sentinel.tools.builtin.lookups import MOCK_CUSTOMERS
from sentinel.tools.registry import Risk, Tool, ToolError

REFUNDS: dict[str, dict[str, Any]] = {}
"""charge_id -> refund record. Mock state; `reset_mock_state()` clears it."""
CLOSED_ACCOUNTS: set[str] = set()
_refund_ids = count(1)


def reset_mock_state() -> None:
    REFUNDS.clear()
    CLOSED_ACCOUNTS.clear()


class IssueRefundArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    charge_id: str
    amount: float
    currency: str
    reason: str


class CloseAccountArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reason: str


def _money(text: str) -> tuple[Decimal, str]:
    amount, currency = text.split()
    return Decimal(amount), currency


async def issue_refund(
    customer_id: str, *, charge_id: str, amount: float, currency: str, reason: str
) -> dict[str, Any]:
    customer = MOCK_CUSTOMERS.get(customer_id)
    if customer is None:
        raise ToolError("customer not found")
    charge = next((c for c in customer["recent_charges"] if c["id"] == charge_id), None)
    if charge is None:
        raise ToolError(f"charge {charge_id} not found for this customer")
    if charge_id in REFUNDS:
        raise ToolError(f"charge {charge_id} was already refunded")
    charged, charged_currency = _money(charge["amount"])
    try:
        requested = Decimal(str(amount))
    except InvalidOperation:
        raise ToolError("invalid amount") from None
    if currency != charged_currency:
        raise ToolError(f"currency {currency} does not match the charge ({charged_currency})")
    if not Decimal(0) < requested <= charged:
        raise ToolError(f"amount must be between 0 and {charged} {charged_currency}")
    refund = {
        "refund_id": f"re_{next(_refund_ids):04d}",
        "charge_id": charge_id,
        "amount": f"{requested:.2f} {currency}",
        "status": "succeeded",
    }
    REFUNDS[charge_id] = refund
    return refund


async def close_account(customer_id: str, *, reason: str) -> dict[str, Any]:
    if customer_id not in MOCK_CUSTOMERS:
        raise ToolError("customer not found")
    CLOSED_ACCOUNTS.add(customer_id)
    return {"status": "closed"}


ISSUE_REFUND = Tool(
    name="issue_refund",
    description="Refund all or part of one of the customer's recent charges.",
    risk=Risk.WRITE,
    fn=issue_refund,
    args_model=IssueRefundArgs,
    proposable=True,
)
CLOSE_ACCOUNT = Tool(
    name="close_account",
    description="Permanently close the customer's account.",
    risk=Risk.DESTRUCTIVE,
    fn=close_account,
    args_model=CloseAccountArgs,
    proposable=True,
)
