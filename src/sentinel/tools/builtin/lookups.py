"""Read-only lookup tools backed by an in-memory mock store. All data is fake."""

from __future__ import annotations

from typing import Any

from sentinel.tools.registry import Risk, Tool

MOCK_CUSTOMERS: dict[str, dict[str, Any]] = {
    "cus_1001": {
        "name": "Test Customer One",
        "email": "test.one@example.com",
        "plan": "Pro (monthly)",
        "status": "active",
        "recent_charges": [
            {"id": "ch_501", "date": "2026-09-01", "amount": "49.00 GBP", "status": "paid"},
            {"id": "ch_502", "date": "2026-09-01", "amount": "49.00 GBP", "status": "paid"},
        ],
        "open_incidents": [],
    },
    "cus_1002": {
        "name": "Test Customer Two",
        "email": "test.two@example.com",
        "plan": "Enterprise (annual)",
        "status": "active",
        "recent_charges": [
            {"id": "ch_601", "date": "2026-08-15", "amount": "12000.00 GBP", "status": "paid"}
        ],
        "open_incidents": [{"id": "inc_77", "title": "Dashboard latency (EU)", "status": "open"}],
    },
}


async def lookup_customer(customer_id: str) -> dict[str, Any]:
    customer = MOCK_CUSTOMERS.get(customer_id)
    if customer is None:
        return {"found": False}
    return {"found": True, **customer}


LOOKUP_CUSTOMER = Tool(
    name="lookup_customer",
    description="Account plan, status, recent charges and open incidents for a customer.",
    risk=Risk.READ_ONLY,
    fn=lookup_customer,
)
