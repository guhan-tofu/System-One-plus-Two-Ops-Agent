from __future__ import annotations

import pytest

from sentinel.agent.models import WorkItem
from sentinel.agent.state import CARD, EMAIL, PHONE, build_state, redact, redact_text


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("mail me at jane.doe+x@example.co.uk now", f"mail me at {EMAIL} now"),
        ("card 4111 1111 1111 1111 please", f"card {CARD} please"),
        ("card 4111-1111-1111-1111.", f"card {CARD}."),
        ("card 4111111111111111", f"card {CARD}"),
        ("call +44 20 7946 0958 today", f"call {PHONE} today"),
        ("call 555-123-4567", f"call {PHONE}"),
        ("call (020) 7946 0958", f"call {PHONE}"),
        ("call 07700900123", f"call {PHONE}"),
    ],
)
def test_redacts_pii(text: str, expected: str) -> None:
    assert redact_text(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "order ORD-10492 arrived broken",
        "charged $49.99 on 2026-09-01",
        "invoice INV-2026-0042",
        "error code 500 at 10:42",
    ],
)
def test_keeps_non_pii(text: str) -> None:
    assert redact_text(text) == text


def test_non_luhn_digit_run_is_not_a_card() -> None:
    assert CARD not in redact_text("ref 1234567812345678")


def test_redact_nested() -> None:
    data = {"email": "a@b.io", "orders": [{"note": "call 555-123-4567"}], "n": 3}
    assert redact(data) == {"email": EMAIL, "orders": [{"note": f"call {PHONE}"}], "n": 3}


def test_build_state_is_minimal_and_redacted() -> None:
    item = WorkItem(
        id="t-1",
        source="ticket",
        subject="Refund for a@b.io",
        body="I was charged twice, card 4111 1111 1111 1111.",
        customer_id="cus_123",
    )
    state = build_state(item)
    assert state == {
        "channel": "ticket",
        "subject": f"Refund for {EMAIL}",
        "message": f"I was charged twice, card {CARD}.",
    }
    assert "cus_123" not in str(state)
