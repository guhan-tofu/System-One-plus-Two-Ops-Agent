"""State builder with PII redaction.

State is what models (Jev and OpenAI, both third parties) see: minimal, and with
emails, phone numbers and card numbers replaced by placeholders. The raw
`WorkItem` stays in code.
"""

from __future__ import annotations

import re
from typing import Any

from sentinel.agent.models import WorkItem

EMAIL = "[EMAIL]"
PHONE = "[PHONE]"
CARD = "[CARD]"

_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
# 13-19 digits, optionally separated by single spaces or dashes; Luhn-checked below.
_CARD_RE = re.compile(r"(?<![\d-])\d(?:[ -]?\d){12,18}(?![\d-])")
# International (+44 20 7946 0958) or grouped national (555-123-4567, (020) 7946 0958)
# numbers. Over-redacting a long digit run is acceptable; leaking a phone is not.
_PHONE_RE = re.compile(
    r"""
    (?<![\w+])
    (?:
        (?:\+\d{1,3}[\s.-]?)?          # optional country code
        (?:\(\d{2,5}\)[\s.-]?)?        # optional (area code)
        \d{2,5}(?:[\s.-]\d{2,5}){1,3}  # grouped digits
      |
        \+?\d{9,15}                   # or one unbroken run
    )
    (?!\w)
    """,
    re.VERBOSE,
)
_MIN_PHONE_DIGITS = 9


def _luhn_ok(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2 == 1:
            d = d * 2 - 9 if d > 4 else d * 2
        total += d
    return total % 10 == 0


def _card(match: re.Match[str]) -> str:
    digits = re.sub(r"\D", "", match.group())
    return CARD if _luhn_ok(digits) else match.group()


def _phone(match: re.Match[str]) -> str:
    digits = re.sub(r"\D", "", match.group())
    return PHONE if len(digits) >= _MIN_PHONE_DIGITS else match.group()


def redact_text(text: str) -> str:
    text = _EMAIL_RE.sub(EMAIL, text)
    text = _CARD_RE.sub(_card, text)
    return _PHONE_RE.sub(_phone, text)


def redact(value: Any) -> Any:
    """Redact every string inside a JSON-like value."""
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, dict):
        return {k: redact(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [redact(v) for v in value]
    return value


def build_state(item: WorkItem) -> dict[str, Any]:
    """Only what decisions need. No IDs, no timestamps, no customer reference."""
    state: dict[str, Any] = {"channel": item.source, "message": redact_text(item.body)}
    if item.subject:
        state["subject"] = redact_text(item.subject)
    return state


def with_enrichment(state: dict[str, Any], key: str, data: Any) -> dict[str, Any]:
    return {**state, key: redact(data)}
