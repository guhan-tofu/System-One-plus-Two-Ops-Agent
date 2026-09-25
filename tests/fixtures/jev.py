"""Shared Jev test questions. Values are invented; no real data."""

from __future__ import annotations

from sentinel.jev.models import Question
from sentinel.jev.questions import choice, noul, score

QUESTIONS: dict[str, Question] = {
    "category": choice(
        "Which team should handle this?",
        {"billing": "Payments", "technical": "Bugs"},
    ),
    "urgency": score("How urgent?", ["Not urgent", "Soon", "Today"]),
    "is_abusive": noul("Is it abusive?"),
}
