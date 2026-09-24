"""Fake Jev responses. Values are invented; no real data."""

from __future__ import annotations

from typing import Any

from sentinel.jev.models import Question
from sentinel.jev.questions import choice, noul, score

FAKE_MODEL_ID = "jev-test-0.0.1"

QUESTIONS: dict[str, Question] = {
    "category": choice(
        "Which team should handle this?",
        {"billing": "Payments", "technical": "Bugs"},
    ),
    "urgency": score("How urgent?", ["Not urgent", "Soon", "Today"]),
    "is_abusive": noul("Is it abusive?"),
}

ANSWERS: dict[str, Any] = {
    "category": {
        "choice": "billing",
        "probabilities": {"billing": 0.9, "technical": 0.1},
        "confidence": 0.9,
    },
    "urgency": {
        "score": 1.4,
        "legend": "Soon",
        "probabilities": {"Not urgent": 0.1, "Soon": 0.4, "Today": 0.5},
        "confidence": 0.5,
    },
    "is_abusive": {"noul": 0.03},
}


def documented_response(**overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": FAKE_MODEL_ID,
        "answers": ANSWERS,
        "usage": {"input_tokens": 42},
        "elapsedMs": 87,
    }
    body.update(overrides)
    return body
