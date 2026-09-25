"""Fake Jev responses in the confirmed thejevai.com shape. Values are invented."""

from __future__ import annotations

import copy
from typing import Any

from sentinel.jev.models import Question
from sentinel.jev.questions import choice, noul, score

REQUESTED_MODEL = "jev-latest"

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
        "type": "choice",
        "choice": "billing",
        "probabilities": {"billing": 0.9, "technical": 0.1},
        "confidence": 0.9,
    },
    "urgency": {
        "type": "score",
        "score": 1.4,
        "legend": {"0": "Not urgent", "1": "Soon", "2": "Today"},
        "probabilities": {"0": 0.1, "1": 0.4, "2": 0.5},
        "confidence": 0.5,
    },
    "is_abusive": {"type": "noul", "noul": 0.03},
}


def confirmed_response(answers: dict[str, Any] | None = None) -> dict[str, Any]:
    """Envelope exactly as returned by thejevai.com (see sentinel.jev.parsing)."""
    return {
        "code": 0,
        "message": "ok",
        "data": {
            "creditsUsed": 1,
            "result": {
                "answers": copy.deepcopy(ANSWERS if answers is None else answers),
                "usage": {"input_tokens": 42, "output_tokens": 7},
                "elapsedMs": 87,
            },
        },
    }


def error_envelope(message: str = "The decision provider rejected the request.") -> dict[str, Any]:
    return {"code": -1, "message": message}
