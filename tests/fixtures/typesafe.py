"""Fake official TypeSafe Jev API responses (documented shape). Values are invented."""

from __future__ import annotations

import copy
import json
from typing import Any

import httpx

from sentinel.jev.confidence import choice_confidence

API_URL = "https://jev.test/v1/systemone"
MODEL_ALIAS = "jev-latest"
MODEL_VERSION = "jev-1.13.0"

# Answers for tests.fixtures.jev.QUESTIONS (category / urgency / is_abusive).
ANSWERS: dict[str, Any] = {
    "category": {
        "type": "choice",
        "choice": "billing",
        "probabilities": {"billing": 0.9, "technical": 0.1},
        "confidence": 0.8,
    },
    "urgency": {
        "type": "score",
        "score": 1.4,
        "legend": {"0": "Not urgent", "1": "Soon", "2": "Today"},
        "probabilities": {"0": 0.1, "1": 0.4, "2": 0.5},
        "confidence": 0.35,
    },
    "is_abusive": {"type": "noul", "noul": 0.03},
}


def api_response(
    answers: dict[str, Any] | None = None, *, model: str | None = MODEL_VERSION
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "answers": copy.deepcopy(ANSWERS if answers is None else answers),
        "usage": {"input_tokens": 378, "output_tokens": 65},
    }
    if model is not None:
        body["model"] = model
    return body


def api_error(message: str, error_type: str = "invalid_request_error") -> dict[str, Any]:
    return {"detail": {"error_type": error_type, "message": message}}


def answer_request(
    request: httpx.Request,
    *,
    picks: dict[str, str] | None = None,
    noul: dict[str, float] | None = None,
    p: float = 0.9,
) -> httpx.Response:
    """Answer whatever request was sent, in the documented TypeSafe shape.

    Choice questions get `picks[qid]` (default: the first option) with probability p
    and Jev's confidence formula;
    score questions put all mass on level 1; noul questions get `noul[qid]`
    (default 0.1).
    """
    questions = json.loads(request.content)["questions"]
    answers: dict[str, Any] = {}
    for qid, q in questions.items():
        if q["type"] == "choice":
            options = list(q["criteria"])
            pick = (picks or {}).get(qid, options[0])
            rest = (1 - p) / (len(options) - 1)
            probs = {o: p if o == pick else rest for o in options}
            answers[qid] = {
                "type": "choice",
                "choice": pick,
                "probabilities": probs,
                "confidence": choice_confidence(list(probs.values())),
            }
        elif q["type"] == "score":
            levels = q["criteria"]
            answers[qid] = {
                "type": "score",
                "score": 1.0,
                "legend": {str(i): text for i, text in enumerate(levels)},
                "probabilities": {str(i): 1.0 if i == 1 else 0.0 for i in range(len(levels))},
                "confidence": 1.0,
            }
        else:
            answers[qid] = {"type": "noul", "noul": (noul or {}).get(qid, 0.1)}
    return httpx.Response(200, json={"model": MODEL_VERSION, "answers": answers})
