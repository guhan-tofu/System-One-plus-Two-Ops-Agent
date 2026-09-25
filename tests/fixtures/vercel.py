"""Fake Vercel AI Gateway evaluation-model responses (confirmed shape). Values are invented."""

from __future__ import annotations

import copy
import json
from typing import Any

import httpx

GATEWAY_URL = "https://gateway.test/v4/ai"
EVAL_URL = f"{GATEWAY_URL}/evaluation-model"
GATEWAY_MODEL = "typesafe-ai/jev"

# Answers for tests.fixtures.jev.QUESTIONS (category / urgency / is_abusive).
ANSWERS: dict[str, Any] = {
    "category": {
        "type": "choice",
        "choice": "billing",
        "probabilities": {"billing": 0.9, "technical": 0.1},
    },
    "urgency": {
        "type": "score",
        "score": 1.4,
        "probabilities": {"0": 0.1, "1": 0.4, "2": 0.5},
    },
    "is_abusive": {"type": "boolean", "probability": 0.03},
}
CONFIDENCE = {"category": 0.8, "urgency": 0.35}


def gateway_response(
    answers: dict[str, Any] | None = None,
    *,
    confidence: dict[str, Any] | None = None,
    cost: str = "0",
    warnings: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "answers": copy.deepcopy(ANSWERS if answers is None else answers),
        "rounding": {"probabilityDecimals": 2, "scoreDecimals": 2},
        "usage": {"inputTokens": 378, "outputTokens": 64},
        "warnings": warnings or [],
        "providerMetadata": {
            "typesafe": {"confidence": dict(CONFIDENCE if confidence is None else confidence)},
            "gateway": {
                "routing": {
                    "originalModelId": GATEWAY_MODEL,
                    "canonicalSlug": GATEWAY_MODEL,
                    "finalProvider": "typesafe-ai",
                },
                "cost": cost,
                "generationId": "gen_fake_0001",
            },
        },
    }


def gateway_error(message: str, error_type: str = "invalid_request_error") -> dict[str, Any]:
    return {"error": {"message": message, "type": error_type}}


def answer_request(
    request: httpx.Request,
    *,
    picks: dict[str, str] | None = None,
    boolean: dict[str, float] | None = None,
    p: float = 0.9,
) -> httpx.Response:
    """Answer whatever evaluation request was sent, in the confirmed gateway shape.

    Choice questions get `picks[qid]` (default: the first option) with probability p;
    score questions put all mass on level 1; boolean questions get `boolean[qid]`
    (default 0.1).
    """
    questions = json.loads(request.content)["questions"]
    answers: dict[str, Any] = {}
    for qid, q in questions.items():
        if q["type"] == "choice":
            options = list(q["criteria"])
            pick = (picks or {}).get(qid, options[0])
            rest = (1 - p) / (len(options) - 1)
            answers[qid] = {
                "type": "choice",
                "choice": pick,
                "probabilities": {o: p if o == pick else rest for o in options},
            }
        elif q["type"] == "score":
            levels = len(q["criteria"])
            answers[qid] = {
                "type": "score",
                "score": 1.0,
                "probabilities": {str(i): 1.0 if i == 1 else 0.0 for i in range(levels)},
            }
        else:
            answers[qid] = {"type": "boolean", "probability": (boolean or {}).get(qid, 0.1)}
    return httpx.Response(200, json={"answers": answers, "warnings": []})
