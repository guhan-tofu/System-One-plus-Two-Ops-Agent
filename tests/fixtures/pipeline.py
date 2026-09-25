"""Builders for pipeline tests: Jev results and a scripted in-memory Jev provider."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from sentinel.agent.models import WorkItem
from sentinel.jev.errors import JevError
from sentinel.jev.models import (
    Answer,
    ChoiceAnswer,
    JevResult,
    NoulAnswer,
    Question,
    ScoreAnswer,
    State,
)

URGENCY_LEVELS = ["Not urgent", "Soon", "Today", "Immediately / outage"]


def choice(value: str, confidence: float, options: list[str]) -> ChoiceAnswer:
    rest = (1 - confidence) / (len(options) - 1)
    probs = {o: (confidence if o == value else rest) for o in options}
    return ChoiceAnswer(type="choice", choice=value, probabilities=probs, confidence=confidence)


def triage_answers(
    *,
    category: str = "billing",
    category_confidence: float = 0.9,
    path: str = "generate",
    path_confidence: float = 0.9,
    abusive: float = 0.02,
    urgency: float = 2.0,
) -> dict[str, Answer]:
    return {
        "category": choice(
            category, category_confidence, ["billing", "technical", "account", "sales"]
        ),
        "urgency": ScoreAnswer(
            type="score",
            score=urgency,
            legend={str(i): lvl for i, lvl in enumerate(URGENCY_LEVELS)},
            probabilities={"0": 0.0, "1": 0.2, "2": 0.6, "3": 0.2},
            confidence=0.7,
        ),
        "path": choice(path, path_confidence, ["lookup", "generate", "human"]),
        "is_abusive": NoulAnswer(type="noul", noul=abusive),
    }


def route_answers(tier: str = "fast", confidence: float = 0.9) -> dict[str, Answer]:
    return {"tier": choice(tier, confidence, ["fast", "strong"])}


def guard_answers(safe: float = 0.97) -> dict[str, Answer]:
    return {"safe_to_run": NoulAnswer(type="noul", noul=safe)}


def verify_answers(supported: float = 0.95, status: str = "complete") -> dict[str, Answer]:
    return {
        "claim_supported": NoulAnswer(type="noul", noul=supported),
        "task_status": choice(status, 0.9, ["complete", "verify_more", "failed"]),
    }


def jev_result(answers: dict[str, Answer], provider: str = "fake_jev") -> JevResult:
    return JevResult(
        provider=provider,
        model=f"{provider}-model-1",
        model_verified=provider != "fake_jev",
        answers=answers,
        usage={"cost_usd": 0.001},
        latency_ms=5.0,
    )


def _stage_for(questions: Mapping[str, Question]) -> str:
    if "category" in questions:
        return "triage"
    if "tier" in questions:
        return "route_model"
    if "safe_to_run" in questions:
        return "guard"
    return "verify"


class ScriptedJev:
    """In-memory JevProvider: returns (or raises) the scripted value per stage.

    Stages default to confident "happy path" answers; override any of them.
    """

    def __init__(self, name: str = "fake_jev", **script: dict[str, Answer] | JevError) -> None:
        self.name = name
        self.script: dict[str, dict[str, Answer] | JevError] = {
            "triage": triage_answers(),
            "route_model": route_answers(),
            "guard": guard_answers(),
            "verify": verify_answers(),
            **script,
        }
        self.calls: list[tuple[State, list[str]]] = []

    def states(self, stage: str) -> list[State]:
        return [s for s, qs in self.calls if _stage_for(dict.fromkeys(qs)) == stage]

    async def evaluate(
        self, state: State, questions: Mapping[str, Question], model: str | None = None
    ) -> JevResult:
        self.calls.append((state, sorted(questions)))
        value = self.script[_stage_for(questions)]
        if isinstance(value, JevError):
            raise value
        return jev_result(value, provider=self.name)

    async def aclose(self) -> None:
        return None


def draft_json(reply: str, *calls: dict[str, Any]) -> str:
    """Structured-output body for the generate stage."""
    return json.dumps({"reply": reply, "tool_calls": list(calls)})


def refund_call(amount: float = 49.0, charge_id: str = "ch_502") -> dict[str, Any]:
    return {
        "tool": "issue_refund",
        "charge_id": charge_id,
        "amount": amount,
        "currency": "GBP",
        "reason": "duplicate charge",
    }


def work_item(**overrides: Any) -> WorkItem:
    data: dict[str, Any] = {
        "id": "item-1",
        "source": "ticket",
        "subject": "Charged twice",
        "body": "I was charged twice. Email me at test.one@example.com or call +44 20 7946 0958.",
        "customer_id": "cus_1001",
    }
    data.update(overrides)
    return WorkItem.model_validate(data)
