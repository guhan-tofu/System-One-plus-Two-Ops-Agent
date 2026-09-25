"""Builders for pipeline tests: Jev results and a scripted in-memory Jev provider."""

from __future__ import annotations

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


def jev_result(answers: dict[str, Answer], provider: str = "fake_jev") -> JevResult:
    return JevResult(
        provider=provider,
        model=f"{provider}-model-1",
        model_verified=provider != "fake_jev",
        answers=answers,
        usage={"cost_usd": 0.001},
        latency_ms=5.0,
    )


class ScriptedJev:
    """In-memory JevProvider: returns (or raises) the scripted value per question set."""

    def __init__(self, name: str = "fake_jev", **script: dict[str, Answer] | JevError) -> None:
        self.name = name
        self.script = script  # keys: "triage", "route_model"
        self.calls: list[tuple[State, list[str]]] = []

    async def evaluate(
        self, state: State, questions: Mapping[str, Question], model: str | None = None
    ) -> JevResult:
        self.calls.append((state, sorted(questions)))
        key = "triage" if "category" in questions else "route_model"
        value = self.script[key]
        if isinstance(value, JevError):
            raise value
        return jev_result(value, provider=self.name)

    async def aclose(self) -> None:
        return None


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
