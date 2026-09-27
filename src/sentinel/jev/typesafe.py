"""Official TypeSafe Jev API (the default Jev provider).

Locked to the documented shape, confirmed by a live probe (2026-09-27):

    POST {JEV_BASE_URL}            (default https://api.typesafe.ai/v1/systemone)
    headers: Authorization: Bearer <JEV_API_KEY>
    body:    {"model", "state", "questions": {id: {"type": "choice"|"score"|"noul", ...}}}

    200: {
      "model": "jev-1.13.0",
      "answers": {
        <id>: {"type": "choice", "choice", "probabilities": {option: p}, "confidence"},
        <id>: {"type": "score", "score", "legend": {"0": text, ...},
               "probabilities": {"0": p, ...}, "confidence"},
        <id>: {"type": "noul", "noul"},
      },
      "usage": {"input_tokens", "output_tokens"}
    }

    401/403 {"detail": {"error_type", "message"}}, 422 validation error,
    429 rate limited / 529 overloaded (both retried, honouring `retry-after`).

Our question and answer models are the API's own, so the request is sent as is.
The API returns a versioned model ID (aliases such as `jev-latest` resolve to it),
which is what gets audited, with `model_verified=True`.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar

import httpx
from pydantic import BaseModel, ConfigDict, ValidationError

from sentinel.config import Settings
from sentinel.jev.checks import check_distribution, probability
from sentinel.jev.errors import JevResponseError
from sentinel.jev.http import HTTPJevProvider
from sentinel.jev.models import (
    Answer,
    ChoiceAnswer,
    ChoiceQuestion,
    JevRequest,
    JevResult,
    NoulAnswer,
    NoulQuestion,
    Question,
    ScoreAnswer,
    ScoreQuestion,
)


class _Strict(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")


class _Choice(_Strict):
    type: str
    choice: str
    probabilities: dict[str, float]
    confidence: float


class _Score(_Strict):
    type: str
    score: float
    legend: dict[str, str]
    probabilities: dict[str, float]
    confidence: float


class _Noul(_Strict):
    type: str
    noul: float


def parse_answer(qid: str, question: Question, data: Any) -> Answer:
    reported = data.get("type") if isinstance(data, Mapping) else None
    if reported != question.type:
        raise JevResponseError(
            f"answer {qid!r}: type {reported!r} does not match question type {question.type!r}"
        )
    try:
        if isinstance(question, ChoiceQuestion):
            choice = _Choice.model_validate(data)
            if choice.choice not in question.criteria:
                raise JevResponseError(
                    f"answer {qid!r}: choice {choice.choice!r} is not one of the offered options"
                )
            check_distribution(qid, choice.probabilities, list(question.criteria))
            return ChoiceAnswer(
                type="choice",
                choice=choice.choice,
                probabilities=choice.probabilities,
                confidence=probability(qid, "confidence", choice.confidence),
            )
        if isinstance(question, ScoreQuestion):
            score = _Score.model_validate(data)
            expected = {str(i): text for i, text in enumerate(question.criteria)}
            if score.legend != expected:
                raise JevResponseError(
                    f"answer {qid!r}: legend does not match the levels we asked about"
                )
            check_distribution(qid, score.probabilities, list(expected))
            if not 0 <= score.score <= len(expected) - 1:
                raise JevResponseError(
                    f"answer {qid!r}: score {score.score} is outside the level range"
                )
            return ScoreAnswer(
                type="score",
                score=score.score,
                legend=expected,
                probabilities=score.probabilities,
                confidence=probability(qid, "confidence", score.confidence),
            )
        if isinstance(question, NoulQuestion):
            noul = _Noul.model_validate(data)
            return NoulAnswer(type="noul", noul=probability(qid, "noul", noul.noul))
    except ValidationError as exc:
        fields = ", ".join(".".join(map(str, e["loc"])) or "<root>" for e in exc.errors())
        raise JevResponseError(f"answer {qid!r} is malformed (fields: {fields})") from None
    raise JevResponseError(f"answer {qid!r}: unsupported question type")  # pragma: no cover


def parse_response(
    raw: Any, questions: Mapping[str, Question], *, requested_model: str, latency_ms: float
) -> JevResult:
    if not isinstance(raw, Mapping):
        raise JevResponseError(f"response body is {type(raw).__name__}, expected an object")
    model = raw.get("model")
    if not isinstance(model, str) or not model:
        raise JevResponseError("response has no model ID")
    answers_raw = raw.get("answers")
    if not isinstance(answers_raw, Mapping):
        raise JevResponseError(f"no answers in response (top-level keys: {sorted(raw)})")
    missing = sorted(set(questions) - set(answers_raw))
    if missing:
        raise JevResponseError(f"response is missing answers for: {missing}")

    answers = {qid: parse_answer(qid, q, answers_raw[qid]) for qid, q in questions.items()}
    usage_raw = raw.get("usage")
    usage = (
        {
            "input_tokens": usage_raw.get("input_tokens"),
            "output_tokens": usage_raw.get("output_tokens"),
        }
        if isinstance(usage_raw, Mapping)
        else None
    )
    return JevResult(
        provider="typesafe",
        model=model,
        model_verified=True,
        answers=answers,
        usage=usage,
        latency_ms=latency_ms,
        metadata={"requested_model": requested_model} if requested_model != model else {},
        raw=dict(raw),
    )


def describe_shape(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        return {"body_type": type(raw).__name__}
    answers = raw.get("answers")
    usage = raw.get("usage")
    return {
        "top_level_keys": sorted(raw),
        "model": raw.get("model") if isinstance(raw.get("model"), str) else None,
        "answer_fields": {
            qid: sorted(a) if isinstance(a, Mapping) else type(a).__name__
            for qid, a in answers.items()
        }
        if isinstance(answers, Mapping)
        else None,
        "usage_keys": sorted(usage) if isinstance(usage, Mapping) else None,
    }


class TypeSafeProvider(HTTPJevProvider):
    name = "typesafe"
    key_env: ClassVar[str] = "JEV_API_KEY"
    validation_statuses: ClassVar[tuple[int, ...]] = (400, 422)
    retry_statuses: ClassVar[tuple[int, ...]] = (529,)

    @classmethod
    def from_settings(
        cls, settings: Settings, client: httpx.AsyncClient | None = None
    ) -> TypeSafeProvider:
        return cls(
            api_key=settings.jev_api_key,
            url=settings.jev_base_url,
            default_model=settings.jev_model,
            client=client,
        )

    def payload(self, request: JevRequest) -> dict[str, Any]:
        return request.model_dump(mode="json", exclude_none=True)

    def parse(self, raw: Any, request: JevRequest, *, latency_ms: float) -> JevResult:
        return parse_response(
            raw, request.questions, requested_model=request.model, latency_ms=latency_ms
        )

    def describe_shape(self, raw: Any) -> dict[str, Any]:
        return describe_shape(raw)
