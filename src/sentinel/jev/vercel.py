"""Official TypeSafe Jev (`typesafe-ai/jev`) through Vercel AI Gateway.

Locked to the shape confirmed by a live probe (2026-09-25), which matches the
AI SDK's EvaluationModelV4 contract:

    POST {AI_GATEWAY_BASE_URL}/evaluation-model
    headers: ai-model-id, ai-evaluation-model-specification-version: 4,
             ai-gateway-protocol-version: 0.0.1, Authorization: Bearer <key>
    body:    {"state", "questions": {id: {"type": "choice"|"score"|"boolean", ...}}}

    200: {
      "answers": {
        <id>: {"type": "choice", "choice", "probabilities": {option: p}},
        <id>: {"type": "score", "score", "probabilities": {"0": p, ...}},
        <id>: {"type": "boolean", "probability"},
      },
      "rounding"?: {...}, "usage"?: {"inputTokens", "outputTokens"}, "warnings": [...],
      "providerMetadata": {
        "typesafe": {"confidence": {<choice/score id>: c}},
        "gateway": {"routing": {"finalProvider", ...}, "cost", "generationId", ...}
      }
    }

Differences from thejevai.com: yes/no questions are `boolean` (answer `probability`)
rather than `noul`; no envelope; no score legend (rebuilt from the question);
confidence lives in provider metadata (computed with Jev's own formulas if absent).
The gateway returns no versioned model ID, so the requested ID is audited with
`model_verified=False`, alongside the generation ID and the upstream provider.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, ClassVar

import httpx
from pydantic import BaseModel, ConfigDict, ValidationError

from sentinel.config import Settings
from sentinel.jev.confidence import choice_confidence, score_confidence
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
from sentinel.log import get_logger

log = get_logger(__name__)

SPEC_VERSION = "4"
PROTOCOL_VERSION = "0.0.1"
WIRE_TYPES = {"choice": "choice", "score": "score", "noul": "boolean"}


class _Strict(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")


class _Choice(_Strict):
    type: str
    choice: str
    probabilities: dict[str, float]


class _Score(_Strict):
    type: str
    score: float
    probabilities: dict[str, float]


class _Boolean(_Strict):
    type: str
    probability: float


def _wire_question(question: Question) -> dict[str, Any]:
    body = question.model_dump(mode="json", exclude_none=True)
    body["type"] = WIRE_TYPES[question.type]
    return body


def _probability(qid: str, name: str, value: float) -> float:
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise JevResponseError(f"answer {qid!r}: {name} {value} is outside [0, 1]")
    return value


def _check_distribution(qid: str, probs: Mapping[str, float], allowed: list[str]) -> None:
    unknown = sorted(set(probs) - set(allowed))
    if unknown:
        raise JevResponseError(f"answer {qid!r}: probabilities for unknown labels {unknown}")
    for label, p in probs.items():
        _probability(qid, f"probability for {label!r}", p)


def _mapping(obj: Any, key: str) -> Mapping[str, Any]:
    value = obj.get(key) if isinstance(obj, Mapping) else None
    return value if isinstance(value, Mapping) else {}


def parse_answer(
    qid: str, question: Question, data: Any, vendor_confidence: Mapping[str, Any]
) -> Answer:
    wire_type = WIRE_TYPES[question.type]
    reported = data.get("type") if isinstance(data, Mapping) else None
    if reported != wire_type:
        raise JevResponseError(
            f"answer {qid!r}: type {reported!r} does not match question type {wire_type!r}"
        )
    confidence = vendor_confidence.get(qid)
    if confidence is not None and not isinstance(confidence, int | float):
        raise JevResponseError(f"answer {qid!r}: confidence is not a number")
    try:
        if isinstance(question, ChoiceQuestion):
            choice = _Choice.model_validate(data)
            options = list(question.criteria)
            if choice.choice not in question.criteria:
                raise JevResponseError(
                    f"answer {qid!r}: choice {choice.choice!r} is not one of the offered options"
                )
            _check_distribution(qid, choice.probabilities, options)
            probs = [choice.probabilities.get(o, 0.0) for o in options]
            return ChoiceAnswer(
                type="choice",
                choice=choice.choice,
                probabilities=choice.probabilities,
                confidence=confidence if confidence is not None else choice_confidence(probs),
            )
        if isinstance(question, ScoreQuestion):
            score = _Score.model_validate(data)
            levels = [str(i) for i in range(len(question.criteria))]
            _check_distribution(qid, score.probabilities, levels)
            if not 0 <= score.score <= len(levels) - 1:
                raise JevResponseError(
                    f"answer {qid!r}: score {score.score} is outside the level range"
                )
            probs = [score.probabilities.get(i, 0.0) for i in levels]
            return ScoreAnswer(
                type="score",
                score=score.score,
                legend=dict(zip(levels, question.criteria, strict=True)),
                probabilities=score.probabilities,
                confidence=confidence if confidence is not None else score_confidence(probs),
            )
        if isinstance(question, NoulQuestion):
            boolean = _Boolean.model_validate(data)
            return NoulAnswer(
                type="noul", noul=_probability(qid, "probability", boolean.probability)
            )
    except ValidationError as exc:
        fields = ", ".join(".".join(map(str, e["loc"])) or "<root>" for e in exc.errors())
        raise JevResponseError(f"answer {qid!r} is malformed (fields: {fields})") from None
    raise JevResponseError(f"answer {qid!r}: unsupported question type")  # pragma: no cover


def parse_response(
    raw: Any, questions: Mapping[str, Question], *, requested_model: str, latency_ms: float
) -> JevResult:
    if not isinstance(raw, Mapping):
        raise JevResponseError(f"response body is {type(raw).__name__}, expected an object")
    answers_raw = raw.get("answers")
    if not isinstance(answers_raw, Mapping):
        raise JevResponseError(f"no answers in response (top-level keys: {sorted(raw)})")
    missing = sorted(set(questions) - set(answers_raw))
    if missing:
        raise JevResponseError(f"response is missing answers for: {missing}")

    provider_meta = _mapping(raw, "providerMetadata")
    vendor_confidence = _mapping(_mapping(provider_meta, "typesafe"), "confidence")
    answers = {
        qid: parse_answer(qid, q, answers_raw[qid], vendor_confidence)
        for qid, q in questions.items()
    }

    gateway = _mapping(provider_meta, "gateway")
    routing = _mapping(gateway, "routing")
    metadata = {
        k: v
        for k, v in {
            "generation_id": gateway.get("generationId"),
            "final_provider": routing.get("finalProvider"),
            "resolved_model": routing.get("canonicalSlug"),
        }.items()
        if isinstance(v, str)
    }
    usage_raw = _mapping(raw, "usage")
    usage: dict[str, Any] = {
        "input_tokens": usage_raw.get("inputTokens"),
        "output_tokens": usage_raw.get("outputTokens"),
        "cost_usd": _cost(gateway.get("cost")),
    }
    warnings = raw.get("warnings")
    if isinstance(warnings, list) and warnings:
        log.warning("jev.vendor_warnings", provider="vercel", warnings=warnings)

    return JevResult(
        provider="vercel",
        model=requested_model,
        model_verified=False,
        answers=answers,
        usage=usage,
        latency_ms=latency_ms,
        metadata=metadata,
        raw=dict(raw),
    )


def _cost(value: Any) -> float | None:
    try:
        cost = float(value)
    except (TypeError, ValueError):
        return None
    return cost if math.isfinite(cost) and cost >= 0 else None


def describe_shape(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        return {"body_type": type(raw).__name__}
    answers = raw.get("answers")
    provider_meta = _mapping(raw, "providerMetadata")
    return {
        "top_level_keys": sorted(raw),
        "answer_fields": {
            qid: sorted(a) if isinstance(a, Mapping) else type(a).__name__
            for qid, a in answers.items()
        }
        if isinstance(answers, Mapping)
        else None,
        "provider_metadata_keys": sorted(provider_meta),
        "confidence_ids": sorted(_mapping(_mapping(provider_meta, "typesafe"), "confidence")),
        "usage_keys": sorted(_mapping(raw, "usage")),
    }


class VercelJevProvider(HTTPJevProvider):
    name = "vercel"
    key_env: ClassVar[str] = "AI_GATEWAY_API_KEY"
    validation_statuses: ClassVar[tuple[int, ...]] = (400, 422)
    # The gateway answers intermittent upstream failures with 503 "Service temporarily
    # unavailable. Please try again shortly." (~50% of calls in the 2026-09-25 eval).
    retry_statuses: ClassVar[tuple[int, ...]] = (503, 529)

    @classmethod
    def from_settings(
        cls, settings: Settings, client: httpx.AsyncClient | None = None
    ) -> VercelJevProvider:
        return cls(
            api_key=settings.ai_gateway_api_key,
            url=f"{settings.ai_gateway_base_url.rstrip('/')}/evaluation-model",
            default_model=settings.jev_gateway_model,
            client=client,
        )

    def payload(self, request: JevRequest) -> dict[str, Any]:
        return {
            "state": request.state,
            "questions": {qid: _wire_question(q) for qid, q in request.questions.items()},
        }

    def headers(self, request: JevRequest) -> dict[str, str]:
        return {
            "ai-model-id": request.model,
            "ai-evaluation-model-specification-version": SPEC_VERSION,
            "ai-gateway-protocol-version": PROTOCOL_VERSION,
            "ai-gateway-auth-method": "api-key",
        }

    def parse(self, raw: Any, request: JevRequest, *, latency_ms: float) -> JevResult:
        return parse_response(
            raw, request.questions, requested_model=request.model, latency_ms=latency_ms
        )

    def describe_shape(self, raw: Any) -> dict[str, Any]:
        return describe_shape(raw)
