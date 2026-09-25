"""Strict parsing of thejevai.com responses into `JevResult`.

Locked to the shape confirmed by `sentinel probe` (2026-09-25):

    {
      "code": 0,
      "message": "ok",
      "data": {
        "creditsUsed": 1,
        "result": {
          "answers": {
            <id>: {"type": "choice", "choice", "probabilities": {option: p}, "confidence"},
            <id>: {"type": "score", "score", "legend": {"0": level, ...},
                   "probabilities": {"0": p, ...}, "confidence"},
            <id>: {"type": "noul", "noul"},
          },
          "usage": {"input_tokens", "output_tokens"},
          "elapsedMs": 852
        }
      }
    }

The vendor does not return a model ID (neither in the body nor in headers). The
requested model is audited instead, with `model_verified=False`; if the vendor
ever starts echoing `model` inside `data` or `data.result`, that value wins and
`model_verified=True`.

Anything else raises `JevResponseError`: we never fill in default answers.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import ValidationError

from sentinel.jev.errors import JevResponseError
from sentinel.jev.models import (
    Answer,
    ChoiceAnswer,
    ChoiceQuestion,
    JevResult,
    NoulAnswer,
    NoulQuestion,
    Question,
    ScoreAnswer,
    ScoreQuestion,
)

OK_CODE = 0
_MESSAGE_LIMIT = 200


def _mapping(obj: Mapping[str, Any], key: str, where: str) -> Mapping[str, Any]:
    value = obj.get(key)
    if not isinstance(value, Mapping):
        raise JevResponseError(f"{where}.{key} is missing or not an object")
    return value


def _number(value: Any) -> float | None:
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    return None


def vendor_message(raw: Any) -> str | None:
    """The envelope's `message`, truncated. Vendor text: informational only."""
    if isinstance(raw, Mapping) and isinstance(raw.get("message"), str):
        return str(raw["message"])[:_MESSAGE_LIMIT]
    return None


def unwrap(raw: Any) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    """Validate the envelope and return `(data, data.result)`."""
    if not isinstance(raw, Mapping):
        raise JevResponseError(f"response body is {type(raw).__name__}, expected an object")
    code = raw.get("code")
    if code != OK_CODE or isinstance(code, bool):
        raise JevResponseError(
            f"vendor returned code {code!r}: {vendor_message(raw) or 'no message'}"
        )
    data = _mapping(raw, "data", "response")
    return data, _mapping(data, "result", "data")


def _check_type(qid: str, question: Question, data: Any) -> None:
    reported = data.get("type") if isinstance(data, Mapping) else None
    if reported != question.type:
        raise JevResponseError(
            f"answer {qid!r}: type {reported!r} does not match question type {question.type!r}"
        )


def _check_choice(qid: str, question: ChoiceQuestion, answer: ChoiceAnswer) -> None:
    if answer.choice not in question.criteria:
        raise JevResponseError(
            f"answer {qid!r}: choice {answer.choice!r} is not one of the offered options"
        )
    unknown = sorted(set(answer.probabilities) - set(question.criteria))
    if unknown:
        raise JevResponseError(f"answer {qid!r}: probabilities for unknown options {unknown}")


def _check_score(qid: str, question: ScoreQuestion, answer: ScoreAnswer) -> None:
    expected = {str(i): level for i, level in enumerate(question.criteria)}
    if dict(answer.legend) != expected:
        raise JevResponseError(f"answer {qid!r}: legend does not match the question's levels")
    unknown = sorted(set(answer.probabilities) - set(expected))
    if unknown:
        raise JevResponseError(f"answer {qid!r}: probabilities for unknown levels {unknown}")
    if not 0 <= answer.score <= len(expected) - 1:
        raise JevResponseError(f"answer {qid!r}: score {answer.score} is outside the level range")


def parse_answer(qid: str, question: Question, data: Any) -> Answer:
    _check_type(qid, question, data)
    try:
        if isinstance(question, ChoiceQuestion):
            choice = ChoiceAnswer.model_validate(data)
            _check_choice(qid, question, choice)
            return choice
        if isinstance(question, ScoreQuestion):
            score = ScoreAnswer.model_validate(data)
            _check_score(qid, question, score)
            return score
        if isinstance(question, NoulQuestion):
            return NoulAnswer.model_validate(data)
    except ValidationError as exc:
        fields = ", ".join(".".join(map(str, e["loc"])) or "<root>" for e in exc.errors())
        raise JevResponseError(f"answer {qid!r} is malformed (fields: {fields})") from None
    raise JevResponseError(f"answer {qid!r}: unsupported question type")  # pragma: no cover


def find_model(data: Mapping[str, Any], result: Mapping[str, Any]) -> str | None:
    for container in (result, data):
        model = container.get("model")
        if isinstance(model, str) and model:
            return model
    return None


def parse_response(
    raw: Any,
    questions: Mapping[str, Question],
    *,
    requested_model: str,
    latency_ms: float,
) -> JevResult:
    data, result = unwrap(raw)
    answers_raw = _mapping(result, "answers", "data.result")
    missing = sorted(set(questions) - set(answers_raw))
    if missing:
        raise JevResponseError(f"response is missing answers for: {missing}")
    answers = {qid: parse_answer(qid, q, answers_raw[qid]) for qid, q in questions.items()}
    returned_model = find_model(data, result)
    usage = result.get("usage")
    return JevResult(
        model=returned_model or requested_model,
        model_verified=returned_model is not None,
        answers=answers,
        usage=dict(usage) if isinstance(usage, Mapping) else None,
        credits_used=_number(data.get("creditsUsed")),
        vendor_elapsed_ms=_number(result.get("elapsedMs")),
        latency_ms=latency_ms,
        raw=dict(raw),
    )


def describe_shape(raw: Any) -> dict[str, Any]:
    """Structural summary of a response (keys and types, no values) for the probe."""
    if not isinstance(raw, Mapping):
        return {"body_type": type(raw).__name__}
    summary: dict[str, Any] = {"top_level_keys": sorted(raw), "code": raw.get("code")}
    data = raw.get("data")
    if not isinstance(data, Mapping):
        return summary
    summary["data_keys"] = sorted(data)
    result = data.get("result")
    if not isinstance(result, Mapping):
        return summary
    summary["result_keys"] = sorted(result)
    answers = result.get("answers")
    if isinstance(answers, Mapping):
        summary["answer_fields"] = {
            qid: sorted(a) if isinstance(a, Mapping) else type(a).__name__
            for qid, a in answers.items()
        }
    summary["model_field_found"] = find_model(data, result) is not None
    usage = result.get("usage")
    summary["usage_keys"] = sorted(usage) if isinstance(usage, Mapping) else None
    return summary
