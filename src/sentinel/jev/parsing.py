"""Defensive parsing of Jev responses into `JevResult`.

Accepted shapes (until the live probe confirms one and we lock it in):
    {"model", "answers": {...}, "usage"?, "elapsedMs"? | "elapsed"?}      <- documented
    {"model"?, "result": {"answers": {...}, "model"?, ...}}
    {"model"?, "result": {<question id>: {...}, ...}}

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

PRIMARY_SHAPE = "answers"
TIMING_FIELDS = ("elapsedMs", "elapsed_ms", "elapsed")


def locate_answers(raw: Mapping[str, Any]) -> tuple[Mapping[str, Any], str]:
    answers = raw.get("answers")
    if isinstance(answers, Mapping):
        return answers, "answers"
    result = raw.get("result")
    if isinstance(result, Mapping):
        nested = result.get("answers")
        if isinstance(nested, Mapping):
            return nested, "result.answers"
        return result, "result"
    raise JevResponseError(
        f"no answers found in response (top-level keys: {sorted(raw)})"  # keys only, no values
    )


def _containers(raw: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    out: list[Mapping[str, Any]] = [raw]
    for key in ("result", "usage"):
        value = raw.get(key)
        if isinstance(value, Mapping):
            out.append(value)
    return out


def find_model(raw: Mapping[str, Any]) -> str:
    for container in _containers(raw):
        model = container.get("model")
        if isinstance(model, str) and model:
            return model
    raise JevResponseError("response has no model ID; refusing an unauditable answer")


def find_timing(raw: Mapping[str, Any]) -> dict[str, float]:
    timing: dict[str, float] = {}
    for container in _containers(raw):
        for name in TIMING_FIELDS:
            value = container.get(name)
            if isinstance(value, int | float) and not isinstance(value, bool):
                timing.setdefault(name, float(value))
    return timing


def parse_answer(qid: str, question: Question, data: Any) -> Answer:
    try:
        if isinstance(question, ChoiceQuestion):
            choice = ChoiceAnswer.model_validate(data)
            if choice.choice not in question.criteria:
                raise JevResponseError(
                    f"answer {qid!r}: choice {choice.choice!r} is not one of the offered options"
                )
            return choice
        if isinstance(question, ScoreQuestion):
            return ScoreAnswer.model_validate(data)
        if isinstance(question, NoulQuestion):
            if isinstance(data, int | float) and not isinstance(data, bool):
                data = {"noul": data}
            return NoulAnswer.model_validate(data)
    except ValidationError as exc:
        fields = ", ".join(".".join(map(str, e["loc"])) or "<root>" for e in exc.errors())
        raise JevResponseError(f"answer {qid!r} is malformed (fields: {fields})") from None
    raise JevResponseError(f"answer {qid!r}: unsupported question type")  # pragma: no cover


def parse_response(raw: Any, questions: Mapping[str, Question], *, latency_ms: float) -> JevResult:
    if not isinstance(raw, Mapping):
        raise JevResponseError(f"response body is {type(raw).__name__}, expected an object")
    answers_raw, shape = locate_answers(raw)
    missing = sorted(set(questions) - set(answers_raw))
    if missing:
        raise JevResponseError(f"response is missing answers for: {missing}")
    answers = {qid: parse_answer(qid, q, answers_raw[qid]) for qid, q in questions.items()}
    usage = raw.get("usage")
    return JevResult(
        model=find_model(raw),
        answers=answers,
        usage=dict(usage) if isinstance(usage, Mapping) else None,
        vendor_timing=find_timing(raw),
        latency_ms=latency_ms,
        shape=shape,
        raw=dict(raw),
    )


def describe_shape(raw: Any) -> dict[str, Any]:
    """Structural summary of a response (keys and types, no values) for the probe."""
    if not isinstance(raw, Mapping):
        return {"body_type": type(raw).__name__}
    summary: dict[str, Any] = {"top_level_keys": sorted(raw)}
    try:
        answers, shape = locate_answers(raw)
    except JevResponseError:
        summary["answers_location"] = None
        return summary
    summary["answers_location"] = shape
    summary["answer_fields"] = {
        qid: sorted(a) if isinstance(a, Mapping) else type(a).__name__ for qid, a in answers.items()
    }
    summary["model_field_found"] = any(isinstance(c.get("model"), str) for c in _containers(raw))
    summary["timing_fields"] = sorted(find_timing(raw))
    usage = raw.get("usage")
    summary["usage_keys"] = sorted(usage) if isinstance(usage, Mapping) else None
    return summary
