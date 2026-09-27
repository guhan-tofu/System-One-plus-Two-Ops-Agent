from __future__ import annotations

import copy
from typing import Any

import pytest

from sentinel.jev.errors import JevResponseError
from sentinel.jev.typesafe import describe_shape, parse_response
from tests.fixtures.jev import QUESTIONS
from tests.fixtures.typesafe import ANSWERS, MODEL_ALIAS, MODEL_VERSION, api_response


def parse(raw: Any) -> Any:
    return parse_response(raw, QUESTIONS, requested_model=MODEL_ALIAS, latency_ms=10.0)


def with_answer(qid: str, **changes: Any) -> dict[str, Any]:
    answers = copy.deepcopy(ANSWERS)
    answers[qid].update(changes)
    return api_response(answers)


def without(qid: str, key: str) -> dict[str, Any]:
    answers = copy.deepcopy(ANSWERS)
    del answers[qid][key]
    return api_response(answers)


def test_documented_shape_all_answer_types() -> None:
    result = parse(api_response())
    assert result.provider == "typesafe"
    assert result.model == MODEL_VERSION and result.model_verified is True
    assert result.metadata == {"requested_model": MODEL_ALIAS}
    assert result.usage == {"input_tokens": 378, "output_tokens": 65}

    category = result.choice("category")
    assert category.choice == "billing" and category.confidence == 0.8
    urgency = result.score("urgency")
    assert urgency.score == 1.4 and urgency.confidence == 0.35
    assert urgency.legend == {"0": "Not urgent", "1": "Soon", "2": "Today"}
    assert result.noul("is_abusive").noul == 0.03


def test_usage_is_optional() -> None:
    raw = api_response()
    del raw["usage"]
    assert parse(raw).usage is None


def test_extra_fields_ignored() -> None:
    raw = api_response()
    raw["request_id"] = "req_1"
    raw["answers"]["category"]["explanation"] = "vendor extra"
    assert parse(raw).choice("category").choice == "billing"


@pytest.mark.parametrize(
    ("raw", "match"),
    [
        ([], "expected an object"),
        (api_response(model=None), "no model ID"),
        (api_response(model=""), "no model ID"),
        ({"model": MODEL_VERSION}, "no answers"),
        (api_response({"category": ANSWERS["category"]}), "missing answers"),
        (with_answer("category", type="score"), "does not match question type"),
        (with_answer("is_abusive", type="boolean"), "does not match question type"),
        (with_answer("category", choice="sales"), "not one of the offered options"),
        (with_answer("category", probabilities={"sales": 1.0}), "unknown labels"),
        (with_answer("category", probabilities={"billing": 1.5}), "outside"),
        (with_answer("category", confidence=1.2), "confidence"),
        (without("category", "confidence"), "malformed"),
        (with_answer("urgency", score=3.0), "outside the level range"),
        (with_answer("urgency", probabilities={"7": 1.0}), "unknown labels"),
        (with_answer("urgency", legend={"0": "Low", "1": "Mid", "2": "High"}), "legend"),
        (without("urgency", "legend"), "malformed"),
        (without("urgency", "confidence"), "malformed"),
        (with_answer("is_abusive", noul=1.3), "outside"),
        (with_answer("is_abusive", noul=float("nan")), "outside"),
        (with_answer("is_abusive", noul="high"), "malformed"),
    ],
)
def test_unexpected_shapes_rejected(raw: Any, match: str) -> None:
    with pytest.raises(JevResponseError, match=match):
        parse(raw)


def test_describe_shape_has_no_values() -> None:
    shape = describe_shape(api_response())
    assert shape == {
        "top_level_keys": ["answers", "model", "usage"],
        "model": MODEL_VERSION,
        "answer_fields": {
            "category": ["choice", "confidence", "probabilities", "type"],
            "urgency": ["confidence", "legend", "probabilities", "score", "type"],
            "is_abusive": ["noul", "type"],
        },
        "usage_keys": ["input_tokens", "output_tokens"],
    }
    assert describe_shape("nope") == {"body_type": "str"}
