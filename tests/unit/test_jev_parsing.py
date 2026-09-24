from __future__ import annotations

import copy
from typing import Any

import pytest

from sentinel.jev.errors import JevResponseError
from sentinel.jev.models import ChoiceAnswer, NoulAnswer, ScoreAnswer
from sentinel.jev.parsing import describe_shape, parse_response
from tests.fixtures.jev import ANSWERS, FAKE_MODEL_ID, QUESTIONS, documented_response


def parse(raw: Any) -> Any:
    return parse_response(raw, QUESTIONS, latency_ms=12.0)


def test_documented_shape_all_answer_types() -> None:
    result = parse(documented_response())
    assert result.model == FAKE_MODEL_ID
    assert result.shape == "answers"
    assert result.vendor_timing == {"elapsedMs": 87.0}
    assert result.usage == {"input_tokens": 42}

    category = result.choice("category")
    assert isinstance(category, ChoiceAnswer)
    assert category.choice == "billing" and category.confidence == 0.9

    urgency = result.score("urgency")
    assert isinstance(urgency, ScoreAnswer)
    assert urgency.score == 1.4 and urgency.legend == "Soon"

    abusive = result.noul("is_abusive")
    assert isinstance(abusive, NoulAnswer) and abusive.noul == 0.03


def test_typed_accessor_rejects_wrong_kind() -> None:
    result = parse(documented_response())
    with pytest.raises(TypeError):
        result.noul("category")


def test_nested_under_result_answers_with_elapsed() -> None:
    raw = {"result": {"model": FAKE_MODEL_ID, "answers": ANSWERS, "elapsed": 0.087}}
    result = parse(raw)
    assert result.shape == "result.answers"
    assert result.model == FAKE_MODEL_ID
    assert result.vendor_timing == {"elapsed": 0.087}


def test_answers_directly_under_result() -> None:
    raw = {"model": FAKE_MODEL_ID, "result": ANSWERS}
    result = parse(raw)
    assert result.shape == "result"
    assert result.choice("category").choice == "billing"


def test_bare_float_noul_accepted() -> None:
    answers = {**ANSWERS, "is_abusive": 0.7}
    assert parse(documented_response(answers=answers)).noul("is_abusive").noul == 0.7


def test_missing_model_rejected() -> None:
    raw = documented_response()
    del raw["model"]
    with pytest.raises(JevResponseError, match="model"):
        parse(raw)


def test_missing_answer_rejected() -> None:
    answers = {k: v for k, v in ANSWERS.items() if k != "urgency"}
    with pytest.raises(JevResponseError, match="urgency"):
        parse(documented_response(answers=answers))


def test_choice_outside_options_rejected() -> None:
    answers = copy.deepcopy(ANSWERS)
    answers["category"]["choice"] = "sales"
    with pytest.raises(JevResponseError, match="not one of"):
        parse(documented_response(answers=answers))


def test_missing_confidence_rejected() -> None:
    answers = copy.deepcopy(ANSWERS)
    del answers["category"]["confidence"]
    with pytest.raises(JevResponseError, match="confidence"):
        parse(documented_response(answers=answers))


def test_out_of_range_probability_rejected() -> None:
    answers = {**ANSWERS, "is_abusive": {"noul": 1.5}}
    with pytest.raises(JevResponseError, match="noul"):
        parse(documented_response(answers=answers))


@pytest.mark.parametrize("raw", [[], "ok", {"model": FAKE_MODEL_ID}])
def test_unrecognised_body_rejected(raw: Any) -> None:
    with pytest.raises(JevResponseError):
        parse(raw)


def test_describe_shape_has_no_values() -> None:
    summary = describe_shape(documented_response())
    assert summary["answers_location"] == "answers"
    assert summary["answer_fields"]["is_abusive"] == ["noul"]
    assert summary["timing_fields"] == ["elapsedMs"]
    assert FAKE_MODEL_ID not in str(summary)
