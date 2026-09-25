from __future__ import annotations

import copy
from typing import Any

import pytest

from sentinel.jev.errors import JevResponseError
from sentinel.jev.models import ChoiceAnswer, NoulAnswer, ScoreAnswer
from sentinel.jev.parsing import describe_shape, parse_response
from tests.fixtures.jev import (
    ANSWERS,
    QUESTIONS,
    REQUESTED_MODEL,
    confirmed_response,
    error_envelope,
)


def parse(raw: Any) -> Any:
    return parse_response(raw, QUESTIONS, requested_model=REQUESTED_MODEL, latency_ms=12.0)


def with_answer(qid: str, **changes: Any) -> dict[str, Any]:
    answers = copy.deepcopy(ANSWERS)
    answers[qid].update(changes)
    return confirmed_response(answers)


def test_confirmed_shape_all_answer_types() -> None:
    result = parse(confirmed_response())
    assert result.model == REQUESTED_MODEL
    assert result.model_verified is False
    assert result.vendor_elapsed_ms == 87.0
    assert result.credits_used == 1.0
    assert result.usage == {"input_tokens": 42, "output_tokens": 7}

    category = result.choice("category")
    assert isinstance(category, ChoiceAnswer)
    assert category.choice == "billing" and category.confidence == 0.9

    urgency = result.score("urgency")
    assert isinstance(urgency, ScoreAnswer)
    assert urgency.score == 1.4
    assert urgency.legend["1"] == "Soon"
    assert urgency.probabilities["2"] == 0.5

    abusive = result.noul("is_abusive")
    assert isinstance(abusive, NoulAnswer) and abusive.noul == 0.03


def test_integer_confidence_accepted() -> None:
    # Observed live: fully confident answers come back as integers 0/1.
    raw = with_answer("category", confidence=1, probabilities={"billing": 1, "technical": 0})
    assert parse(raw).choice("category").confidence == 1.0


@pytest.mark.parametrize("where", ["result", "data"])
def test_vendor_model_id_used_when_present(where: str) -> None:
    raw = confirmed_response()
    target = raw["data"]["result"] if where == "result" else raw["data"]
    target["model"] = "jev-1.13.0"
    result = parse(raw)
    assert result.model == "jev-1.13.0"
    assert result.model_verified is True


def test_typed_accessor_rejects_wrong_kind() -> None:
    result = parse(confirmed_response())
    with pytest.raises(TypeError):
        result.noul("category")


@pytest.mark.parametrize(
    "raw",
    [
        error_envelope(),
        {**confirmed_response(), "code": 1},
        {**confirmed_response(), "code": False},
        {k: v for k, v in confirmed_response().items() if k != "code"},
    ],
)
def test_non_ok_code_rejected(raw: dict[str, Any]) -> None:
    with pytest.raises(JevResponseError, match="code"):
        parse(raw)


def test_error_message_surfaced() -> None:
    with pytest.raises(JevResponseError, match="rejected the request"):
        parse(error_envelope())


@pytest.mark.parametrize(
    "raw",
    [
        [],
        "ok",
        {"code": 0, "message": "ok"},
        {"code": 0, "data": {"creditsUsed": 1}},
        {"code": 0, "data": {"result": {"elapsedMs": 1}}},
        # Previously-guessed shapes are no longer accepted.
        {"model": "m", "answers": ANSWERS},
        {"model": "m", "result": {"answers": ANSWERS}},
    ],
)
def test_unrecognised_body_rejected(raw: Any) -> None:
    with pytest.raises(JevResponseError):
        parse(raw)


def test_missing_answer_rejected() -> None:
    answers = {k: v for k, v in ANSWERS.items() if k != "urgency"}
    with pytest.raises(JevResponseError, match="urgency"):
        parse(confirmed_response(answers))


@pytest.mark.parametrize(
    ("qid", "reported"), [("category", "score"), ("is_abusive", "choice"), ("urgency", None)]
)
def test_answer_type_mismatch_rejected(qid: str, reported: str | None) -> None:
    raw = with_answer(qid, type=reported)
    with pytest.raises(JevResponseError, match="does not match question type"):
        parse(raw)


def test_bare_float_noul_rejected() -> None:
    answers = {**copy.deepcopy(ANSWERS), "is_abusive": 0.7}
    with pytest.raises(JevResponseError):
        parse(confirmed_response(answers))


def test_choice_outside_options_rejected() -> None:
    with pytest.raises(JevResponseError, match="not one of"):
        parse(with_answer("category", choice="sales"))


def test_choice_probability_for_unknown_option_rejected() -> None:
    raw = with_answer("category", probabilities={"billing": 0.9, "sales": 0.1})
    with pytest.raises(JevResponseError, match="unknown options"):
        parse(raw)


def test_missing_confidence_rejected() -> None:
    raw = confirmed_response()
    del raw["data"]["result"]["answers"]["category"]["confidence"]
    with pytest.raises(JevResponseError, match="confidence"):
        parse(raw)


def test_out_of_range_probability_rejected() -> None:
    with pytest.raises(JevResponseError, match="noul"):
        parse(with_answer("is_abusive", noul=1.5))


def test_score_legend_mismatch_rejected() -> None:
    raw = with_answer("urgency", legend={"0": "Not urgent", "1": "Later", "2": "Today"})
    with pytest.raises(JevResponseError, match="legend"):
        parse(raw)


def test_score_probability_for_unknown_level_rejected() -> None:
    raw = with_answer("urgency", probabilities={"0": 0.1, "1": 0.4, "3": 0.5})
    with pytest.raises(JevResponseError, match="unknown levels"):
        parse(raw)


@pytest.mark.parametrize("value", [-0.1, 2.5])
def test_score_out_of_range_rejected(value: float) -> None:
    with pytest.raises(JevResponseError, match="outside the level range"):
        parse(with_answer("urgency", score=value))


def test_score_list_probabilities_rejected() -> None:
    with pytest.raises(JevResponseError, match="probabilities"):
        parse(with_answer("urgency", probabilities=[0.1, 0.4, 0.5]))


def test_describe_shape_has_no_values() -> None:
    summary = describe_shape(confirmed_response())
    assert summary["top_level_keys"] == ["code", "data", "message"]
    assert summary["code"] == 0
    assert summary["data_keys"] == ["creditsUsed", "result"]
    assert summary["result_keys"] == ["answers", "elapsedMs", "usage"]
    assert summary["answer_fields"]["is_abusive"] == ["noul", "type"]
    assert summary["model_field_found"] is False
    assert "billing" not in str(summary["answer_fields"])
