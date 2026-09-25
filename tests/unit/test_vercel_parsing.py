from __future__ import annotations

import copy
from typing import Any

import pytest

from sentinel.jev.errors import JevResponseError
from sentinel.jev.vercel import describe_shape, parse_response
from tests.fixtures.jev import QUESTIONS
from tests.fixtures.vercel import ANSWERS, GATEWAY_MODEL, gateway_response


def parse(raw: Any) -> Any:
    return parse_response(raw, QUESTIONS, requested_model=GATEWAY_MODEL, latency_ms=10.0)


def with_answer(qid: str, **changes: Any) -> dict[str, Any]:
    answers = copy.deepcopy(ANSWERS)
    answers[qid].update(changes)
    return gateway_response(answers)


def test_confirmed_shape_all_answer_types() -> None:
    result = parse(gateway_response())
    assert result.provider == "vercel"
    assert result.model == GATEWAY_MODEL and result.model_verified is False
    assert result.metadata == {
        "generation_id": "gen_fake_0001",
        "final_provider": "typesafe-ai",
        "resolved_model": GATEWAY_MODEL,
    }
    assert result.usage == {"input_tokens": 378, "output_tokens": 64, "cost_usd": 0.0}

    category = result.choice("category")
    assert category.choice == "billing" and category.confidence == 0.8  # vendor's value

    urgency = result.score("urgency")
    assert urgency.score == 1.4 and urgency.confidence == 0.35
    assert urgency.legend == {"0": "Not urgent", "1": "Soon", "2": "Today"}  # from question

    assert result.noul("is_abusive").noul == 0.03  # boolean -> noul


def test_confidence_computed_when_vendor_omits_it() -> None:
    result = parse(gateway_response(confidence={}))
    assert result.choice("category").confidence == pytest.approx(0.8)  # (2*0.9-1)/1
    # mode "2": E|x-2| = 0.1*2 + 0.4*1 = 0.6; D_3 = 2/3
    assert result.score("urgency").confidence == pytest.approx(1 - 0.6 / (2 / 3))


def test_live_probe_confidence_matches_formula() -> None:
    # Values from the live probe (2026-09-25): vendor confidence 0.25 for this score.
    questions = {"s": QUESTIONS["urgency"]}
    raw = gateway_response(
        {"s": {"type": "score", "score": 1.5, "probabilities": {"0": 0.02, "1": 0.46, "2": 0.52}}},
        confidence={},
    )
    result = parse_response(raw, questions, requested_model=GATEWAY_MODEL, latency_ms=1.0)
    assert result.score("s").confidence == pytest.approx(0.25, abs=0.01)


@pytest.mark.parametrize(
    ("qid", "wire_type"), [("is_abusive", "noul"), ("category", "score"), ("urgency", None)]
)
def test_type_mismatch_rejected(qid: str, wire_type: str | None) -> None:
    with pytest.raises(JevResponseError, match="does not match question type"):
        parse(with_answer(qid, type=wire_type))


@pytest.mark.parametrize(
    ("raw", "match"),
    [
        ([], "expected an object"),
        ({"result": {}}, "no answers"),
        (gateway_response({"category": ANSWERS["category"]}), "missing answers"),
    ],
)
def test_bad_bodies_rejected(raw: Any, match: str) -> None:
    with pytest.raises(JevResponseError, match=match):
        parse(raw)


@pytest.mark.parametrize(
    ("qid", "changes", "match"),
    [
        ("category", {"choice": "sales"}, "not one of"),
        ("category", {"probabilities": {"billing": 0.9, "sales": 0.1}}, "unknown labels"),
        ("category", {"probabilities": {"billing": 1.2, "technical": 0.1}}, "outside"),
        ("urgency", {"probabilities": {"0": 0.5, "3": 0.5}}, "unknown labels"),
        ("urgency", {"score": 2.5}, "level range"),
        ("is_abusive", {"probability": -0.1}, "outside"),
    ],
)
def test_invalid_answers_rejected(qid: str, changes: dict[str, Any], match: str) -> None:
    with pytest.raises(JevResponseError, match=match):
        parse(with_answer(qid, **changes))


@pytest.mark.parametrize(
    ("qid", "drop"),
    [("category", "probabilities"), ("urgency", "score"), ("is_abusive", "probability")],
)
def test_missing_fields_rejected(qid: str, drop: str) -> None:
    raw = gateway_response()
    del raw["answers"][qid][drop]
    with pytest.raises(JevResponseError, match=drop):
        parse(raw)


def test_non_numeric_confidence_rejected() -> None:
    with pytest.raises(JevResponseError, match="confidence"):
        parse(gateway_response(confidence={"category": "high"}))


@pytest.mark.parametrize(("cost", "expected"), [("0.000015876", 0.000015876), ("n/a", None)])
def test_cost(cost: str, expected: float | None) -> None:
    assert parse(gateway_response(cost=cost)).usage["cost_usd"] == expected


def test_describe_shape_has_no_values() -> None:
    summary = describe_shape(gateway_response())
    assert summary["answer_fields"]["is_abusive"] == ["probability", "type"]
    assert summary["confidence_ids"] == ["category", "urgency"]
    assert "billing" not in str(summary) and "gen_fake" not in str(summary)
