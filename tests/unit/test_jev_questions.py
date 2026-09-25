from __future__ import annotations

import pytest
from pydantic import ValidationError

from sentinel.jev.models import JevRequest
from sentinel.jev.questions import GUARD, ROUTE_MODEL, TRIAGE, VERIFY, choice, noul, score


def test_catalog_builds() -> None:
    for catalog in (TRIAGE, ROUTE_MODEL, GUARD, VERIFY):
        JevRequest(model="m", state="s", questions=catalog)


def test_choice_option_bounds() -> None:
    with pytest.raises(ValidationError):
        choice("q", {"only": "one"})
    too_many = {f"o{i}": "d" for i in range(256)}
    with pytest.raises(ValidationError):
        choice("q", too_many)
    choice("q", {f"o{i}": "d" for i in range(255)})


@pytest.mark.parametrize("n", [1, 11])
def test_score_level_bounds(n: int) -> None:
    with pytest.raises(ValidationError):
        score("q", [f"l{i}" for i in range(n)])


def test_score_levels_unique() -> None:
    with pytest.raises(ValidationError):
        score("q", ["a", "a"])


def test_noul_criteria_keys() -> None:
    with pytest.raises(ValueError, match="true"):
        noul("q", {"yes": "a", "no": "b"})
    q = noul("q", {"true": "a", "false": "b"})
    assert q.criteria is not None and q.criteria.true == "a"


def test_empty_instructions_rejected() -> None:
    with pytest.raises(ValidationError):
        noul("")


def test_payload_wire_format() -> None:
    req = JevRequest(
        model="jev-latest",
        state={"subject": "hi"},
        questions={"a": noul("q?"), "b": noul("q?", {"true": "y", "false": "n"})},
    )
    assert req.model_dump(mode="json", exclude_none=True) == {
        "model": "jev-latest",
        "state": {"subject": "hi"},
        "questions": {
            "a": {"type": "noul", "instructions": "q?"},
            "b": {"type": "noul", "instructions": "q?", "criteria": {"true": "y", "false": "n"}},
        },
    }


def test_request_needs_a_question() -> None:
    with pytest.raises(ValidationError):
        JevRequest(model="m", state="s", questions={})
