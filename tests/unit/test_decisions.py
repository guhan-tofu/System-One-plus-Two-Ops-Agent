"""Triage and routing rules: each rule has an allow and an escalate/override branch."""

from __future__ import annotations

from pathlib import Path

import pytest

from sentinel.agent.routing import decide_tier
from sentinel.agent.triage import decide_triage
from sentinel.policy.engine import RouteModelPolicy, Thresholds, TriagePolicy
from tests.fixtures.pipeline import jev_result, route_answers, triage_answers

TRIAGE_POLICY = TriagePolicy(category_min_confidence=0.75, abusive_escalate_at=0.6)
ROUTE_POLICY = RouteModelPolicy(min_confidence=0.6, default_tier="strong")
REPO_THRESHOLDS = Path(__file__).parents[2] / "policies" / "thresholds.yaml"


def test_triage_allows_confident_routine_item() -> None:
    d = decide_triage(jev_result(triage_answers()), TRIAGE_POLICY)
    assert not d.escalate
    assert d.category == "billing" and d.path == "generate"
    assert d.urgency_label == "Today"


@pytest.mark.parametrize(("confidence", "escalate"), [(0.75, False), (0.749, True)])
def test_category_confidence_threshold(confidence: float, escalate: bool) -> None:
    d = decide_triage(jev_result(triage_answers(category_confidence=confidence)), TRIAGE_POLICY)
    assert d.escalate is escalate


@pytest.mark.parametrize(("path", "escalate"), [("lookup", False), ("human", True)])
def test_human_path_always_escalates(path: str, escalate: bool) -> None:
    d = decide_triage(jev_result(triage_answers(path=path, path_confidence=0.99)), TRIAGE_POLICY)
    assert d.escalate is escalate


@pytest.mark.parametrize(("abusive", "escalate"), [(0.59, False), (0.6, True)])
def test_abusive_threshold(abusive: float, escalate: bool) -> None:
    d = decide_triage(jev_result(triage_answers(abusive=abusive)), TRIAGE_POLICY)
    assert d.escalate is escalate


def test_all_reasons_reported() -> None:
    d = decide_triage(
        jev_result(triage_answers(category_confidence=0.5, path="human", abusive=0.9)),
        TRIAGE_POLICY,
    )
    assert len(d.escalate_reasons) == 3


@pytest.mark.parametrize(
    ("suggested", "confidence", "tier", "overridden"),
    [
        ("fast", 0.9, "fast", False),
        ("fast", 0.6, "fast", False),
        ("fast", 0.59, "strong", True),
        ("strong", 0.3, "strong", True),
    ],
)
def test_route_model(suggested: str, confidence: float, tier: str, overridden: bool) -> None:
    d = decide_tier(jev_result(route_answers(suggested, confidence)), ROUTE_POLICY)
    assert d.tier == tier
    assert d.overridden is overridden
    assert d.suggested == suggested


def test_repo_thresholds_match_plan() -> None:
    t = Thresholds.load(REPO_THRESHOLDS)
    assert t.triage.category_min_confidence == 0.75
    assert t.triage.abusive_escalate_at == 0.6
    assert t.route_model.min_confidence == 0.6
    assert t.route_model.default_tier == "strong"


def test_thresholds_missing_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        Thresholds.load(tmp_path / "missing.yaml")
