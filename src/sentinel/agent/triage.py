"""Triage: one Jev request (TRIAGE questions), then deterministic thresholds."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from sentinel.jev.models import JevResult
from sentinel.jev.questions import TRIAGE
from sentinel.policy.engine import TriagePolicy

QUESTIONS = TRIAGE
Path = Literal["lookup", "generate", "human"]


class TriageDecision(BaseModel):
    model_config = ConfigDict(frozen=True)

    category: str
    category_confidence: float
    urgency: float
    """Expected level index, 0 (not urgent) .. 3 (immediately / outage)."""
    urgency_label: str
    path: Path
    path_confidence: float
    abusive: float
    """Probability the message is abusive or threatening."""
    escalate_reasons: list[str]

    @property
    def escalate(self) -> bool:
        return bool(self.escalate_reasons)


def decide_triage(result: JevResult, policy: TriagePolicy) -> TriageDecision:
    category = result.choice("category")
    urgency = result.score("urgency")
    path = result.choice("path")
    abusive = result.noul("is_abusive").noul

    reasons: list[str] = []
    if category.confidence < policy.category_min_confidence:
        reasons.append(
            f"triage.category confidence {category.confidence:.3f} "
            f"< {policy.category_min_confidence}"
        )
    if path.confidence < policy.path_min_confidence:
        reasons.append(
            f"triage.path confidence {path.confidence:.3f} < {policy.path_min_confidence}"
        )
    if path.choice == "human":
        reasons.append("triage.path is human")
    if abusive >= policy.abusive_escalate_at:
        reasons.append(f"triage.is_abusive {abusive:.3f} >= {policy.abusive_escalate_at}")

    return TriageDecision(
        category=category.choice,
        category_confidence=category.confidence,
        urgency=urgency.score,
        urgency_label=urgency.legend[str(round(urgency.score))],
        path=path.choice,  # pydantic checks it is one of the Path literals
        path_confidence=path.confidence,
        abusive=abusive,
        escalate_reasons=reasons,
    )
