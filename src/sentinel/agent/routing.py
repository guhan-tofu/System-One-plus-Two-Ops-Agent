"""Model routing: Jev picks an allowlisted tier; low confidence falls back to policy."""

from __future__ import annotations

from typing import cast

from pydantic import BaseModel, ConfigDict

from sentinel.config import LLMTier
from sentinel.jev.models import JevResult
from sentinel.jev.questions import ROUTE_MODEL
from sentinel.policy.engine import RouteModelPolicy

QUESTIONS = ROUTE_MODEL


class RouteDecision(BaseModel):
    model_config = ConfigDict(frozen=True)

    tier: LLMTier
    suggested: str
    confidence: float
    overridden: bool
    """True when policy replaced Jev's suggestion (low confidence)."""


def decide_tier(result: JevResult, policy: RouteModelPolicy) -> RouteDecision:
    answer = result.choice("tier")
    confident = answer.confidence >= policy.min_confidence
    return RouteDecision(
        tier=cast(LLMTier, answer.choice) if confident else policy.default_tier,
        suggested=answer.choice,
        confidence=answer.confidence,
        overridden=not confident,
    )
