"""Deterministic policy: thresholds loaded from policies/thresholds.yaml.

Code applies these to Jev's answers. Model outputs are signals; these rules decide.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

from sentinel.config import LLMTier


class _Policy(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class TriagePolicy(_Policy):
    category_min_confidence: float = Field(ge=0, le=1)
    abusive_escalate_at: float = Field(ge=0, le=1)


class RouteModelPolicy(_Policy):
    min_confidence: float = Field(ge=0, le=1)
    default_tier: LLMTier


class Thresholds(_Policy):
    triage: TriagePolicy
    route_model: RouteModelPolicy

    @classmethod
    def load(cls, path: Path) -> Thresholds:
        """Missing or invalid policy is an error: we never run on implicit defaults."""
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        return cls.model_validate(data)
