"""Deterministic policy loaded from policies/*.yaml.

Code applies these to Jev's answers and to proposed tool calls. Model outputs are
signals; these rules decide. Missing or invalid policy is an error: we never run
on implicit defaults.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from sentinel.config import LLMTier
from sentinel.tools.registry import Risk, ToolRegistry


class _Policy(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


def _load_yaml(path: Path) -> Any:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


# --- thresholds (policies/thresholds.yaml) --------------------------------------


class TriagePolicy(_Policy):
    category_min_confidence: float = Field(ge=0, le=1)
    path_min_confidence: float = Field(ge=0, le=1)
    abusive_escalate_at: float = Field(ge=0, le=1)


class RouteModelPolicy(_Policy):
    min_confidence: float = Field(ge=0, le=1)
    default_tier: LLMTier


class GuardPolicy(_Policy):
    min_safe: float = Field(ge=0, le=1)


class VerifyPolicy(_Policy):
    claim_supported_min: float = Field(ge=0, le=1)


class Thresholds(_Policy):
    triage: TriagePolicy
    route_model: RouteModelPolicy
    guard: GuardPolicy
    verify: VerifyPolicy

    @classmethod
    def load(cls, path: Path) -> Thresholds:
        return cls.model_validate(_load_yaml(path))


# --- tool rules (policies/tools.yaml) --------------------------------------------

PolicyAction = Literal["allow", "guard", "approval", "block"]
"""allow: run without a guard (read-only). guard: policy permits, Jev must confirm.
approval: a human must approve. block: never run (unknown tool, bad call)."""


class ToolRule(_Policy):
    risk: Risk
    max_auto_amount: float | None = Field(default=None, ge=0)
    auto_currency: str | None = None


class PolicyVerdict(_Policy):
    action: PolicyAction
    reasons: list[str] = Field(default_factory=list)


class ToolPolicy(_Policy):
    tools: dict[str, ToolRule]

    @classmethod
    def load(cls, path: Path) -> ToolPolicy:
        return cls.model_validate(_load_yaml(path))

    def check_registry(self, registry: ToolRegistry) -> None:
        """Every registered tool needs a rule, and the declared risks must agree."""
        for name in registry.names():
            rule = self.tools.get(name)
            if rule is None:
                raise ValueError(f"tool {name!r} has no rule in policies/tools.yaml")
            if rule.risk is not registry.get(name).risk:
                raise ValueError(
                    f"tool {name!r}: registry risk {registry.get(name).risk} "
                    f"!= policy risk {rule.risk}"
                )

    def check(self, tool: str, args: Mapping[str, Any], *, has_customer: bool) -> PolicyVerdict:
        rule = self.tools.get(tool)
        if rule is None:
            return PolicyVerdict(action="block", reasons=[f"{tool}: no policy rule (unknown tool)"])
        if not has_customer:
            return PolicyVerdict(action="block", reasons=[f"{tool}: no customer reference"])
        if rule.risk is Risk.READ_ONLY:
            return PolicyVerdict(action="allow")
        if rule.risk is Risk.DESTRUCTIVE:
            return PolicyVerdict(
                action="approval", reasons=[f"{tool}: destructive tools always need approval"]
            )

        reasons: list[str] = []
        if rule.max_auto_amount is not None:
            amount = args.get("amount")
            if not isinstance(amount, int | float) or isinstance(amount, bool) or amount <= 0:
                return PolicyVerdict(action="block", reasons=[f"{tool}: invalid amount"])
            if amount > rule.max_auto_amount:
                reasons.append(f"{tool}: amount {amount} > auto limit {rule.max_auto_amount}")
        if rule.auto_currency is not None and args.get("currency") != rule.auto_currency:
            reasons.append(f"{tool}: currency {args.get('currency')!r} is not {rule.auto_currency}")
        if reasons:
            return PolicyVerdict(action="approval", reasons=reasons)
        return PolicyVerdict(action="guard")
