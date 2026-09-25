"""Guard: deterministic policy first, then Jev (PLAN.md section 8).

A call runs only if policy allows it and, for write tools, Jev's "safe to run"
probability clears the threshold. Jev can never override a policy "approval" or
"block": it is not even asked.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from sentinel.agent.models import ProposedCall
from sentinel.jev.models import JevResult
from sentinel.jev.questions import GUARD
from sentinel.policy.engine import GuardPolicy, PolicyVerdict
from sentinel.tools.registry import Tool

QUESTIONS = GUARD
GuardAction = Literal["run", "approval", "block"]


class GuardDecision(BaseModel):
    model_config = ConfigDict(frozen=True)

    call: ProposedCall
    action: GuardAction
    policy: PolicyVerdict
    safe: float | None = None
    """Jev's probability that the call is safe to run (only asked for write tools)."""
    reasons: list[str] = Field(default_factory=list)


def guard_state(state: dict[str, Any], call: ProposedCall, tool: Tool) -> dict[str, Any]:
    keys: tuple[str, ...] = ("channel", "subject", "message", "account")
    context: dict[str, Any] = {k: state[k] for k in keys if k in state}
    return {
        **context,
        "proposed_tool_call": {
            "tool": call.tool,
            "description": tool.description,
            "risk": tool.risk.value,
            "args": call.args,
        },
    }


def decide_guard(
    call: ProposedCall, verdict: PolicyVerdict, result: JevResult | None, policy: GuardPolicy
) -> GuardDecision:
    if verdict.action == "block":
        return GuardDecision(call=call, action="block", policy=verdict, reasons=verdict.reasons)
    if verdict.action == "approval":
        return GuardDecision(call=call, action="approval", policy=verdict, reasons=verdict.reasons)
    if verdict.action == "allow":
        return GuardDecision(call=call, action="run", policy=verdict)
    if result is None:  # policy wants a guard but none was obtained: never run blind
        return GuardDecision(
            call=call, action="approval", policy=verdict, reasons=[f"{call.tool}: no guard answer"]
        )
    safe = result.noul("safe_to_run").noul
    if safe < policy.min_safe:
        return GuardDecision(
            call=call,
            action="approval",
            policy=verdict,
            safe=safe,
            reasons=[f"guard.{call.tool} safe_to_run {safe:.3f} < {policy.min_safe}"],
        )
    return GuardDecision(call=call, action="run", policy=verdict, safe=safe)
