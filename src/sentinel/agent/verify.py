"""Verify: does the draft's claim match what the tools actually did?

Deterministic first: any failed tool escalates, whatever Jev says. Then Jev
(PLAN.md section 8): claim_supported >= threshold, and, when tools ran,
task_status == complete.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict

from sentinel.jev.models import JevResult
from sentinel.jev.questions import VERIFY
from sentinel.policy.engine import VerifyPolicy
from sentinel.tools.registry import ToolResult

QUESTIONS = VERIFY


class VerifyDecision(BaseModel):
    model_config = ConfigDict(frozen=True)

    claim_supported: float | None
    task_status: str | None
    escalate_reasons: list[str]

    @property
    def escalate(self) -> bool:
        return bool(self.escalate_reasons)


def failed_tools(results: list[ToolResult]) -> list[str]:
    return [f"execute: {r.tool} failed ({r.error})" for r in results if not r.ok]


def verify_state(state: dict[str, Any], draft: str, results: list[ToolResult]) -> dict[str, Any]:
    return {
        "customer_message": state.get("message"),
        "account": state.get("account"),
        "draft": draft,
        "tool_results": [
            {"tool": r.tool, "args": r.args, "ok": r.ok, "data": r.data, "error": r.error}
            for r in results
        ],
    }


def decide_verify(
    results: list[ToolResult], result: JevResult, policy: VerifyPolicy
) -> VerifyDecision:
    reasons = failed_tools(results)
    supported = result.noul("claim_supported").noul
    status = result.choice("task_status").choice
    if supported < policy.claim_supported_min:
        reasons.append(f"verify.claim_supported {supported:.3f} < {policy.claim_supported_min}")
    if results and status != "complete":
        reasons.append(f"verify.task_status is {status}")
    return VerifyDecision(claim_supported=supported, task_status=status, escalate_reasons=reasons)
