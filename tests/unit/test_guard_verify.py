"""Guard and verify rules: each has an allow and an escalate branch."""

from __future__ import annotations

import pytest

from sentinel.agent.guard import decide_guard
from sentinel.agent.models import ProposedCall
from sentinel.agent.verify import decide_verify
from sentinel.policy.engine import GuardPolicy, PolicyVerdict, VerifyPolicy
from sentinel.tools.registry import ToolResult
from tests.fixtures.pipeline import guard_answers, jev_result, verify_answers

GUARD = GuardPolicy(min_safe=0.9)
VERIFY = VerifyPolicy(claim_supported_min=0.85)
CALL = ProposedCall(tool="issue_refund", args={"amount": 49.0})


def verdict(action: str) -> PolicyVerdict:
    return PolicyVerdict(action=action, reasons=[f"policy says {action}"])  # type: ignore[arg-type]


@pytest.mark.parametrize(("safe", "action"), [(0.9, "run"), (0.899, "approval")])
def test_guard_threshold(safe: float, action: str) -> None:
    d = decide_guard(CALL, verdict("guard"), jev_result(guard_answers(safe)), GUARD)
    assert d.action == action and d.safe == safe


@pytest.mark.parametrize("policy_action", ["approval", "block"])
def test_jev_never_overrides_policy(policy_action: str) -> None:
    d = decide_guard(CALL, verdict(policy_action), jev_result(guard_answers(1.0)), GUARD)
    assert d.action == policy_action
    assert d.safe is None  # Jev's answer is not even considered


def test_read_only_runs_without_jev() -> None:
    assert decide_guard(CALL, verdict("allow"), None, GUARD).action == "run"


def test_missing_guard_answer_needs_approval() -> None:
    assert decide_guard(CALL, verdict("guard"), None, GUARD).action == "approval"


def ok(tool: str = "issue_refund") -> ToolResult:
    return ToolResult(tool=tool, args={}, ok=True, data={"status": "succeeded"}, latency_ms=1)


def failed(tool: str = "issue_refund") -> ToolResult:
    return ToolResult(tool=tool, args={}, ok=False, error="charge not found", latency_ms=1)


@pytest.mark.parametrize(("supported", "escalate"), [(0.85, False), (0.849, True)])
def test_verify_claim_threshold(supported: float, escalate: bool) -> None:
    d = decide_verify([ok()], jev_result(verify_answers(supported)), VERIFY)
    assert d.escalate is escalate


@pytest.mark.parametrize(
    ("results", "status", "escalate"),
    [
        ([ok()], "complete", False),
        ([ok()], "verify_more", True),
        ([ok()], "failed", True),
        ([], "verify_more", False),  # no actions taken: task_status is not used
    ],
)
def test_verify_task_status(results: list[ToolResult], status: str, escalate: bool) -> None:
    d = decide_verify(results, jev_result(verify_answers(0.95, status)), VERIFY)
    assert d.escalate is escalate


def test_failed_tool_escalates_even_if_jev_is_satisfied() -> None:
    d = decide_verify([failed()], jev_result(verify_answers(1.0, "complete")), VERIFY)
    assert d.escalate
    assert d.escalate_reasons == ["execute: issue_refund failed (charge not found)"]


def test_no_tools_uses_reply_question_without_task_status() -> None:
    from sentinel.agent.verify import questions_for
    from sentinel.jev.models import NoulAnswer
    from sentinel.jev.questions import VERIFY as VERIFY_QUESTIONS
    from sentinel.jev.questions import VERIFY_REPLY

    assert questions_for([]) is VERIFY_REPLY and questions_for([ok()]) is VERIFY_QUESTIONS
    assert set(VERIFY_REPLY) == {"claim_supported"}
    reply_only = jev_result({"claim_supported": NoulAnswer(type="noul", noul=0.9)})
    d = decide_verify([], reply_only, VERIFY)
    assert not d.escalate and d.task_status is None
    low = jev_result({"claim_supported": NoulAnswer(type="noul", noul=0.5)})
    assert decide_verify([], low, VERIFY).escalate
