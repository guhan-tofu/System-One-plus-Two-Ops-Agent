"""Phase 4 acceptance, through the whole pipeline:

(a) a destructive tool never runs without approval, whatever Jev says;
(b) a draft claiming success after a failed tool result is escalated.
Plus the other guard / execute / verify branches.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any

import pytest
import respx
from pydantic import SecretStr

from sentinel.agent.models import Outcome
from sentinel.agent.pipeline import Pipeline
from sentinel.jev.errors import JevOverloadedError
from sentinel.llm.openai_client import OpenAIClient
from sentinel.policy.engine import Thresholds, ToolPolicy
from sentinel.storage.audit import AuditLog
from sentinel.storage.db import make_engine
from sentinel.storage.review import ReviewQueue
from sentinel.tools.builtin import default_registry
from sentinel.tools.builtin.actions import (
    CLOSE_ACCOUNT,
    CLOSED_ACCOUNTS,
    ISSUE_REFUND,
    REFUNDS,
    reset_mock_state,
)
from sentinel.tools.builtin.lookups import LOOKUP_CUSTOMER
from sentinel.tools.registry import ToolError, ToolRegistry
from tests.fixtures.openai import RESPONSES_URL, response_body
from tests.fixtures.pipeline import (
    ScriptedJev,
    draft_json,
    guard_answers,
    refund_call,
    triage_answers,
    verify_answers,
    work_item,
)

POLICIES = Path(__file__).parents[2] / "policies"
FAKE_KEY = "fake-openai-key-for-tests"  # pragma: allowlist secret
CLAIMED = "Good news: I've refunded the duplicate charge of 49.00 GBP."


@pytest.fixture(autouse=True)
def _clean_mock_store() -> Iterator[None]:
    reset_mock_state()
    yield
    reset_mock_state()


@pytest.fixture
def db(tmp_path: Path) -> str:
    return f"sqlite:///{tmp_path / 'sentinel.db'}"


@pytest.fixture
async def llm() -> AsyncIterator[OpenAIClient]:
    client = OpenAIClient(
        api_key=SecretStr(FAKE_KEY), models={"fast": "fake-fast", "strong": "fake-strong"}
    )
    yield client
    await client.aclose()


async def run(
    llm: OpenAIClient,
    db: str,
    jev: ScriptedJev,
    *calls: dict[str, Any],
    tools: ToolRegistry | None = None,
) -> Outcome:
    respx.post(RESPONSES_URL).respond(200, json=response_body(draft_json(CLAIMED, *calls)))
    engine = make_engine(db)
    pipeline = Pipeline(
        jev=jev,
        llm=llm,
        audit=AuditLog(engine),
        queue=ReviewQueue(engine),
        thresholds=Thresholds.load(POLICIES / "thresholds.yaml"),
        tools=tools or default_registry(),
        tool_policy=ToolPolicy.load(POLICIES / "tools.yaml"),
    )
    return await pipeline.run(work_item())


def records(outcome: Outcome, stage: str) -> list[Any]:
    return [r for r in outcome.trace if r.stage == stage]


def lookup_jev(**script: Any) -> ScriptedJev:
    return ScriptedJev(triage=triage_answers(path="lookup"), **script)


# --- (a) destructive tools never run without approval ------------------------------


@respx.mock
async def test_destructive_tool_never_runs_even_if_jev_says_safe(
    llm: OpenAIClient, db: str
) -> None:
    jev = lookup_jev(guard=guard_answers(1.0))  # Jev would wave it through
    outcome = await run(llm, db, jev, {"tool": "close_account", "reason": "customer asked"})

    assert outcome.status == "awaiting_approval"
    assert not CLOSED_ACCOUNTS  # never executed
    assert records(outcome, "execute") == []
    assert jev.states("guard") == []  # Jev is not even asked
    assert outcome.reasons == ["close_account: destructive tools always need approval"]

    pending = ReviewQueue(make_engine(db)).pending("approval")
    assert [r.id for r in pending] == [outcome.review_id]
    assert pending[0].payload["tool_calls"][0]["tool"] == "close_account"


@respx.mock
async def test_destructive_alongside_safe_call_runs_nothing(llm: OpenAIClient, db: str) -> None:
    outcome = await run(
        llm, db, lookup_jev(), refund_call(), {"tool": "close_account", "reason": "asked"}
    )
    assert outcome.status == "awaiting_approval"
    assert not REFUNDS and not CLOSED_ACCOUNTS


@respx.mock
async def test_refund_over_limit_needs_approval_regardless_of_jev(
    llm: OpenAIClient, db: str
) -> None:
    jev = lookup_jev(guard=guard_answers(1.0))
    outcome = await run(llm, db, jev, refund_call(amount=150.0))
    assert outcome.status == "awaiting_approval"
    assert "auto limit" in outcome.reasons[0]
    assert REFUNDS == {} and jev.states("guard") == []


# --- (b) a failed tool behind a success claim is escalated ------------------------


@respx.mock
async def test_claimed_success_after_failed_tool_is_escalated(llm: OpenAIClient, db: str) -> None:
    async def failing_refund(customer_id: str, **kwargs: Any) -> dict[str, Any]:
        raise ToolError("payment processor declined the refund")

    tools = ToolRegistry(
        [LOOKUP_CUSTOMER, CLOSE_ACCOUNT, dataclasses.replace(ISSUE_REFUND, fn=failing_refund)]
    )
    jev = lookup_jev(verify=verify_answers(1.0, "complete"))  # Jev would call it supported
    outcome = await run(llm, db, jev, refund_call(), tools=tools)

    assert outcome.status == "escalated"
    assert outcome.reasons == [
        "execute: issue_refund failed (payment processor declined the refund)"
    ]
    assert outcome.draft == CLAIMED  # the claim is kept for the reviewer, never sent
    [verified] = records(outcome, "verify")
    assert verified.status == "escalate" and verified.detail["deterministic"] is True
    assert jev.states("verify") == []  # deterministic rule wins; Jev not consulted
    review = ReviewQueue(make_engine(db)).get(outcome.review_id or 0)
    assert review is not None and review.kind == "review"
    assert review.payload["tool_results"][0]["ok"] is False


@respx.mock
async def test_real_tool_failure_is_escalated(llm: OpenAIClient, db: str) -> None:
    # The charge exists but the amount is more than was charged: the mock tool refuses.
    outcome = await run(llm, db, lookup_jev(), refund_call(amount=60.0))
    assert outcome.status == "escalated"
    assert "between 0 and 49.00" in outcome.reasons[0]
    assert REFUNDS == {}


@respx.mock
async def test_stops_at_first_failed_call(llm: OpenAIClient, db: str) -> None:
    outcome = await run(
        llm, db, lookup_jev(), refund_call(charge_id="ch_missing"), refund_call(charge_id="ch_501")
    )
    assert outcome.status == "escalated"
    assert [r.detail["args"]["charge_id"] for r in records(outcome, "execute")] == ["ch_missing"]
    assert REFUNDS == {}


@respx.mock
async def test_jev_unsupported_claim_is_escalated(llm: OpenAIClient, db: str) -> None:
    jev = lookup_jev(verify=verify_answers(0.3, "complete"))
    outcome = await run(llm, db, jev, refund_call())
    assert outcome.status == "escalated"
    assert outcome.reasons == ["verify.claim_supported 0.300 < 0.85"]
    assert "ch_502" in REFUNDS  # the refund did happen; the reply is what is held back


# --- happy path and the remaining guard branches ------------------------------------


@respx.mock
async def test_guarded_refund_runs_and_is_verified(llm: OpenAIClient, db: str) -> None:
    jev = lookup_jev()
    outcome = await run(llm, db, jev, refund_call())

    assert outcome.status == "ready_to_send"
    assert outcome.draft == CLAIMED
    assert [c.tool for c in outcome.tool_calls] == ["issue_refund"]
    assert REFUNDS["ch_502"]["status"] == "succeeded"
    assert [r.stage for r in outcome.trace][-4:] == ["guard", "execute", "verify", "decide"]

    [guard_state] = jev.states("guard")
    assert guard_state["proposed_tool_call"]["tool"] == "issue_refund"  # type: ignore[index]
    assert "test.one@example.com" not in json.dumps(guard_state)
    [verify_state] = jev.states("verify")
    tool_results = verify_state["tool_results"]  # type: ignore[index]
    assert tool_results[0]["ok"] is True and verify_state["draft"] == CLAIMED  # type: ignore[index]
    assert ReviewQueue(make_engine(db)).pending() == []


@respx.mock
async def test_low_guard_confidence_needs_approval(llm: OpenAIClient, db: str) -> None:
    outcome = await run(llm, db, lookup_jev(guard=guard_answers(0.5)), refund_call())
    assert outcome.status == "awaiting_approval"
    assert outcome.reasons == ["guard.issue_refund safe_to_run 0.500 < 0.9"]
    assert REFUNDS == {}


@respx.mock
async def test_guard_jev_failure_never_runs_blind(llm: OpenAIClient, db: str) -> None:
    jev = lookup_jev(guard=JevOverloadedError("HTTP 529: overloaded", status_code=529))
    outcome = await run(llm, db, jev, refund_call())
    assert outcome.status == "awaiting_approval"
    assert REFUNDS == {}


@respx.mock
async def test_invalid_call_is_blocked(llm: OpenAIClient, db: str) -> None:
    outcome = await run(llm, db, lookup_jev(), refund_call(amount=-10.0))
    assert outcome.status == "escalated"
    assert outcome.reasons == ["issue_refund: invalid amount"]
    assert REFUNDS == {}


@respx.mock
async def test_no_tool_calls_still_verified(llm: OpenAIClient, db: str) -> None:
    jev = lookup_jev(verify=verify_answers(0.95, "verify_more"))
    outcome = await run(llm, db, jev)
    assert outcome.status == "ready_to_send"
    assert records(outcome, "guard") == [] and records(outcome, "execute") == []
    [(state, questions)] = [c for c in jev.calls if "claim_supported" in c[1]]
    assert questions == ["claim_supported"]  # VERIFY_REPLY: no task_status question
    assert "account" in state and "customer_message" in state  # type: ignore[operator]
