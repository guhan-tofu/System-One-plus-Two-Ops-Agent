from __future__ import annotations

import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
import respx
from pydantic import SecretStr

from sentinel.agent.pipeline import Pipeline
from sentinel.jev.errors import JevOverloadedError
from sentinel.llm.openai_client import OpenAIClient
from sentinel.policy.engine import Thresholds, ToolPolicy
from sentinel.storage.audit import AuditLog
from sentinel.storage.db import make_engine
from sentinel.storage.review import ReviewQueue
from sentinel.tools.builtin import default_registry
from sentinel.tools.registry import Risk, Tool, ToolRegistry
from tests.fixtures.openai import RESPONSES_URL, error_body, response_body
from tests.fixtures.pipeline import (
    ScriptedJev,
    draft_json,
    route_answers,
    triage_answers,
    work_item,
)

POLICIES = Path(__file__).parents[2] / "policies"
THRESHOLDS = POLICIES / "thresholds.yaml"
FAKE_KEY = "fake-openai-key-for-tests"  # pragma: allowlist secret
DRAFT = "Thanks for reaching out. Our billing team will review the duplicate charge."


@pytest.fixture
def audit(tmp_path: Path) -> AuditLog:
    return AuditLog(make_engine(f"sqlite:///{tmp_path / 'audit.db'}"))


@pytest.fixture
def queue(tmp_path: Path) -> ReviewQueue:
    return ReviewQueue(make_engine(f"sqlite:///{tmp_path / 'audit.db'}"))


@pytest.fixture
async def llm() -> AsyncIterator[OpenAIClient]:
    client = OpenAIClient(
        api_key=SecretStr(FAKE_KEY), models={"fast": "fake-fast", "strong": "fake-strong"}
    )
    yield client
    await client.aclose()


def make_pipeline(
    jev: ScriptedJev,
    llm: OpenAIClient,
    audit: AuditLog,
    *,
    fallback: ScriptedJev | None = None,
    tools: ToolRegistry | None = None,
    queue: ReviewQueue | None = None,
) -> Pipeline:
    return Pipeline(
        jev=jev,
        jev_fallback=fallback,
        llm=llm,
        audit=audit,
        thresholds=Thresholds.load(THRESHOLDS),
        tools=tools or default_registry(),
        tool_policy=ToolPolicy.load(POLICIES / "tools.yaml"),
        queue=queue or ReviewQueue(audit._engine),
    )


def stages(outcome: Any) -> list[str]:
    return [r.stage for r in outcome.trace]


@respx.mock
async def test_lookup_path_produces_draft(llm: OpenAIClient, audit: AuditLog) -> None:
    route = respx.post(RESPONSES_URL).respond(200, json=response_body(draft_json(DRAFT)))
    jev = ScriptedJev(
        triage=triage_answers(path="lookup"), route_model=route_answers("strong", 0.8)
    )
    outcome = await make_pipeline(jev, llm, audit).run(work_item())

    assert outcome.status == "ready_to_send"
    assert outcome.tier == "strong"
    assert outcome.draft == DRAFT
    assert stages(outcome) == [
        "ingest", "build_state", "triage", "enrich", "route_model", "generate", "verify",
        "decide",
    ]  # fmt: skip

    # Models only ever see redacted, minimal state.
    for state, _ in jev.calls:
        text = json.dumps(state)
        assert "test.one@example.com" not in text and "7946" not in text
        assert "cus_1001" not in text
    route_state = jev.calls[1][0]
    assert route_state["account"]["plan"] == "Pro (monthly)"
    assert "name" not in route_state["account"] and "email" not in route_state["account"]

    sent = json.loads(route.calls.last.request.content)
    assert sent["model"] == "fake-strong"
    assert "Pro (monthly)" in sent["input"]
    for leaked in ("test.one@example.com", "Test Customer One", "cus_1001", "7946 0958"):
        assert leaked not in sent["input"]


@respx.mock
async def test_every_stage_is_audited_with_model_id(llm: OpenAIClient, audit: AuditLog) -> None:
    respx.post(RESPONSES_URL).respond(
        200, json=response_body(draft_json(DRAFT), model="fake-model-2026")
    )
    jev = ScriptedJev(triage=triage_answers(path="lookup"), route_model=route_answers())
    outcome = await make_pipeline(jev, llm, audit).run(work_item())

    events = audit.events(outcome.run_id)
    assert [e.stage for e in events] == stages(outcome)
    assert all(e.item_id == "item-1" for e in events)
    by_stage = {e.stage: e for e in events}
    assert by_stage["triage"].model == "fake_jev-model-1"
    assert by_stage["triage"].provider == "fake_jev"
    assert by_stage["triage"].detail["answers"]["category"]["choice"] == "billing"
    assert by_stage["route_model"].model == "fake_jev-model-1"
    assert by_stage["generate"].model == "fake-model-2026"
    assert by_stage["generate"].model_verified is True
    assert by_stage["decide"].detail["outcome"] == "ready_to_send"


@respx.mock
async def test_generate_path_skips_enrich(llm: OpenAIClient, audit: AuditLog) -> None:
    route = respx.post(RESPONSES_URL).respond(200, json=response_body(draft_json(DRAFT)))
    jev = ScriptedJev(triage=triage_answers(path="generate"), route_model=route_answers("fast"))
    outcome = await make_pipeline(jev, llm, audit).run(work_item())

    assert outcome.status == "ready_to_send" and outcome.tier == "fast"
    enrich = next(r for r in outcome.trace if r.stage == "enrich")
    assert enrich.detail["ran"] is False
    assert json.loads(route.calls.last.request.content)["model"] == "fake-fast"
    assert "account" not in jev.calls[1][0]


@pytest.mark.parametrize(
    ("triage_kw", "reason"),
    [
        ({"category_confidence": 0.5}, "category confidence"),
        ({"path": "lookup", "path_confidence": 0.05}, "path confidence"),
        ({"path": "human"}, "path is human"),
        ({"abusive": 0.95}, "is_abusive"),
    ],
)
@respx.mock
async def test_triage_escalations_stop_before_llm(
    llm: OpenAIClient, audit: AuditLog, triage_kw: dict[str, Any], reason: str
) -> None:
    route = respx.post(RESPONSES_URL).respond(200, json=response_body(draft_json(DRAFT)))
    jev = ScriptedJev(triage=triage_answers(**triage_kw), route_model=route_answers())
    outcome = await make_pipeline(jev, llm, audit).run(work_item())

    assert outcome.status == "escalated"
    assert any(reason in r for r in outcome.reasons)
    assert outcome.draft is None
    assert stages(outcome) == ["ingest", "build_state", "triage", "decide"]
    assert route.call_count == 0
    assert len(jev.calls) == 1


@respx.mock
async def test_low_route_confidence_uses_policy_default(llm: OpenAIClient, audit: AuditLog) -> None:
    route = respx.post(RESPONSES_URL).respond(200, json=response_body(draft_json(DRAFT)))
    jev = ScriptedJev(triage=triage_answers(), route_model=route_answers("fast", 0.4))
    outcome = await make_pipeline(jev, llm, audit).run(work_item())
    assert outcome.tier == "strong"
    assert json.loads(route.calls.last.request.content)["model"] == "fake-strong"
    routed = next(r for r in outcome.trace if r.stage == "route_model")
    assert routed.detail["decision"]["overridden"] is True


@pytest.mark.parametrize("stage", ["triage", "route_model"])
@respx.mock
async def test_jev_failure_escalates_without_fallback(
    llm: OpenAIClient, audit: AuditLog, stage: str
) -> None:
    respx.post(RESPONSES_URL).respond(200, json=response_body(draft_json(DRAFT)))
    script: dict[str, Any] = {"triage": triage_answers(), "route_model": route_answers()}
    script[stage] = JevOverloadedError("HTTP 529: overloaded", status_code=529)
    outcome = await make_pipeline(ScriptedJev(**script), llm, audit).run(work_item())

    assert outcome.status == "escalated"
    assert outcome.reasons == [f"{stage}: Jev unavailable (JevOverloadedError)"]
    failed = next(r for r in outcome.trace if r.stage == stage)
    assert failed.status == "error"


@respx.mock
async def test_jev_failure_uses_fallback_provider(llm: OpenAIClient, audit: AuditLog) -> None:
    respx.post(RESPONSES_URL).respond(200, json=response_body(draft_json(DRAFT)))
    primary = ScriptedJev(
        triage=JevOverloadedError("HTTP 529: overloaded", status_code=529),
        route_model=route_answers(),
    )
    fallback = ScriptedJev(
        name="llm_fallback", triage=triage_answers(), route_model=route_answers()
    )
    outcome = await make_pipeline(primary, llm, audit, fallback=fallback).run(work_item())

    assert outcome.status == "ready_to_send"
    triaged = next(r for r in outcome.trace if r.stage == "triage")
    assert triaged.provider == "llm_fallback"
    assert "JevOverloadedError" in triaged.detail["primary_error"]
    routed = next(r for r in outcome.trace if r.stage == "route_model")
    assert routed.provider == "fake_jev" and routed.detail["primary_error"] is None


@respx.mock
async def test_llm_failure_escalates(llm: OpenAIClient, audit: AuditLog) -> None:
    respx.post(RESPONSES_URL).respond(401, json=error_body("bad key"))
    jev = ScriptedJev(triage=triage_answers(), route_model=route_answers())
    outcome = await make_pipeline(jev, llm, audit).run(work_item())
    assert outcome.status == "escalated"
    assert outcome.reasons == ["generate: draft failed"]
    assert FAKE_KEY not in outcome.model_dump_json()


@respx.mock
async def test_missing_customer_reference(llm: OpenAIClient, audit: AuditLog) -> None:
    respx.post(RESPONSES_URL).respond(200, json=response_body(draft_json(DRAFT)))
    jev = ScriptedJev(triage=triage_answers(path="lookup"), route_model=route_answers())
    outcome = await make_pipeline(jev, llm, audit).run(work_item(customer_id=None))
    assert outcome.status == "ready_to_send"
    assert jev.calls[1][0]["account"] == {"found": False}


@respx.mock
async def test_tool_failure_escalates(llm: OpenAIClient, audit: AuditLog) -> None:
    async def broken(customer_id: str) -> dict[str, Any]:
        raise ConnectionError("crm down")

    tools = ToolRegistry([Tool("lookup_customer", "broken", Risk.READ_ONLY, broken)])
    jev = ScriptedJev(triage=triage_answers(path="lookup"), route_model=route_answers())
    outcome = await make_pipeline(jev, llm, audit, tools=tools).run(work_item())
    assert outcome.status == "escalated"
    assert outcome.reasons == ["enrich: lookup_customer failed"]


async def test_registry_policy_mismatch_refuses_to_start(
    llm: OpenAIClient, audit: AuditLog
) -> None:
    async def refund(customer_id: str) -> dict[str, Any]:
        raise AssertionError("must never run")

    # lookup_customer is read_only in policies/tools.yaml; a registry claiming
    # otherwise must not start, so enrich can never run a side-effecting tool.
    tools = ToolRegistry([Tool("lookup_customer", "x", Risk.DESTRUCTIVE, refund)])
    with pytest.raises(ValueError, match="risk"):
        make_pipeline(ScriptedJev(), llm, audit, tools=tools)


async def test_audit_failure_stops_the_pipeline(llm: OpenAIClient, queue: ReviewQueue) -> None:
    class BrokenAudit(AuditLog):
        def __init__(self) -> None:
            pass

        def record(self, *args: Any, **kwargs: Any) -> None:
            raise OSError("disk full")

    jev = ScriptedJev(triage=triage_answers(), route_model=route_answers())
    with pytest.raises(OSError):
        await make_pipeline(jev, llm, BrokenAudit(), queue=queue).run(work_item())
    assert jev.calls == []
