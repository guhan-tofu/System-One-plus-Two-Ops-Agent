"""Phase 6 acceptance: end to end through HTTP with mocked providers.

Jev is the in-memory ScriptedJev; OpenAI is mocked with respx; the database is
a real SQLite file. Starlette's TestClient runs background tasks before it
returns, so each POST /items has finished processing when the call returns.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import respx
from fastapi.testclient import TestClient
from pydantic import SecretStr
from sqlalchemy import Engine

from sentinel.agent.pipeline import Pipeline
from sentinel.api.app import create_app
from sentinel.api.service import Service
from sentinel.config import Settings
from sentinel.llm.openai_client import OpenAIClient
from sentinel.policy.engine import Thresholds, ToolPolicy
from sentinel.storage.audit import AuditLog
from sentinel.storage.db import make_engine
from sentinel.storage.items import ItemStore
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
)

POLICIES = Path(__file__).parents[2] / "policies"
TOKEN = "test-api-token"  # noqa: S105  # pragma: allowlist secret
AUTH = {"Authorization": f"Bearer {TOKEN}"}
DRAFT = "I've refunded the duplicate charge of 49.00 GBP."
ITEM = {
    "id": "t-1",
    "source": "ticket",
    "subject": "Charged twice",
    "body": "I was charged twice this month, please refund one. Email: a@b.io",
    "customer_id": "cus_1001",
}


@pytest.fixture(autouse=True)
def _clean() -> Iterator[None]:
    reset_mock_state()
    yield
    reset_mock_state()


@pytest.fixture
def engine(tmp_path: Path) -> Engine:
    return make_engine(f"sqlite:///{tmp_path / 'api.db'}")


def make_client(
    engine: Engine,
    jev: ScriptedJev,
    *calls: dict[str, Any],
    tools: ToolRegistry | None = None,
) -> TestClient:
    respx.post(RESPONSES_URL).respond(200, json=response_body(draft_json(DRAFT, *calls)))
    llm = OpenAIClient(
        api_key=SecretStr("fake-openai-key"),  # pragma: allowlist secret
        models={"fast": "fake-fast", "strong": "fake-strong"},
    )
    queue = ReviewQueue(engine)
    audit = AuditLog(engine)
    pipeline = Pipeline(
        jev=jev,
        llm=llm,
        audit=audit,
        queue=queue,
        thresholds=Thresholds.load(POLICIES / "thresholds.yaml"),
        tools=tools or default_registry(),
        tool_policy=ToolPolicy.load(POLICIES / "tools.yaml"),
    )
    service = Service(pipeline=pipeline, items=ItemStore(engine), queue=queue, audit=audit)
    settings = Settings(sentinel_api_token=SecretStr(TOKEN))
    return TestClient(create_app(settings, service=service))


def lookup_jev(**script: Any) -> ScriptedJev:
    return ScriptedJev(triage=triage_answers(path="lookup"), **script)


def submit(client: TestClient, **overrides: Any) -> dict[str, Any]:
    response = client.post("/items", json={**ITEM, **overrides}, headers=AUTH)
    assert response.status_code == 202, response.text
    assert response.json()["status"] == "queued"
    item = client.get(f"/items/{response.json()['id']}", headers=AUTH)
    assert item.status_code == 200
    body: dict[str, Any] = item.json()
    return body


def audit_stages(engine: Engine, item_id: str) -> list[tuple[str, dict[str, Any]]]:
    from sqlalchemy.orm import Session

    from sentinel.storage.db import AuditEvent

    with Session(engine) as session:
        rows = session.query(AuditEvent).filter_by(item_id=item_id).order_by(AuditEvent.id)
        return [(r.stage, r.detail) for r in rows]


# --- auth -----------------------------------------------------------------------------


def test_refuses_to_start_without_token() -> None:
    with pytest.raises(RuntimeError, match="SENTINEL_API_TOKEN"):
        create_app(Settings())


@respx.mock
def test_auth_required(engine: Engine) -> None:
    with make_client(engine, lookup_jev()) as client:
        assert client.get("/healthz").status_code == 200
        assert client.post("/items", json=ITEM).status_code == 401
        bad = {"Authorization": "Bearer wrong"}
        assert client.post("/items", json=ITEM, headers=bad).status_code == 401
        assert client.get("/review", headers=bad).status_code == 401
        assert client.post("/review/1/approve", json={"by": "x"}).status_code == 401
    assert REFUNDS == {}


# --- items -----------------------------------------------------------------------------


@respx.mock
def test_item_processed_end_to_end(engine: Engine) -> None:
    with make_client(engine, lookup_jev(), refund_call()) as client:
        item = submit(client)

    assert item["status"] == "ready_to_send"
    outcome = item["outcome"]
    assert outcome["draft"] == DRAFT
    assert [c["tool"] for c in outcome["tool_calls"]] == ["issue_refund"]
    assert [r["stage"] for r in outcome["trace"]][-4:] == ["guard", "execute", "verify", "decide"]
    assert REFUNDS["ch_502"]["status"] == "succeeded"
    assert [s for s, _ in audit_stages(engine, "t-1")][-1] == "decide"


@respx.mock
def test_item_errors(engine: Engine) -> None:
    with make_client(engine, lookup_jev()) as client:
        submit(client)
        assert client.post("/items", json=ITEM, headers=AUTH).status_code == 409
        assert client.get("/items/nope", headers=AUTH).status_code == 404
        assert client.post("/items", json={"id": "x"}, headers=AUTH).status_code == 422


@respx.mock
def test_pipeline_crash_marks_item_failed(engine: Engine) -> None:
    class Exploding(ScriptedJev):
        async def evaluate(self, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("bug")

    with make_client(engine, Exploding()) as client:
        item = submit(client)
    assert item["status"] == "failed" and item["error"] == "RuntimeError"


# --- review: approval -------------------------------------------------------------------


@respx.mock
def test_approval_flow_runs_tool_after_human_approves(engine: Engine) -> None:
    with make_client(engine, lookup_jev(guard=guard_answers(0.5)), refund_call()) as client:
        item = submit(client)
        assert item["status"] == "awaiting_approval"
        assert REFUNDS == {}  # nothing ran yet
        review_id = item["review_id"]

        pending = client.get("/review", params={"kind": "approval"}, headers=AUTH).json()
        assert [r["id"] for r in pending] == [review_id]
        detail = client.get(f"/review/{review_id}", headers=AUTH).json()
        assert detail["payload"]["tool_calls"][0]["tool"] == "issue_refund"
        assert detail["reasons"] == ["guard.issue_refund safe_to_run 0.500 < 0.9"]

        approved = client.post(
            f"/review/{review_id}/approve", json={"by": "agent@example.test"}, headers=AUTH
        )
        assert approved.status_code == 200
        assert approved.json()["status"] == "ready_to_send"
        assert REFUNDS["ch_502"]["status"] == "succeeded"

        again = client.post(f"/review/{review_id}/approve", json={"by": "x"}, headers=AUTH)
        assert again.status_code == 409
        assert client.get("/review", headers=AUTH).json() == []

    stages = audit_stages(engine, "t-1")
    review_records = [d for s, d in stages if s == "review"]
    assert review_records == [
        {"review_id": review_id, "decision": "approved", "by": "agent@example.test"}
    ]
    assert [s for s, _ in stages][-4:] == ["enrich", "execute", "verify", "decide"]


@respx.mock
def test_destructive_tool_runs_only_after_approval(engine: Engine) -> None:
    close = {"tool": "close_account", "reason": "customer asked"}
    with make_client(engine, lookup_jev(guard=guard_answers(1.0)), close) as client:
        item = submit(client)
        assert item["status"] == "awaiting_approval"
        assert not CLOSED_ACCOUNTS
        response = client.post(
            f"/review/{item['review_id']}/approve", json={"by": "lead"}, headers=AUTH
        )
    assert response.json()["status"] == "ready_to_send"
    assert sorted(CLOSED_ACCOUNTS) == ["cus_1001"]


@respx.mock
def test_reject_runs_nothing(engine: Engine) -> None:
    with make_client(engine, lookup_jev(guard=guard_answers(0.5)), refund_call()) as client:
        item = submit(client)
        response = client.post(
            f"/review/{item['review_id']}/reject",
            json={"by": "agent", "note": "not a duplicate"},
            headers=AUTH,
        )
        assert response.json()["status"] == "rejected"
        review = client.get(f"/review/{item['review_id']}", headers=AUTH).json()
    assert review["status"] == "rejected" and review["note"] == "not a duplicate"
    assert REFUNDS == {}
    expected = {"review_id": item["review_id"], "kind": "approval", "outcome": "rejected"}
    assert ("review", {**expected, "by": "agent"}) in audit_stages(engine, "t-1")


@respx.mock
def test_approved_call_that_fails_is_escalated(engine: Engine) -> None:
    async def failing(customer_id: str, **kwargs: Any) -> dict[str, Any]:
        raise ToolError("processor declined")

    tools = ToolRegistry(
        [LOOKUP_CUSTOMER, CLOSE_ACCOUNT, dataclasses.replace(ISSUE_REFUND, fn=failing)]
    )
    jev = lookup_jev(guard=guard_answers(0.5))
    with make_client(engine, jev, refund_call(), tools=tools) as client:
        item = submit(client)
        response = client.post(
            f"/review/{item['review_id']}/approve", json={"by": "agent"}, headers=AUTH
        )
        body = response.json()
        assert body["status"] == "escalated"
        assert body["outcome"]["reasons"] == ["execute: issue_refund failed (processor declined)"]
        # A new review entry for the human, since the reply must not be sent.
        [pending] = client.get("/review", headers=AUTH).json()
        assert pending["kind"] == "review" and pending["id"] == body["review_id"]


@respx.mock
def test_approval_never_unblocks_a_blocked_call(engine: Engine) -> None:
    with make_client(engine, lookup_jev()) as client:
        submit(client)
        # A tampered/legacy approval entry holding a call the policy blocks.
        review_id = ReviewQueue(engine).enqueue(
            run_id="r",
            item_id="t-1",
            kind="approval",
            reasons=["test"],
            payload={"draft": DRAFT, "tool_calls": [{"tool": "drop_database", "args": {}}]},
        )
        response = client.post(f"/review/{review_id}/approve", json={"by": "x"}, headers=AUTH)
    assert response.json()["status"] == "escalated"
    assert "no policy rule" in response.json()["outcome"]["reasons"][0]


# --- review: escalations ----------------------------------------------------------------


@respx.mock
def test_escalated_item_resolved_by_human(engine: Engine) -> None:
    jev = ScriptedJev(triage=triage_answers(path="human"))
    with make_client(engine, jev) as client:
        item = submit(client)
        assert item["status"] == "escalated"
        [pending] = client.get("/review", params={"kind": "review"}, headers=AUTH).json()
        assert pending["reasons"] == ["triage.path is human"]
        response = client.post(
            f"/review/{pending['id']}/approve", json={"by": "agent"}, headers=AUTH
        )
    assert response.json()["status"] == "resolved"


@respx.mock
def test_unknown_review(engine: Engine) -> None:
    with make_client(engine, lookup_jev()) as client:
        assert client.get("/review/999", headers=AUTH).status_code == 404
        response = client.post("/review/999/approve", json={"by": "x"}, headers=AUTH)
        assert response.status_code == 409
        response = client.post("/review/999/approve", json={"by": ""}, headers=AUTH)
        assert response.status_code == 422
