from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import httpx
import pytest
import respx
from typer.testing import CliRunner

from sentinel.cli import app
from tests.fixtures.openai import RESPONSES_URL, response_body
from tests.fixtures.pipeline import draft_json
from tests.fixtures.vercel import EVAL_URL, GATEWAY_URL, answer_request

FAKE_KEY = "fake-key-for-cli-tests"  # pragma: allowlist secret
SAMPLES = Path(__file__).parents[2] / "samples"
DRAFT = "Thanks, our billing team will look into the duplicate charge."


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    db = tmp_path / "audit.db"
    for var, value in {
        "AI_GATEWAY_API_KEY": FAKE_KEY,
        "AI_GATEWAY_BASE_URL": GATEWAY_URL,
        "OPENAI_API_KEY": FAKE_KEY,
        "OPENAI_MODEL_FAST": "fake-fast",
        "OPENAI_MODEL_STRONG": "fake-strong",
        "DATABASE_URL": f"sqlite:///{db}",
    }.items():
        monkeypatch.setenv(var, value)
    return db


def jev_answer(request: httpx.Request, triage_path: str = "lookup") -> httpx.Response:
    """Official Jev via the gateway: billing, the given path, strong tier, supported claim."""
    return answer_request(
        request,
        picks={"category": "billing", "path": triage_path, "tier": "strong",
               "task_status": "complete"},
        boolean={"is_abusive": 0.01, "claim_supported": 0.95},
        p=0.96,
    )  # fmt: skip


@respx.mock
def test_run_file_prints_full_trace(_env: Path) -> None:
    respx.post(EVAL_URL).mock(side_effect=jev_answer)
    respx.post(RESPONSES_URL).respond(
        200, json=response_body(draft_json(DRAFT), model="fake-strong-2026")
    )

    result = CliRunner().invoke(app, ["run-file", str(SAMPLES / "ticket.json")])

    assert result.exit_code == 0, result.output
    out = result.stdout
    for stage in (
        "ingest", "build_state", "triage", "enrich", "route_model", "generate", "verify", "decide",
    ):  # fmt: skip
        assert f"[{stage:<11}]" in out
    assert "category=billing (0.95)" in out
    assert "vercel:typesafe-ai/jev (unverified)" in out
    assert "lookup_customer found=True" in out
    assert "tier=strong" in out
    assert "openai:fake-strong-2026" in out
    assert "claim_supported=0.95" in out
    assert "READY TO SEND" in out and DRAFT in out
    assert FAKE_KEY not in result.output

    rows = sqlite3.connect(_env).execute("select stage, model from audit_events").fetchall()
    assert [r[0] for r in rows][-1] == "decide"
    assert ("triage", "typesafe-ai/jev") in rows and ("generate", "fake-strong-2026") in rows


@respx.mock
def test_run_file_json_output() -> None:
    respx.post(EVAL_URL).mock(side_effect=lambda r: jev_answer(r, triage_path="human"))
    result = CliRunner().invoke(app, ["run-file", "--json", str(SAMPLES / "ticket.json")])
    assert result.exit_code == 0, result.output
    outcome = json.loads(result.stdout)
    assert outcome["status"] == "escalated"
    assert outcome["reasons"] == ["triage.path is human"]


def test_run_file_misconfigured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY")
    result = CliRunner().invoke(app, ["run-file", str(SAMPLES / "ticket.json")])
    assert result.exit_code == 1
    assert "OPENAI_API_KEY" in result.output


@pytest.mark.parametrize("name", ["ticket", "outage", "how_to", "abusive"])
def test_samples_are_valid_work_items(name: str) -> None:
    from sentinel.agent.models import WorkItem

    WorkItem.model_validate_json((SAMPLES / f"{name}.json").read_text())
