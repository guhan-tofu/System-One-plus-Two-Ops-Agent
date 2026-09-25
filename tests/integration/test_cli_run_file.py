from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx
from typer.testing import CliRunner

from sentinel.cli import app
from tests.fixtures.jev import confirmed_response
from tests.fixtures.openai import RESPONSES_URL, response_body
from tests.fixtures.pipeline import draft_json

JEV_URL = "https://jev.test/v1/systemone"
FAKE_KEY = "fake-key-for-cli-tests"  # pragma: allowlist secret
SAMPLES = Path(__file__).parents[2] / "samples"
DRAFT = "Thanks, our billing team will look into the duplicate charge."


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    db = tmp_path / "audit.db"
    for var, value in {
        "JEV_PROVIDER": "thejevai",
        "JEV_API_KEY": FAKE_KEY,
        "JEV_BASE_URL": JEV_URL,
        "OPENAI_API_KEY": FAKE_KEY,
        "OPENAI_MODEL_FAST": "fake-fast",
        "OPENAI_MODEL_STRONG": "fake-strong",
        "DATABASE_URL": f"sqlite:///{db}",
    }.items():
        monkeypatch.setenv(var, value)
    return db


def jev_answer(request: httpx.Request, triage_path: str = "lookup") -> httpx.Response:
    """Answer whichever question set was sent, in thejevai's confirmed shape."""
    questions = json.loads(request.content)["questions"]
    answers: dict[str, Any] = {}
    if "category" in questions:
        answers = {
            "category": {
                "type": "choice",
                "choice": "billing",
                "probabilities": {
                    "billing": 0.97,
                    "technical": 0.01,
                    "account": 0.01,
                    "sales": 0.01,
                },
                "confidence": 0.96,
            },
            "urgency": {
                "type": "score",
                "score": 1.6,
                "legend": {
                    "0": "Not urgent",
                    "1": "Soon",
                    "2": "Today",
                    "3": "Immediately / outage",
                },
                "probabilities": {"0": 0.05, "1": 0.3, "2": 0.6, "3": 0.05},
                "confidence": 0.6,
            },
            "path": {
                "type": "choice",
                "choice": triage_path,
                "probabilities": {
                    p: (0.9 if p == triage_path else 0.05) for p in ("lookup", "generate", "human")
                },
                "confidence": 0.85,
            },
            "is_abusive": {"type": "noul", "noul": 0.01},
        }
    elif "claim_supported" in questions:
        answers = {
            "claim_supported": {"type": "noul", "noul": 0.95},
            "task_status": {
                "type": "choice",
                "choice": "complete",
                "probabilities": {"complete": 0.9, "verify_more": 0.05, "failed": 0.05},
                "confidence": 0.85,
            },
        }
    else:
        answers = {
            "tier": {
                "type": "choice",
                "choice": "strong",
                "probabilities": {"fast": 0.2, "strong": 0.8},
                "confidence": 0.6,
            }
        }
    return httpx.Response(200, json=confirmed_response(answers))


@respx.mock
def test_run_file_prints_full_trace(_env: Path) -> None:
    respx.post(JEV_URL).mock(side_effect=jev_answer)
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
    assert "category=billing (0.96)" in out
    assert "thejevai:jev-latest (unverified)" in out
    assert "lookup_customer found=True" in out
    assert "tier=strong" in out
    assert "openai:fake-strong-2026" in out
    assert "claim_supported=0.95" in out
    assert "READY TO SEND" in out and DRAFT in out
    assert FAKE_KEY not in result.output

    rows = sqlite3.connect(_env).execute("select stage, model from audit_events").fetchall()
    assert [r[0] for r in rows][-1] == "decide"
    assert ("triage", "jev-latest") in rows and ("generate", "fake-strong-2026") in rows


@respx.mock
def test_run_file_json_output() -> None:
    respx.post(JEV_URL).mock(side_effect=lambda r: jev_answer(r, triage_path="human"))
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
