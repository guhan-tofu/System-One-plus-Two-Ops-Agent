from __future__ import annotations

import json

import pytest
import respx
from typer.testing import CliRunner

from sentinel.cli import app
from tests.fixtures.vercel import EVAL_URL, GATEWAY_MODEL, GATEWAY_URL, answer_request

FAKE_KEY = "fake-gateway-key-for-tests"  # pragma: allowlist secret


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AI_GATEWAY_API_KEY", FAKE_KEY)
    monkeypatch.setenv("AI_GATEWAY_BASE_URL", GATEWAY_URL)


def run_probe(*args: str) -> tuple[int, str]:
    result = CliRunner().invoke(app, ["probe", *args])
    return result.exit_code, result.stdout


@respx.mock
def test_probe_prints_shape_and_raw() -> None:
    respx.post(EVAL_URL).mock(side_effect=answer_request)
    code, out = run_probe()
    assert code == 0
    report = json.loads(out)
    assert report["parse"] == "ok"
    assert report["provider"] == "vercel"
    assert report["audited_model"] == GATEWAY_MODEL
    assert report["model_verified"] is False
    assert report["shape"]["answer_fields"]["probe_noul"] == ["probability", "type"]
    raw_answers = report["raw_response"]["answers"]
    assert raw_answers["probe_noul"] == {"type": "boolean", "probability": 0.1}
    assert FAKE_KEY not in out


@respx.mock
def test_probe_model_override() -> None:
    route = respx.post(EVAL_URL).mock(side_effect=answer_request)
    code, _ = run_probe("--model", "typesafe-ai/jev-next")
    assert code == 0
    assert route.calls.last.request.headers["ai-model-id"] == "typesafe-ai/jev-next"


@respx.mock
def test_probe_reports_unparseable_shape() -> None:
    respx.post(EVAL_URL).respond(200, json={"data": {"something": "else"}})
    code, out = run_probe()
    assert code == 1
    report = json.loads(out)
    assert report["parse"].startswith("failed")
    assert report["shape"]["top_level_keys"] == ["data"]


@respx.mock
def test_probe_http_error() -> None:
    respx.post(EVAL_URL).respond(401)
    assert run_probe()[0] == 1


def test_probe_without_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AI_GATEWAY_API_KEY")
    assert run_probe()[0] == 1


def test_probe_needs_an_http_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JEV_PROVIDER", "llm_fallback")
    monkeypatch.setenv("OPENAI_API_KEY", FAKE_KEY)
    monkeypatch.setenv("OPENAI_MODEL_FAST", "fake-fast")
    result = CliRunner().invoke(app, ["probe"])
    assert result.exit_code == 1
    assert "HTTP Jev provider" in result.output
