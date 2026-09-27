from __future__ import annotations

import json

import pytest
import respx
from typer.testing import CliRunner

from sentinel.cli import app
from tests.fixtures import vercel
from tests.fixtures.typesafe import API_URL, MODEL_ALIAS, MODEL_VERSION, answer_request

FAKE_KEY = "fake-jev-key-for-tests"  # pragma: allowlist secret


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JEV_API_KEY", FAKE_KEY)
    monkeypatch.setenv("JEV_BASE_URL", API_URL)


def run_probe(*args: str) -> tuple[int, str]:
    result = CliRunner().invoke(app, ["probe", *args])
    return result.exit_code, result.stdout


@respx.mock
def test_probe_prints_shape_and_raw() -> None:
    respx.post(API_URL).mock(side_effect=answer_request)
    code, out = run_probe()
    assert code == 0
    report = json.loads(out)
    assert report["parse"] == "ok"
    assert report["provider"] == "typesafe"
    assert report["requested_model"] == MODEL_ALIAS
    assert report["audited_model"] == MODEL_VERSION
    assert report["model_verified"] is True
    assert report["shape"]["answer_fields"]["probe_noul"] == ["noul", "type"]
    assert report["raw_response"]["answers"]["probe_noul"] == {"type": "noul", "noul": 0.1}
    assert FAKE_KEY not in out


@respx.mock
def test_probe_model_override() -> None:
    route = respx.post(API_URL).mock(side_effect=answer_request)
    code, _ = run_probe("--model", "jev-preview")
    assert code == 0
    assert json.loads(route.calls.last.request.content)["model"] == "jev-preview"


@respx.mock
def test_probe_reports_unparseable_shape() -> None:
    respx.post(API_URL).respond(200, json={"data": {"something": "else"}})
    code, out = run_probe()
    assert code == 1
    report = json.loads(out)
    assert report["parse"].startswith("failed")
    assert report["shape"]["top_level_keys"] == ["data"]


@respx.mock
def test_probe_http_error() -> None:
    respx.post(API_URL).respond(401)
    assert run_probe()[0] == 1


def test_probe_without_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("JEV_API_KEY")
    result = CliRunner().invoke(app, ["probe"])
    assert result.exit_code == 1
    assert "JEV_API_KEY is not set" in result.output


@respx.mock
def test_probe_vercel(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JEV_PROVIDER", "vercel")
    monkeypatch.setenv("AI_GATEWAY_API_KEY", FAKE_KEY)
    monkeypatch.setenv("AI_GATEWAY_BASE_URL", vercel.GATEWAY_URL)
    respx.post(vercel.EVAL_URL).mock(side_effect=vercel.answer_request)
    code, out = run_probe()
    assert code == 0
    report = json.loads(out)
    assert report["provider"] == "vercel"
    assert report["audited_model"] == vercel.GATEWAY_MODEL
    assert report["model_verified"] is False


def test_probe_needs_an_http_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JEV_PROVIDER", "llm_fallback")
    monkeypatch.setenv("OPENAI_API_KEY", FAKE_KEY)
    monkeypatch.setenv("OPENAI_MODEL_FAST", "fake-fast")
    result = CliRunner().invoke(app, ["probe"])
    assert result.exit_code == 1
    assert "HTTP Jev provider" in result.output
