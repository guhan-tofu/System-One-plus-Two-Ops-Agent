from __future__ import annotations

import json

import pytest
import respx
from typer.testing import CliRunner

from sentinel.cli import app
from tests.fixtures.jev import confirmed_response

URL = "https://jev.test/v1/systemone"
FAKE_KEY = "fake-jev-key-for-tests"  # pragma: allowlist secret

PROBE_ANSWERS = {
    "probe_choice": {
        "type": "choice",
        "choice": "technical",
        "probabilities": {"technical": 0.8, "billing": 0.2},
        "confidence": 0.8,
    },
    "probe_score": {
        "type": "score",
        "score": 1.0,
        "legend": {"0": "Not urgent", "1": "Soon", "2": "Today"},
        "probabilities": {"0": 0.2, "1": 0.6, "2": 0.2},
        "confidence": 0.6,
    },
    "probe_noul": {"type": "noul", "noul": 0.01},
}


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JEV_API_KEY", FAKE_KEY)
    monkeypatch.setenv("JEV_BASE_URL", URL)


def run_probe() -> tuple[int, str]:
    result = CliRunner().invoke(app, ["probe"])
    return result.exit_code, result.stdout


@respx.mock
def test_probe_prints_shape_and_raw() -> None:
    respx.post(URL).respond(200, json=confirmed_response(PROBE_ANSWERS))
    code, out = run_probe()
    assert code == 0
    report = json.loads(out)
    assert report["parse"] == "ok"
    assert report["audited_model"] == "jev-latest"
    assert report["model_verified"] is False
    assert report["shape"]["result_keys"] == ["answers", "elapsedMs", "usage"]
    raw_answers = report["raw_response"]["data"]["result"]["answers"]
    assert raw_answers["probe_noul"] == {"type": "noul", "noul": 0.01}
    assert FAKE_KEY not in out


@respx.mock
def test_probe_reports_unparseable_shape() -> None:
    respx.post(URL).respond(200, json={"data": {"something": "else"}})
    code, out = run_probe()
    assert code == 1
    report = json.loads(out)
    assert report["parse"].startswith("failed")
    assert report["shape"]["top_level_keys"] == ["data"]


@respx.mock
def test_probe_http_error() -> None:
    respx.post(URL).respond(401)
    assert run_probe()[0] == 1


def test_probe_without_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("JEV_API_KEY")
    assert run_probe()[0] == 1
