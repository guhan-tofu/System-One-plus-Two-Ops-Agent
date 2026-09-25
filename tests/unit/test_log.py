from __future__ import annotations

import json

import pytest

from sentinel.log import REDACTED, configure_logging, get_logger, redact_secrets


def test_redact_secrets_nested() -> None:
    event = {
        "event": "call",
        "api_key": "x",
        "headers": {"Authorization": "Bearer x", "Accept": "json"},
        "items": [{"token": "x", "ok": 1}],
    }
    out = redact_secrets(None, "info", event)
    assert out["api_key"] == REDACTED
    assert out["headers"] == {"Authorization": REDACTED, "Accept": "json"}
    assert out["items"] == [{"token": REDACTED, "ok": 1}]
    assert out["event"] == "call"


def test_token_counts_not_redacted() -> None:
    event = {"input_tokens": 12, "cached_input_tokens": 3, "tokens": "x", "access_token": "x"}
    out = redact_secrets(None, "info", event)
    assert out["input_tokens"] == 12 and out["cached_input_tokens"] == 3
    assert out["tokens"] == REDACTED
    assert out["access_token"] == REDACTED


def test_json_output_is_scrubbed(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging("INFO")
    fake = "fake-value"  # pragma: allowlist secret
    get_logger("t").info("hello", jev_api_key=fake, stage="triage")
    line = capsys.readouterr().err.strip().splitlines()[-1]
    record = json.loads(line)
    assert record["event"] == "hello"
    assert record["stage"] == "triage"
    assert record["jev_api_key"] == REDACTED
    assert fake not in line
