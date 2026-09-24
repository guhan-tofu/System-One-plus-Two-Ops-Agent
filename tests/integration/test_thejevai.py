from __future__ import annotations

import json
from collections.abc import AsyncIterator

import httpx
import pytest
import respx
from pydantic import SecretStr
from tenacity import wait_none

from sentinel.config import Settings
from sentinel.jev.errors import (
    JevAuthError,
    JevConfigError,
    JevHTTPError,
    JevOverloadedError,
    JevRateLimitError,
    JevResponseError,
    JevTimeoutError,
    JevValidationError,
)
from sentinel.jev.provider import JevProvider, build_provider
from sentinel.jev.thejevai import TheJevAIProvider
from tests.fixtures.jev import FAKE_MODEL_ID, QUESTIONS, documented_response

URL = "https://jev.test/v1/systemone"
FAKE_KEY = "fake-jev-key-for-tests"  # pragma: allowlist secret


@pytest.fixture
async def provider() -> AsyncIterator[TheJevAIProvider]:
    p = TheJevAIProvider(
        api_key=SecretStr(FAKE_KEY),
        base_url=URL,
        default_model="jev-latest",
        retry_wait=wait_none(),
    )
    yield p
    await p.aclose()


@respx.mock
async def test_success_sends_documented_request(provider: TheJevAIProvider) -> None:
    route = respx.post(URL).respond(200, json=documented_response())
    result = await provider.evaluate({"subject": "refund"}, QUESTIONS)

    assert result.model == FAKE_MODEL_ID
    assert result.choice("category").choice == "billing"
    assert result.latency_ms >= 0

    sent = route.calls.last.request
    assert sent.headers["Authorization"] == f"Bearer {FAKE_KEY}"
    body = json.loads(sent.content)
    assert body["model"] == "jev-latest"
    assert body["state"] == {"subject": "refund"}
    assert body["questions"]["category"]["type"] == "choice"
    assert body["questions"]["urgency"]["criteria"] == ["Not urgent", "Soon", "Today"]
    assert body["questions"]["is_abusive"] == {"type": "noul", "instructions": "Is it abusive?"}


@respx.mock
async def test_model_override(provider: TheJevAIProvider) -> None:
    route = respx.post(URL).respond(200, json=documented_response())
    await provider.evaluate("s", QUESTIONS, model="jev-1.13.0")
    assert json.loads(route.calls.last.request.content)["model"] == "jev-1.13.0"


@respx.mock
async def test_401_not_retried(provider: TheJevAIProvider) -> None:
    route = respx.post(URL).respond(401, json={"error": "unauthorized"})
    with pytest.raises(JevAuthError) as exc_info:
        await provider.evaluate("s", QUESTIONS)
    assert route.call_count == 1
    assert FAKE_KEY not in str(exc_info.value)


@pytest.mark.parametrize(
    ("body", "field"),
    [
        (
            {"error": {"message": "bad criteria", "param": "questions.urgency.criteria"}},
            "questions.urgency.criteria",
        ),
        (
            {"detail": [{"loc": ["body", "questions", "x"], "msg": "field required"}]},
            "body.questions.x",
        ),
        ({"errors": [{"path": "state", "message": "too long"}]}, "state"),
    ],
)
@respx.mock
async def test_422_surfaces_field_not_retried(
    provider: TheJevAIProvider, body: dict[str, object], field: str
) -> None:
    route = respx.post(URL).respond(422, json=body)
    with pytest.raises(JevValidationError) as exc_info:
        await provider.evaluate("s", QUESTIONS)
    assert exc_info.value.field == field
    assert field in str(exc_info.value)
    assert route.call_count == 1


@respx.mock
async def test_422_non_json_body(provider: TheJevAIProvider) -> None:
    respx.post(URL).respond(422, text="nope")
    with pytest.raises(JevValidationError, match="nope"):
        await provider.evaluate("s", QUESTIONS)


@pytest.mark.parametrize("status", [429, 529])
@respx.mock
async def test_retryable_then_success(provider: TheJevAIProvider, status: int) -> None:
    route = respx.post(URL)
    route.side_effect = [
        httpx.Response(status),
        httpx.Response(status),
        httpx.Response(200, json=documented_response()),
    ]
    result = await provider.evaluate("s", QUESTIONS)
    assert result.model == FAKE_MODEL_ID
    assert route.call_count == 3


@pytest.mark.parametrize(("status", "error"), [(429, JevRateLimitError), (529, JevOverloadedError)])
@respx.mock
async def test_retryable_gives_up_after_4_attempts(
    provider: TheJevAIProvider, status: int, error: type[Exception]
) -> None:
    route = respx.post(URL).respond(status)
    with pytest.raises(error):
        await provider.evaluate("s", QUESTIONS)
    assert route.call_count == 4


@pytest.mark.parametrize("status", [400, 403, 500, 503])
@respx.mock
async def test_other_statuses_not_retried(provider: TheJevAIProvider, status: int) -> None:
    route = respx.post(URL).respond(status)
    with pytest.raises((JevHTTPError, JevAuthError)):
        await provider.evaluate("s", QUESTIONS)
    assert route.call_count == 1


@respx.mock
async def test_timeout_not_retried(provider: TheJevAIProvider) -> None:
    route = respx.post(URL).mock(side_effect=httpx.ReadTimeout("slow"))
    with pytest.raises(JevTimeoutError):
        await provider.evaluate("s", QUESTIONS)
    assert route.call_count == 1


@respx.mock
async def test_non_json_success_rejected(provider: TheJevAIProvider) -> None:
    respx.post(URL).respond(200, text="<html>")
    with pytest.raises(JevResponseError):
        await provider.evaluate("s", QUESTIONS)


@respx.mock
async def test_key_never_logged(
    provider: TheJevAIProvider, capsys: pytest.CaptureFixture[str]
) -> None:
    from sentinel.log import configure_logging

    configure_logging("DEBUG")
    route = respx.post(URL)
    route.side_effect = [httpx.Response(429), httpx.Response(200, json=documented_response())]
    await provider.evaluate("secret-state-text", QUESTIONS)
    err = capsys.readouterr().err
    assert "jev.retry" in err and "jev.evaluate" in err
    assert FAKE_KEY not in err
    assert "secret-state-text" not in err


def test_timeouts_configured() -> None:
    from sentinel.jev.thejevai import DEFAULT_TIMEOUT

    assert DEFAULT_TIMEOUT.connect == 5.0
    assert DEFAULT_TIMEOUT.read == 10.0


def test_missing_key_rejected() -> None:
    with pytest.raises(JevConfigError):
        TheJevAIProvider(api_key=SecretStr(""), base_url=URL, default_model="m")


async def test_factory_builds_thejevai() -> None:
    settings = Settings(jev_api_key=SecretStr(FAKE_KEY), jev_provider="thejevai")
    p = build_provider(settings)
    assert isinstance(p, TheJevAIProvider)
    assert isinstance(p, JevProvider)
    await p.aclose()


@pytest.mark.parametrize("name", ["typesafe", "llm_fallback"])
def test_factory_unimplemented_providers(name: str) -> None:
    settings = Settings(jev_api_key=SecretStr(FAKE_KEY), jev_provider=name)  # type: ignore[arg-type]
    with pytest.raises(JevConfigError):
        build_provider(settings)
