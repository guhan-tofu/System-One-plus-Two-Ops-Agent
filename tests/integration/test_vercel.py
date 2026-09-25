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
    JevRateLimitError,
    JevValidationError,
)
from sentinel.jev.provider import build_provider
from sentinel.jev.vercel import VercelJevProvider
from tests.fixtures.jev import QUESTIONS
from tests.fixtures.vercel import EVAL_URL, GATEWAY_MODEL, gateway_error, gateway_response

FAKE_KEY = "fake-gateway-key-for-tests"  # pragma: allowlist secret


@pytest.fixture
async def provider() -> AsyncIterator[VercelJevProvider]:
    p = VercelJevProvider(
        api_key=SecretStr(FAKE_KEY),
        url=EVAL_URL,
        default_model=GATEWAY_MODEL,
        retry_wait=wait_none(),
    )
    yield p
    await p.aclose()


@respx.mock
async def test_request_matches_gateway_contract(provider: VercelJevProvider) -> None:
    route = respx.post(EVAL_URL).respond(200, json=gateway_response())
    result = await provider.evaluate({"subject": "refund"}, QUESTIONS)
    assert result.provider == "vercel"

    sent = route.calls.last.request
    assert sent.headers["Authorization"] == f"Bearer {FAKE_KEY}"
    assert sent.headers["ai-model-id"] == GATEWAY_MODEL
    assert sent.headers["ai-evaluation-model-specification-version"] == "4"
    assert sent.headers["ai-gateway-protocol-version"] == "0.0.1"
    body = json.loads(sent.content)
    assert set(body) == {"state", "questions"}  # model goes in a header, not the body
    assert body["state"] == {"subject": "refund"}
    assert body["questions"]["is_abusive"] == {"type": "boolean", "instructions": "Is it abusive?"}
    assert body["questions"]["category"]["criteria"] == {"billing": "Payments", "technical": "Bugs"}
    assert body["questions"]["urgency"]["criteria"] == ["Not urgent", "Soon", "Today"]


@respx.mock
async def test_noul_criteria_sent_as_boolean_criteria(provider: VercelJevProvider) -> None:
    from sentinel.jev.questions import GUARD

    route = respx.post(EVAL_URL).respond(
        200,
        json=gateway_response(
            {"safe_to_run": {"type": "boolean", "probability": 0.2}}, confidence={}
        ),
    )
    result = await provider.evaluate("s", GUARD)
    sent = json.loads(route.calls.last.request.content)["questions"]["safe_to_run"]
    assert sent["type"] == "boolean"
    assert set(sent["criteria"]) == {"true", "false"}
    assert result.noul("safe_to_run").noul == 0.2


@respx.mock
async def test_model_override_goes_in_header(provider: VercelJevProvider) -> None:
    route = respx.post(EVAL_URL).respond(200, json=gateway_response())
    result = await provider.evaluate("s", QUESTIONS, model="typesafe-ai/jev-next")
    assert route.calls.last.request.headers["ai-model-id"] == "typesafe-ai/jev-next"
    assert result.model == "typesafe-ai/jev-next"


@pytest.mark.parametrize(
    ("status", "body", "match"),
    [
        (401, gateway_error("Missing Authorization header", "authentication_error"), "Missing"),
        (
            403,
            gateway_error(
                "AI Gateway requires a valid credit card", "customer_verification_required"
            ),
            "credit card",
        ),
    ],
)
@respx.mock
async def test_auth_errors_surface_vendor_message(
    provider: VercelJevProvider, status: int, body: dict[str, object], match: str
) -> None:
    route = respx.post(EVAL_URL).respond(status, json=body)
    with pytest.raises(JevAuthError, match=match) as exc_info:
        await provider.evaluate("s", QUESTIONS)
    assert route.call_count == 1
    assert FAKE_KEY not in str(exc_info.value)


@respx.mock
async def test_400_is_a_validation_error(provider: VercelJevProvider) -> None:
    route = respx.post(EVAL_URL).respond(
        400, json=gateway_error("Unsupported gateway protocol version")
    )
    with pytest.raises(JevValidationError, match="Unsupported gateway protocol version"):
        await provider.evaluate("s", QUESTIONS)
    assert route.call_count == 1


@respx.mock
async def test_429_retried(provider: VercelJevProvider) -> None:
    route = respx.post(EVAL_URL)
    route.side_effect = [httpx.Response(429), httpx.Response(200, json=gateway_response())]
    await provider.evaluate("s", QUESTIONS)
    assert route.call_count == 2


@respx.mock
async def test_429_gives_up(provider: VercelJevProvider) -> None:
    route = respx.post(EVAL_URL).respond(429)
    with pytest.raises(JevRateLimitError):
        await provider.evaluate("s", QUESTIONS)
    assert route.call_count == 4


@respx.mock
async def test_state_and_key_never_logged(
    provider: VercelJevProvider, capsys: pytest.CaptureFixture[str]
) -> None:
    from sentinel.log import configure_logging

    configure_logging("DEBUG")
    respx.post(EVAL_URL).respond(200, json=gateway_response())
    await provider.evaluate("secret-state-text", QUESTIONS)
    err = capsys.readouterr().err
    assert "jev.evaluate" in err and "gen_fake_0001" in err
    assert FAKE_KEY not in err and "secret-state-text" not in err


async def test_factory_default_is_vercel() -> None:
    settings = Settings(ai_gateway_api_key=SecretStr(FAKE_KEY))
    p = build_provider(settings)
    assert isinstance(p, VercelJevProvider)
    assert p._url == "https://ai-gateway.vercel.sh/v4/ai/evaluation-model"
    await p.aclose()


def test_missing_key() -> None:
    with pytest.raises(JevConfigError, match="AI_GATEWAY_API_KEY"):
        build_provider(Settings())
