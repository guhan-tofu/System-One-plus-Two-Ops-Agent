from __future__ import annotations

import json
from collections.abc import AsyncIterator

import httpx
import pytest
import respx
from pydantic import SecretStr
from tenacity import RetryCallState, wait_none

from sentinel.config import Settings
from sentinel.jev.errors import (
    JevAuthError,
    JevConfigError,
    JevOverloadedError,
    JevRateLimitError,
    JevValidationError,
)
from sentinel.jev.http import MAX_RETRY_AFTER_S, wait_retry_after
from sentinel.jev.provider import build_provider
from sentinel.jev.typesafe import TypeSafeProvider
from tests.fixtures.jev import QUESTIONS
from tests.fixtures.typesafe import API_URL, MODEL_ALIAS, MODEL_VERSION, api_error, api_response

FAKE_KEY = "fake-jev-key-for-tests"  # pragma: allowlist secret


@pytest.fixture
async def provider() -> AsyncIterator[TypeSafeProvider]:
    p = TypeSafeProvider(
        api_key=SecretStr(FAKE_KEY),
        url=API_URL,
        default_model=MODEL_ALIAS,
        retry_wait=wait_none(),
    )
    yield p
    await p.aclose()


@respx.mock
async def test_request_matches_api_contract(provider: TypeSafeProvider) -> None:
    route = respx.post(API_URL).respond(200, json=api_response())
    result = await provider.evaluate({"subject": "refund"}, QUESTIONS)

    sent = route.calls.last.request
    assert sent.headers["Authorization"] == f"Bearer {FAKE_KEY}"
    body = json.loads(sent.content)
    assert set(body) == {"model", "state", "questions"}
    assert body["model"] == MODEL_ALIAS
    assert body["state"] == {"subject": "refund"}
    assert body["questions"]["is_abusive"] == {"type": "noul", "instructions": "Is it abusive?"}
    assert body["questions"]["category"]["criteria"] == {"billing": "Payments", "technical": "Bugs"}
    assert body["questions"]["urgency"]["criteria"] == ["Not urgent", "Soon", "Today"]

    assert result.provider == "typesafe"
    assert result.model == MODEL_VERSION  # the versioned ID, not the alias we asked for
    assert result.model_verified is True
    assert result.metadata == {"requested_model": MODEL_ALIAS}
    assert result.usage == {"input_tokens": 378, "output_tokens": 65}
    assert result.choice("category").confidence == 0.8  # vendor-reported
    assert result.score("urgency").confidence == 0.35
    assert result.noul("is_abusive").noul == 0.03


@respx.mock
async def test_noul_criteria_sent_as_is(provider: TypeSafeProvider) -> None:
    from sentinel.jev.questions import GUARD

    route = respx.post(API_URL).respond(
        200, json=api_response({"safe_to_run": {"type": "noul", "noul": 0.2}})
    )
    result = await provider.evaluate("s", GUARD)
    sent = json.loads(route.calls.last.request.content)["questions"]["safe_to_run"]
    assert sent["type"] == "noul"
    assert set(sent["criteria"]) == {"true", "false"}
    assert result.noul("safe_to_run").noul == 0.2


@respx.mock
async def test_model_override_goes_in_body(provider: TypeSafeProvider) -> None:
    route = respx.post(API_URL).respond(200, json=api_response())
    result = await provider.evaluate("s", QUESTIONS, model=MODEL_VERSION)
    assert json.loads(route.calls.last.request.content)["model"] == MODEL_VERSION
    assert result.model == MODEL_VERSION
    assert result.metadata == {}


@pytest.mark.parametrize("status", [401, 403])
@respx.mock
async def test_auth_errors_surface_vendor_message(provider: TypeSafeProvider, status: int) -> None:
    # Shape observed live for a bad key.
    route = respx.post(API_URL).respond(
        status, json=api_error("Invalid API key.", "authentication_error")
    )
    with pytest.raises(JevAuthError, match="Invalid API key") as exc_info:
        await provider.evaluate("s", QUESTIONS)
    assert route.call_count == 1
    assert FAKE_KEY not in str(exc_info.value)


@respx.mock
async def test_400_usage_error_is_a_validation_error(provider: TypeSafeProvider) -> None:
    # Shape observed live for an unknown model.
    route = respx.post(API_URL).respond(
        400, json=api_error("Unknown model: nope", "api_usage_error")
    )
    with pytest.raises(JevValidationError, match="Unknown model: nope"):
        await provider.evaluate("s", QUESTIONS)
    assert route.call_count == 1


@respx.mock
async def test_422_surfaces_field(provider: TypeSafeProvider) -> None:
    respx.post(API_URL).respond(
        422,
        json={"detail": [{"loc": ["body", "questions", "urgency"], "msg": "too few levels"}]},
    )
    with pytest.raises(JevValidationError, match="too few levels") as exc_info:
        await provider.evaluate("s", QUESTIONS)
    assert exc_info.value.field == "questions.urgency"


@pytest.mark.parametrize(("status", "error"), [(429, JevRateLimitError), (529, JevOverloadedError)])
@respx.mock
async def test_transient_statuses_retried_then_give_up(
    provider: TypeSafeProvider, status: int, error: type[Exception]
) -> None:
    route = respx.post(API_URL)
    route.side_effect = [httpx.Response(status), httpx.Response(200, json=api_response())]
    await provider.evaluate("s", QUESTIONS)
    assert route.call_count == 2

    route.side_effect = None
    route.return_value = httpx.Response(status, headers={"retry-after": "2"})
    with pytest.raises(error) as exc_info:
        await provider.evaluate("s", QUESTIONS)
    assert route.call_count == 2 + 4
    assert getattr(exc_info.value, "retry_after", None) == 2.0


@respx.mock
async def test_503_not_retried(provider: TypeSafeProvider) -> None:
    route = respx.post(API_URL).respond(503)
    with pytest.raises(Exception, match="HTTP 503"):
        await provider.evaluate("s", QUESTIONS)
    assert route.call_count == 1


def _retry_state(exc: BaseException) -> RetryCallState:
    state = RetryCallState(retry_object=None, fn=None, args=(), kwargs={})  # type: ignore[arg-type]
    state.set_exception((type(exc), exc, None))
    return state


@pytest.mark.parametrize(
    ("retry_after", "expected"),
    [(None, 0.5), (0.1, 0.5), (3.0, 3.0), (600.0, MAX_RETRY_AFTER_S)],
)
def test_wait_honours_retry_after_with_a_cap(retry_after: float | None, expected: float) -> None:
    wait = wait_retry_after(lambda _state: 0.5)  # type: ignore[arg-type]
    exc = JevRateLimitError("HTTP 429", status_code=429, retry_after=retry_after)
    assert wait(_retry_state(exc)) == expected


@pytest.mark.parametrize("header", ["soon", "-1", "Wed, 21 Oct 2026 07:28:00 GMT"])
@respx.mock
async def test_unusable_retry_after_ignored(provider: TypeSafeProvider, header: str) -> None:
    respx.post(API_URL).respond(429, headers={"retry-after": header})
    with pytest.raises(JevRateLimitError) as exc_info:
        await provider.evaluate("s", QUESTIONS)
    assert exc_info.value.retry_after is None


@respx.mock
async def test_state_and_key_never_logged(
    provider: TypeSafeProvider, capsys: pytest.CaptureFixture[str]
) -> None:
    from sentinel.log import configure_logging

    configure_logging("DEBUG")
    respx.post(API_URL).respond(200, json=api_response())
    await provider.evaluate("secret-state-text", QUESTIONS)
    err = capsys.readouterr().err
    assert "jev.evaluate" in err and MODEL_VERSION in err
    assert FAKE_KEY not in err and "secret-state-text" not in err


async def test_factory_default_is_typesafe() -> None:
    settings = Settings(jev_api_key=SecretStr(FAKE_KEY))
    p = build_provider(settings)
    assert isinstance(p, TypeSafeProvider)
    assert p._url == "https://api.typesafe.ai/v1/systemone"
    assert p.build_request("s", QUESTIONS).model == "jev-latest"
    await p.aclose()


def test_missing_key() -> None:
    with pytest.raises(JevConfigError, match="JEV_API_KEY"):
        build_provider(Settings())
