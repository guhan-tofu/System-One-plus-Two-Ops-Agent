"""Provider contract: the same tests pass whichever JEV_PROVIDER is configured.

Each backend knows how to fake its own wire format; the assertions below never
look at provider-specific details.
"""

from __future__ import annotations

import math
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass
from typing import Any

import httpx
import pytest
import respx
from tenacity import wait_none

from sentinel.config import Settings, get_settings
from sentinel.jev import thejevai
from sentinel.jev.errors import (
    JevAuthError,
    JevConfigError,
    JevRateLimitError,
    JevResponseError,
    JevTimeoutError,
    JevValidationError,
)
from sentinel.jev.models import ChoiceQuestion, JevResult, ScoreQuestion
from sentinel.jev.provider import JevProvider, build_provider
from tests.fixtures.jev import QUESTIONS, confirmed_response
from tests.fixtures.openai import FAST_RETRY, RESPONSES_URL, error_body, judge, response_body

JEV_URL = "https://jev.test/v1/systemone"
FAKE_KEY = "fake-key-for-contract-tests"  # pragma: allowlist secret


@dataclass(frozen=True)
class Backend:
    name: str
    url: str
    env: dict[str, str]
    key_var: str

    def ok(self) -> Any:
        if self.name == "thejevai":
            return httpx.Response(200, json=confirmed_response())
        return judge

    def status(self, code: int) -> httpx.Response:
        if self.name == "thejevai":
            return httpx.Response(code)
        return httpx.Response(code, headers=FAST_RETRY, json=error_body("error"))

    def invalid(self, field: str) -> httpx.Response:
        if self.name == "thejevai":
            return httpx.Response(422, json={"error": {"message": "bad", "param": field}})
        return httpx.Response(400, json=error_body("bad", param=field))

    def garbage(self) -> httpx.Response:
        if self.name == "thejevai":
            return httpx.Response(200, text="<html>")
        return httpx.Response(200, json=response_body("not json"))


BACKENDS = {
    "thejevai": Backend(
        name="thejevai",
        url=JEV_URL,
        env={"JEV_API_KEY": FAKE_KEY, "JEV_BASE_URL": JEV_URL},
        key_var="JEV_API_KEY",
    ),
    "llm_fallback": Backend(
        name="llm_fallback",
        url=RESPONSES_URL,
        env={"OPENAI_API_KEY": FAKE_KEY, "OPENAI_MODEL_FAST": "fake-fast"},
        key_var="OPENAI_API_KEY",
    ),
}


@pytest.fixture(params=sorted(BACKENDS))
def backend(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> Iterator[Backend]:
    b = BACKENDS[request.param]
    monkeypatch.setenv("JEV_PROVIDER", b.name)
    for var, value in b.env.items():
        monkeypatch.setenv(var, value)
    monkeypatch.setattr(thejevai, "default_retry_wait", wait_none)
    get_settings.cache_clear()
    yield b


@pytest.fixture
async def provider(backend: Backend) -> AsyncIterator[JevProvider]:
    p = build_provider(get_settings())
    yield p
    await p.aclose()


def assert_valid(result: JevResult, provider_name: str) -> None:
    assert result.provider == provider_name
    assert result.model
    assert result.latency_ms >= 0
    assert set(result.answers) == set(QUESTIONS)

    category_q = QUESTIONS["category"]
    assert isinstance(category_q, ChoiceQuestion)
    category = result.choice("category")
    assert category.choice in category_q.criteria
    assert set(category.probabilities) <= set(category_q.criteria)
    assert math.isclose(sum(category.probabilities.values()), 1.0, abs_tol=0.02)
    assert 0.0 <= category.confidence <= 1.0

    urgency_q = QUESTIONS["urgency"]
    assert isinstance(urgency_q, ScoreQuestion)
    urgency = result.score("urgency")
    levels = len(urgency_q.criteria)
    assert 0.0 <= urgency.score <= levels - 1
    assert urgency.legend == {str(i): lvl for i, lvl in enumerate(urgency_q.criteria)}
    assert math.isclose(sum(urgency.probabilities.values()), 1.0, abs_tol=0.02)
    assert 0.0 <= urgency.confidence <= 1.0

    assert 0.0 <= result.noul("is_abusive").noul <= 1.0


@respx.mock
async def test_all_answer_types(backend: Backend, provider: JevProvider) -> None:
    respx.post(backend.url).mock(side_effect=backend.ok())
    assert isinstance(provider, JevProvider)
    assert provider.name == backend.name
    assert_valid(await provider.evaluate({"subject": "refund"}, QUESTIONS), backend.name)


@respx.mock
async def test_auth_failure_not_retried(backend: Backend, provider: JevProvider) -> None:
    route = respx.post(backend.url).mock(return_value=backend.status(401))
    with pytest.raises(JevAuthError) as exc_info:
        await provider.evaluate("s", QUESTIONS)
    assert FAKE_KEY not in str(exc_info.value)
    assert route.call_count <= len(QUESTIONS)  # one attempt per request, no retries


@respx.mock
async def test_invalid_request_surfaces_field(backend: Backend, provider: JevProvider) -> None:
    respx.post(backend.url).mock(return_value=backend.invalid("questions.urgency"))
    with pytest.raises(JevValidationError) as exc_info:
        await provider.evaluate("s", QUESTIONS)
    assert exc_info.value.field == "questions.urgency"


@respx.mock
async def test_rate_limit_retried_then_success(backend: Backend, provider: JevProvider) -> None:
    ok = backend.ok()
    calls = {"n": 0}

    def flaky(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] <= 2:
            return backend.status(429)
        return ok(request) if callable(ok) else ok

    respx.post(backend.url).mock(side_effect=flaky)
    assert_valid(await provider.evaluate("s", QUESTIONS), backend.name)


@respx.mock
async def test_rate_limit_gives_up(backend: Backend, provider: JevProvider) -> None:
    respx.post(backend.url).mock(return_value=backend.status(429))
    with pytest.raises(JevRateLimitError):
        await provider.evaluate("s", QUESTIONS)


@respx.mock
async def test_timeout(
    backend: Backend, provider: JevProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    import anyio

    async def no_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr(anyio, "sleep", no_sleep)
    respx.post(backend.url).mock(side_effect=httpx.ReadTimeout("slow"))
    with pytest.raises(JevTimeoutError):
        await provider.evaluate("s", QUESTIONS)


@respx.mock
async def test_unparseable_response_never_defaulted(
    backend: Backend, provider: JevProvider
) -> None:
    respx.post(backend.url).mock(return_value=backend.garbage())
    with pytest.raises(JevResponseError):
        await provider.evaluate("s", QUESTIONS)


def test_missing_key(backend: Backend, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(backend.key_var)
    get_settings.cache_clear()
    with pytest.raises(JevConfigError):
        build_provider(Settings())
