from __future__ import annotations

import json
from collections.abc import AsyncIterator

import httpx
import pytest
import respx
from pydantic import BaseModel, SecretStr

from sentinel.config import Settings
from sentinel.llm.openai_client import (
    DEFAULT_TIMEOUT,
    LLMAuthError,
    LLMConfigError,
    LLMRateLimitError,
    LLMRequestError,
    LLMResponseError,
    LLMServerError,
    LLMTimeoutError,
    OpenAIClient,
)
from sentinel.llm.pricing import ModelPrice, PriceTable
from tests.fixtures.openai import (
    FAKE_OPENAI_MODEL,
    FAST_RETRY,
    RESPONSES_URL,
    error_body,
    response_body,
)

FAKE_KEY = "fake-openai-key-for-tests"  # pragma: allowlist secret
MODELS = {"fast": "fake-fast", "strong": "fake-strong"}


class Verdict(BaseModel):
    ok: bool


@pytest.fixture
async def llm() -> AsyncIterator[OpenAIClient]:
    client = OpenAIClient(
        api_key=SecretStr(FAKE_KEY),
        models=MODELS,  # type: ignore[arg-type]
        prices=PriceTable(prices={FAKE_OPENAI_MODEL: ModelPrice(input=2.0, output=10.0)}),
    )
    yield client
    await client.aclose()


@respx.mock
async def test_generate_uses_tier_model_and_accounts_tokens(llm: OpenAIClient) -> None:
    route = respx.post(RESPONSES_URL).respond(
        200, json=response_body("Hello!", input_tokens=1000, output_tokens=500)
    )
    gen = await llm.generate("strong", instructions="Be brief.", input="Say hi")

    assert gen.text == "Hello!"
    assert gen.tier == "strong"
    assert gen.requested_model == "fake-strong"
    assert gen.model == FAKE_OPENAI_MODEL
    assert gen.usage.input_tokens == 1000 and gen.usage.output_tokens == 500
    assert gen.cost_usd == pytest.approx((1000 * 2.0 + 500 * 10.0) / 1_000_000)
    assert gen.latency_ms >= 0

    sent = route.calls.last.request
    assert sent.headers["Authorization"] == f"Bearer {FAKE_KEY}"
    body = json.loads(sent.content)
    assert body["model"] == "fake-strong"
    assert body["store"] is False
    assert body["instructions"] == "Be brief."
    assert body["input"] == "Say hi"


@respx.mock
async def test_unpriced_model_cost_is_none(llm: OpenAIClient) -> None:
    respx.post(RESPONSES_URL).respond(200, json=response_body("x", model="other-model"))
    gen = await llm.generate("fast", instructions="i", input="x")
    assert gen.cost_usd is None


@respx.mock
async def test_max_output_tokens_passed(llm: OpenAIClient) -> None:
    route = respx.post(RESPONSES_URL).respond(200, json=response_body("x"))
    await llm.generate("fast", instructions="i", input="x", max_output_tokens=50)
    assert json.loads(route.calls.last.request.content)["max_output_tokens"] == 50


@respx.mock
async def test_structured_output(llm: OpenAIClient) -> None:
    route = respx.post(RESPONSES_URL).respond(200, json=response_body('{"ok": true}'))
    parsed, gen = await llm.structured("fast", instructions="i", input="x", schema=Verdict)
    assert parsed == Verdict(ok=True)
    assert gen.model == FAKE_OPENAI_MODEL
    fmt = json.loads(route.calls.last.request.content)["text"]["format"]
    assert fmt["type"] == "json_schema" and fmt["strict"] is True


@respx.mock
async def test_structured_output_schema_mismatch(llm: OpenAIClient) -> None:
    respx.post(RESPONSES_URL).respond(200, json=response_body('{"ok": "maybe"}'))
    with pytest.raises(LLMResponseError, match="schema"):
        await llm.structured("fast", instructions="i", input="x", schema=Verdict)


@respx.mock
async def test_refusal(llm: OpenAIClient) -> None:
    respx.post(RESPONSES_URL).respond(200, json=response_body("", refusal=True))
    with pytest.raises(LLMResponseError, match="refused"):
        await llm.structured("fast", instructions="i", input="x", schema=Verdict)


@respx.mock
async def test_incomplete_response(llm: OpenAIClient) -> None:
    respx.post(RESPONSES_URL).respond(200, json=response_body("Hel", status="incomplete"))
    with pytest.raises(LLMResponseError, match="max_output_tokens"):
        await llm.generate("fast", instructions="i", input="x")


async def test_unconfigured_tier() -> None:
    client = OpenAIClient(api_key=SecretStr(FAKE_KEY), models={"fast": "m"})
    with pytest.raises(LLMConfigError, match="OPENAI_MODEL_STRONG"):
        await client.generate("strong", instructions="i", input="x")
    await client.aclose()


def test_missing_key() -> None:
    with pytest.raises(LLMConfigError):
        OpenAIClient(api_key=SecretStr(""), models=MODELS)  # type: ignore[arg-type]


@respx.mock
async def test_401_not_retried(llm: OpenAIClient) -> None:
    route = respx.post(RESPONSES_URL).respond(401, json=error_body("Incorrect API key"))
    with pytest.raises(LLMAuthError) as exc_info:
        await llm.generate("fast", instructions="i", input="x")
    assert route.call_count == 1
    assert FAKE_KEY not in str(exc_info.value)


@respx.mock
async def test_400_surfaces_param_not_retried(llm: OpenAIClient) -> None:
    route = respx.post(RESPONSES_URL).respond(
        400, json=error_body("Unsupported value", param="temperature")
    )
    with pytest.raises(LLMRequestError) as exc_info:
        await llm.generate("fast", instructions="i", input="x")
    assert exc_info.value.param == "temperature"
    assert "Unsupported value" in str(exc_info.value)
    assert route.call_count == 1


@respx.mock
async def test_429_retried_then_success(llm: OpenAIClient) -> None:
    route = respx.post(RESPONSES_URL)
    route.side_effect = [
        httpx.Response(429, headers=FAST_RETRY, json=error_body("slow down")),
        httpx.Response(200, json=response_body("ok")),
    ]
    assert (await llm.generate("fast", instructions="i", input="x")).text == "ok"
    assert route.call_count == 2


@respx.mock
async def test_429_gives_up_after_4_attempts(llm: OpenAIClient) -> None:
    route = respx.post(RESPONSES_URL).respond(429, headers=FAST_RETRY, json=error_body("slow"))
    with pytest.raises(LLMRateLimitError):
        await llm.generate("fast", instructions="i", input="x")
    assert route.call_count == 4


@respx.mock
async def test_500_gives_up(llm: OpenAIClient) -> None:
    respx.post(RESPONSES_URL).respond(500, headers=FAST_RETRY, json=error_body("boom"))
    with pytest.raises(LLMServerError) as exc_info:
        await llm.generate("fast", instructions="i", input="x")
    assert exc_info.value.status_code == 500


@respx.mock
async def test_timeout(llm: OpenAIClient, monkeypatch: pytest.MonkeyPatch) -> None:
    import anyio

    async def no_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr(anyio, "sleep", no_sleep)
    respx.post(RESPONSES_URL).mock(side_effect=httpx.ReadTimeout("slow"))
    with pytest.raises(LLMTimeoutError):
        await llm.generate("fast", instructions="i", input="x")


@respx.mock
async def test_prompt_and_key_never_logged(
    llm: OpenAIClient, capsys: pytest.CaptureFixture[str]
) -> None:
    from sentinel.log import configure_logging

    configure_logging("DEBUG")
    respx.post(RESPONSES_URL).respond(200, json=response_body("secret-output-text"))
    await llm.generate("fast", instructions="secret-instructions", input="secret-state-text")
    err = capsys.readouterr().err
    assert "llm.generate" in err and FAKE_OPENAI_MODEL in err
    for secret in (FAKE_KEY, "secret-instructions", "secret-state-text", "secret-output-text"):
        assert secret not in err


def test_timeouts_configured() -> None:
    assert DEFAULT_TIMEOUT.connect == 5.0


async def test_from_settings_reads_tiers_and_prices(tmp_path: object) -> None:
    settings = Settings(
        openai_api_key=SecretStr(FAKE_KEY),
        openai_model_fast="fake-fast",
        openai_model_strong="fake-strong",
    )
    client = OpenAIClient.from_settings(settings)
    assert client.model_for("fast") == "fake-fast"
    assert client.model_for("strong") == "fake-strong"
    await client.aclose()
