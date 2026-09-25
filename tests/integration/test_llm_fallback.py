from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest
import respx
from pydantic import SecretStr

from sentinel.config import Settings
from sentinel.jev.errors import JevConfigError, JevResponseError, JevValidationError
from sentinel.jev.llm_fallback import LLMFallbackProvider
from sentinel.llm.openai_client import OpenAIClient
from sentinel.llm.pricing import ModelPrice, PriceTable
from tests.fixtures.jev import QUESTIONS
from tests.fixtures.openai import (
    FAKE_OPENAI_MODEL,
    RESPONSES_URL,
    error_body,
    judge,
    labels_from_request,
    response_body,
)

FAKE_KEY = "fake-openai-key-for-tests"  # pragma: allowlist secret


@pytest.fixture
async def provider() -> AsyncIterator[LLMFallbackProvider]:
    llm = OpenAIClient(
        api_key=SecretStr(FAKE_KEY),
        models={"fast": "fake-fast", "strong": "fake-strong"},
        prices=PriceTable(prices={FAKE_OPENAI_MODEL: ModelPrice(input=1.0, output=1.0)}),
    )
    p = LLMFallbackProvider(llm, tier="fast")
    yield p
    await p.aclose()


def answering(**by_first_label: dict[str, Any]) -> Any:
    """Side effect returning a fixed payload per question, keyed by its first allowed answer."""

    def respond(request: httpx.Request) -> httpx.Response:
        labels = labels_from_request(request)
        payload = by_first_label["noul" if labels is None else labels[0]]
        return httpx.Response(200, json=response_body(json.dumps(payload)))

    return respond


@respx.mock
async def test_one_call_per_question(provider: LLMFallbackProvider) -> None:
    route = respx.post(RESPONSES_URL).mock(side_effect=judge)
    result = await provider.evaluate({"subject": "refund"}, QUESTIONS)

    assert route.call_count == len(QUESTIONS)
    assert result.provider == "llm_fallback"
    assert result.model == FAKE_OPENAI_MODEL
    assert result.model_verified is True
    assert result.usage is not None
    assert result.usage["input_tokens"] == 100 * len(QUESTIONS)
    assert result.usage["cost_usd"] == pytest.approx(120 * len(QUESTIONS) / 1_000_000)
    for call in route.calls:
        body = json.loads(call.request.content)
        assert body["model"] == "fake-fast"
        assert body["store"] is False
        assert '{"subject": "refund"}' in body["input"]
        assert "untrusted" in body["input"]


@respx.mock
async def test_derives_jev_semantics(provider: LLMFallbackProvider) -> None:
    respx.post(RESPONSES_URL).mock(
        side_effect=answering(
            billing={
                "distribution": [
                    {"answer": "technical", "probability": 0.6},
                    {"answer": "billing", "probability": 0.2},
                ]
            },
            **{
                "Not urgent": {
                    "distribution": [
                        {"answer": "Soon", "probability": 0.5},
                        {"answer": "Today", "probability": 0.5},
                    ]
                }
            },
            noul={"probability_yes": 0.25},
        )
    )
    result = await provider.evaluate("s", QUESTIONS)

    category = result.choice("category")
    assert category.choice == "technical"
    assert category.probabilities == pytest.approx(
        {"billing": 0.25, "technical": 0.75}
    )  # normalised
    assert category.confidence == pytest.approx(0.5)  # (2 * 0.75 - 1) / 1

    urgency = result.score("urgency")
    assert urgency.score == pytest.approx(1.5)
    assert urgency.legend == {"0": "Not urgent", "1": "Soon", "2": "Today"}
    assert urgency.probabilities == {"0": 0.0, "1": 0.5, "2": 0.5}  # omitted level -> 0
    assert urgency.confidence == pytest.approx(1 - 0.5 / (2 / 3))

    assert result.noul("is_abusive").noul == 0.25


@pytest.mark.parametrize(
    ("distribution", "match"),
    [
        (
            [{"answer": "billing", "probability": 0.5}, {"answer": "billing", "probability": 0.5}],
            "twice",
        ),
        ([{"answer": "billing", "probability": 0.0}], "sum to zero"),
        ([{"answer": "billing", "probability": 1.5}], "out of range"),
    ],
)
@respx.mock
async def test_bad_distribution_rejected(
    provider: LLMFallbackProvider, distribution: list[dict[str, Any]], match: str
) -> None:
    respx.post(RESPONSES_URL).mock(
        side_effect=answering(
            billing={"distribution": distribution},
            **{"Not urgent": {"distribution": [{"answer": "Soon", "probability": 1}]}},
            noul={"probability_yes": 0.1},
        )
    )
    with pytest.raises(JevResponseError, match=match):
        await provider.evaluate("s", QUESTIONS)


@respx.mock
async def test_noul_out_of_range_rejected(provider: LLMFallbackProvider) -> None:
    respx.post(RESPONSES_URL).respond(200, json=response_body('{"probability_yes": 1.2}'))
    with pytest.raises(JevResponseError, match="probability_yes"):
        await provider.evaluate("s", {"is_abusive": QUESTIONS["is_abusive"]})


@respx.mock
async def test_schema_constrains_answers(provider: LLMFallbackProvider) -> None:
    route = respx.post(RESPONSES_URL).mock(side_effect=judge)
    await provider.evaluate("s", {"category": QUESTIONS["category"]})
    assert labels_from_request(route.calls.last.request) == ["billing", "technical"]


@respx.mock
async def test_request_error_surfaces_field(provider: LLMFallbackProvider) -> None:
    respx.post(RESPONSES_URL).respond(400, json=error_body("Invalid schema", param="text.format"))
    with pytest.raises(JevValidationError) as exc_info:
        await provider.evaluate("s", QUESTIONS)
    assert exc_info.value.field == "text.format"


def test_requires_tier_model() -> None:
    settings = Settings(jev_provider="llm_fallback", openai_api_key=SecretStr(FAKE_KEY))
    with pytest.raises(JevConfigError, match="OPENAI_MODEL_FAST"):
        LLMFallbackProvider.from_settings(settings)


def test_requires_openai_key() -> None:
    settings = Settings(jev_provider="llm_fallback", openai_model_fast="m")
    with pytest.raises(JevConfigError, match="OPENAI_API_KEY"):
        LLMFallbackProvider.from_settings(settings)
