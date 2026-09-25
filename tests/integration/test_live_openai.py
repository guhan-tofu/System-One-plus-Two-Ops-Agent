"""Real calls to OpenAI. Run with: uv run pytest --run-live -m live"""

from __future__ import annotations

import pytest

from sentinel.config import Settings, get_settings
from sentinel.jev.llm_fallback import LLMFallbackProvider
from sentinel.jev.questions import TRIAGE
from sentinel.llm.openai_client import OpenAIClient


def live_settings(monkeypatch: pytest.MonkeyPatch) -> Settings:
    monkeypatch.undo()  # use the developer's real .env / environment for live runs
    get_settings.cache_clear()
    settings = Settings()
    if not settings.openai_api_key.get_secret_value() or not settings.openai_model_fast:
        pytest.skip("OPENAI_API_KEY / OPENAI_MODEL_FAST not set")
    return settings


@pytest.mark.live
async def test_live_generate(monkeypatch: pytest.MonkeyPatch) -> None:
    llm = OpenAIClient.from_settings(live_settings(monkeypatch))
    try:
        gen = await llm.generate(
            "fast", instructions="Reply with one short sentence.", input="Greet a test user."
        )
    finally:
        await llm.aclose()
    assert gen.text
    assert gen.model
    assert gen.usage.input_tokens > 0 and gen.usage.output_tokens > 0


@pytest.mark.live
async def test_live_fallback_triage(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = live_settings(monkeypatch).model_copy(update={"jev_fallback_tier": "fast"})
    provider = LLMFallbackProvider.from_settings(settings)
    try:
        result = await provider.evaluate(
            "Test ticket: I was charged twice for my subscription this month.", TRIAGE
        )
    finally:
        await provider.aclose()
    assert result.provider == "llm_fallback"
    assert result.model_verified
    assert set(result.answers) == set(TRIAGE)
    assert result.choice("category").choice == "billing"
