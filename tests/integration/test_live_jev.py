"""Real calls to thejevai.com. Run with: uv run pytest --run-live -m live"""

from __future__ import annotations

import pytest

from sentinel.config import get_settings
from sentinel.jev.questions import TRIAGE
from sentinel.jev.thejevai import TheJevAIProvider


@pytest.mark.live
async def test_live_triage_roundtrip(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.undo()  # use the developer's real .env for live runs
    from sentinel.config import Settings

    get_settings.cache_clear()
    settings = Settings()
    if not settings.jev_api_key.get_secret_value():
        pytest.skip("JEV_API_KEY not set")
    provider = TheJevAIProvider.from_settings(settings)
    try:
        result = await provider.evaluate(
            "Test ticket: I was charged twice for my subscription this month.", TRIAGE
        )
    finally:
        await provider.aclose()
    assert result.model
    assert set(result.answers) == set(TRIAGE)
    assert result.shape == "answers"
