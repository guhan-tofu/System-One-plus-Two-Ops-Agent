"""Real Jev calls. Run with: uv run pytest --run-live -m live"""

from __future__ import annotations

import pytest

from sentinel.config import Settings, get_settings
from sentinel.jev.provider import build_provider
from sentinel.jev.questions import TRIAGE


def live_settings(monkeypatch: pytest.MonkeyPatch, provider: str) -> Settings:
    monkeypatch.undo()  # use the developer's real .env / environment for live runs
    get_settings.cache_clear()
    settings = Settings(jev_provider=provider)  # type: ignore[arg-type]
    if not settings.ai_gateway_api_key.get_secret_value():
        pytest.skip(f"no key configured for {provider}")
    return settings


@pytest.mark.live
@pytest.mark.parametrize("provider", ["vercel"])
async def test_live_triage_roundtrip(monkeypatch: pytest.MonkeyPatch, provider: str) -> None:
    jev = build_provider(live_settings(monkeypatch, provider))
    try:
        result = await jev.evaluate(
            "Test ticket: I was charged twice for my subscription this month.", TRIAGE
        )
    finally:
        await jev.aclose()
    assert result.provider == provider
    assert result.model
    assert set(result.answers) == set(TRIAGE)
    assert result.choice("category").choice == "billing"
