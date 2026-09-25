from __future__ import annotations

import pytest
from pydantic import ValidationError

from sentinel.config import Settings, get_settings


def test_defaults() -> None:
    s = Settings()
    assert s.jev_provider == "vercel"
    assert s.jev_gateway_model == "typesafe-ai/jev"
    assert s.ai_gateway_api_key.get_secret_value() == ""
    assert s.database_url.startswith("sqlite")
    assert s.jev_fallback_tier == "fast"
    assert str(s.model_tiers_path) == "policies/model_tiers.yaml"


def test_reads_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JEV_PROVIDER", "llm_fallback")
    monkeypatch.setenv("OPENAI_MODEL_FAST", "fake-fast-model")
    s = get_settings()
    assert s.jev_provider == "llm_fallback"
    assert s.openai_model_fast == "fake-fast-model"


@pytest.mark.parametrize("name", ["mystery", "thejevai"])
def test_rejects_unknown_provider(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    monkeypatch.setenv("JEV_PROVIDER", name)
    with pytest.raises(ValidationError):
        Settings()


def test_secrets_not_in_repr(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = "fake-test-value-not-a-real-key"  # pragma: allowlist secret
    monkeypatch.setenv("AI_GATEWAY_API_KEY", fake)
    monkeypatch.setenv("OPENAI_API_KEY", fake)
    s = Settings()
    assert fake not in repr(s)
    assert fake not in str(s.model_dump())
    assert s.ai_gateway_api_key.get_secret_value() == fake
