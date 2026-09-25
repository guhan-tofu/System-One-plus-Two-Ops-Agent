"""Application settings loaded from the environment / `.env`.

Secrets are held as `SecretStr` so they never appear in reprs, logs or tracebacks.
Read them only at the point of use via `.get_secret_value()`.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

JevProviderName = Literal["vercel", "typesafe", "llm_fallback"]
LLMTier = Literal["fast", "strong"]
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Default: official TypeSafe Jev via Vercel AI Gateway.
    ai_gateway_api_key: SecretStr = SecretStr("")
    ai_gateway_base_url: str = "https://ai-gateway.vercel.sh/v4/ai"
    jev_gateway_model: str = "typesafe-ai/jev"

    jev_provider: JevProviderName = "vercel"

    openai_api_key: SecretStr = SecretStr("")
    openai_model_fast: str = ""
    openai_model_strong: str = ""
    model_tiers_path: Path = Path("policies/model_tiers.yaml")
    """Per-model token prices used for cost accounting."""

    jev_fallback_tier: LLMTier = "fast"
    """OpenAI tier used when JEV_PROVIDER=llm_fallback emulates Jev."""
    jev_max_rps: float | None = 2.0
    """Client-side cap on Jev requests/second (Vercel free tier throttles ~3 req/s)."""
    jev_breaker_failures: int = 5
    """Consecutive transient Jev failures that open the circuit breaker."""
    jev_breaker_reset_s: float = 30.0
    """How long an open circuit refuses calls before one trial request."""
    api_max_items_per_minute: int = 60
    """POST /items limit for the HTTP API (429 above it)."""
    jev_on_failure: Literal["human", "llm_fallback"] = "human"
    """When the Jev provider fails: escalate to a human, or retry on llm_fallback."""

    thresholds_path: Path = Path("policies/thresholds.yaml")
    tools_policy_path: Path = Path("policies/tools.yaml")

    database_url: str = "sqlite:///./sentinel.db"

    sentinel_api_token: SecretStr = SecretStr("")
    """Bearer token for the HTTP API (SENTINEL_API_TOKEN). Required to serve."""
    log_level: LogLevel = "INFO"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
