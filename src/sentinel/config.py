"""Application settings loaded from the environment / `.env`.

Secrets are held as `SecretStr` so they never appear in reprs, logs or tracebacks.
Read them only at the point of use via `.get_secret_value()`.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

JevProviderName = Literal["thejevai", "typesafe", "llm_fallback"]
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    jev_api_key: SecretStr = SecretStr("")
    jev_base_url: str = "https://thejevai.com/v1/systemone"
    jev_model: str = "jev-latest"
    jev_provider: JevProviderName = "thejevai"

    openai_api_key: SecretStr = SecretStr("")
    openai_model_fast: str = ""
    openai_model_strong: str = ""

    database_url: str = "sqlite:///./sentinel.db"
    log_level: LogLevel = "INFO"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
