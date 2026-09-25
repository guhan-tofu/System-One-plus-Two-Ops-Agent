"""Token prices (USD per 1M tokens) loaded from policies/model_tiers.yaml."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

from sentinel.log import get_logger

log = get_logger(__name__)

PER_TOKENS = 1_000_000


class ModelPrice(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    input: float = Field(ge=0)
    output: float = Field(ge=0)
    cached_input: float | None = Field(default=None, ge=0)


class TokenUsage(BaseModel):
    model_config = ConfigDict(frozen=True)

    input_tokens: int = 0
    cached_input_tokens: int = 0
    """Subset of `input_tokens` served from the prompt cache."""
    output_tokens: int = 0
    reasoning_tokens: int = 0
    """Subset of `output_tokens` spent on hidden reasoning."""

    def __add__(self, other: TokenUsage) -> TokenUsage:
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            cached_input_tokens=self.cached_input_tokens + other.cached_input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
        )


class PriceTable(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    prices: dict[str, ModelPrice] = Field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> PriceTable:
        if not path.is_file():
            log.warning("llm.prices_missing", path=str(path))
            return cls()
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return cls.model_validate({"prices": data.get("prices") or {}})

    def lookup(self, *models: str) -> ModelPrice | None:
        for model in models:
            if model in self.prices:
                return self.prices[model]
        return None

    def cost(self, usage: TokenUsage, *models: str) -> float | None:
        """Cost in USD, or None if none of `models` has a price."""
        price = self.lookup(*models)
        if price is None:
            return None
        cached_rate = price.input if price.cached_input is None else price.cached_input
        uncached = usage.input_tokens - usage.cached_input_tokens
        total = (
            uncached * price.input
            + usage.cached_input_tokens * cached_rate
            + usage.output_tokens * price.output
        )
        return total / PER_TOKENS
