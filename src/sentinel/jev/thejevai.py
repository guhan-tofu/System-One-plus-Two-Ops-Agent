"""HTTP provider for thejevai.com (an independent, untrusted reseller of Jev).

Kept as a fallback; the default is the official Jev via Vercel AI Gateway
(`sentinel.jev.vercel`). Wire format: `sentinel.jev.parsing`.
"""

from __future__ import annotations

from typing import Any, ClassVar

import httpx

from sentinel.config import Settings
from sentinel.jev.http import DEFAULT_TIMEOUT, MAX_ATTEMPTS, HTTPJevProvider
from sentinel.jev.models import JevRequest, JevResult
from sentinel.jev.parsing import describe_shape, parse_response

__all__ = ["DEFAULT_TIMEOUT", "MAX_ATTEMPTS", "TheJevAIProvider"]


class TheJevAIProvider(HTTPJevProvider):
    name = "thejevai"
    key_env: ClassVar[str] = "JEV_API_KEY"

    @classmethod
    def from_settings(
        cls, settings: Settings, client: httpx.AsyncClient | None = None
    ) -> TheJevAIProvider:
        return cls(
            api_key=settings.jev_api_key,
            url=settings.jev_base_url,
            default_model=settings.jev_model,
            client=client,
        )

    def payload(self, request: JevRequest) -> dict[str, Any]:
        return request.to_payload()

    def parse(self, raw: Any, request: JevRequest, *, latency_ms: float) -> JevResult:
        return parse_response(
            raw, request.questions, requested_model=request.model, latency_ms=latency_ms
        )

    def describe_shape(self, raw: Any) -> dict[str, Any]:
        return describe_shape(raw)
