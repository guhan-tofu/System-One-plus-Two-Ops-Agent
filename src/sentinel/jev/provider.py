"""The `JevProvider` interface and the factory that picks an implementation.

Everything that talks to System One goes through here, so the provider can be
swapped with a config change: the official TypeSafe Jev API (default), the same
Jev via Vercel AI Gateway (`vercel`), or OpenAI emulation (`llm_fallback`).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol, runtime_checkable

from sentinel.config import Settings
from sentinel.jev.models import JevResult, Question, State


@runtime_checkable
class JevProvider(Protocol):
    name: str

    async def evaluate(
        self,
        state: State,
        questions: Mapping[str, Question],
        model: str | None = None,
    ) -> JevResult:
        """Answer every question about `state`. Raises `JevError` on failure."""
        ...

    async def aclose(self) -> None: ...


def build_provider(settings: Settings) -> JevProvider:
    match settings.jev_provider:
        case "typesafe":
            from sentinel.jev.typesafe import TypeSafeProvider

            return TypeSafeProvider.from_settings(settings)
        case "vercel":
            from sentinel.jev.vercel import VercelJevProvider

            return VercelJevProvider.from_settings(settings)
        case "llm_fallback":
            from sentinel.jev.llm_fallback import LLMFallbackProvider

            return LLMFallbackProvider.from_settings(settings)
