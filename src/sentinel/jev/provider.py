"""The `JevProvider` interface and the factory that picks an implementation.

Everything that talks to System One goes through here, so the vendor can be
swapped (Vercel AI Gateway -> thejevai -> OpenAI emulation) with a config change.
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
        case "vercel":
            from sentinel.jev.vercel import VercelJevProvider

            return VercelJevProvider.from_settings(settings)
        case "thejevai":
            from sentinel.jev.thejevai import TheJevAIProvider

            return TheJevAIProvider.from_settings(settings)
        case "typesafe":
            from sentinel.jev.typesafe import TypeSafeProvider

            return TypeSafeProvider.from_settings(settings)
        case "llm_fallback":
            from sentinel.jev.llm_fallback import LLMFallbackProvider

            return LLMFallbackProvider.from_settings(settings)
