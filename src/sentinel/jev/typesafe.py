"""Placeholder for the official TypeSafe Jev API. Implemented once we have access."""

from __future__ import annotations

from collections.abc import Mapping

from sentinel.config import Settings
from sentinel.jev.errors import JevConfigError
from sentinel.jev.models import JevResult, Question, State


class TypeSafeProvider:
    name = "typesafe"

    @classmethod
    def from_settings(cls, settings: Settings) -> TypeSafeProvider:
        raise JevConfigError("typesafe provider is not implemented yet")

    async def evaluate(
        self,
        state: State,
        questions: Mapping[str, Question],
        model: str | None = None,
    ) -> JevResult:
        raise JevConfigError("typesafe provider is not implemented yet")

    async def aclose(self) -> None:
        return None
