"""Resilience wrappers for any JevProvider: client-side rate limit and circuit breaker.

Wrapping order in the pipeline: CircuitBreaker(RateLimited(provider)), so an open
circuit fails fast without waiting for a rate-limit slot.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Mapping
from typing import Literal

from sentinel.jev.errors import (
    JevError,
    JevHTTPError,
    JevRetryableError,
    JevTimeoutError,
    JevTransportError,
)
from sentinel.jev.models import JevResult, Question, State
from sentinel.jev.provider import JevProvider
from sentinel.log import get_logger

log = get_logger(__name__)

Clock = Callable[[], float]


class JevCircuitOpenError(JevError):
    """The provider failed repeatedly; calls are refused until the cool-down ends."""


def is_transient(exc: JevError) -> bool:
    """Failures that waiting may fix (vs. bad requests, auth or config problems)."""
    if isinstance(exc, JevRetryableError | JevTimeoutError | JevTransportError):
        return True
    return isinstance(exc, JevHTTPError) and exc.status_code >= 500


class RateLimitedJev:
    """Spaces request starts to at most `rate` per second (shared by all callers)."""

    def __init__(self, inner: JevProvider, *, rate: float, clock: Clock = time.monotonic) -> None:
        if rate <= 0:
            raise ValueError("rate must be positive")
        self.inner = inner
        self.name = inner.name
        self._interval = 1 / rate
        self._clock = clock
        self._lock = asyncio.Lock()
        self._next = 0.0

    async def evaluate(
        self, state: State, questions: Mapping[str, Question], model: str | None = None
    ) -> JevResult:
        async with self._lock:
            now = self._clock()
            wait = self._next - now
            if wait > 0:
                await asyncio.sleep(wait)
            self._next = max(now, self._next) + self._interval
        return await self.inner.evaluate(state, questions, model)

    async def aclose(self) -> None:
        await self.inner.aclose()


class CircuitBreakerJev:
    """closed -> (N consecutive transient failures) -> open -> (cool-down) -> half-open.

    Half-open lets one trial call through: success closes the circuit, failure
    re-opens it for another cool-down.
    """

    def __init__(
        self,
        inner: JevProvider,
        *,
        failure_threshold: int = 5,
        reset_after_s: float = 30.0,
        clock: Clock = time.monotonic,
    ) -> None:
        if failure_threshold < 1:
            raise ValueError("failure_threshold must be >= 1")
        self.inner = inner
        self.name = inner.name
        self._threshold = failure_threshold
        self._reset_after = reset_after_s
        self._clock = clock
        self._failures = 0
        self._opened_at: float | None = None
        self._trial_in_flight = False

    @property
    def state(self) -> Literal["closed", "open", "half_open"]:
        if self._opened_at is None:
            return "closed"
        if self._clock() - self._opened_at >= self._reset_after:
            return "half_open"
        return "open"

    async def evaluate(
        self, state: State, questions: Mapping[str, Question], model: str | None = None
    ) -> JevResult:
        current = self.state
        if current == "open" or (current == "half_open" and self._trial_in_flight):
            raise JevCircuitOpenError(f"{self.name}: circuit open after repeated failures")
        trial = current == "half_open"
        self._trial_in_flight = trial
        try:
            result = await self.inner.evaluate(state, questions, model)
        except JevError as exc:
            if is_transient(exc):
                self._record_failure(trial)
            elif trial:
                self._trial_in_flight = False
            raise
        self._close()
        return result

    def _record_failure(self, trial: bool) -> None:
        self._failures += 1
        self._trial_in_flight = False
        if trial or self._failures >= self._threshold:
            if self._opened_at is None or trial:
                log.warning("jev.circuit_open", provider=self.name, failures=self._failures)
            self._opened_at = self._clock()

    def _close(self) -> None:
        if self._opened_at is not None:
            log.info("jev.circuit_closed", provider=self.name)
        self._failures = 0
        self._opened_at = None
        self._trial_in_flight = False

    async def aclose(self) -> None:
        await self.inner.aclose()


def harden(
    provider: JevProvider,
    *,
    max_rps: float | None,
    failure_threshold: int,
    reset_after_s: float,
) -> JevProvider:
    inner: JevProvider = provider if max_rps is None else RateLimitedJev(provider, rate=max_rps)
    return CircuitBreakerJev(
        inner, failure_threshold=failure_threshold, reset_after_s=reset_after_s
    )
