from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any

import pytest

from sentinel.jev.errors import (
    JevAuthError,
    JevError,
    JevHTTPError,
    JevOverloadedError,
    JevTimeoutError,
    JevValidationError,
)
from sentinel.jev.models import JevResult, Question, State
from sentinel.jev.resilience import (
    CircuitBreakerJev,
    JevCircuitOpenError,
    RateLimitedJev,
    harden,
    is_transient,
)
from tests.fixtures.pipeline import jev_result, triage_answers


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class Flaky:
    """Raises the queued errors in order, then succeeds."""

    name = "flaky"

    def __init__(self, *errors: JevError) -> None:
        self.errors = list(errors)
        self.calls = 0

    async def evaluate(
        self, state: State, questions: Mapping[str, Question], model: str | None = None
    ) -> JevResult:
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        return jev_result(triage_answers())

    async def aclose(self) -> None:
        return None


OVERLOADED = JevOverloadedError("HTTP 529", status_code=529)


async def call(jev: Any) -> JevResult:
    result: JevResult = await jev.evaluate("s", {})
    return result


@pytest.mark.parametrize(
    ("exc", "transient"),
    [
        (OVERLOADED, True),
        (JevTimeoutError("t"), True),
        (JevHTTPError("HTTP 503", status_code=503), True),
        (JevHTTPError("HTTP 402", status_code=402), False),
        (JevAuthError("401"), False),
        (JevValidationError("422"), False),
    ],
)
def test_transient_classification(exc: JevError, transient: bool) -> None:
    assert is_transient(exc) is transient


async def test_breaker_opens_after_threshold_and_fails_fast() -> None:
    clock = Clock()
    inner = Flaky(OVERLOADED, OVERLOADED, OVERLOADED)
    breaker = CircuitBreakerJev(inner, failure_threshold=3, reset_after_s=30, clock=clock)

    for _ in range(3):
        with pytest.raises(JevOverloadedError):
            await call(breaker)
    assert breaker.state == "open"

    with pytest.raises(JevCircuitOpenError):
        await call(breaker)
    assert inner.calls == 3  # the open circuit did not touch the provider


async def test_half_open_trial_success_closes() -> None:
    clock = Clock()
    inner = Flaky(OVERLOADED)
    breaker = CircuitBreakerJev(inner, failure_threshold=1, reset_after_s=30, clock=clock)
    with pytest.raises(JevOverloadedError):
        await call(breaker)
    clock.now += 30
    assert breaker.state == "half_open"
    await call(breaker)
    assert breaker.state == "closed"


async def test_half_open_trial_failure_reopens() -> None:
    clock = Clock()
    inner = Flaky(OVERLOADED, OVERLOADED)
    breaker = CircuitBreakerJev(inner, failure_threshold=1, reset_after_s=30, clock=clock)
    with pytest.raises(JevOverloadedError):
        await call(breaker)
    clock.now += 30
    with pytest.raises(JevOverloadedError):
        await call(breaker)
    assert breaker.state == "open"
    clock.now += 29
    with pytest.raises(JevCircuitOpenError):
        await call(breaker)


async def test_only_one_trial_in_half_open() -> None:
    clock = Clock()
    gate = asyncio.Event()

    class Slow(Flaky):
        async def evaluate(self, *args: Any, **kwargs: Any) -> JevResult:
            await gate.wait()
            return await super().evaluate(*args, **kwargs)

    breaker = CircuitBreakerJev(Slow(), failure_threshold=1, reset_after_s=1, clock=clock)
    breaker._opened_at = clock.now - 5  # already past the cool-down
    trial = asyncio.create_task(call(breaker))
    await asyncio.sleep(0)
    with pytest.raises(JevCircuitOpenError):
        await call(breaker)
    gate.set()
    await trial
    assert breaker.state == "closed"


async def test_non_transient_errors_do_not_open() -> None:
    inner = Flaky(*(JevValidationError("bad") for _ in range(5)))
    breaker = CircuitBreakerJev(inner, failure_threshold=2, clock=Clock())
    for _ in range(5):
        with pytest.raises(JevValidationError):
            await call(breaker)
    assert breaker.state == "closed"


async def test_success_resets_the_count() -> None:
    inner = Flaky(OVERLOADED)
    breaker = CircuitBreakerJev(inner, failure_threshold=2, clock=Clock())
    with pytest.raises(JevOverloadedError):
        await call(breaker)
    await call(breaker)
    inner.errors = [OVERLOADED]
    with pytest.raises(JevOverloadedError):
        await call(breaker)
    assert breaker.state == "closed"  # 1 failure since the last success, threshold 2


async def test_rate_limiter_spaces_starts(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = Clock()
    waits: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        waits.append(seconds)
        clock.now += seconds

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    limited = RateLimitedJev(Flaky(), rate=2.0, clock=clock)
    for _ in range(3):
        await call(limited)
    assert waits == [pytest.approx(0.5), pytest.approx(0.5)]
    clock.now += 10  # idle: no backlog of credit is kept beyond one interval
    await call(limited)
    assert len(waits) == 2


def test_harden_composes_and_keeps_name() -> None:
    wrapped = harden(Flaky(), max_rps=2, failure_threshold=3, reset_after_s=5)
    assert isinstance(wrapped, CircuitBreakerJev)
    assert isinstance(wrapped.inner, RateLimitedJev)
    assert wrapped.name == "flaky"
    assert isinstance(
        harden(Flaky(), max_rps=None, failure_threshold=3, reset_after_s=5).inner, Flaky
    )
