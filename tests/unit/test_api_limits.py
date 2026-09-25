from __future__ import annotations

import pytest
from pydantic import ValidationError

from sentinel.agent.models import MAX_BODY_CHARS, WorkItem
from sentinel.api.app import SlidingWindowLimiter


def test_sliding_window() -> None:
    now = [0.0]
    limiter = SlidingWindowLimiter(2, window_s=60, clock=lambda: now[0])
    assert limiter.acquire() is None
    now[0] = 10
    assert limiter.acquire() is None
    now[0] = 20
    assert limiter.acquire() == pytest.approx(40)  # first slot frees at t=60
    now[0] = 60
    assert limiter.acquire() is None


def test_work_item_size_limits() -> None:
    WorkItem(id="x", source="ticket", body="a" * MAX_BODY_CHARS)
    with pytest.raises(ValidationError):
        WorkItem(id="x", source="ticket", body="a" * (MAX_BODY_CHARS + 1))
    with pytest.raises(ValidationError):
        WorkItem(id="x" * 129, source="ticket", body="hi")
