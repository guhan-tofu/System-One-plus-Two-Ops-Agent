from __future__ import annotations

from pathlib import Path

import pytest

from sentinel.storage.db import make_engine
from sentinel.storage.review import ReviewNotPendingError, ReviewQueue


@pytest.fixture
def queue(tmp_path: Path) -> ReviewQueue:
    return ReviewQueue(make_engine(f"sqlite:///{tmp_path / 'q.db'}"))


def test_enqueue_and_decide(queue: ReviewQueue) -> None:
    rid = queue.enqueue(
        run_id="r1", item_id="i1", kind="approval", reasons=["why"], payload={"draft": "hi"}
    )
    queue.enqueue(run_id="r2", item_id="i2", kind="review", reasons=["x"], payload={})

    assert [r.id for r in queue.pending("approval")] == [rid]
    assert len(queue.pending()) == 2

    row = queue.decide(rid, approve=True, by="agent@example.test", note="ok")
    assert row.status == "approved" and row.decided_by == "agent@example.test"
    assert row.decided_at is not None
    assert [r.kind for r in queue.pending()] == ["review"]


def test_cannot_decide_twice(queue: ReviewQueue) -> None:
    rid = queue.enqueue(run_id="r", item_id="i", kind="review", reasons=[], payload={})
    queue.decide(rid, approve=False, by="a")
    with pytest.raises(ReviewNotPendingError):
        queue.decide(rid, approve=True, by="b")
    with pytest.raises(ReviewNotPendingError):
        queue.decide(9999, approve=True, by="b")
