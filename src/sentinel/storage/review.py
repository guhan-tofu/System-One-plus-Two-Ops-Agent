"""Human review queue: escalations and tool calls awaiting approval.

Phase 4 enqueues and records decisions. Acting on an approval (running the
approved calls) arrives with the review endpoints in Phase 6.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from sqlalchemy import Engine, select
from sqlalchemy.orm import Session

from sentinel.storage.db import ReviewItem

ReviewKind = Literal["approval", "review"]
ReviewStatus = Literal["pending", "approved", "rejected"]


class ReviewNotPendingError(Exception):
    pass


class ReviewQueue:
    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def enqueue(
        self,
        *,
        run_id: str,
        item_id: str,
        kind: ReviewKind,
        reasons: list[str],
        payload: dict[str, Any],
    ) -> int:
        with Session(self._engine, expire_on_commit=False) as session, session.begin():
            row = ReviewItem(
                run_id=run_id,
                item_id=item_id,
                kind=kind,
                status="pending",
                reasons=reasons,
                payload=payload,
            )
            session.add(row)
            session.flush()
            return row.id

    def get(self, review_id: int) -> ReviewItem | None:
        with Session(self._engine) as session:
            return session.get(ReviewItem, review_id)

    def pending(self, kind: ReviewKind | None = None) -> list[ReviewItem]:
        with Session(self._engine) as session:
            query = select(ReviewItem).where(ReviewItem.status == "pending")
            if kind is not None:
                query = query.where(ReviewItem.kind == kind)
            return list(session.scalars(query.order_by(ReviewItem.id)))

    def decide(
        self, review_id: int, *, approve: bool, by: str, note: str | None = None
    ) -> ReviewItem:
        with Session(self._engine, expire_on_commit=False) as session, session.begin():
            row = session.get(ReviewItem, review_id)
            if row is None or row.status != "pending":
                raise ReviewNotPendingError(review_id)
            row.status = "approved" if approve else "rejected"
            row.decided_at = datetime.now(UTC)
            row.decided_by = by
            row.note = note
            return row
