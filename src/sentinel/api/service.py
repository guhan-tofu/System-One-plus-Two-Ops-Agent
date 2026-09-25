"""Service layer behind the HTTP API: processing items and acting on reviews."""

from __future__ import annotations

from typing import Literal

from sentinel.agent.models import StageRecord, WorkItem
from sentinel.agent.pipeline import Pipeline
from sentinel.log import get_logger
from sentinel.storage.audit import AuditLog
from sentinel.storage.db import ItemRow, ReviewItem
from sentinel.storage.items import ItemStatus, ItemStore
from sentinel.storage.review import ReviewNotPendingError, ReviewQueue

log = get_logger(__name__)


class ReviewConflictError(Exception):
    """The review is not pending (already decided) or does not exist."""


class Service:
    def __init__(
        self, *, pipeline: Pipeline, items: ItemStore, queue: ReviewQueue, audit: AuditLog
    ) -> None:
        self.pipeline = pipeline
        self.items = items
        self.queue = queue
        self.audit = audit

    def submit(self, item: WorkItem) -> None:
        self.items.create(item)

    async def process(self, item_id: str) -> None:
        """Background task: run the pipeline and store the outcome. Never raises."""
        self.items.set_status(item_id, "processing")
        try:
            outcome = await self.pipeline.run(self.items.work_item(item_id))
        except Exception as exc:  # the item must end in a visible state, not hang
            log.error("item.failed", item_id=item_id, error=type(exc).__name__)
            self.items.set_status(item_id, "failed", error=type(exc).__name__)
            return
        self.items.set_outcome(item_id, outcome)

    async def decide(
        self,
        review_id: int,
        *,
        decision: Literal["approve", "reject"],
        by: str,
        note: str | None,
    ) -> ItemRow:
        """Record a human decision and act on it; returns the item's new state."""
        try:
            review = self.queue.decide(review_id, approve=decision == "approve", by=by, note=note)
        except ReviewNotPendingError:
            raise ReviewConflictError(review_id) from None

        if decision == "approve" and review.kind == "approval":
            await self._run_approved(review, by)
        else:
            status: ItemStatus = "rejected" if decision == "reject" else "resolved"
            self._audit_decision(review, status, by)
            self.items.set_status(review.item_id, status)

        row = self.items.get(review.item_id)
        if row is None:  # reviews are only created for stored items
            raise ReviewConflictError(review_id)
        return row

    async def _run_approved(self, review: ReviewItem, by: str) -> None:
        item = self.items.work_item(review.item_id)
        self.items.set_status(review.item_id, "processing")
        try:
            outcome = await self.pipeline.resume_approved(
                item, review.id, review.payload, approver=by
            )
        except Exception as exc:
            log.error("review.execute_failed", review_id=review.id, error=type(exc).__name__)
            self.items.set_status(review.item_id, "failed", error=type(exc).__name__)
            return
        self.items.set_outcome(review.item_id, outcome)

    def _audit_decision(self, review: ReviewItem, status: str, by: str) -> None:
        self.audit.record(
            review.run_id,
            review.item_id,
            StageRecord(
                stage="review",
                status="ok",
                detail={"review_id": review.id, "kind": review.kind, "outcome": status, "by": by},
            ),
        )
