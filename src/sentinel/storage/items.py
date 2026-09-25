"""Work items submitted to the service, with their status and latest outcome.

Statuses: queued -> processing -> ready_to_send | awaiting_approval | escalated | failed,
then, after a human decision: ready_to_send | escalated | rejected | resolved.
"""

from __future__ import annotations

from typing import Any, Literal

from sqlalchemy import Engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from sentinel.agent.models import Outcome, WorkItem
from sentinel.storage.db import ItemRow

ItemStatus = Literal[
    "queued",
    "processing",
    "ready_to_send",
    "awaiting_approval",
    "escalated",
    "failed",
    "rejected",
    "resolved",
]


class DuplicateItemError(Exception):
    pass


class ItemStore:
    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def create(self, item: WorkItem) -> None:
        try:
            with Session(self._engine) as session, session.begin():
                session.add(
                    ItemRow(id=item.id, status="queued", work_item=item.model_dump(mode="json"))
                )
        except IntegrityError:
            raise DuplicateItemError(item.id) from None

    def get(self, item_id: str) -> ItemRow | None:
        with Session(self._engine) as session:
            return session.get(ItemRow, item_id)

    def work_item(self, item_id: str) -> WorkItem:
        row = self.get(item_id)
        if row is None:
            raise KeyError(item_id)
        return WorkItem.model_validate(row.work_item)

    def set_status(self, item_id: str, status: ItemStatus, *, error: str | None = None) -> None:
        self._update(item_id, status=status, error=error)

    def set_outcome(self, item_id: str, outcome: Outcome) -> None:
        self._update(
            item_id,
            status=outcome.status,
            outcome=outcome.model_dump(mode="json"),
            review_id=outcome.review_id,
            error=None,
        )

    def _update(self, item_id: str, **fields: Any) -> None:
        with Session(self._engine) as session, session.begin():
            row = session.get(ItemRow, item_id)
            if row is None:
                raise KeyError(item_id)
            for key, value in fields.items():
                setattr(row, key, value)
