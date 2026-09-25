"""Audit log: every pipeline decision is persisted with the model ID that made it.

Writes fail loudly. If a decision cannot be audited the pipeline must stop, not
carry on unrecorded.
"""

from __future__ import annotations

from sqlalchemy import Engine, select
from sqlalchemy.orm import Session

from sentinel.agent.models import StageRecord
from sentinel.storage.db import AuditEvent


class AuditLog:
    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def record(self, run_id: str, item_id: str, record: StageRecord) -> None:
        with Session(self._engine) as session, session.begin():
            session.add(
                AuditEvent(
                    run_id=run_id,
                    item_id=item_id,
                    stage=record.stage,
                    status=record.status,
                    provider=record.provider,
                    model=record.model,
                    model_verified=record.model_verified,
                    latency_ms=record.latency_ms,
                    cost_usd=record.cost_usd,
                    detail=record.detail,
                )
            )

    def events(self, run_id: str) -> list[AuditEvent]:
        with Session(self._engine) as session:
            rows = session.scalars(
                select(AuditEvent).where(AuditEvent.run_id == run_id).order_by(AuditEvent.id)
            )
            return list(rows)
