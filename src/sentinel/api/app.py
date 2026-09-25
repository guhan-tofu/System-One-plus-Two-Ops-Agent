"""HTTP API (FastAPI).

POST /items              submit a work item (processed in the background) -> 202
GET  /items/{id}         status and latest outcome
GET  /review             pending review queue (optionally ?kind=approval|review)
GET  /review/{id}        one review entry with its payload
POST /review/{id}/approve  approve: held tool calls run (policy re-checked), then verify
POST /review/{id}/reject   reject: nothing runs
GET  /healthz            liveness (no auth)

Every endpoint except /healthz requires `Authorization: Bearer <SENTINEL_API_TOKEN>`.
"""

from __future__ import annotations

import secrets
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Annotated, Any, Literal

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Query, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from sentinel.agent.models import WorkItem
from sentinel.api.service import ReviewConflictError, Service
from sentinel.config import Settings, get_settings
from sentinel.storage.db import ItemRow, ReviewItem
from sentinel.storage.items import DuplicateItemError


class ItemAccepted(BaseModel):
    id: str
    status: str


class ItemView(BaseModel):
    id: str
    status: str
    review_id: int | None
    error: str | None
    outcome: dict[str, Any] | None
    created_at: datetime
    updated_at: datetime

    @classmethod
    def of(cls, row: ItemRow) -> ItemView:
        return cls(
            id=row.id,
            status=row.status,
            review_id=row.review_id,
            error=row.error,
            outcome=row.outcome,
            created_at=row.created_at,
            updated_at=row.updated_at,
        )


class ReviewSummary(BaseModel):
    id: int
    item_id: str
    kind: str
    status: str
    reasons: list[str]
    created_at: datetime

    @classmethod
    def of(cls, row: ReviewItem) -> ReviewSummary:
        return cls(
            id=row.id,
            item_id=row.item_id,
            kind=row.kind,
            status=row.status,
            reasons=row.reasons,
            created_at=row.created_at,
        )


class ReviewView(ReviewSummary):
    payload: dict[str, Any]
    decided_by: str | None
    decided_at: datetime | None
    note: str | None

    @classmethod
    def of(cls, row: ReviewItem) -> ReviewView:
        return cls(
            **ReviewSummary.of(row).model_dump(),
            payload=row.payload,
            decided_by=row.decided_by,
            decided_at=row.decided_at,
            note=row.note,
        )


class Decision(BaseModel):
    by: str = Field(min_length=1, max_length=128)
    """Who decided (recorded in the review row and the audit log)."""
    note: str | None = Field(default=None, max_length=1000)


_bearer = HTTPBearer(auto_error=False)


def _authorized(
    request: Request,
    creds: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
) -> None:
    token: str = request.app.state.api_token
    if creds is None or not secrets.compare_digest(creds.credentials, token):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid or missing token")


def _service(request: Request) -> Service:
    current: Service = request.app.state.service
    return current


Auth = Depends(_authorized)
Svc = Annotated[Service, Depends(_service)]


def create_app(settings: Settings | None = None, *, service: Service | None = None) -> FastAPI:
    settings = settings or get_settings()
    token = settings.sentinel_api_token.get_secret_value()
    if not token:
        raise RuntimeError("SENTINEL_API_TOKEN is not set; refusing to serve an open API")

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if service is None:
            app.state.service = _build_service(settings)
        else:
            app.state.service = service
        try:
            yield
        finally:
            await app.state.service.pipeline.aclose()

    app = FastAPI(title="Sentinel", lifespan=lifespan)
    app.state.api_token = token

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/items", status_code=status.HTTP_202_ACCEPTED, dependencies=[Auth])
    async def submit_item(
        item: WorkItem, background: BackgroundTasks, service: Svc
    ) -> ItemAccepted:
        try:
            service.submit(item)
        except DuplicateItemError:
            raise HTTPException(status.HTTP_409_CONFLICT, "item id already exists") from None
        background.add_task(service.process, item.id)
        return ItemAccepted(id=item.id, status="queued")

    @app.get("/items/{item_id}", dependencies=[Auth])
    async def get_item(item_id: str, service: Svc) -> ItemView:
        row = service.items.get(item_id)
        if row is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "item not found")
        return ItemView.of(row)

    @app.get("/review", dependencies=[Auth])
    async def list_reviews(
        service: Svc, kind: Annotated[Literal["approval", "review"] | None, Query()] = None
    ) -> list[ReviewSummary]:
        return [ReviewSummary.of(r) for r in service.queue.pending(kind)]

    @app.get("/review/{review_id}", dependencies=[Auth])
    async def get_review(review_id: int, service: Svc) -> ReviewView:
        row = service.queue.get(review_id)
        if row is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "review not found")
        return ReviewView.of(row)

    async def _decide(
        review_id: int, decision: Literal["approve", "reject"], body: Decision, service: Service
    ) -> ItemView:
        try:
            row = await service.decide(review_id, decision=decision, by=body.by, note=body.note)
        except ReviewConflictError:
            raise HTTPException(
                status.HTTP_409_CONFLICT, "review not found or already decided"
            ) from None
        return ItemView.of(row)

    @app.post("/review/{review_id}/approve", dependencies=[Auth])
    async def approve(review_id: int, body: Decision, service: Svc) -> ItemView:
        return await _decide(review_id, "approve", body, service)

    @app.post("/review/{review_id}/reject", dependencies=[Auth])
    async def reject(review_id: int, body: Decision, service: Svc) -> ItemView:
        return await _decide(review_id, "reject", body, service)

    return app


def _build_service(settings: Settings) -> Service:
    from sentinel.agent.pipeline import build_pipeline
    from sentinel.storage.audit import AuditLog
    from sentinel.storage.db import make_engine
    from sentinel.storage.items import ItemStore
    from sentinel.storage.review import ReviewQueue

    engine = make_engine(settings.database_url)
    return Service(
        pipeline=build_pipeline(settings),
        items=ItemStore(engine),
        queue=ReviewQueue(engine),
        audit=AuditLog(engine),
    )
