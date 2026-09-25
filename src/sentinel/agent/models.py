"""Pipeline data types: inbound items, per-stage trace records, and the outcome."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from sentinel.config import LLMTier

Source = Literal["ticket", "alert", "email"]
StageName = Literal[
    "ingest", "build_state", "triage", "enrich", "route_model", "generate", "decide"
]
StageStatus = Literal["ok", "escalate", "error"]
OutcomeStatus = Literal["draft_ready", "escalated"]


class WorkItem(BaseModel):
    """An inbound item as received. Holds raw (unredacted) data: never sent to a model."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=1)
    source: Source
    subject: str = ""
    body: str = Field(min_length=1)
    customer_id: str | None = None
    """Reference into our own systems, used by tools. Not included in model state."""
    received_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class StageRecord(BaseModel):
    """One pipeline stage's result. Every record is written to the audit log."""

    model_config = ConfigDict(frozen=True)

    stage: StageName
    status: StageStatus
    detail: dict[str, Any] = Field(default_factory=dict)
    provider: str | None = None
    model: str | None = None
    """Model ID returned by the provider (or requested, if `model_verified` is False)."""
    model_verified: bool | None = None
    latency_ms: float | None = None
    cost_usd: float | None = None


class Outcome(BaseModel):
    model_config = ConfigDict(frozen=True)

    run_id: str
    item_id: str
    status: OutcomeStatus
    reasons: list[str] = Field(default_factory=list)
    """Why the item was escalated (empty when a draft is ready)."""
    tier: LLMTier | None = None
    draft: str | None = None
    """Proposed reply. Phase 3 never sends it: guard/verify come in Phase 4."""
    trace: list[StageRecord]
