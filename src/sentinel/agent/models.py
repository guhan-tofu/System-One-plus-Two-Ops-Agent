"""Pipeline data types: inbound items, per-stage trace records, and the outcome."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from sentinel.config import LLMTier

Source = Literal["ticket", "alert", "email"]
StageName = Literal[
    "ingest",
    "build_state",
    "triage",
    "enrich",
    "route_model",
    "generate",
    "guard",
    "execute",
    "verify",
    "review",
    "decide",
]
StageStatus = Literal["ok", "escalate", "error"]
OutcomeStatus = Literal["ready_to_send", "awaiting_approval", "escalated"]

MAX_BODY_CHARS = 20_000
"""Keeps state well inside Jev's 32k-token state limit; longer items are rejected."""


class WorkItem(BaseModel):
    """An inbound item as received. Holds raw (unredacted) data: never sent to a model."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=1, max_length=128)
    source: Source
    subject: str = Field(default="", max_length=500)
    body: str = Field(min_length=1, max_length=MAX_BODY_CHARS)
    customer_id: str | None = Field(default=None, max_length=128)
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


class ProposedCall(BaseModel):
    """A tool call proposed by the drafting model. Code decides whether it runs."""

    model_config = ConfigDict(frozen=True)

    tool: str
    args: dict[str, Any]


class Outcome(BaseModel):
    model_config = ConfigDict(frozen=True)

    run_id: str
    item_id: str
    status: OutcomeStatus
    reasons: list[str] = Field(default_factory=list)
    """Why the item went to a human (empty when ready to send)."""
    review_id: int | None = None
    """Human review queue entry, for escalated / awaiting_approval items."""
    tier: LLMTier | None = None
    draft: str | None = None
    """Proposed reply. Verified against tool results when status is ready_to_send."""
    tool_calls: list[ProposedCall] = Field(default_factory=list)
    trace: list[StageRecord]
