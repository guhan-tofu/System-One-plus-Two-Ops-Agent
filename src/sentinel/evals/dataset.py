"""Labeled eval items (evals/datasets/*.jsonl). All data is fake."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from sentinel.agent.models import Source, WorkItem


class TriageLabels(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    category: Literal["billing", "technical", "account", "sales"]
    urgency: int = Field(ge=0, le=3)
    """Level index: 0 not urgent .. 3 immediately / outage."""
    path: Literal["lookup", "generate", "human"]
    is_abusive: bool


class EvalItem(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    source: Source = "ticket"
    subject: str = ""
    body: str
    labels: TriageLabels

    def work_item(self) -> WorkItem:
        return WorkItem(id=self.id, source=self.source, subject=self.subject, body=self.body)


def load_dataset(path: Path) -> list[EvalItem]:
    items = [
        EvalItem.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    ids = [i.id for i in items]
    if len(set(ids)) != len(ids):
        raise ValueError(f"duplicate item ids in {path}")
    return items
