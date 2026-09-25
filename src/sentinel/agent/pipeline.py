"""Orchestrates the read-only pipeline and returns an `Outcome` (PLAN.md section 4).

ingest -> build_state -> triage -> [enrich] -> route_model -> generate -> decide

- Every stage is appended to the trace and written to the audit log as it happens.
- Any failure escalates to a human with a reason; nothing is ever defaulted silently.
- Phase 3 has no side effects: the draft is returned, never sent.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Mapping
from typing import Any

from sentinel.agent import generate, routing, triage
from sentinel.agent.models import (
    Outcome,
    StageName,
    StageRecord,
    StageStatus,
    WorkItem,
)
from sentinel.agent.state import build_state, with_enrichment
from sentinel.agent.triage import TriageDecision
from sentinel.config import LLMTier, Settings
from sentinel.jev.errors import JevError
from sentinel.jev.models import JevResult, Question
from sentinel.jev.provider import JevProvider, build_provider
from sentinel.llm.openai_client import LLMError, OpenAIClient
from sentinel.log import get_logger
from sentinel.policy.engine import Thresholds
from sentinel.storage.audit import AuditLog
from sentinel.storage.db import make_engine
from sentinel.tools.builtin import default_registry
from sentinel.tools.registry import Risk, ToolRegistry

log = get_logger(__name__)

ENRICH_TOOL = "lookup_customer"
# Only what drafting/decisions need. Names and contact details stay in code.
ENRICH_FIELDS = ("found", "plan", "status", "recent_charges", "open_incidents")


class _Run:
    """Collects the trace for one item and audits each record as it is added."""

    def __init__(self, item: WorkItem, audit: AuditLog) -> None:
        self.run_id = uuid.uuid4().hex
        self.item = item
        self.audit = audit
        self.trace: list[StageRecord] = []

    def add(self, record: StageRecord) -> None:
        self.audit.record(self.run_id, self.item.id, record)
        self.trace.append(record)
        log.info(
            "pipeline.stage",
            run_id=self.run_id,
            item_id=self.item.id,
            stage=record.stage,
            status=record.status,
            model=record.model,
        )

    def stage(self, stage: StageName, status: StageStatus, **detail: Any) -> None:
        self.add(StageRecord(stage=stage, status=status, detail=detail))

    def jev_stage(
        self, stage: StageName, status: StageStatus, result: JevResult, **detail: Any
    ) -> None:
        cost = result.usage.get("cost_usd") if result.usage else None
        self.add(
            StageRecord(
                stage=stage,
                status=status,
                detail={
                    "answers": {k: a.model_dump() for k, a in result.answers.items()},
                    "provider_metadata": result.metadata,
                    **detail,
                },
                provider=result.provider,
                model=result.model,
                model_verified=result.model_verified,
                latency_ms=result.latency_ms,
                cost_usd=cost if isinstance(cost, int | float) else None,
            )
        )

    def escalate(self, reasons: list[str]) -> Outcome:
        self.stage("decide", "escalate", outcome="escalated", reasons=reasons, **self._totals())
        return Outcome(
            run_id=self.run_id,
            item_id=self.item.id,
            status="escalated",
            reasons=reasons,
            trace=self.trace,
        )

    def draft_ready(self, tier: LLMTier, draft: str) -> Outcome:
        self.stage("decide", "ok", outcome="draft_ready", **self._totals())
        return Outcome(
            run_id=self.run_id,
            item_id=self.item.id,
            status="draft_ready",
            tier=tier,
            draft=draft,
            trace=self.trace,
        )

    def _totals(self) -> dict[str, Any]:
        costs = [r.cost_usd for r in self.trace if r.model is not None]
        known = [c for c in costs if c is not None]
        return {
            "total_cost_usd": sum(known) if known else None,
            "cost_complete": len(known) == len(costs),
        }


class Pipeline:
    def __init__(
        self,
        *,
        jev: JevProvider,
        llm: OpenAIClient,
        audit: AuditLog,
        thresholds: Thresholds,
        tools: ToolRegistry,
        jev_fallback: JevProvider | None = None,
    ) -> None:
        self._jev = jev
        self._jev_fallback = jev_fallback
        self._llm = llm
        self._audit = audit
        self._thresholds = thresholds
        self._tools = tools

    async def aclose(self) -> None:
        await self._jev.aclose()
        if self._jev_fallback is not None:
            await self._jev_fallback.aclose()
        await self._llm.aclose()

    async def run(self, item: WorkItem) -> Outcome:
        run = _Run(item, self._audit)
        started = time.perf_counter()
        run.stage("ingest", "ok", source=item.source, has_customer_id=item.customer_id is not None)

        state = build_state(item)
        run.stage("build_state", "ok", fields=sorted(state))

        # --- triage (Jev, one request) ---
        try:
            result, primary_error = await self._ask(triage.QUESTIONS, state)
        except JevError as exc:
            run.stage("triage", "error", error=_describe(exc))
            return run.escalate([f"triage: Jev unavailable ({type(exc).__name__})"])
        decision = triage.decide_triage(result, self._thresholds.triage)
        run.jev_stage(
            "triage",
            "escalate" if decision.escalate else "ok",
            result,
            decision=decision.model_dump(),
            primary_error=primary_error,
        )
        if decision.escalate:
            return run.escalate(decision.escalate_reasons)

        # --- enrich (read-only tools, only when triage asks for data) ---
        if decision.path == "lookup":
            enriched = await self._enrich(run, item, state)
            if enriched is None:
                return run.escalate([f"enrich: {ENRICH_TOOL} failed"])
            state = enriched
        else:
            run.stage("enrich", "ok", ran=False, reason=f"triage.path is {decision.path}")

        # --- route_model (Jev picks a tier; policy decides on low confidence) ---
        try:
            result, primary_error = await self._ask(routing.QUESTIONS, state)
        except JevError as exc:
            run.stage("route_model", "error", error=_describe(exc))
            return run.escalate([f"route_model: Jev unavailable ({type(exc).__name__})"])
        route = routing.decide_tier(result, self._thresholds.route_model)
        run.jev_stage(
            "route_model", "ok", result, decision=route.model_dump(), primary_error=primary_error
        )

        # --- generate (OpenAI draft; not sent) ---
        draft = await self._generate(run, route.tier, state, decision)
        if draft is None:
            return run.escalate(["generate: draft failed"])

        log.info(
            "pipeline.done", run_id=run.run_id, ms=round((time.perf_counter() - started) * 1000)
        )
        return run.draft_ready(route.tier, draft)

    async def _ask(
        self, questions: Mapping[str, Question], state: dict[str, Any]
    ) -> tuple[JevResult, str | None]:
        """Primary Jev provider, then the fallback provider if configured."""
        try:
            return await self._jev.evaluate(state, questions), None
        except JevError as exc:
            if self._jev_fallback is None:
                raise
            log.warning("jev.fallback", primary=self._jev.name, error=type(exc).__name__)
            return await self._jev_fallback.evaluate(state, questions), _describe(exc)

    async def _enrich(
        self, run: _Run, item: WorkItem, state: dict[str, Any]
    ) -> dict[str, Any] | None:
        tool = self._tools.get(ENRICH_TOOL)
        if tool.risk is not Risk.READ_ONLY:  # enrich must never have side effects
            raise RuntimeError(f"enrich tool {tool.name!r} is not read-only")
        if item.customer_id is None:
            run.stage("enrich", "ok", tool=tool.name, ran=False, reason="no customer reference")
            return with_enrichment(state, "account", {"found": False})
        started = time.perf_counter()
        try:
            data = await tool.fn(item.customer_id)
        except Exception as exc:
            run.stage("enrich", "error", tool=tool.name, error=type(exc).__name__)
            return None
        run.add(
            StageRecord(
                stage="enrich",
                status="ok",
                detail={
                    "tool": tool.name,
                    "ran": True,
                    "found": data.get("found"),
                    "fields": sorted(data),
                },
                latency_ms=(time.perf_counter() - started) * 1000,
            )
        )
        minimal = {k: data[k] for k in ENRICH_FIELDS if k in data}
        return with_enrichment(state, "account", minimal)

    async def _generate(
        self, run: _Run, tier: LLMTier, state: dict[str, Any], decision: TriageDecision
    ) -> str | None:
        try:
            gen = await self._llm.generate(
                tier,
                instructions=generate.INSTRUCTIONS,
                input=generate.render_input(state, decision),
            )
        except LLMError as exc:
            run.stage("generate", "error", tier=tier, error=_describe(exc))
            return None
        run.add(
            StageRecord(
                stage="generate",
                status="ok",
                detail={
                    "tier": tier,
                    "requested_model": gen.requested_model,
                    "usage": gen.usage.model_dump(),
                    "draft": gen.text,
                },
                provider="openai",
                model=gen.model,
                model_verified=True,
                latency_ms=gen.latency_ms,
                cost_usd=gen.cost_usd,
            )
        )
        return gen.text


def _describe(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"


def build_pipeline(settings: Settings) -> Pipeline:
    """Wire the pipeline from settings. Raises JevError/LLMError on misconfiguration."""
    jev = build_provider(settings)
    fallback: JevProvider | None = None
    if settings.jev_on_failure == "llm_fallback" and settings.jev_provider != "llm_fallback":
        fallback = build_provider(settings.model_copy(update={"jev_provider": "llm_fallback"}))
    return Pipeline(
        jev=jev,
        jev_fallback=fallback,
        llm=OpenAIClient.from_settings(settings),
        audit=AuditLog(make_engine(settings.database_url)),
        thresholds=Thresholds.load(settings.thresholds_path),
        tools=default_registry(),
    )
