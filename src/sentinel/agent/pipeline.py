"""Orchestrates the pipeline and returns an `Outcome` (PLAN.md section 4).

ingest -> build_state -> triage -> [enrich] -> route_model -> generate
       -> guard -> execute -> verify -> decide

- Every stage is appended to the trace and written to the audit log as it happens.
- Any failure escalates to the human review queue with a reason; nothing is ever
  defaulted silently.
- Tools run only after the guard (deterministic policy first, then Jev); a draft
  is ready to send only after verify has checked it against the tool results.
- Sending the reply is not implemented yet: `ready_to_send` is the final state.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Mapping
from typing import Any

from sentinel.agent import generate, guard, routing, triage, verify
from sentinel.agent.guard import GuardDecision
from sentinel.agent.models import (
    Outcome,
    ProposedCall,
    StageName,
    StageRecord,
    StageStatus,
    WorkItem,
)
from sentinel.agent.state import build_state, redact, with_enrichment
from sentinel.agent.triage import TriageDecision
from sentinel.config import LLMTier, Settings
from sentinel.jev.errors import JevError
from sentinel.jev.models import JevResult, Question
from sentinel.jev.provider import JevProvider, build_provider
from sentinel.llm.openai_client import LLMError, OpenAIClient
from sentinel.log import get_logger
from sentinel.policy.engine import Thresholds, ToolPolicy
from sentinel.storage.audit import AuditLog
from sentinel.storage.db import make_engine
from sentinel.storage.review import ReviewKind, ReviewQueue
from sentinel.tools.builtin import default_registry
from sentinel.tools.registry import Risk, ToolRegistry, ToolResult

log = get_logger(__name__)

ENRICH_TOOL = "lookup_customer"
# Only what drafting/decisions need. Names and contact details stay in code.
ENRICH_FIELDS = ("found", "plan", "status", "recent_charges", "open_incidents")


class _Run:
    """Collects the trace for one item and audits each record as it is added."""

    def __init__(self, item: WorkItem, audit: AuditLog, queue: ReviewQueue) -> None:
        self.run_id = uuid.uuid4().hex
        self.item = item
        self.audit = audit
        self.queue = queue
        self.trace: list[StageRecord] = []
        self.tier: LLMTier | None = None
        self.draft: str | None = None
        self.calls: list[ProposedCall] = []

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

    def escalate(
        self, reasons: list[str], *, kind: ReviewKind = "review", **payload: Any
    ) -> Outcome:
        """Send the item to the human review queue and finish the run."""
        review_id = self.queue.enqueue(
            run_id=self.run_id,
            item_id=self.item.id,
            kind=kind,
            reasons=reasons,
            payload={
                "draft": self.draft,
                "tool_calls": [c.model_dump() for c in self.calls],
                **payload,
            },
        )
        status = "awaiting_approval" if kind == "approval" else "escalated"
        self.stage(
            "decide",
            "escalate",
            outcome=status,
            reasons=reasons,
            review_id=review_id,
            **self._totals(),
        )
        return Outcome(
            run_id=self.run_id,
            item_id=self.item.id,
            status=status,
            reasons=reasons,
            review_id=review_id,
            tier=self.tier,
            draft=self.draft,
            tool_calls=self.calls,
            trace=self.trace,
        )

    def ready_to_send(self) -> Outcome:
        self.stage("decide", "ok", outcome="ready_to_send", **self._totals())
        return Outcome(
            run_id=self.run_id,
            item_id=self.item.id,
            status="ready_to_send",
            tier=self.tier,
            draft=self.draft,
            tool_calls=self.calls,
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
        tool_policy: ToolPolicy,
        queue: ReviewQueue,
        jev_fallback: JevProvider | None = None,
    ) -> None:
        tool_policy.check_registry(tools)  # every tool needs a rule; risks must agree
        self._jev = jev
        self._jev_fallback = jev_fallback
        self._llm = llm
        self._audit = audit
        self._thresholds = thresholds
        self._tools = tools
        self._tool_policy = tool_policy
        self._queue = queue

    async def aclose(self) -> None:
        await self._jev.aclose()
        if self._jev_fallback is not None:
            await self._jev_fallback.aclose()
        await self._llm.aclose()

    async def run(self, item: WorkItem) -> Outcome:
        run = _Run(item, self._audit, self._queue)
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

        run.tier = route.tier

        # --- generate (OpenAI: reply + proposed tool calls) ---
        if not await self._generate(run, route.tier, state, decision):
            return run.escalate(["generate: draft failed"])

        # --- guard (policy first, then Jev) -> execute -> verify ---
        decisions = await self._guard(run, item, state)
        blocked = [r for d in decisions if d.action == "block" for r in d.reasons]
        if blocked:
            return run.escalate(blocked, guard=[d.model_dump() for d in decisions])
        needs_approval = [r for d in decisions if d.action == "approval" for r in d.reasons]
        if needs_approval:
            return run.escalate(
                needs_approval, kind="approval", guard=[d.model_dump() for d in decisions]
            )

        results = await self._execute(run, item)
        reasons = await self._verify(run, state, results)
        if reasons:
            return run.escalate(reasons, tool_results=[r.model_dump() for r in results])

        log.info(
            "pipeline.done", run_id=run.run_id, ms=round((time.perf_counter() - started) * 1000)
        )
        return run.ready_to_send()

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
        result = await self._tools.execute(tool.name, item.customer_id, {})
        if not result.ok:
            run.stage("enrich", "error", tool=tool.name, error=result.error)
            return None
        data = result.data or {}
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
                latency_ms=result.latency_ms,
            )
        )
        minimal = {k: data[k] for k in ENRICH_FIELDS if k in data}
        return with_enrichment(state, "account", minimal)

    async def _generate(
        self, run: _Run, tier: LLMTier, state: dict[str, Any], decision: TriageDecision
    ) -> bool:
        try:
            output, gen = await self._llm.structured(
                tier,
                instructions=generate.INSTRUCTIONS,
                input=generate.render_input(state, decision, self._tools),
                schema=generate.draft_schema(self._tools),
            )
        except LLMError as exc:
            run.stage("generate", "error", tier=tier, error=_describe(exc))
            return False
        run.draft, run.calls = generate.parse_draft(output)
        run.add(
            StageRecord(
                stage="generate",
                status="ok",
                detail={
                    "tier": tier,
                    "requested_model": gen.requested_model,
                    "usage": gen.usage.model_dump(),
                    "draft": run.draft,
                    "tool_calls": [c.model_dump() for c in run.calls],
                },
                provider="openai",
                model=gen.model,
                model_verified=True,
                latency_ms=gen.latency_ms,
                cost_usd=gen.cost_usd,
            )
        )
        return True

    async def _guard(self, run: _Run, item: WorkItem, state: dict[str, Any]) -> list[GuardDecision]:
        decisions: list[GuardDecision] = []
        for call in run.calls:
            verdict = self._tool_policy.check(
                call.tool, call.args, has_customer=item.customer_id is not None
            )
            result: JevResult | None = None
            primary_error: str | None = None
            if verdict.action == "guard":  # policy permits; Jev must also be confident
                tool = self._tools.get(call.tool)
                try:
                    result, primary_error = await self._ask(
                        guard.QUESTIONS, redact(guard.guard_state(state, call, tool))
                    )
                except JevError as exc:
                    run.stage("guard", "error", tool=call.tool, error=_describe(exc))
            decision = guard.decide_guard(call, verdict, result, self._thresholds.guard)
            status: StageStatus = "ok" if decision.action == "run" else "escalate"
            detail = {"decision": decision.model_dump(), "primary_error": primary_error}
            if result is not None:
                run.jev_stage("guard", status, result, **detail)
            else:
                run.stage("guard", status, **detail)
            decisions.append(decision)
        return decisions

    async def _execute(self, run: _Run, item: WorkItem) -> list[ToolResult]:
        """Run the guarded calls in order; stop at the first failure."""
        results: list[ToolResult] = []
        if item.customer_id is None:  # policy blocks every call without one
            return results
        for call in run.calls:
            result = await self._tools.execute(call.tool, item.customer_id, call.args)
            run.add(
                StageRecord(
                    stage="execute",
                    status="ok" if result.ok else "error",
                    detail=result.model_dump(exclude={"latency_ms"}),
                    latency_ms=result.latency_ms,
                )
            )
            results.append(result)
            if not result.ok:
                break
        return results

    async def _verify(
        self, run: _Run, state: dict[str, Any], results: list[ToolResult]
    ) -> list[str]:
        """Escalation reasons (empty when the draft is supported by the results)."""
        failed = verify.failed_tools(results)
        if failed:  # deterministic: a failed tool always escalates, Jev is not asked
            run.stage("verify", "escalate", deterministic=True, reasons=failed)
            return failed
        try:
            result, primary_error = await self._ask(
                verify.QUESTIONS, redact(verify.verify_state(state, run.draft or "", results))
            )
        except JevError as exc:
            run.stage("verify", "error", error=_describe(exc))
            return [f"verify: Jev unavailable ({type(exc).__name__})"]
        decision = verify.decide_verify(results, result, self._thresholds.verify)
        run.jev_stage(
            "verify",
            "escalate" if decision.escalate else "ok",
            result,
            decision=decision.model_dump(),
            primary_error=primary_error,
        )
        return decision.escalate_reasons


def _describe(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"


def build_pipeline(settings: Settings) -> Pipeline:
    """Wire the pipeline from settings. Raises JevError/LLMError on misconfiguration."""
    jev = build_provider(settings)
    fallback: JevProvider | None = None
    if settings.jev_on_failure == "llm_fallback" and settings.jev_provider != "llm_fallback":
        fallback = build_provider(settings.model_copy(update={"jev_provider": "llm_fallback"}))
    engine = make_engine(settings.database_url)
    return Pipeline(
        jev=jev,
        jev_fallback=fallback,
        llm=OpenAIClient.from_settings(settings),
        audit=AuditLog(engine),
        queue=ReviewQueue(engine),
        thresholds=Thresholds.load(settings.thresholds_path),
        tools=default_registry(),
        tool_policy=ToolPolicy.load(settings.tools_policy_path),
    )
