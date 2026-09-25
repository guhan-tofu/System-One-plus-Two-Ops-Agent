"""Run a Jev provider over an eval dataset and score it against the labels."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

from sentinel.agent.state import build_state
from sentinel.agent.triage import decide_triage
from sentinel.evals.dataset import EvalItem
from sentinel.evals.metrics import (
    CalibrationBin,
    brier_binary,
    brier_multiclass,
    calibration_bins,
    expected_calibration_error,
    mean,
    percentile,
)
from sentinel.jev.errors import JevError
from sentinel.jev.models import JevResult
from sentinel.jev.provider import JevProvider
from sentinel.jev.questions import TRIAGE
from sentinel.policy.engine import TriagePolicy

CATEGORIES = ["billing", "technical", "account", "sales"]
PATHS = ["lookup", "generate", "human"]
URGENCY_LEVELS = ["0", "1", "2", "3"]


@dataclass(frozen=True)
class ItemRun:
    item: EvalItem
    result: JevResult | None
    error: str | None = None


@dataclass(frozen=True)
class ProviderRun:
    provider: str
    runs: list[ItemRun]
    wall_s: float


async def run_provider(
    provider: JevProvider,
    items: list[EvalItem],
    *,
    concurrency: int = 4,
    max_rate: float | None = None,
) -> ProviderRun:
    """Ask TRIAGE for every item (same redacted state as the pipeline builds).

    `max_rate` caps request starts per second: Vercel AI Gateway's free tier
    answers bursts of ~3 req/s with 503/429 and no Retry-After.
    """
    semaphore = asyncio.Semaphore(concurrency)
    pace = asyncio.Lock()
    next_start = 0.0

    async def throttle() -> None:
        nonlocal next_start
        if max_rate is None:
            return
        async with pace:
            now = time.perf_counter()
            wait = next_start - now
            if wait > 0:
                await asyncio.sleep(wait)
            next_start = max(now, next_start) + 1 / max_rate

    async def one(item: EvalItem) -> ItemRun:
        async with semaphore:
            await throttle()
            try:
                result = await provider.evaluate(build_state(item.work_item()), TRIAGE)
            except JevError as exc:
                return ItemRun(item=item, result=None, error=f"{type(exc).__name__}: {exc}")
            return ItemRun(item=item, result=result)

    started = time.perf_counter()
    runs = await asyncio.gather(*(one(item) for item in items))
    return ProviderRun(
        provider=provider.name, runs=list(runs), wall_s=time.perf_counter() - started
    )


# --- scoring ------------------------------------------------------------------------


@dataclass
class QuestionScore:
    question: str
    n: int = 0
    correct: int = 0
    brier: list[float] = field(default_factory=list)
    calibration: list[tuple[float, bool]] = field(default_factory=list)
    abs_error: list[float] = field(default_factory=list)
    """Urgency only: |expected level - label|."""
    confusion: dict[tuple[str, str], int] = field(default_factory=dict)
    """(label, predicted) -> count."""
    misses: list[tuple[str, str, str]] = field(default_factory=list)
    """(item id, label, predicted)."""

    def add(self, item_id: str, label: str, predicted: str) -> None:
        self.n += 1
        self.correct += label == predicted
        self.confusion[(label, predicted)] = self.confusion.get((label, predicted), 0) + 1
        if label != predicted:
            self.misses.append((item_id, label, predicted))

    @property
    def accuracy(self) -> float | None:
        return self.correct / self.n if self.n else None

    @property
    def mean_brier(self) -> float | None:
        return mean(self.brier)

    @property
    def bins(self) -> list[CalibrationBin]:
        return calibration_bins(self.calibration)

    @property
    def ece(self) -> float | None:
        return expected_calibration_error(self.bins)


@dataclass
class PolicyScore:
    """What the section 8 triage policy would do with these answers."""

    n: int = 0
    proceeded: int = 0
    proceeded_category_correct: int = 0
    proceeded_path_correct: int = 0
    needs_human: int = 0
    needs_human_escalated: int = 0
    unneeded_escalations: int = 0


@dataclass
class ProviderScore:
    provider: str
    models: list[str]
    n_items: int
    errors: list[tuple[str, str]]
    latencies_ms: list[float]
    costs: list[float | None]
    wall_s: float
    questions: dict[str, QuestionScore]
    policy: PolicyScore

    @property
    def p50_ms(self) -> float | None:
        return percentile(self.latencies_ms, 50)

    @property
    def p95_ms(self) -> float | None:
        return percentile(self.latencies_ms, 95)

    @property
    def cost_per_item(self) -> float | None:
        """Mean cost of successful items; None if any cost is unknown."""
        if not self.costs or any(c is None for c in self.costs):
            return None
        return sum(c for c in self.costs if c is not None) / len(self.costs)


def _argmax(probs: dict[str, float], labels: list[str]) -> tuple[str, float]:
    best = max(labels, key=lambda k: probs.get(k, 0.0))
    return best, probs.get(best, 0.0)


def score_run(run: ProviderRun, policy: TriagePolicy) -> ProviderScore:
    qs = {q: QuestionScore(q) for q in ("category", "path", "urgency", "is_abusive")}
    ps = PolicyScore()
    errors: list[tuple[str, str]] = []
    latencies: list[float] = []
    costs: list[float | None] = []
    models: set[str] = set()

    for item_run in run.runs:
        item, result = item_run.item, item_run.result
        if result is None:
            errors.append((item.id, item_run.error or "unknown error"))
            continue
        labels = item.labels
        models.add(result.model)
        latencies.append(result.latency_ms)
        cost: Any = result.usage.get("cost_usd") if result.usage else None
        costs.append(float(cost) if isinstance(cost, int | float) else None)

        for qid, options, label in (
            ("category", CATEGORIES, labels.category),
            ("path", PATHS, labels.path),
        ):
            answer = result.choice(qid)
            score = qs[qid]
            score.add(item.id, label, answer.choice)
            score.brier.append(brier_multiclass(answer.probabilities, label, options))
            score.calibration.append(
                (answer.probabilities.get(answer.choice, 0.0), answer.choice == label)
            )

        urgency = result.score("urgency")
        level = min(max(round(urgency.score), 0), 3)
        u = qs["urgency"]
        u.add(item.id, str(labels.urgency), str(level))
        u.abs_error.append(abs(urgency.score - labels.urgency))
        u.brier.append(brier_multiclass(urgency.probabilities, str(labels.urgency), URGENCY_LEVELS))
        top, p_top = _argmax(urgency.probabilities, URGENCY_LEVELS)
        u.calibration.append((p_top, top == str(labels.urgency)))

        p_abusive = result.noul("is_abusive").noul
        a = qs["is_abusive"]
        a.add(item.id, str(labels.is_abusive), str(p_abusive >= 0.5))
        a.brier.append(brier_binary(p_abusive, labels.is_abusive))
        a.calibration.append((p_abusive, labels.is_abusive))

        decision = decide_triage(result, policy)
        needs_human = labels.path == "human" or labels.is_abusive
        ps.n += 1
        ps.needs_human += needs_human
        if decision.escalate:
            ps.needs_human_escalated += needs_human
            ps.unneeded_escalations += not needs_human
        else:
            ps.proceeded += 1
            ps.proceeded_category_correct += decision.category == labels.category
            ps.proceeded_path_correct += decision.path == labels.path

    return ProviderScore(
        provider=run.provider,
        models=sorted(models),
        n_items=len(run.runs),
        errors=errors,
        latencies_ms=latencies,
        costs=costs,
        wall_s=run.wall_s,
        questions=qs,
        policy=ps,
    )
