"""Markdown report comparing providers side by side."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import datetime
from pathlib import Path

from sentinel.evals.dataset import EvalItem
from sentinel.evals.runner import CATEGORIES, PATHS, ProviderScore, QuestionScore

QUESTION_NOTES = {
    "category": "choice over 4 teams; Brier is multiclass (0 best, 2 worst)",
    "path": "choice lookup / generate / human; Brier is multiclass",
    "urgency": "score 0-3; accuracy = round(score) == label; MAE on the expected level",
    "is_abusive": "noul; predicted yes at p >= 0.5; Brier is binary (0 best, 1 worst)",
}


def _pct(x: float | None) -> str:
    return "-" if x is None else f"{100 * x:.1f}%"


def _num(x: float | None, digits: int = 3) -> str:
    return "-" if x is None else f"{x:.{digits}f}"


def _ratio(a: int, b: int) -> str:
    return f"{a}/{b} ({100 * a / b:.0f}%)" if b else "-"


def _row(label: str, scores: Sequence[ProviderScore], cell: Callable[[ProviderScore], str]) -> str:
    return f"| {label} | " + " | ".join(cell(s) for s in scores) + " |"


def render(
    *,
    dataset: Path,
    items: list[EvalItem],
    scores: list[ProviderScore],
    generated_at: datetime,
    thresholds_path: Path,
) -> str:
    by_id = {i.id: i for i in items}
    names = [s.provider for s in scores]
    out: list[str] = [
        f"# Jev eval: `{dataset.name}`",
        "",
        f"Generated {generated_at:%Y-%m-%d %H:%M UTC} · {len(items)} labeled items · "
        f"triage policy from `{thresholds_path}`",
        "",
        "Labels are hand-written for fake tickets; treat small differences as noise "
        f"(n={len(items)}). Latency is client-side wall clock per request.",
        "",
        "## Summary",
        "",
        "| | " + " | ".join(names) + " |",
        "|---|" + "---|" * len(names),
        _row("Model(s)", scores, lambda s: ", ".join(s.models) or "-"),
        _row("Errors", scores, lambda s: f"{len(s.errors)}/{s.n_items}"),
        _row(
            "Latency p50 / p95", scores, lambda s: f"{_num(s.p50_ms, 0)} / {_num(s.p95_ms, 0)} ms"
        ),
        _row(
            "Cost per item",
            scores,
            lambda s: "unknown" if s.cost_per_item is None else f"${s.cost_per_item:.6f}",
        ),
        _row("Wall time", scores, lambda s: f"{s.wall_s:.1f} s"),
    ]
    for q in ("category", "path", "urgency", "is_abusive"):
        per = [sc.questions[q] for sc in scores]
        out.append(f"| **{q}** accuracy | " + " | ".join(_pct(x.accuracy) for x in per) + " |")
        out.append(f"| {q} Brier | " + " | ".join(_num(x.mean_brier) for x in per) + " |")
        out.append(f"| {q} ECE | " + " | ".join(_num(x.ece) for x in per) + " |")
        if q == "urgency":
            out.append("| urgency MAE | " + " | ".join(_num(_mae(x)) for x in per) + " |")
    out += [
        _row(
            "**Policy**: proceeds automatically",
            scores,
            lambda s: _ratio(s.policy.proceeded, s.policy.n),
        ),
        _row(
            "needs-human items escalated",
            scores,
            lambda s: _ratio(s.policy.needs_human_escalated, s.policy.needs_human),
        ),
        _row(
            "escalated without needing a human",
            scores,
            lambda s: _ratio(s.policy.unneeded_escalations, s.policy.n - s.policy.needs_human),
        ),
        _row(
            "category correct when proceeding",
            scores,
            lambda s: _ratio(s.policy.proceeded_category_correct, s.policy.proceeded),
        ),
        _row(
            "path correct when proceeding",
            scores,
            lambda s: _ratio(s.policy.proceeded_path_correct, s.policy.proceeded),
        ),
        "",
        "Needs a human = labeled path `human` or abusive. Question notes: "
        + "; ".join(f"**{q}**: {n}" for q, n in QUESTION_NOTES.items())
        + ".",
        "",
    ]

    for s in scores:
        out += [f"## {s.provider}", ""]
        if s.errors:
            out += ["**Errors** (first 5):", ""]
            out += [f"- `{item_id}`: {err}" for item_id, err in s.errors[:5]]
            out.append("")
        for q in ("category", "path", "urgency", "is_abusive"):
            out += _calibration_table(q, s.questions[q])
        for q, options in (("category", CATEGORIES), ("path", PATHS)):
            out += _confusion(q, s.questions[q], options)
        out += _misses(s, by_id)
    return "\n".join(out).rstrip() + "\n"


def _mae(q: QuestionScore) -> float | None:
    return sum(q.abs_error) / len(q.abs_error) if q.abs_error else None


def _calibration_table(question: str, q: QuestionScore) -> list[str]:
    event = "p(yes)" if question == "is_abusive" else "p(predicted)"
    rows = [
        f"### {question}: calibration ({event}, 10 bins)",
        "",
        "| bin | n | mean predicted | observed |",
        "|---|---|---|---|",
    ]
    for b in q.bins:
        if b.n:
            rows.append(
                f"| {b.lo:.1f}-{b.hi:.1f} | {b.n} | {_num(b.mean_predicted, 2)} | "
                f"{_num(b.observed, 2)} |"
            )
    return [*rows, ""]


def _confusion(question: str, q: QuestionScore, options: list[str]) -> list[str]:
    rows = [
        f"### {question}: confusion (rows = label, columns = predicted)",
        "",
        "| | " + " | ".join(options) + " |",
        "|---|" + "---|" * len(options),
    ]
    for label in options:
        rows.append(
            f"| **{label}** | "
            + " | ".join(str(q.confusion.get((label, p), 0)) for p in options)
            + " |"
        )
    return [*rows, ""]


def _misses(s: ProviderScore, by_id: dict[str, EvalItem]) -> list[str]:
    rows = ["### Disagreements with labels", ""]
    any_miss = False
    for question in ("category", "path", "is_abusive"):
        for item_id, label, predicted in s.questions[question].misses:
            any_miss = True
            subject = by_id[item_id].subject if item_id in by_id else ""
            rows.append(
                f"- `{item_id}` {question}: label **{label}**, got **{predicted}** ({subject})"
            )
    if not any_miss:
        rows.append("None for category, path or is_abusive.")
    return [*rows, ""]
