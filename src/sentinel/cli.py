"""`sentinel` command-line entry point. Subcommands are added phase by phase."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Annotated, Any

import typer

from sentinel import __version__
from sentinel.agent.models import Outcome, StageRecord, WorkItem
from sentinel.config import get_settings
from sentinel.jev.errors import JevError
from sentinel.jev.models import Question
from sentinel.jev.parsing import describe_shape, parse_response
from sentinel.jev.questions import choice, noul, score
from sentinel.llm.openai_client import LLMError
from sentinel.log import configure_logging

app = typer.Typer(no_args_is_help=True, add_completion=False)

# Fake, harmless state: the probe must never send real customer data.
PROBE_STATE = (
    "Test ticket: a customer says the dashboard has been loading slowly since this morning."
)
PROBE_QUESTIONS: dict[str, Question] = {
    "probe_choice": choice(
        "Which team should handle this?",
        {"technical": "Bugs, errors, performance", "billing": "Payments and invoices"},
    ),
    "probe_score": score("How urgent is this?", ["Not urgent", "Soon", "Today"]),
    "probe_noul": noul("Is the message abusive?"),
}


@app.callback()
def main() -> None:
    """Sentinel ops agent."""


@app.command()
def version() -> None:
    """Print the Sentinel version."""
    typer.echo(__version__)


@app.command()
def probe(
    model: Annotated[str | None, typer.Option(help="Override JEV_MODEL.")] = None,
) -> None:
    """Make one live Jev call with fake data and print the raw response shape."""
    settings = get_settings()
    configure_logging(settings.log_level)
    try:
        report = asyncio.run(_probe(model))
    except JevError as exc:
        typer.secho(f"probe failed: {type(exc).__name__}: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from None
    typer.echo(json.dumps(report, indent=2, sort_keys=True))
    if report["parse"] != "ok":
        raise typer.Exit(1)


async def _probe(model: str | None) -> dict[str, Any]:
    from sentinel.jev.thejevai import TheJevAIProvider

    provider = TheJevAIProvider.from_settings(get_settings())
    try:
        request = provider.build_request(PROBE_STATE, PROBE_QUESTIONS, model)
        raw, latency_ms = await provider.post_raw(request)
    finally:
        await provider.aclose()

    report: dict[str, Any] = {
        "requested_model": request.model,
        "latency_ms": round(latency_ms, 1),
        "shape": describe_shape(raw),
        "raw_response": raw,
    }
    try:
        result = parse_response(
            raw, request.questions, requested_model=request.model, latency_ms=latency_ms
        )
    except JevError as exc:
        report["parse"] = f"failed: {exc}"
    else:
        report["parse"] = "ok"
        report["audited_model"] = result.model
        report["model_verified"] = result.model_verified
    return report


@app.command("run-file")
def run_file(
    path: Annotated[Path, typer.Argument(exists=True, dir_okay=False, help="WorkItem JSON file.")],
    as_json: Annotated[
        bool, typer.Option("--json", help="Print the full Outcome as JSON.")
    ] = False,
) -> None:
    """Run one work item through the pipeline and print the decision trace."""
    settings = get_settings()
    configure_logging(settings.log_level)
    item = WorkItem.model_validate_json(path.read_text(encoding="utf-8"))
    try:
        outcome = asyncio.run(_run_item(item))
    except (JevError, LLMError) as exc:
        typer.secho(f"run failed: {type(exc).__name__}: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from None
    if as_json:
        typer.echo(outcome.model_dump_json(indent=2))
    else:
        typer.echo(format_trace(outcome))


async def _run_item(item: WorkItem) -> Outcome:
    from sentinel.agent.pipeline import build_pipeline

    pipeline = build_pipeline(get_settings())
    try:
        return await pipeline.run(item)
    finally:
        await pipeline.aclose()


def _stage_summary(record: StageRecord) -> str:
    d = record.detail
    match record.stage:
        case "ingest":
            ref = "yes" if d.get("has_customer_id") else "no"
            return f"source={d.get('source')} customer_ref={ref}"
        case "build_state":
            return f"fields={','.join(d.get('fields', []))} (PII redacted)"
        case "triage" if "decision" in d:
            t = d["decision"]
            return (
                f"category={t['category']} ({t['category_confidence']:.2f})  "
                f"urgency={t['urgency_label']} ({t['urgency']:.2f})  "
                f"path={t['path']} ({t['path_confidence']:.2f})  abusive={t['abusive']:.2f}"
            )
        case "route_model" if "decision" in d:
            r = d["decision"]
            note = " -> policy default" if r["overridden"] else ""
            conf = f"{r['confidence']:.2f}"
            return f"tier={r['tier']} (Jev suggested {r['suggested']}, conf {conf}{note})"
        case "enrich":
            if not d.get("ran"):
                return f"skipped: {d.get('reason')}"
            return f"{d.get('tool')} found={d.get('found')}"
        case "generate" if "draft" in d:
            u = d["usage"]
            return f"tier={d['tier']} tokens in/out={u['input_tokens']}/{u['output_tokens']}"
        case "decide":
            cost = d.get("total_cost_usd")
            cost_s = "unknown" if cost is None else f"${cost:.5f}"
            if not d.get("cost_complete", True) and cost is not None:
                cost_s += " (partial)"
            return f"outcome={d.get('outcome')} total_cost={cost_s}"
    return str(d.get("error", ""))


def format_trace(outcome: Outcome) -> str:
    lines = [f"run {outcome.run_id}  item {outcome.item_id}", ""]
    for r in outcome.trace:
        meta = []
        if r.provider:
            model = r.model or "?"
            meta.append(f"{r.provider}:{model}" + ("" if r.model_verified else " (unverified)"))
        if r.latency_ms is not None:
            meta.append(f"{r.latency_ms:.0f}ms")
        if r.cost_usd is not None:
            meta.append(f"${r.cost_usd:.5f}")
        head = f"[{r.stage:<11}] {r.status:<8}"
        lines.append(head + ("  " + "  ".join(meta) if meta else ""))
        summary = _stage_summary(r)
        if summary:
            lines.append(f"{'':15}{summary}")
    lines.append("")
    if outcome.status == "escalated":
        lines.append("ESCALATED to human queue:")
        lines += [f"  - {reason}" for reason in outcome.reasons]
    else:
        lines.append(
            f"DRAFT READY (tier {outcome.tier}; not sent, guard/verify arrive in Phase 4):"
        )
        lines += ["", *(f"  {line}" for line in (outcome.draft or "").splitlines())]
    return "\n".join(lines)
