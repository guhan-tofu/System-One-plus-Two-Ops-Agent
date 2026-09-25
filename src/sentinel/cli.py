"""`sentinel` command-line entry point. Subcommands are added phase by phase."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any

import typer

from sentinel import __version__
from sentinel.agent.models import Outcome, StageRecord, WorkItem
from sentinel.config import get_settings
from sentinel.jev.errors import JevConfigError, JevError
from sentinel.jev.models import Question
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
    model: Annotated[str | None, typer.Option(help="Override JEV_GATEWAY_MODEL.")] = None,
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
    from sentinel.jev.http import HTTPJevProvider
    from sentinel.jev.provider import build_provider

    settings = get_settings()
    provider = build_provider(settings)
    try:
        if not isinstance(provider, HTTPJevProvider):
            raise JevConfigError(f"probe needs an HTTP Jev provider, not {provider.name!r}")
        request = provider.build_request(PROBE_STATE, PROBE_QUESTIONS, model)
        raw, latency_ms = await provider.post_raw(request)
    finally:
        await provider.aclose()

    report: dict[str, Any] = {
        "provider": provider.name,
        "requested_model": request.model,
        "latency_ms": round(latency_ms, 1),
        "shape": provider.describe_shape(raw),
        "raw_response": raw,
    }
    try:
        result = provider.parse(raw, request, latency_ms=latency_ms)
    except JevError as exc:
        report["parse"] = f"failed: {exc}"
    else:
        report["parse"] = "ok"
        report["audited_model"] = result.model
        report["model_verified"] = result.model_verified
        report["metadata"] = result.metadata
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
            calls = ", ".join(c["tool"] for c in d.get("tool_calls", [])) or "none"
            return (
                f"tier={d['tier']} tokens in/out={u['input_tokens']}/{u['output_tokens']}"
                f"  proposed tools: {calls}"
            )
        case "guard" if "decision" in d:
            g = d["decision"]
            safe = "" if g.get("safe") is None else f"  jev safe={g['safe']:.2f}"
            why = f"  ({'; '.join(g['reasons'])})" if g.get("reasons") else ""
            return (
                f"{g['call']['tool']}: policy={g['policy']['action']}{safe} -> {g['action']}{why}"
            )
        case "execute":
            if d.get("ok"):
                return f"{d.get('tool')} ok"
            return f"{d.get('tool')} FAILED: {d.get('error')}"
        case "verify":
            if d.get("deterministic"):
                return "failed tool -> escalate (Jev not asked)"
            if "decision" in d:
                v = d["decision"]
                status = f"  task_status={v['task_status']}" if v["task_status"] else ""
                return f"claim_supported={v['claim_supported']:.2f}{status}"
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
    if outcome.tool_calls:
        lines.append("Tool calls:")
        lines += [f"  - {c.tool} {json.dumps(c.args)}" for c in outcome.tool_calls]
        lines.append("")
    if outcome.status == "ready_to_send":
        lines.append(f"READY TO SEND (tier {outcome.tier}; verified against tool results):")
    else:
        label = "AWAITING APPROVAL" if outcome.status == "awaiting_approval" else "ESCALATED"
        lines.append(f"{label} (review #{outcome.review_id}):")
        lines += [f"  - {reason}" for reason in outcome.reasons]
    if outcome.draft:
        lines += ["", *(f"  {line}" for line in outcome.draft.splitlines())]
    return "\n".join(lines)


@app.command("eval")
def eval_command(
    dataset: Annotated[
        Path, typer.Option(exists=True, dir_okay=False, help="Labeled JSONL dataset.")
    ] = Path("evals/datasets/triage.jsonl"),
    provider: Annotated[
        list[str] | None,
        typer.Option("--provider", "-p", help="Jev provider to evaluate (repeatable)."),
    ] = None,
    out: Annotated[Path, typer.Option(help="Report directory.")] = Path("evals/reports"),
    concurrency: Annotated[int, typer.Option(min=1, max=16)] = 4,
    limit: Annotated[int | None, typer.Option(min=1, help="Only the first N items.")] = None,
    rate: Annotated[
        float | None,
        typer.Option(min=0.1, help="Max requests/second per provider (gateway free tier)."),
    ] = None,
) -> None:
    """Evaluate Jev providers on a labeled dataset and write a markdown report."""
    from sentinel.evals.dataset import load_dataset
    from sentinel.evals.report import render

    settings = get_settings()
    configure_logging(settings.log_level)
    items = load_dataset(dataset)[:limit]
    names = provider or ["vercel", "llm_fallback"]
    scores = asyncio.run(_eval(names, items, concurrency, rate))

    now = datetime.now(UTC)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{now:%Y%m%d-%H%M%S}-{dataset.stem}"
    report_path = out / f"{stem}.md"
    report_path.write_text(
        render(
            dataset=dataset,
            items=items,
            scores=scores,
            generated_at=now,
            thresholds_path=settings.thresholds_path,
        ),
        encoding="utf-8",
    )
    typer.echo(f"report: {report_path}")
    for s in scores:
        acc = s.questions["category"].accuracy
        typer.echo(
            f"  {s.provider}: errors {len(s.errors)}/{s.n_items}, "
            f"category accuracy {'-' if acc is None else f'{100 * acc:.1f}%'}"
        )


async def _eval(
    names: list[str], items: list[Any], concurrency: int, rate: float | None
) -> list[Any]:
    from sentinel.evals.runner import ItemRun, ProviderRun, run_provider, score_run
    from sentinel.jev.provider import build_provider
    from sentinel.policy.engine import Thresholds

    settings = get_settings()
    policy = Thresholds.load(settings.thresholds_path).triage
    scores = []
    for name in names:
        try:
            jev = build_provider(settings.model_copy(update={"jev_provider": name}))
        except (JevError, ValueError) as exc:
            error = f"provider not available: {exc}"
            run = ProviderRun(
                provider=name,
                runs=[ItemRun(item=i, result=None, error=error) for i in items],
                wall_s=0.0,
            )
        else:
            try:
                run = await run_provider(jev, items, concurrency=concurrency, max_rate=rate)
            finally:
                await jev.aclose()
        scores.append(score_run(run, policy))
    return scores


@app.command()
def serve(
    host: Annotated[str, typer.Option()] = "127.0.0.1",
    port: Annotated[int, typer.Option()] = 8000,
) -> None:
    """Run the HTTP API (requires SENTINEL_API_TOKEN)."""
    import uvicorn

    from sentinel.api.app import create_app

    settings = get_settings()
    configure_logging(settings.log_level)
    try:
        api = create_app(settings)
    except RuntimeError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from None
    uvicorn.run(api, host=host, port=port, log_level=settings.log_level.lower())
