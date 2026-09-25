"""`sentinel` command-line entry point. Subcommands are added phase by phase."""

from __future__ import annotations

import asyncio
import json
from typing import Annotated, Any

import typer

from sentinel import __version__
from sentinel.config import get_settings
from sentinel.jev.errors import JevError
from sentinel.jev.models import Question
from sentinel.jev.parsing import describe_shape, parse_response
from sentinel.jev.questions import choice, noul, score
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
