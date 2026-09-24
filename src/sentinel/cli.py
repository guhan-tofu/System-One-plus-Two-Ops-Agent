"""`sentinel` command-line entry point. Subcommands are added phase by phase."""

from __future__ import annotations

import typer

from sentinel import __version__

app = typer.Typer(no_args_is_help=True, add_completion=False)


@app.callback()
def main() -> None:
    """Sentinel ops agent."""


@app.command()
def version() -> None:
    """Print the Sentinel version."""
    typer.echo(__version__)
