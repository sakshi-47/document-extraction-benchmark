"""Command-line entry point."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from docfit.hardware import detect_device
from docfit.settings import load_settings
from docfit.train.config import load_run_config

app = typer.Typer(add_completion=False, help="Cost/accuracy frontier for document extraction.")
console = Console()


@app.command()
def hardware() -> None:
    """Report the detected accelerator and the precision that follows from it."""
    device = detect_device()
    console.print(device.describe())
    if device.kind == "cuda" and not device.supports_bf16:
        console.print(
            "[yellow]Pre-Ampere GPU: bfloat16 unavailable. Training in float16 "
            "with gradient scaling.[/yellow]"
        )


@app.command()
def config() -> None:
    """Show resolved settings. Safe to paste: it holds no credentials."""
    settings = load_settings()
    table = Table("setting", "value")
    for key, value in settings.model_dump().items():
        table.add_row(key, str(value))
    console.print(table)


@app.command()
def validate(path: Path) -> None:
    """Validate a run config without training."""
    run = load_run_config(path)
    console.print(f"[green]valid[/green]  {run.name}  ({run.base_model})")
    console.print(f"effective batch size: {run.effective_batch_size}")


if __name__ == "__main__":
    app()
