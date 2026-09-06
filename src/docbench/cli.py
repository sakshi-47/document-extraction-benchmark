"""Command-line entry point.

The only place that reads a dotenv file or creates directories, so importing
any library module stays free of side effects.
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from docbench.hardware import detect_device
from docbench.settings import load_settings
from docbench.train.config import load_run_config

app = typer.Typer(add_completion=False, help="Document extraction benchmark.")
console = Console()


def _bootstrap():
    try:
        from dotenv import load_dotenv
    except ImportError:
        pass
    else:
        load_dotenv()
    return load_settings()


@app.command()
def config() -> None:
    """Show resolved settings. Safe to paste: it holds no credentials."""
    settings = _bootstrap()
    table = Table("setting", "value")
    for key, value in settings.model_dump().items():
        table.add_row(key, str(value))
    console.print(table)


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
def conditions() -> None:
    """List registered degradation conditions."""
    from docbench.degrade.transforms import CONDITIONS

    for name in sorted(CONDITIONS):
        console.print(f"- {name}")


@app.command()
def validate(path: Path) -> None:
    """Validate a training run config without training."""
    run = load_run_config(path)
    console.print(f"[green]valid[/green]  {run.name}  ({run.base_model})")
    console.print(f"effective batch size: {run.effective_batch_size}")


if __name__ == "__main__":
    app()
