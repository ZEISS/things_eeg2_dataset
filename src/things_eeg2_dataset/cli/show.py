import os
import subprocess
from importlib.resources import as_file, files
from pathlib import Path

import typer
from rich import print as rprint
from rich.panel import Panel


def run_streamlit(app_path: Path, project_dir: Path) -> None:
    if not app_path.exists():
        rprint(f"[bold red]Error:[/bold red] Could not find Streamlit app at {app_path}")
        raise typer.Exit(code=1)

    welcome_msg = (
        f"[bold]Project Directory:[/bold] {project_dir}\n"
        "[yellow]Press Ctrl+C to stop the server[/yellow]"
    )
    rprint(Panel(welcome_msg, title="🚀 THINGS EEG2 EXPLORER", border_style="green"))

    os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"

    cmd = [
        "streamlit",
        "run",
        str(app_path),
        "--",
        f"--project-dir={project_dir}",
    ]

    try:
        subprocess.run(cmd, check=True)  # noqa: S603
    except KeyboardInterrupt:
        rprint("\n[green]Explorer stopped.[/green]")
    except subprocess.CalledProcessError:
        rprint("\n[bold red]Streamlit crashed.[/bold red]")
        raise typer.Exit(code=1) from None


def resolve_streamlit_app() -> Path:
    try:
        traversable = files("things_eeg2_dataset.visualization").joinpath("app.py")
        with as_file(traversable) as app_path:
            return app_path
    except (ImportError, TypeError):
        return Path(__file__).parent.parent / "visualization" / "app.py"
