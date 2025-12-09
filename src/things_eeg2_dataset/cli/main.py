from pathlib import Path
from typing import Annotated

import typer
from rich import print  # noqa: A004

from things_eeg2_dataset import __version__

app = typer.Typer(
    name="things-eeg2",
    help="Unified CLI for THINGS-EEG2 processing & dataloader tools.",
)

DEFAULT_SUBJECTS = list(range(1, 11))
DEFAULT_MODELS: list[str] = []


def version_callback(value: bool) -> None:
    if value:
        print(f"{__package__.split('.')[0]} version: [green]{__version__}[/green]")
        raise typer.Exit()


@app.callback()
def callback(
    version: Annotated[
        bool | None,
        typer.Option(
            "--version", help="Show the version and exit.", callback=version_callback
        ),
    ] = None,
    verbose: bool = typer.Option(False, help="Enable verbose output"),
) -> None:
    pass


@app.command(name="download")
def download(
    project_dir: Path = typer.Option(..., "--project-dir", help="Path to project."),
    subjects: list[int] = typer.Option(
        DEFAULT_SUBJECTS, "--subjects", help="List of subject numbers to download."
    ),
    overwrite: bool = typer.Option(
        False, "--overwrite", help="Overwrite existing downloaded data."
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Don't write data to disk."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """
    Download the THINGS-EEG2 raw dataset.
    """
    from things_eeg2_dataset.processing.pipeline import (  # noqa: PLC0415
        PipelineConfig,
        ThingsEEGPipeline,
    )

    config = PipelineConfig(
        project_dir=project_dir,
        subjects=subjects,
        models=[],
        overwrite=overwrite,
        dry_run=dry_run,
        verbose=verbose,
    )

    pipeline = ThingsEEGPipeline(config)
    pipeline.step_download_data()


# ---- Typer command ----
@app.command(name="process")
def process(  # noqa: PLR0913
    project_dir: Path = typer.Option(..., "--project-dir", help="Path to project."),
    subjects: list[int] = typer.Option(
        DEFAULT_SUBJECTS, "--subjects", help="List of subject numbers to process."
    ),
    models: list[str] = typer.Option(
        DEFAULT_MODELS, "--models", help="List of models to use."
    ),
    processed_dir_name: str = typer.Option("processed"),
    sfreq: int = typer.Option(250, "--sfreq", help="Downsampling frequency in Hz."),
    device: str = typer.Option(
        "cuda:0", "--device", help="Device for model inference."
    ),
    force: bool = typer.Option(
        False, "--force", help="Overwrite existing processed data."
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Don't write data to disk."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
    skip_download: bool = typer.Option(False, "--skip-download"),
    skip_preprocessing: bool = typer.Option(False, "--skip-preprocessing"),
    create_embeddings: bool = typer.Option(False, "--create-embeddings"),
    skip_merging: bool = typer.Option(False, "--skip-merging"),
) -> None:
    """
    Run the full THINGS-EEG2 raw processing pipeline.
    """
    from things_eeg2_dataset.processing.pipeline import (  # noqa: PLC0415
        PipelineConfig,
        ThingsEEGPipeline,
    )

    config = PipelineConfig(
        project_dir=project_dir,
        subjects=subjects,
        models=models,
        processed_dir_name=processed_dir_name,
        sfreq=sfreq,
        device=device,
        overwrite=force,
        dry_run=dry_run,
        verbose=verbose,
        skip_download=skip_download,
        skip_processing=skip_preprocessing,
        create_embeddings=create_embeddings,
        skip_merging=skip_merging,
    )

    pipeline = ThingsEEGPipeline(config)
    status = pipeline.run()
    raise typer.Exit(code=status)


@app.command(name="load")
def load() -> None:
    """
    Run the THINGS-EEG2 dataloader tasks.
    """
    typer.echo("Dataloader not yet implemented.")


# ---------------------------------------------------------------------
if __name__ == "__main__":
    app()
