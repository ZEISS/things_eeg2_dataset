import logging
import subprocess
from pathlib import Path
from typing import Annotated

import typer
from rich import print  # noqa: A004

from things_eeg2_dataset import __version__
from things_eeg2_dataset.cli.logger import setup_logging
from things_eeg2_dataset.dataloader.sample_info import get_info_for_sample

setup_logging()

logger = logging.getLogger(__name__)

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
    processed_dir: Path = typer.Option(Path("processed")),
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
        processed_dir=processed_dir,
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
    pipeline.run()
    raise typer.Exit(code=0)


@app.command(name="info")
def info(
    project_dir: Path = typer.Option(
        ..., "--project-dir", help="Path to project root."
    ),
    subject: int = typer.Option(..., "--subject", help="Subject number (e.g., 1)."),
    session: int = typer.Option(..., "--session", help="Session number (e.g., 1)."),
    data_index: int = typer.Option(
        ...,
        "--data-index",
        help="0-based index of the numpy array element you want information about.",
    ),
    partition: str = typer.Option(
        "training", "--partition", help="Partition ('training' or 'test')."
    ),
) -> None:
    """
    Load and display information for a specific sample based on metadata.
    """
    info = get_info_for_sample(
        project_dir=project_dir,
        subject=subject,
        session=session,
        data_idx=data_index,
        partition=partition,
    )

    print(info)


@app.command(name="show")
def show(
    project_dir: Path = typer.Option(
        ..., "--project-dir", help="Path to project root."
    ),
) -> None:
    """
    Visualize EEG data for a specific sample.
    """

    subprocess.run(  # noqa: S603
        [  # noqa: S607
            "streamlit",
            "run",
            "src/things_eeg2_dataset/visualization/app.py",
            "--",
            f"--project-dir={project_dir}",
        ],
        check=False,
    )


if __name__ == "__main__":
    app()
