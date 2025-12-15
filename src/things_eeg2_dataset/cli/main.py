import logging
from enum import Enum
from pathlib import Path
from typing import Annotated

import typer
from rich import print  # noqa: A004

from things_eeg2_dataset import __version__
from things_eeg2_dataset.cli.logger import setup_logging


class Partition(str, Enum):
    TRAINING = "training"
    TEST = "test"


logger = logging.getLogger(__name__)

app = typer.Typer(
    name="things-eeg2",
    help="Unified CLI for THINGS-EEG2 processing & dataloader tools.",
    no_args_is_help=True,
)


DEFAULT_SUBJECTS = list(range(1, 11))
DEFAULT_MODELS: list[str] = []
DEFAULT_PROJECT_DIR = Path.home() / "things_eeg2"


def version_callback(value: bool) -> None:
    if value:
        print(f"{__package__.split('.')[0]} version: [green]{__version__}[/green]")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool | None,
        typer.Option(
            "--version",
            help="Show the application's version and exit.",
            callback=version_callback,
        ),
    ] = None,
    verbose: int = typer.Option(
        0,
        "--verbose",
        "-v",
        count=True,
        help="Increase verbosity (-v, -vv, -vvv).",
    ),
) -> None:
    setup_logging(verbosity=verbose)


@app.command(name="download")
def download(
    project_dir: Path = typer.Option(
        DEFAULT_PROJECT_DIR, "--project-dir", help="Path to project."
    ),
    subjects: list[int] = typer.Option(
        DEFAULT_SUBJECTS, "--subjects", help="List of subject numbers to download."
    ),
    overwrite: bool = typer.Option(
        False, "--overwrite", help="Overwrite existing downloaded data."
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Don't write data to disk."),
) -> None:
    """
    Download the THINGS-EEG2 raw dataset.
    """

    from things_eeg2_dataset.processing.pipeline import (  # noqa: PLC0415
        _init_pipeline,
    )

    pipeline = _init_pipeline(
        project_dir=project_dir,
        subjects=subjects,
        force=overwrite,
        dry_run=dry_run,
        skip_download=False,
        skip_preprocessing=True,  # Not needed for downloading
        create_embeddings=False,  # Not needed for downloading
    )
    pipeline.step_download_data()


@app.command(name="preprocess")
def preprocess(  # noqa: PLR0913
    project_dir: Path = typer.Option(
        DEFAULT_PROJECT_DIR, "--project-dir", help="Path to project."
    ),
    subjects: list[int] = typer.Option(
        DEFAULT_SUBJECTS, "--subjects", help="List of subject numbers to process."
    ),
    sfreq: int = typer.Option(250, "--sfreq", help="Downsampling frequency in Hz."),
    force: bool = typer.Option(
        False, "--force", help="Overwrite existing processed data."
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Don't write data to disk."),
    profile: bool = typer.Option(
        False,
        "--profile",
        help="Enable pyinstrument profiling for the preprocessing step.",
    ),
    open_report: bool = typer.Option(
        True, "--open/--no-open", help="Open profiling report automatically."
    ),
) -> None:
    """
    Preprocess the THINGS-EEG2 raw dataset.
    """

    from things_eeg2_dataset.cli.profiling import run_with_profiling  # noqa: PLC0415
    from things_eeg2_dataset.processing.pipeline import (  # noqa: PLC0415
        _init_pipeline,
    )

    pipeline = _init_pipeline(
        project_dir=project_dir,
        subjects=subjects,
        models=[],  # Not needed for preprocessing
        sfreq=sfreq,
        device="cuda:0",  # Not needed for preprocessing
        force=force,
        dry_run=dry_run,
        skip_download=True,  # Not needed for preprocessing
        skip_preprocessing=False,
        create_embeddings=False,  # Not needed for preprocessing
    )

    if profile:
        run_with_profiling(
            fn=pipeline.step_process_eeg,
            output_dir=project_dir / "profiling",
            label="preprocessing",
            open_report=open_report,
        )
        return

    pipeline.step_process_eeg()


@app.command(name="view-profile")
def view_profile(
    project_dir: Path = typer.Option(DEFAULT_PROJECT_DIR, "--project-dir"),
    list_all: bool = typer.Option(
        False, "--list", help="List all available reports without opening."
    ),
    serve: bool = typer.Option(
        False,
        "--serve",
        help="Host the reports on a local web server (good for remote SSH).",
    ),
    port: int = typer.Option(8000, "--port", help="Port to use for the web server."),
) -> None:
    """
    Open the most recent profiling report in your web browser.
    """
    from things_eeg2_dataset.cli.profiling import show_profiling  # noqa: PLC0415

    show_profiling(
        project_dir=project_dir,
        serve=serve,
        list_all=list_all,
        port=port,
    )


@app.command(name="embed")
def embed(
    project_dir: Path = typer.Option(
        DEFAULT_PROJECT_DIR, "--project-dir", help="Path to project."
    ),
    models: list[str] = typer.Option(
        ..., "--models", help="List of models to generate embeddings for."
    ),
    device: str = typer.Option(
        "cuda:0", "--device", help="Device for model inference."
    ),
    force: bool = typer.Option(False, "--force", help="Overwrite existing embeddings."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Don't write data to disk."),
) -> None:
    """
    Generate model embeddings for the THINGS-EEG2 dataset.
    """
    from things_eeg2_dataset.processing.pipeline import (  # noqa: PLC0415
        _init_pipeline,
    )

    pipeline = _init_pipeline(
        project_dir=project_dir,
        subjects=[],  # Not needed for embedding
        models=models,
        sfreq=250,  # Not needed for embedding
        device=device,
        force=force,
        dry_run=dry_run,
        skip_download=True,  # Not needed for embedding
        skip_preprocessing=True,  # Not needed for embedding
        create_embeddings=True,
    )
    pipeline.step_generate_embeddings()


@app.command(name="pipeline")
def pipeline(  # noqa: PLR0913
    project_dir: Path = typer.Option(
        DEFAULT_PROJECT_DIR, "--project-dir", help="Path to project."
    ),
    subjects: list[int] = typer.Option(
        DEFAULT_SUBJECTS, "--subjects", help="List of subject numbers to process."
    ),
    models: list[str] = typer.Option(
        DEFAULT_MODELS, "--models", help="List of models to use."
    ),
    sfreq: int = typer.Option(250, "--sfreq", help="Downsampling frequency in Hz."),
    device: str = typer.Option(
        "cuda:0", "--device", help="Device for model inference."
    ),
    force: bool = typer.Option(
        False, "--force", help="Overwrite existing processed data."
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Don't write data to disk."),
    skip_download: bool = typer.Option(False, "--skip-download"),
    skip_preprocessing: bool = typer.Option(False, "--skip-preprocessing"),
    create_embeddings: bool = typer.Option(False, "--create-embeddings"),
) -> None:
    """
    Run the full THINGS-EEG2 raw processing pipeline.
    """

    from things_eeg2_dataset.processing.pipeline import (  # noqa: PLC0415
        _init_pipeline,
    )

    pipeline = _init_pipeline(
        project_dir=project_dir,
        subjects=subjects,
        models=models,
        sfreq=sfreq,
        device=device,
        force=force,
        dry_run=dry_run,
        skip_download=skip_download,
        skip_preprocessing=skip_preprocessing,
        create_embeddings=create_embeddings,
    )
    pipeline.run()
    raise typer.Exit(code=0)


@app.command(name="info")
def info(
    project_dir: Path = typer.Option(
        DEFAULT_PROJECT_DIR, "--project-dir", help="Path to project root."
    ),
    subject: int = typer.Option(..., "--subject", help="Subject number (e.g., 1)."),
    session: int = typer.Option(..., "--session", help="Session number (e.g., 1)."),
    data_index: int = typer.Option(
        ...,
        "--data-index",
        help="0-based index of the numpy array element you want information about.",
    ),
    partition: Partition = typer.Option(
        Partition.TRAINING, "--partition", help="Partition ('training' or 'test')."
    ),
) -> None:
    """
    Load and display information for a specific sample based on metadata.
    """

    from things_eeg2_dataset.dataloader.sample_info import (  # noqa: PLC0415
        get_info_for_sample,
    )

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
        DEFAULT_PROJECT_DIR, "--project-dir", help="Path to project root."
    ),
) -> None:
    """
    Visualize EEG data for a specific sample.
    """

    from things_eeg2_dataset.cli.show import (  # noqa: PLC0415
        resolve_streamlit_app,
        run_streamlit,
    )

    app_path = resolve_streamlit_app()
    run_streamlit(app_path, project_dir)


if __name__ == "__main__":
    app()
