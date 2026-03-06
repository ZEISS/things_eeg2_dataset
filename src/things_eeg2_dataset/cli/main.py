import logging
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Annotated, List

import typer
from typer import BadParameter

from rich import print as rprint
from rich.prompt import Prompt
from rich.table import Table

from things_eeg2_dataset import __version__
from things_eeg2_dataset.cli.logger import setup_logging
from things_eeg2_dataset.verification.dependency_ver import (
    verify_dependencies,
    auto_verify_and_install_dependencies,
)
from things_eeg2_dataset.visualization.report_generator import (
    ReportGenerator,
    StageReport,
    StageStatus,
)

logger = logging.getLogger(__name__)

app = typer.Typer(
    name="things-eeg2",
    help="Unified CLI for THINGS-EEG2 processing & dataloader tools.",
    no_args_is_help=True,
)

DEFAULT_SUBJECTS = list(range(1, 11))
DEFAULT_MODELS: list[str] = []
DEFAULT_PROJECT_DIR = Path.home() / "things_eeg2"

class EmbeddingModel(Enum):
    """Available embedding models for the THINGS-EEG2 dataset."""
    OPEN_CLIP_VIT_H_14 = "open_clip_vit_h_14"
    OPENAI_CLIP_VIT_L_14 = "openai_clip_vit_l_14"
    DINO_V2 = "dino_v2"
    IP_ADAPTER = "ip_adapter"
    SIGLIP = "siglip"
    SIGLIP2 = "siglip2"

class Partition(Enum):
    """Dataset partitions."""
    TRAINING = "training"
    TEST = "test"

def parse_models(models: List[str] | None) -> List[EmbeddingModel] | None:
    """
    Implement the list of models to generate embeddings for.

    Args:
        models (List[str] | None): List of model names or values provided by the user.

    Returns:
        List[EmbeddingModel] | None: List of EmbeddingModel enum instances or None.

    Raises:
        BadParameter: If an invalid model name or value is provided.
    """
    if not models:
        return None

    # Map available model names and values to their corresponding enum instances
    available_models = {model.name: model for model in EmbeddingModel}
    available_models.update({model.value: model for model in EmbeddingModel})

    parsed_models = []
    for model in models:
        model_upper = model.upper()
        if model_upper in available_models:
            parsed_models.append(available_models[model_upper])
        elif model in available_models:
            parsed_models.append(available_models[model])
        else:
            raise BadParameter(
                f"Invalid model: {model}. Available models: {', '.join(m.value for m in EmbeddingModel)}"
            )

    return parsed_models

def version_callback(value: bool) -> None:
    if value:
        rprint(f"{__package__.split('.')[0]} version: [cyan]{__version__}[/cyan]")
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

@app.command(name="pipeline")
def pipeline(  # noqa: PLR0913
    project_dir: Path = typer.Option(
        DEFAULT_PROJECT_DIR, "--project-dir", help="Path to project."
    ),
    subjects: list[int] = typer.Option(
        DEFAULT_SUBJECTS, "--subjects", help="List of subject numbers to process."
    ),
    models: list[str] = typer.Option(
        None,
        "--models",
        help="List of models to use for embedding generation. Use 'all' to select all models.",
        callback=parse_models,
    ),
    sfreq: int = typer.Option(250, "--sfreq", help="Downsampling frequency in Hz."),
    device: str = typer.Option(
        "cuda:0", "--device", help="Device for model inference."
    ),
    overwrite: bool = typer.Option(
        False, "--overwrite", help="Overwrite existing processed data."
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Don't write data to disk."),
    skip_download: bool = typer.Option(False, "--skip-download"),
    skip_preprocessing: bool = typer.Option(False, "--skip-preprocessing"),
    skip_embeddings: bool = typer.Option(False, "--skip-embeddings"),
    interactive: bool = typer.Option(True, "--interactive/--no-interactive", help="Enable interactive model selection for embedding generation."),
) -> None:
    """
    Run the full THINGS-EEG2 raw processing pipeline.
    """
    
    # Auto-verify and install dependencies
    if not auto_verify_and_install_dependencies():
        logger.error("Failed to verify/install required dependencies.")
        raise typer.Exit(code=1)

    from things_eeg2_dataset.processing.pipeline import (  # noqa: PLC0415
        _init_pipeline,
    )

    if not skip_embeddings and models is None and interactive:
        models = None
    
    elif not skip_embeddings and models is None:
        models = [
            EmbeddingModel.OPEN_CLIP_VIT_H_14,
            EmbeddingModel.OPENAI_CLIP_VIT_L_14,
            EmbeddingModel.DINO_V2,
            EmbeddingModel.IP_ADAPTER,
            EmbeddingModel.SIGLIP,
            EmbeddingModel.SIGLIP2,
        ]

        logger.info("No models specified, defaulting to all available models.")

    if skip_embeddings:
        models = []

    pipeline = _init_pipeline(
        project_dir=project_dir,
        subjects=subjects,
        models=models,
        sfreq=sfreq,
        device=device,
        overwrite=overwrite,
        dry_run=dry_run,
        skip_download=skip_download,
        skip_preprocessing=skip_preprocessing,
        skip_embeddings=skip_embeddings,
        interactive=interactive,
        is_full_pipeline=True,
    )
    pipeline.run()
    raise typer.Exit(code=0)

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
    generate_report: bool = typer.Option(
        True, "--generate-report/--no-generate-report", help="Generate a summary report."
    ),
) -> None:
    """
    Download the THINGS-EEG2 raw dataset.
    """
    
    # Auto-verify and install dependencies
    if not auto_verify_and_install_dependencies():
        logger.error("Failed to verify/install required dependencies.")
        raise typer.Exit(code=1)

    from things_eeg2_dataset.processing.pipeline import (  # noqa: PLC0415
        _init_pipeline,
    )

    # Initialize the pipeline
    pipeline = _init_pipeline(
        project_dir=project_dir,
        subjects=subjects,
        overwrite=overwrite,
        dry_run=dry_run,
        skip_download=False,
        skip_preprocessing=True,  
        skip_embeddings=True,  
    )

    overall_status = "completed"
    
    try:
        # Perform the download (this populates the stage report internally)
        pipeline.step_download_data()
        
        rprint("[bold cyan]✓[/bold cyan] Download completed successfully.")

    except Exception as e:
        logger.error(f"Download failed: {e}")
        overall_status = "failed"
        raise typer.Exit(code=1)
    
    finally:
        # Generate final report if requested
        if generate_report:
            # Update subjects processed count from the stage report
            if pipeline.report_generator.report.stages:
                last_stage = pipeline.report_generator.report.stages[-1]
                if last_stage.downloaded_subjects:
                    pipeline.report_generator.set_subjects_processed(len(last_stage.downloaded_subjects))
                else:
                    pipeline.report_generator.set_subjects_processed(0)
            
            pipeline.report_generator.finalize(overall_status=overall_status)
            report_path = pipeline.report_generator.save_text_report()
            json_path = pipeline.report_generator.save_json_report()
            rprint(f"[bold cyan]✓[/bold cyan] Report saved to: {report_path}")
            rprint(f"[bold cyan]✓[/bold cyan] JSON report saved to: {json_path}")

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

    rprint(info)


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


@app.command(name="preprocess")
def preprocess(  # noqa: PLR0913
    project_dir: Path = typer.Option(
        DEFAULT_PROJECT_DIR, "--project-dir", help="Path to project."
    ),
    subjects: list[int] = typer.Option(
        DEFAULT_SUBJECTS, "--subjects", help="List of subject numbers to process."
    ),
    sfreq: int = typer.Option(250, "--sfreq", help="Downsampling frequency in Hz."),
    overwrite: bool = typer.Option(
        False, "--overwrite", help="Overwrite existing processed data."
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
    
    # Auto-verify and install dependencies
    if not auto_verify_and_install_dependencies():
        logger.error("Failed to verify/install required dependencies.")
        raise typer.Exit(code=1)

    from things_eeg2_dataset.cli.profiling import run_with_profiling  # noqa: PLC0415
    from things_eeg2_dataset.processing.pipeline import (  # noqa: PLC0415
        _init_pipeline,
    )

    pipeline = _init_pipeline(
        project_dir=project_dir,
        subjects=subjects,
        models=[],
        sfreq=sfreq,
        device="cuda:0",
        overwrite=overwrite,
        dry_run=dry_run,
        skip_download=True,
        skip_preprocessing=False,
        skip_embeddings=True,
        interactive=True,
        is_full_pipeline=False,
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
    pipeline.run_single_step("EEG Preprocessing")

@app.command(name="embed")
def embed(
    project_dir: Path = typer.Option(
        DEFAULT_PROJECT_DIR, "--project-dir", help="Path to project."
    ),
    models: list[str] = typer.Option(
        None,
        "--models",
        help="List of models to generate embeddings for.",
        callback=parse_models,
    ),
    device: str = typer.Option(
        "cuda:0", "--device", help="Device for model inference."
    ),
    overwrite: bool = typer.Option(
        False, "--overwrite", help="Overwrite existing embeddings."
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Don't write data to disk."),
    interactive: bool = typer.Option(True, "--interactive/--no-interactive", help="Enable interactive model selection for embedding generation."),
    sequential: bool = typer.Option(True, "--sequential/--parallel", help="Run embedding generation sequentially to reduce memory usage."),
) -> None:
    """
    Generate model embeddings for the THINGS-EEG2 dataset.
    """
    
    # Auto-verify and install dependencies
    if not auto_verify_and_install_dependencies():
        logger.error("Failed to verify/install required dependencies.")
        raise typer.Exit(code=1)
    
    from things_eeg2_dataset.processing.pipeline import (  # noqa: PLC0415
        _init_pipeline,
    )

    if models is None and interactive:
        models = interactive_model_selection()

    if models is None or len(models) == 0:
        models = [
            EmbeddingModel.OPEN_CLIP_VIT_H_14,
            EmbeddingModel.OPENAI_CLIP_VIT_L_14,
            EmbeddingModel.DINO_V2,
            EmbeddingModel.IP_ADAPTER,
            EmbeddingModel.SIGLIP,
            EmbeddingModel.SIGLIP2,
        ]
        logger.info("No models specified, defaulting to all available models.")

    pipeline = _init_pipeline(
        project_dir=project_dir,
        subjects=[],  
        models=models,
        device=device,
        overwrite=overwrite,
        dry_run=dry_run,
        skip_download=True, 
        skip_preprocessing=True,
        skip_embeddings=False,
        interactive=interactive,
        is_full_pipeline=False,
    )
    pipeline.step_generate_embeddings()
    pipeline.run_single_step("Embedding Generation")

def interactive_model_selection() -> list[EmbeddingModel]:
    """
    Prompt the user to select which embedding models to generate.
    """
    available_models = [
        EmbeddingModel.OPEN_CLIP_VIT_H_14,
        EmbeddingModel.OPENAI_CLIP_VIT_L_14,
        EmbeddingModel.DINO_V2,
        EmbeddingModel.IP_ADAPTER,
        EmbeddingModel.SIGLIP,
        EmbeddingModel.SIGLIP2,
    ]

    table = Table(title="Available embedding models")
    table.add_column("Identifier", style="cyan", justify="center")
    table.add_column("Model", style="cyan")
    table.add_column("Value", style="cyan")

    for idx, model in enumerate(available_models, start=1):
        table.add_row(str(idx), model.name, model.value)

    rprint(table)
    rprint("\n[bold cyan]Please select the desired model for embedding generation...[/bold cyan]")
    rprint("[dim]Enter model number(s) separated by spaces or 'all' for all models.[/dim]\n")

    while True:
        selection = Prompt.ask("[bold cyan]Your selection[/bold cyan]")

        if selection.lower() == "all":
            return available_models
        
        try:
            indices = [int(x.strip()) for x in selection.split()]

            if all(1 <= idx <= len(available_models) for idx in indices):
                selected_models = [available_models[idx - 1] for idx in indices]
            
                rprint(f"\n[bold cyan]✓[/bold cyan] Selected models:")
                for model in selected_models:
                    rprint(f" • {model.value}")

                return selected_models
            else:
                rprint(f"[bold cyan]Error[/bold cyan]: Invalid selection. Please enter valid model numbers or 'all'.\n")

        except ValueError:
            rprint(f"[bold cyan]Error[/bold cyan]: Invalid input. Please enter numbers separated by spaces or 'all'.\n")

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

@app.command(name="validate", hidden=True)
def validate(
    project_dir: Path = typer.Option(
        DEFAULT_PROJECT_DIR, "--project-dir", help="Path to project."
    ),
    subjects: list[int] = typer.Option(
        DEFAULT_SUBJECTS, "--subjects", help="List of subject numbers to validate."
    ),
    stage: str = typer.Option(
        "both",
        "--stage",
        help="Validation stage: 'raw' for downloaded data, 'processed' for preprocessed data, 'both' for both stages.",
    ),
    sfreq: int = typer.Option(
        250, "--sfreq", help="Sampling frequency (needed for processed data validation)."
    ),
) -> None:
    """
    Validate data integrity without running the full pipeline.
    
    Checks for file existence, correct shapes, NaN/inf values, and other data quality issues.
    """
    from things_eeg2_dataset.verification.data_validation import (  # noqa: PLC0415
        validate_raw_data,
        validate_processed_data,
    )

    if stage.lower() not in ["raw", "processed", "both"]:
        rprint("[bold red]Error:[/bold red] Invalid stage. Must be 'raw', 'processed', or 'both'.")
        raise typer.Exit(code=1)

    all_passed = True

    # Validate raw data
    if stage.lower() in ["raw", "both"]:
        rprint("\n[bold cyan]Validating raw data...[/bold cyan]")
        raw_report = validate_raw_data(
            project_dir=project_dir,
            subjects=subjects,
        )
        
        if raw_report.has_failures:
            all_passed = False

    # Validate processed data
    if stage.lower() in ["processed", "both"]:
        rprint("\n[bold cyan]Validating processed data...[/bold cyan]")
        processed_report = validate_processed_data(
            project_dir=project_dir,
            subjects=subjects,
            sfreq=sfreq,
        )
        
        if processed_report.has_failures:
            all_passed = False

    # Exit with appropriate code
    if all_passed:
        rprint("\n[bold green]✓ All validation checks passed![/bold green]")
        raise typer.Exit(code=0)
    else:
        rprint("\n[bold red]✗ Validation failed. See details above.[/bold red]")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()

