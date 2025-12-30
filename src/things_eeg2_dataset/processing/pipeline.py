"""
Complete THINGS-EEG2 data processing pipeline.
Orchestrates EEG preprocessing, embedding generation, and index merging.
"""

import logging
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from rich import print as rprint
from rich.table import Table

from things_eeg2_dataset.cli.main import EmbeddingModel, Partition
from things_eeg2_dataset.paths import layout
from things_eeg2_dataset.processing import (
    Downloader,
    RawProcessor,
    build_embedder,
)

logger = logging.getLogger(__name__)

# --- Configuration & Path Management ---

NUM_SESSIONS = 4


class PipelineError(Exception):
    """Custom exception for pipeline errors."""

    pass


@dataclass(frozen=True)
class PipelineConfig:
    """Immutable configuration for the pipeline execution."""

    project_dir: Path
    subjects: list[int]
    models: list[EmbeddingModel] = field(default_factory=list)
    sfreq: int = 250
    device: str = "cuda:0"
    overwrite: bool = False
    dry_run: bool = False
    skip_download: bool = False
    skip_processing: bool = False
    create_embeddings: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "project_dir", self.project_dir.resolve())


def _init_pipeline(  # noqa: PLR0913
    project_dir: Path,
    subjects: list[int],
    overwrite: bool,
    dry_run: bool,
    skip_download: bool,
    skip_preprocessing: bool,
    create_embeddings: bool,
    device: str = "cuda:0",
    sfreq: int = 250,
    models: list[EmbeddingModel] | None = None,
) -> "ThingsEEGPipeline":  # type: ignore
    from things_eeg2_dataset.processing.pipeline import (  # noqa: PLC0415
        PipelineConfig,
        ThingsEEGPipeline,
    )

    models = models or []

    config = PipelineConfig(
        project_dir=project_dir,
        subjects=subjects,
        models=models,
        sfreq=sfreq,
        device=device,
        overwrite=overwrite,
        dry_run=dry_run,
        skip_download=skip_download,
        skip_processing=skip_preprocessing,
        create_embeddings=create_embeddings,
    )

    return ThingsEEGPipeline(config)


def get_git_commit_hash() -> str:
    try:
        if not shutil.which("git"):
            return "git_not_found"
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],  # noqa: S607
                stderr=subprocess.DEVNULL,
            )
            .decode("ascii")
            .strip()
        )
    except subprocess.CalledProcessError:
        return "unknown"


# --- Main Pipeline Class ---
class ThingsEEGPipeline:
    def __init__(self, config: PipelineConfig) -> None:
        self.cfg = config
        self._log_config()

    def run(self) -> None:
        logger.info("PIPELINE START")

        # 1. Download
        if not self.cfg.skip_download:
            self.step_download_data()
        else:
            logger.info("=== Raw Data Download (SKIPPED) ===")

        # Pre-flight check
        if not self.validate_pipeline_inputs():
            # raise RuntimeError("Raw data check failed. Aborting pipeline.")
            logger.error("Raw data check failed. Aborting pipeline.")

        # 2. EEG Processing
        if not self.cfg.skip_processing:
            self.step_process_eeg()
        else:
            logger.info("=== EEG Preprocessing (SKIPPED) ===")

        # 3. Embeddings
        if self.cfg.create_embeddings:
            self.step_generate_embeddings()
        else:
            logger.info("=== Embedding Generation (SKIPPED) ===")

        # 4. Validation & Versioning
        self.validate_pipeline_outputs()
        self._write_version_file()

        logger.info("Pipeline completed successfully.")

    def _log_config(self) -> None:
        """Pretty prints the configuration using a Rich Table."""
        if logger.isEnabledFor(logging.DEBUG):
            table = Table(
                title="Configuration Settings",
                show_header=True,
                header_style="bold magenta",
            )
            table.add_column("Field", style="dim", width=25)
            table.add_column("Value")

            for field_name, value in self.cfg.__dict__.items():
                table.add_row(field_name, str(value))

            rprint(table)

    def step_download_data(self) -> None:
        logger.info("=== Raw Data Download ===")
        downloader = Downloader(
            project_dir=self.cfg.project_dir,
            subjects=self.cfg.subjects,
            overwrite=self.cfg.overwrite,
            dry_run=self.cfg.dry_run,
            timeout=300,
        )
        downloader.print_summary()

        # Execute downloads
        raw_res = downloader.download_raw_data()
        src_res = downloader.download_source_data()
        img_res = downloader.download_images()

        # Brief summary
        logger.info(f"Raw data success: {sum(raw_res.values())}/{len(raw_res)}")
        logger.info(f"Source data success: {sum(src_res.values())}/{len(src_res)}")
        logger.info(f"Images downloaded: {img_res}")

    def step_process_eeg(self) -> None:
        logger.info("=== EEG Preprocessing ===")
        processor = RawProcessor(
            subjects=self.cfg.subjects,
            project_dir=self.cfg.project_dir,
            sfreq=self.cfg.sfreq,
        )
        processor.run(overwrite=self.cfg.overwrite, dry_run=self.cfg.dry_run)

    def step_generate_embeddings(self) -> None:
        logger.info("=== Embedding Generation ===")

        for model_name in self.cfg.models:
            logger.info(f"Generating: [blue]{model_name}[/blue]")
            try:
                embedder = build_embedder(
                    model_type=model_name,
                    project_dir=self.cfg.project_dir,
                    overwrite=self.cfg.overwrite,
                    dry_run=self.cfg.dry_run,
                    device=self.cfg.device,
                )
                embedder.generate_and_store_embeddings(dry_run=self.cfg.dry_run)
            except Exception as e:
                logger.error(f"Failed to generate {model_name}: {e}")
                if not self.cfg.dry_run:
                    raise

    def validate_pipeline_outputs(self) -> None:
        logger.info("=== Final Validation ===")

        # 1. Check EEG files
        for sub in self.cfg.subjects:
            s_str = f"sub-{sub:02d}"
            train = (layout.get_eeg_train_file(self.cfg.project_dir, sub)).exists()
            test = (layout.get_eeg_test_file(self.cfg.project_dir, sub)).exists()
            if not (train and test):
                logger.warning(f"Missing EEG data for {s_str}")
                if not self.cfg.dry_run:
                    error_msg = f"Train file exists: {train}, Test file exists: {test}\nCheck paths:\n {layout.get_eeg_train_file(self.cfg.project_dir, sub)}\n {layout.get_eeg_test_file(self.cfg.project_dir, sub)}"
                    raise PipelineError(f"Missing EEG data for {s_str}.\n{error_msg}")

        # 2. Check Embeddings
        for model in self.cfg.models:
            for partition in [Partition.TRAINING, Partition.TEST]:
                for full in [True, False]:
                    emb_file = layout.get_embedding_file(
                        self.cfg.project_dir, model, partition, full
                    )
                    if not emb_file.exists():
                        logger.warning(
                            f"Missing embeddings for {model} ({partition.value}, full={full})"
                        )
                        if not self.cfg.dry_run:
                            raise PipelineError(
                                f"Missing embeddings for {model} ({partition.value}, full={full})"
                            )

    def validate_pipeline_inputs(self) -> bool:
        if not layout.get_raw_dir(self.cfg.project_dir).exists():
            logger.error("Raw data directory missing")
            return False

        missing = []
        for sub in self.cfg.subjects:
            subject_dir = layout.get_raw_subject_dir(self.cfg.project_dir, sub)
            # Check that data for all four sessions exists
            sessions = [
                subject_dir / "ses-01",
                subject_dir / "ses-02",
                subject_dir / "ses-03",
                subject_dir / "ses-04",
            ]
            # Assert that all session subdirectories are present
            if not all(session.exists() for session in sessions):
                missing.append(sub)
                logger.error(f"Missing session directories for subject {sub}")
                continue
            # Check that data is contained in the session subdirectories
            if not any(
                list(session.glob("*.set")) + list(session.glob("*.npy"))
                for session in sessions
            ):
                missing.append(sub)
                continue

        if missing:
            logger.error(f"Missing raw data for subjects: {missing}")
            logger.error(
                f"Tried to find data in: {layout.get_raw_dir(self.cfg.project_dir)}"
            )
            return False
        return True

    def _write_version_file(self) -> None:
        v_file = layout.get_version_file(self.cfg.project_dir)
        commit_hash = get_git_commit_hash()

        logger.info(f"Writing data version file with commit hash: {commit_hash}")

        if not self.cfg.dry_run:
            with v_file.open("w") as f:
                f.write(f"Generated with commit: {commit_hash}\n")
