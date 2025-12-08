"""
Complete THINGS-EEG2 data processing pipeline.
Orchestrates EEG preprocessing, embedding generation, and index merging.
"""

import logging
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from things_eeg2_dataset.processing import (
    Downloader,
    RawProcessor,
    build_embedder,
)

# --- Configuration & Path Management ---

NUM_SESSIONS = 4


@dataclass(frozen=True)
class PipelineConfig:
    """Immutable configuration for the pipeline execution."""

    project_dir: Path
    subjects: list[int]
    models: list[str]
    processed_dir_name: str = "processed"
    sfreq: int = 250
    device: str = "cuda:0"
    overwrite: bool = False
    dry_run: bool = False
    skip_download: bool = False
    skip_processing: bool = False
    create_embeddings: bool = False
    skip_merging: bool = False
    verbose: bool = False


@dataclass(frozen=True)
class ProjectPaths:
    """Centralized path definitions."""

    root: Path
    processed_name: str

    @property
    def raw_data(self) -> Path:
        return self.root / "raw_data"

    @property
    def images(self) -> Path:
        return self.root / "Image_set"

    @property
    def embeddings(self) -> Path:
        return self.root / "embeddings"

    @property
    def processed(self) -> Path:
        return self.root / self.processed_name

    def make_structure(self) -> None:
        """Creates the directory tree."""
        dirs = [
            self.raw_data,
            self.images,
            self.images / "training_images",
            self.images / "test_images",
            self.images / "embeddings",
            self.processed,
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)


# --- Utilities ---


def setup_logging(verbose: bool = False) -> logging.Logger:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        force=True,
    )
    return logging.getLogger(__name__)


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
        self.paths = ProjectPaths(config.project_dir, config.processed_dir_name)
        self.logger = setup_logging(config.verbose)

    def log_step(self, title: str) -> None:
        self.logger.info(f"\n{'=' * 80}\n{title}\n{'=' * 80}")

    def run(self) -> int:
        self.log_step("THINGS-EEG2 PROCESSING PIPELINE")
        self.logger.info(f"Config: {self.cfg}")

        try:
            # 0. Setup
            self.paths.make_structure()

            # 1. Download
            if not self.cfg.skip_download:
                self.step_download_data()

            # Pre-flight check
            if not self._check_raw_data():
                return 1

            # 2. EEG Processing
            if not self.cfg.skip_processing:
                self.step_process_eeg()

            # 3. Embeddings
            if self.cfg.create_embeddings:
                self.step_generate_embeddings()

            # 4. Validation & Versioning
            self.step_final_validation()
            self._write_version_file()

            self.logger.info("Pipeline completed successfully.")
            return 0

        except Exception:
            self.logger.exception("Pipeline failed critically.")
            return 1

    def step_download_data(self) -> None:
        self.log_step("STEP 0: Raw Data Download")
        downloader = Downloader(
            project_dir=self.paths.root,
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
        self.logger.info(f"Raw data success: {sum(raw_res.values())}/{len(raw_res)}")
        self.logger.info(f"Source data success: {sum(src_res.values())}/{len(src_res)}")
        self.logger.info(f"Images downloaded: {img_res}")

    def step_process_eeg(self) -> None:
        self.log_step("STEP 1: EEG Preprocessing")
        processor = RawProcessor(
            subjects=self.cfg.subjects,
            project_dir=str(self.paths.root),
            processed_dir_name=self.cfg.processed_dir_name,
            sfreq=self.cfg.sfreq,
            mvnn_dim="epochs",
        )
        processor.run(overwrite=self.cfg.overwrite, dry_run=self.cfg.dry_run)

    def step_generate_embeddings(self) -> None:
        self.log_step("STEP 2: Embedding Generation")

        for model_name in self.cfg.models:
            self.logger.info(f"Generating: {model_name}")
            try:
                embedder = build_embedder(
                    model_type=model_name,
                    data_path=str(self.paths.images),
                    force=self.cfg.overwrite,
                    dry_run=self.cfg.dry_run,
                    device=self.cfg.device,
                )
                embedder.generate_and_store_embeddings()
            except Exception as e:
                self.logger.error(f"Failed to generate {model_name}: {e}")
                if not self.cfg.dry_run:
                    raise

    def step_final_validation(self) -> None:
        self.log_step("STEP 4: Final Validation")

        # 1. Check EEG files
        for sub in self.cfg.subjects:
            s_str = f"sub-{sub:02d}"
            train = (
                self.paths.processed / f"preprocessed_eeg_training_{s_str}.npy"
            ).exists()
            test = (
                self.paths.processed / f"preprocessed_eeg_test_{s_str}.npy"
            ).exists()
            if not (train and test):
                self.logger.warning(f"Missing EEG data for {s_str}")

        # 2. Check Embeddings
        for model in self.cfg.models:
            if not (self.paths.embeddings / f"{model}_embeddings.pt").exists():
                self.logger.warning(f"Missing embeddings for {model}")

    def _check_raw_data(self) -> bool:
        if not self.paths.raw_data.exists():
            self.logger.error("Raw data directory missing")
            return False

        missing = []
        for sub in self.cfg.subjects:
            subject_dir = self.paths.raw_data / f"sub-{sub:02d}"
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
                self.logger.error(f"Missing session directories for subject {sub}")
                continue
            # Check that data is contained in the session subdirectories
            if not any(
                list(session.glob("*.set")) + list(session.glob("*.npy"))
                for session in sessions
            ):
                missing.append(sub)
                continue

        if missing:
            self.logger.error(f"Missing raw data for subjects: {missing}")
            self.logger.error(f"Tried to find data in: {self.paths.raw_data}")
            return False
        return True

    def _write_version_file(self) -> None:
        v_file = self.paths.root / "DATA_VERSION.txt"
        if not self.cfg.dry_run:
            with v_file.open("w") as f:
                f.write(f"Generated with commit: {get_git_commit_hash()}\n")
