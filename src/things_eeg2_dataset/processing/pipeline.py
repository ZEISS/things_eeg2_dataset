"""
Complete THINGS-EEG2 data processing pipeline.
Orchestrates EEG preprocessing, embedding generation, and index merging.
"""

import logging
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime

import torch
from rich import print as rprint
from rich.table import Table

from things_eeg2_dataset.cli.main import EmbeddingModel, Partition
from things_eeg2_dataset.paths import layout
from things_eeg2_dataset.processing import (
    Downloader,
    RawProcessor,
    build_embedder,
)
from things_eeg2_dataset.verification.data_validation import (
    validate_raw_data,
    validate_processed_data,
    ValidationStatus,
)
from things_eeg2_dataset.visualization.report_generator import (
    ReportGenerator,
    StageReport,
    StageStatus,
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
    models: list[EmbeddingModel] | None = None
    sfreq: int = 250
    device: str = "cuda:0"
    overwrite: bool = False
    dry_run: bool = False
    skip_download: bool = False
    skip_processing: bool = False
    skip_embeddings: bool = False
    interactive: bool = True
    generate_stage_reports: bool = True
    is_full_pipeline: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "project_dir", self.project_dir.resolve())


def _init_pipeline(  # noqa: PLR0913
    project_dir: Path,
    subjects: list[int],
    overwrite: bool,
    dry_run: bool,
    skip_download: bool,
    skip_preprocessing: bool,
    skip_embeddings: bool,
    interactive: bool = True,
    device: str = "cuda:0",
    sfreq: int = 250,
    models: list[EmbeddingModel] | None = None,
    generate_stage_reports: bool = False,
    is_full_pipeline: bool = False,
) -> "ThingsEEGPipeline":  # type: ignore
    from things_eeg2_dataset.processing.pipeline import (  # noqa: PLC0415
        PipelineConfig,
        ThingsEEGPipeline,
    )

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
        skip_embeddings=skip_embeddings,
        interactive=interactive,
        generate_stage_reports=generate_stage_reports,
        is_full_pipeline=is_full_pipeline,
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
        
        # Initialize report generator
        commit_hash = get_git_commit_hash()
        pipeline_version = f"commit:{commit_hash[:8]}"
        
        self.report_generator = ReportGenerator(
            output_dir=self.cfg.project_dir,
            pipeline_version=pipeline_version,
            dataset_name="THINGS-EEG2"
        )
        
        # Set paths
        input_path = str(layout.get_raw_dir(self.cfg.project_dir))
        output_path = str(layout.get_processed_dir(self.cfg.project_dir))
        self.report_generator.set_paths(input_path, output_path)
        
        # Set dataset info
        self.report_generator.set_dataset_info(
            n_subjects=len(self.cfg.subjects),
            n_sessions=NUM_SESSIONS
        )

    def _prompt_user(self, step_name: str) -> bool:
        """Prompts the user before continuing with the next step of the pipeline.
        
        Args:
            step_name: Name of the pipeline step to confirm.

        Returns:
            True if the user confirms, False to skip step.
        """

        if self.cfg.dry_run:
            logger.info(f"Dry run mode: Automatically confirming {step_name}")
            return True
        
        print()

        while True:
            user_input = input(
                f"Do you want to proceed with the next step: {step_name}? (y/n): "
            ).strip().lower()

            if user_input in ["y", "yes"]:
                return True
            
            elif user_input in ["n", "no"]:
                logger.info(f"Step {step_name} skipped by user. Proceeding to the next step.")
                return False
            
            else:
                logger.info(f"Invalid input: '{user_input}'. Please enter 'y' or 'n'.")

    def _prompt_stage_report(self, stage_name: str) -> bool:
        """Prompt user if they want to generate a report for the completed stage.
        
        Only prompts in full pipeline mode when --stage-reports flag is used.
        Individual commands never prompt for stage reports.
        
        Args:
            stage_name: Name of the stage that just completed.
            
        Returns:
            True if user wants a stage report, False otherwise.
        """
        # Individual commands: never prompt for stage reports
        if not self.cfg.is_full_pipeline:
            return False
        
        # Full pipeline: respect the --stage-reports flag
        if not self.cfg.generate_stage_reports:
            return False
        
        # --stage-reports flag is set, prompt user
        print()
        
        while True:
            user_input = input(
                f"Generate a report for '{stage_name}'? (y/n): "
            ).strip().lower()
            
            if user_input in ["y", "yes"]:
                return True
            elif user_input in ["n", "no"]:
                return False
            else:
                logger.info(f"Invalid input: '{user_input}'. Please enter 'y' or 'n'.")

    def _create_stage_report(self, stage_name: str) -> StageReport:
        """Create a new stage report."""
        return StageReport(
            name=stage_name,
            status=StageStatus.IN_PROGRESS,
            start_time=datetime.now().isoformat()
        )
    
    def _prompt_final_report(self) -> bool:
        """Prompt user if they want to generate a final report.
        
        Prompts for both full pipeline and individual command execution.
        """
        print()
        
        if self.cfg.is_full_pipeline:
            prompt_text = "Generate a final report for the entire pipeline execution? (y/n): "
        else:
            prompt_text = "Generate a report for this operation? (y/n): "
        
        while True:
            user_input = input(prompt_text).strip().lower()
            
            if user_input in ["y", "yes"]:
                return True
            elif user_input in ["n", "no"]:
                return False
            else:
                logger.info(f"Invalid input: '{user_input}'. Please enter 'y' or 'n'.")

    def _finalize_stage_report(
        self, 
        stage_report: StageReport, 
        status: StageStatus,
        error: Exception | None = None,
        files_produced: list[str] | None = None
    ) -> None:
        """Finalize a stage report with end time and status."""
        stage_report.status = status
        stage_report.end_time = datetime.now().isoformat()
        
        # Calculate elapsed time
        start = datetime.fromisoformat(stage_report.start_time)
        end = datetime.fromisoformat(stage_report.end_time)
        stage_report.elapsed_time = (end - start).total_seconds()
        
        if error:
            stage_report.error_log = str(error)
        
        if files_produced:
            stage_report.files_produced = files_produced
        
        self.report_generator.add_stage(stage_report)
        
        # Generate individual stage report if user wants it
        if self._prompt_stage_report(stage_report.name):
            self._generate_individual_stage_report(stage_report)
    
    def _populate_validation_results(
        self,
        stage_report: StageReport,
        validation_report,  # ValidationReport from data_validation module
    ) -> None:
        """Populate stage report with validation results.
        
        Args:
            stage_report: The stage report to update
            validation_report: The validation report from data validation
        """
        # Set overall validation status
        if validation_report.has_failures:
            stage_report.validation_status = "failed"
        elif validation_report.has_warnings:
            stage_report.validation_status = "warning"
        else:
            stage_report.validation_status = "passed"
        
        # Count checks by severity
        stage_report.validation_checks_passed = sum(
            1 for issue in validation_report.issues 
            if issue.severity == ValidationStatus.PASSED
        )
        stage_report.validation_checks_failed = sum(
            1 for issue in validation_report.issues 
            if issue.severity == ValidationStatus.FAILED
        )
        stage_report.validation_warnings = sum(
            1 for issue in validation_report.issues 
            if issue.severity == ValidationStatus.WARNING
        )
        
        # Store detailed issues (only warnings and failures for brevity)
        stage_report.validation_issues = [
            {
                "severity": issue.severity.value,
                "check_name": issue.check_name,
                "message": issue.message,
                "details": issue.details,
            }
            for issue in validation_report.issues
            if issue.severity != ValidationStatus.PASSED
        ]

    def _generate_individual_stage_report(self, stage_report: StageReport) -> None:
        """Generate a report for a single stage.
        
        Args:
            stage_report: The stage report to save.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stage_name_clean = stage_report.name.replace(" ", "_").replace(":", "_").lower()
        
        txt_filename = f"stage_{stage_name_clean}_{timestamp}.txt"
        json_filename = f"stage_{stage_name_clean}_{timestamp}.json"
        
        txt_path = self.report_generator.save_stage_report_text(stage_report, txt_filename)
        json_path = self.report_generator.save_stage_report_json(stage_report, json_filename)
        
        logger.info(f"Stage report saved to: {txt_path}")
        logger.info(f"Stage JSON report saved to: {json_path}")

    def run(self) -> None:
        logger.info("PIPELINE START")
        overall_status = "completed"

        try:
            # 1. Download
            if not self.cfg.skip_download:
                if self._prompt_user("data download"):
                    self.step_download_data()
                else:
                    logger.info("=== Data Download (SKIPPED BY USER) ===")
                    stage_report = self._create_stage_report("Data Download")
                    stage_report.parameters = {
                        "subjects": self.cfg.subjects,
                        "overwrite": self.cfg.overwrite
                    }
                    self._finalize_stage_report(stage_report, StageStatus.SKIPPED)
            else:
                logger.info("=== Data Download (SKIPPED) ===")
                stage_report = self._create_stage_report("Data Download")
                stage_report.parameters = {"skip_download": True}
                self._finalize_stage_report(stage_report, StageStatus.SKIPPED)

            # Pre-flight check
            if not self.validate_pipeline_inputs():
                logger.error("Raw data check failed. Aborting pipeline.")
                if not self.cfg.dry_run:
                    overall_status = "failed"
                    raise PipelineError("Raw data validation failed.")

            # 2. EEG Processing
            if not self.cfg.skip_processing:
                if self._prompt_user("EEG preprocessing"):
                    self.step_process_eeg()
                else:
                    logger.info("=== EEG Preprocessing (SKIPPED BY USER) ===")
                    stage_report = self._create_stage_report("EEG Preprocessing")
                    stage_report.parameters = {
                        "subjects": self.cfg.subjects,
                        "sfreq": self.cfg.sfreq,
                        "overwrite": self.cfg.overwrite
                    }
                    self._finalize_stage_report(stage_report, StageStatus.SKIPPED)
            else:
                logger.info("=== EEG Preprocessing (SKIPPED) ===")
                stage_report = self._create_stage_report("EEG Preprocessing")
                stage_report.parameters = {"skip_processing": True}
                self._finalize_stage_report(stage_report, StageStatus.SKIPPED)

            # 3. Embeddings
            if not self.cfg.skip_embeddings:
                if self._prompt_user("embedding generation"):
                    # User confirmed - now show model selection if needed
                    if (self.cfg.models is None or len(self.cfg.models) == 0) and self.cfg.interactive:
                        from things_eeg2_dataset.cli.main import interactive_model_selection  # noqa: PLC0415
                        models = interactive_model_selection()
                        object.__setattr__(self.cfg, "models", models)
                    elif self.cfg.models is None or len(self.cfg.models) == 0:
                        # Non-interactive fallback
                        models = [
                            EmbeddingModel.OPEN_CLIP_VIT_H_14,
                            EmbeddingModel.OPENAI_CLIP_VIT_L_14,
                            EmbeddingModel.DINO_V2,
                            EmbeddingModel.IP_ADAPTER,
                        ]
                        object.__setattr__(self.cfg, "models", models)
                        logger.info("Using all available models.")
                    
                    self.step_generate_embeddings()
                else:
                    logger.info("=== Embedding Generation (SKIPPED BY USER) ===")
                    stage_report = self._create_stage_report("Embedding Generation")
                    stage_report.parameters = {
                        "models": [m.value for m in (self.cfg.models or [])],
                        "device": self.cfg.device
                    }
                    self._finalize_stage_report(stage_report, StageStatus.SKIPPED)
            else:
                logger.info("=== Embedding Generation (SKIPPED) ===")
                stage_report = self._create_stage_report("Embedding Generation")
                stage_report.parameters = {"skip_embeddings": True}
                self._finalize_stage_report(stage_report, StageStatus.SKIPPED)

            # 4. Validation & Versioning
            self.validate_pipeline_outputs()
            self._write_version_file()

            logger.info("Pipeline completed successfully.")

        except Exception as e:
            logger.error(f"Pipeline failed with error: {e}")
            overall_status = "failed"
            raise
        
        finally:
            # Prompt user if they want a final pipeline report
            if self._prompt_final_report():
                self.report_generator.finalize(overall_status=overall_status)
                txt_path = self.report_generator.save_text_report()
                json_path = self.report_generator.save_json_report()
                
                logger.info(f"Final pipeline report saved to: {txt_path}")
                logger.info(f"Final JSON report saved to: {json_path}")

    def run_single_step(self, step_name: str) -> None:
        """Run a single step and optionally generate a report.
        
        This is used by individual CLI commands (preprocess, embed, etc.)
        
        Args:
            step_name: Name of the step being run
        """
        overall_status = "completed"
        
        try:
            pass
            
        except Exception as e:
            logger.error(f"Step '{step_name}' failed with error: {e}")
            overall_status = "failed"
            raise
        
        finally:
            # Prompt user if they want a final report
            if self._prompt_final_report():
                self.report_generator.finalize(overall_status=overall_status)
                txt_path = self.report_generator.save_text_report()
                json_path = self.report_generator.save_json_report()
                
                logger.info(f"Report saved to: {txt_path}")
                logger.info(f"JSON report saved to: {json_path}")

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
        
        stage_report = self._create_stage_report("Data Download")
        stage_report.parameters = {
            "subjects": self.cfg.subjects,
            "overwrite": self.cfg.overwrite,
            "timeout": 300
        }
        stage_report.requested_subjects = self.cfg.subjects
        
        try:
            downloader = Downloader(
                project_dir=self.cfg.project_dir,
                subjects=self.cfg.subjects,
                overwrite=self.cfg.overwrite,
                dry_run=self.cfg.dry_run,
                timeout=300,
            )
            downloader.print_summary()

            success = downloader.download_all()

            if not success:
                logger.error("One or more downloads failed. Check logs for details.")
                if not self.cfg.dry_run:
                    raise PipelineError("Download failed")
            
            # Populate download statistics
            stage_report.downloaded_subjects = self._get_downloaded_subjects()
            stage_report.missing_files = self._check_missing_files()
            stage_report.total_download_size = self._calculate_download_size()
            stage_report.source_data_downloaded = self._check_source_data_downloaded()
            
            # Track downloaded files for the report
            downloaded_files = []
            for subject in self.cfg.subjects:
                subject_dir = layout.get_raw_subject_dir(self.cfg.project_dir, subject)
                if subject_dir.exists():
                    downloaded_files.extend([str(f) for f in subject_dir.rglob("*") if f.is_file()])
            
            # Validate downloaded data
            logger.info("Validating downloaded data...")
            validation_report = validate_raw_data(
                project_dir=self.cfg.project_dir,
                subjects=self.cfg.subjects,
            )
            
            # Store validation results in stage report
            self._populate_validation_results(stage_report, validation_report)
            
            # Fail the stage if validation failed
            if validation_report.has_failures:
                self._finalize_stage_report(
                    stage_report, 
                    StageStatus.FAILED,
                    error=Exception("Data validation failed after download"),
                    files_produced=downloaded_files[:100]  # Limit to first 100 files
                )
                raise Exception("Downloaded data validation failed. See report for details.")
            
            self._finalize_stage_report(
                stage_report, 
                StageStatus.EXECUTED,
                files_produced=downloaded_files[:100]  # Limit to first 100 files
            )
            
        except Exception as e:
            self._finalize_stage_report(stage_report, StageStatus.FAILED, error=e)
            raise

    def step_process_eeg(self) -> None:
        """Process raw EEG data for all subjects."""
        if self.cfg.skip_processing:
            logger.info("Skipping EEG preprocessing (--skip-preprocessing flag)")
            return

        preprocessed_dir = layout.get_processed_dir(self.cfg.project_dir)
        self.report_generator.set_paths(
            input_path=str(layout.get_raw_dir(self.cfg.project_dir)),
            output_path=str(preprocessed_dir)
        )

        logger.info("Starting EEG preprocessing...")

        missing_subjects = []
        for subject in self.cfg.subjects:
            subject_dir = layout.get_raw_subject_dir(self.cfg.project_dir, subject)
            if not subject_dir.exists():
                missing_subjects.append(subject)

        if missing_subjects:
            logger.error(f"Missing raw data for subjects: {missing_subjects}")        

        stage_report = self._create_stage_report("EEG Preprocessing")
        stage_report.parameters = {
            "subjects": self.cfg.subjects,
            "sfreq": self.cfg.sfreq,
            "overwrite": self.cfg.overwrite,
            "num_sessions": NUM_SESSIONS,
            "dry_run": self.cfg.dry_run,
        }
        
        # Populate preprocessing-specific fields
        stage_report.sampling_frequency = self.cfg.sfreq
        stage_report.original_sampling_frequency = 1000  # ORIGINAL_SAMPLING_FREQUENCY from epoching.py
        stage_report.num_sessions_processed = NUM_SESSIONS
        stage_report.num_channels = 63  # Number of channels after channel selection (see chan_order in epoching.py)
        
        # Define processing tasks
        stage_report.processing_tasks = [
            "Channel selection (63 EEG channels)",
            "Event extraction and target trial rejection",
            f"Low-pass filtering at {self.cfg.sfreq / 3.0:.2f} Hz (anti-aliasing)",
            f"Frequency downsampling from {1000} Hz to {self.cfg.sfreq} Hz",
            "Epoching with time window [-0.2, 1.0] seconds",
            "Baseline correction (from epoch start to stimulus onset)",
            "Data sorting by image conditions",
            "Multivariate Noise Normalization (MVNN)",
        ]
        
        # Filter information
        stage_report.filter_type = "Low-pass FIR filter (anti-aliasing)"
        stage_report.filter_frequencies = {
            "l_freq": None,  # No high-pass filter
            "h_freq": self.cfg.sfreq / 3.0  # Low-pass at target_freq / 3.0
        }
        
        # Epoching information
        stage_report.epoch_time_window = {
            "tmin": -0.2,
            "tmax": 1.0
        }
        
        # Baseline correction information
        stage_report.baseline_correction = {
            "start": "None",  # From start of epoch
            "end": 0.0,  # To stimulus onset
            "description": "Baseline period from epoch start to stimulus onset (t=0)"
        }
        
        try:
            processor = RawProcessor(
                subjects=self.cfg.subjects,
                project_dir=self.cfg.project_dir,
                sfreq=self.cfg.sfreq,
            )

            processor.run(
                overwrite=self.cfg.overwrite,
                dry_run=self.cfg.dry_run,
            )

            processed_files: list[str] = []
            if not self.cfg.dry_run:
                for subject in self.cfg.subjects:
                    processed_files.extend(
                        self._collect_processed_files_for_subject(subject)
                    )

            logger.info("EEG preprocessing completed")
            
            # Validate processed data
            if not self.cfg.dry_run:
                logger.info("Validating processed data...")
                validation_report = validate_processed_data(
                    project_dir=self.cfg.project_dir,
                    subjects=self.cfg.subjects,
                    sfreq=self.cfg.sfreq,
                )
                
                # Store validation results in stage report
                self._populate_validation_results(stage_report, validation_report)
                
                # Fail the stage if validation failed
                if validation_report.has_failures:
                    self._finalize_stage_report(
                        stage_report,
                        StageStatus.FAILED,
                        error=Exception("Data validation failed after preprocessing"),
                        files_produced=processed_files or None,
                    )
                    raise Exception("Processed data validation failed. See report for details.")
            
            self._finalize_stage_report(
                stage_report,
                StageStatus.EXECUTED,
                files_produced=processed_files or None,
            )
            
        except Exception as e:
            self._finalize_stage_report(stage_report, StageStatus.FAILED, error=e)
            raise

    def step_generate_embeddings(self) -> None:
        """Generate embeddings for all specified models."""
        
        if not self.cfg.models or len(self.cfg.models) == 0:
            logger.info("No embedding models specified, skipping embedding generation.")
            return

        logger.info("=== Embedding Generation ===")

        missing_subjects = []
        for subject in self.cfg.subjects:
            train_file = layout.get_eeg_train_file(self.cfg.project_dir, subject)
            test_file = layout.get_eeg_test_file(self.cfg.project_dir, subject)
            if not train_file.exists() or not test_file.exists():
                missing_subjects.append(subject)

        if missing_subjects:
            logger.error(f"Missing raw data for subjects: {missing_subjects}")  

        embeddings_dir = layout.get_embeddings_dir(self.cfg.project_dir)
        self.report_generator.set_paths(
            input_path=str(layout.get_processed_dir(self.cfg.project_dir)),
            output_path=str(embeddings_dir)
            )

        for model_name in self.cfg.models:
            logger.info(f"Generating: [blue]{model_name}[/blue]")
            
            stage_report = self._create_stage_report(f"Embedding: {model_name.value}")
            stage_report.parameters = {
                "model": model_name.value,
                "device": self.cfg.device,
                "overwrite": self.cfg.overwrite
            }
            
            # Populate embedding-specific fields
            stage_report.model_name = model_name.value
            stage_report.batch_size = 40  # From BaseEmbedder.store_embeddings
            stage_report.precision = "float16"  # Models use half precision
            
            # Set embedding dimensions and variants based on model type
            if model_name == EmbeddingModel.OPEN_CLIP_VIT_H_14:
                stage_report.embedding_dimension = {
                    "pooled": "(1024,)",
                    "full": "(257, 1024)"
                }
                stage_report.embedding_variants = ["pooled", "full sequence"]
            elif model_name == EmbeddingModel.OPENAI_CLIP_VIT_L_14:
                stage_report.embedding_dimension = {
                    "pooled": "(768,)",
                    "full": "(257, 768)"
                }
                stage_report.embedding_variants = ["pooled", "full sequence"]
            elif model_name == EmbeddingModel.DINO_V2:
                stage_report.embedding_dimension = {
                    "pooled": "(1024,)",
                    "full": "(257, 1024)"
                }
                stage_report.embedding_variants = ["pooled", "full sequence"]
            elif model_name == EmbeddingModel.IP_ADAPTER:
                stage_report.embedding_dimension = {
                    "pooled": "(1024,)",
                    "full": "(257, 1280)"
                }
                stage_report.embedding_variants = ["pooled", "full sequence"]

            elif model_name == EmbeddingModel.SIGLIP:
                stage_report.embedding_dimension = {
                    "pooled": "(768,)",
                    "full": "(197, 768)"
                }
                stage_report.embedding_variants = ["pooled", "full sequence"]

            elif model_name == EmbeddingModel.SIGLIP2:
                stage_report.embedding_dimension = {
                    "pooled": "(768,)",
                    "full": "(197, 768)"
                }
            
            # Count images to process
            train_images_dir = layout.get_training_images_dir(self.cfg.project_dir)
            test_images_dir = layout.get_test_images_dir(self.cfg.project_dir)
            
            try:
                # Count training images (1654 categories * 10 images each)
                num_train_images = sum(1 for _ in train_images_dir.rglob("*.png")) + \
                                   sum(1 for _ in train_images_dir.rglob("*.jpg"))
                
                # Count test images (200 categories * 1 image each)
                num_test_images = sum(1 for _ in test_images_dir.rglob("*.png")) + \
                                  sum(1 for _ in test_images_dir.rglob("*.jpg"))
                
                stage_report.num_images_processed = {
                    "training": num_train_images,
                    "test": num_test_images
                }
            except Exception as e:
                logger.warning(f"Could not count images: {e}")
                stage_report.num_images_processed = {
                    "training": 16540,  # Expected: 1654 * 10
                    "test": 200  # Expected: 200 * 1
                }
            
            try:
                embedder = build_embedder(
                    model_type=model_name,
                    project_dir=self.cfg.project_dir,
                    overwrite=self.cfg.overwrite,
                    dry_run=self.cfg.dry_run,
                    device=self.cfg.device,
                )
                embedder.generate_and_store_embeddings(dry_run=self.cfg.dry_run)

                # Clean up after each model to free memory
                del embedder
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                
                # Track produced embedding files
                embedding_files = []
                for partition in [Partition.TRAINING, Partition.TEST]:
                    for full in [True, False]:
                        emb_file = layout.get_embedding_file(
                            self.cfg.project_dir, model_name, partition, full
                        )
                        if emb_file.exists():
                            embedding_files.append(str(emb_file))
                
                self._finalize_stage_report(
                    stage_report,
                    StageStatus.EXECUTED,
                    files_produced=embedding_files
                )
                
            except Exception as e:
                logger.error(f"Failed to generate {model_name}: {e}")
                self._finalize_stage_report(stage_report, StageStatus.FAILED, error=e)
                if not self.cfg.dry_run:
                    raise

    def _collect_processed_files_for_subject(self, subject: int) -> list[str]:
        """Return all key processed files recorded for a subject."""
        file_getters = (
            layout.get_eeg_train_file,
            layout.get_eeg_test_file,
            layout.get_eeg_train_image_conditions_file,
            layout.get_eeg_test_image_conditions_file,
            layout.get_metadata_file,
        )
        files: list[str] = []
        for getter in file_getters:
            path = getter(self.cfg.project_dir, subject)
            if path.exists():
                files.append(str(path))
        return files

    def _get_downloaded_subjects(self) -> list[int]:
        """Return list of subjects that have been successfully downloaded.
        
        A subject is considered downloaded if it has all 4 session directories
        with data files present.
        """
        downloaded = []
        for subject in self.cfg.subjects:
            subject_dir = layout.get_raw_subject_dir(self.cfg.project_dir, subject)
            if not subject_dir.exists():
                continue
            
            # Check for all 4 sessions
            sessions = [
                subject_dir / "ses-01",
                subject_dir / "ses-02",
                subject_dir / "ses-03",
                subject_dir / "ses-04",
            ]
            
            # Verify all sessions exist and have data files
            if all(session.exists() for session in sessions):
                has_data = any(
                    list(session.glob("*.set")) or list(session.glob("*.npy"))
                    for session in sessions
                )
                if has_data:
                    downloaded.append(subject)
        
        return downloaded

    def _check_missing_files(self) -> list[str]:
        """Check for missing files in the downloaded data.
        
        Returns a list of descriptive strings about missing files/directories.
        """
        missing = []
        
        for subject in self.cfg.subjects:
            subject_dir = layout.get_raw_subject_dir(self.cfg.project_dir, subject)
            
            if not subject_dir.exists():
                missing.append(f"sub-{subject:02d}: Missing subject directory")
                continue
            
            # Check for all 4 sessions
            for session_num in range(1, 5):
                session_dir = subject_dir / f"ses-{session_num:02d}"
                if not session_dir.exists():
                    missing.append(f"sub-{subject:02d}: Missing session {session_num}")
                    continue
                
                # Check for data files in session
                set_files = list(session_dir.glob("*.set"))
                npy_files = list(session_dir.glob("*.npy"))
                
                if not set_files and not npy_files:
                    missing.append(f"sub-{subject:02d}/ses-{session_num:02d}: No data files found")
        
        return missing

    def _calculate_download_size(self) -> float:
        """Calculate total size of downloaded files in GB.
        
        Returns the total size in gigabytes.
        """
        total_size = 0
        
        for subject in self.cfg.subjects:
            subject_dir = layout.get_raw_subject_dir(self.cfg.project_dir, subject)
            if subject_dir.exists():
                for file_path in subject_dir.rglob("*"):
                    if file_path.is_file():
                        total_size += file_path.stat().st_size
        
        # Convert bytes to GB
        return total_size / (1024 ** 3)

    def _check_source_data_downloaded(self) -> bool:
        """Check if source data directory exists and contains data.
        
        Returns True if source_data directory exists with content.
        """
        source_dir = self.cfg.project_dir / "raw_data" / "source_data"
        
        if not source_dir.exists():
            return False
        
        # Check if there's any content in source_data
        has_content = any(source_dir.iterdir())
        return has_content

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
        if self.cfg.models:
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