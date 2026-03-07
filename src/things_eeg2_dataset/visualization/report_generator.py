"""
Report generator module
Generates detailed execution reports for the processing pipeline
"""

import json
import platform
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import psutil


class StageStatus(Enum):
    """Status of a pipeline stage."""

    EXECUTED = "executed"
    SKIPPED = "skipped"
    FAILED = "failed"
    IN_PROGRESS = "in_progress"


@dataclass
class StageReport:
    """Report for a single pipeline stage."""

    name: str
    status: StageStatus
    start_time: str | None = None
    end_time: str | None = None
    elapsed_time: float | None = None  # in seconds
    parameters: dict[str, Any] = None  # type: ignore
    error_log: str | None = None
    files_produced: list[str] = None  # type: ignore

    # Fields for the downloading step of the process
    requested_subjects: list[int] | None = None
    downloaded_subjects: list[int] | None = None
    source_data_downloaded: bool | None = None
    missing_files: list[str] = field(default_factory=list)  # type: ignore  # noqa: F821, RUF009
    total_download_size: float | None = None  # in GB

    # Fields for the preprocessing step
    sampling_frequency: int | None = None  # Target sampling frequency in Hz
    original_sampling_frequency: int | None = None  # Original sampling frequency
    processing_tasks: list[str] | None = None  # List of processing tasks performed
    filter_type: str | None = None  # Type of filter applied (e.g., "low-pass FIR")
    filter_frequencies: dict[str, float] | None = (
        None  # Filter frequencies (l_freq, h_freq)
    )
    epoch_time_window: dict[str, float] | None = (
        None  # Epoching time window (tmin, tmax)
    )
    baseline_correction: dict[str, Any] | None = None  # Baseline correction parameters
    num_channels: int | None = None  # Number of EEG channels
    num_sessions_processed: int | None = None  # Number of sessions processed

    # Fields for the embedding generation step
    model_name: str | None = None  # Name of the embedding model
    batch_size: int | None = None  # Batch size used for embedding generation
    embedding_dimension: dict[str, Any] | None = (
        None  # Embedding dimensions (pooled, full)
    )
    num_images_processed: dict[str, int] | None = None  # Number of images per partition
    precision: str | None = None  # Model precision (e.g., "float16", "float32")
    embedding_variants: list[str] | None = (
        None  # Types of embeddings generated (pooled, full)
    )

    # Fields for validation results
    validation_status: str | None = (
        None  # Overall validation status: "passed", "warning", "failed"
    )
    validation_checks_passed: int | None = None  # Number of validation checks passed
    validation_checks_failed: int | None = None  # Number of validation checks failed
    validation_warnings: int | None = None  # Number of validation warnings
    validation_issues: list[dict[str, Any]] = field(  # type: ignore  # noqa: F821, RUF009
        default_factory=list
    )  # List of validation issues

    def __post_init__(self):  # noqa: ANN204
        if self.parameters is None:
            self.parameters = {}
        if self.files_produced is None:
            self.files_produced = []


@dataclass
class PipelineReport:
    """Complete pipeline execution report."""

    pipeline_version: str
    dataset_name: str
    execution_start: str
    execution_end: str | None = None
    total_elapsed_time: float | None = None  # Time in seconds

    # Input/Output paths
    input_data_path: str | None = None
    output_data_path: str | None = None

    # Dataset info
    n_subjects: int | None = None
    n_sessions: int | None = None
    n_subjects_processed: int | None = None  # Number of subjects of the dataset

    # Stage reports
    stages: list[StageReport] = None  # type: ignore

    # System info
    system_info: dict[str, Any] = None  # type: ignore

    # Overall status
    overall_status: str = "in_progress"

    def __post_init__(self):  # noqa: ANN204
        if self.stages is None:
            self.stages = []
        if self.system_info is None:
            self.system_info = self._collect_system_info()
        if self.n_subjects_processed is None:
            self.n_subjects_processed = 0

    @staticmethod
    def _collect_system_info() -> dict[str, Any]:
        """Collect system information."""
        return {
            "platform": platform.platform(),
            "system": platform.system(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "cpu_count": psutil.cpu_count(logical=False),
            "cpu_count_logical": psutil.cpu_count(logical=True),
            "total_memory_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "hostname": platform.node(),
        }


class ReportGenerator:
    """Generates and manages pipeline execution reports."""

    def __init__(self, output_dir: Path, pipeline_version: str, dataset_name: str):  # noqa: ANN204
        """
        Initialize the report generator.

        Args:
            output_dir: Directory where reports will be saved
            pipeline_version: Version of the pipeline
            dataset_name: Name of the dataset being processed
        """
        self.output_dir = Path(output_dir)
        self.reports_dir = self.output_dir / "reports"
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        self.report = PipelineReport(
            pipeline_version=pipeline_version,
            dataset_name=dataset_name,
            execution_start=datetime.now().isoformat(),  # noqa: DTZ005
        )

    def set_paths(self, input_path: str, output_path: str):  # noqa: ANN201
        """Set input and output data paths."""
        self.report.input_data_path = input_path
        self.report.output_data_path = output_path

    def set_dataset_info(self, n_subjects: int, n_sessions: int):  # noqa: ANN201
        """Set dataset information."""
        self.report.n_subjects = n_subjects
        self.report.n_sessions = n_sessions

    def set_subjects_processed(self, n_subjects_processed: int):  # noqa: ANN201
        """Set the number of subjects processed."""
        self.report.n_subjects_processed = n_subjects_processed

    def add_stage(self, stage_report: StageReport):  # noqa: ANN201
        """Add a stage report."""
        self.report.stages.append(stage_report)

    def finalize(self, overall_status: str = "completed"):  # noqa: ANN201
        """Finalize the report."""
        self.report.execution_end = datetime.now().isoformat()  # noqa: DTZ005

        # Calculate total elapsed time
        start = datetime.fromisoformat(self.report.execution_start)
        end = datetime.fromisoformat(self.report.execution_end)
        self.report.total_elapsed_time = (end - start).total_seconds()

        self.report.overall_status = overall_status

    def generate_text_report(self) -> str:  # noqa: PLR0912, PLR0915
        """Generate a text report."""
        lines = []
        lines.append("=" * 80)
        lines.append("THINGS-EEG2 DATASET CLI APPLICATION REPORT")
        lines.append("=" * 80)
        lines.append("")

        # Header information
        lines.append(f"Pipeline Version: {self.report.pipeline_version}")
        lines.append(f"Dataset Name: {self.report.dataset_name}")
        lines.append(f"Overall Status: {self.report.overall_status.upper()}")
        lines.append("")

        # Execution time
        lines.append("-" * 80)
        lines.append("EXECUTION TIME")
        lines.append("-" * 80)
        lines.append(f"Start Time: {self.report.execution_start}")
        if self.report.execution_end:
            lines.append(f"End Time: {self.report.execution_end}")
            lines.append(
                f"Total Elapsed Time: {self._format_time(self.report.total_elapsed_time)}"  # type: ignore
            )
        lines.append("")

        # Data paths
        lines.append("-" * 80)
        lines.append("DATA PATHS")
        lines.append("-" * 80)
        lines.append(f"Input Data Path: {self.report.input_data_path or 'N/A'}")
        lines.append(f"Output Data Path: {self.report.output_data_path or 'N/A'}")
        lines.append("")

        # Dataset information
        lines.append("-" * 80)
        lines.append("DATASET INFORMATION")
        lines.append("-" * 80)
        lines.append(
            f"Number of Subjects Processed: {self.report.n_subjects_processed or 'N/A'}"
        )
        lines.append(f"Number of Sessions: {self.report.n_sessions or 'N/A'}")
        lines.append("")

        # Stage summary
        lines.append("-" * 80)
        lines.append("STAGES SUMMARY")
        lines.append("-" * 80)
        executed = sum(
            1 for s in self.report.stages if s.status == StageStatus.EXECUTED
        )
        skipped = sum(1 for s in self.report.stages if s.status == StageStatus.SKIPPED)
        failed = sum(1 for s in self.report.stages if s.status == StageStatus.FAILED)

        lines.append(f"Total Stages: {len(self.report.stages)}")
        lines.append(f"  - Executed: {executed}")
        lines.append(f"  - Skipped: {skipped}")
        lines.append(f"  - Failed: {failed}")
        lines.append("")

        # Detailed stage information
        lines.append("-" * 80)
        lines.append("DETAILED STAGE REPORTS")
        lines.append("-" * 80)

        for i, stage in enumerate(self.report.stages, 1):
            lines.append("")
            lines.append(f"[{i}] {stage.name}")
            lines.append(f"    Status: {stage.status.value.upper()}")

            if stage.start_time:
                lines.append(f"    Start Time: {stage.start_time}")
            if stage.end_time:
                lines.append(f"    End Time: {stage.end_time}")
            if stage.elapsed_time is not None:
                lines.append(
                    f"    Elapsed Time: {self._format_time(stage.elapsed_time)}"  # type: ignore
                )

            if stage.parameters:
                lines.append("    Parameters:")
                for key, value in stage.parameters.items():
                    if key.lower() == "embedding":
                        lines.append(f"      - Model used: {value}")
                    else:
                        lines.append(f"      - {key}: {value}")

            # Download-specific statistics
            if stage.requested_subjects is not None:
                lines.append("    Download Statistics:")
                lines.append(
                    f"      - Requested: {len(stage.requested_subjects or [])} subjects"
                )
                lines.append(
                    f"      - Downloaded: {len(stage.downloaded_subjects or [])} subjects"
                )
                if stage.total_download_size is not None:
                    lines.append(
                        f"      - Total Size: {stage.total_download_size:.2f} GB"
                    )
                if stage.missing_files:
                    lines.append(
                        f"      - Missing Files: {len(stage.missing_files)} issues detected"
                    )

            # Preprocessing-specific statistics
            if stage.sampling_frequency is not None:
                lines.append("    Preprocessing Statistics:")
                lines.append(
                    f"      - Sampling Frequency: {stage.original_sampling_frequency} Hz → {stage.sampling_frequency} Hz"
                )
                if stage.num_channels:
                    lines.append(f"      - EEG Channels: {stage.num_channels}")
                if stage.filter_frequencies:
                    h_freq = stage.filter_frequencies.get("h_freq", "N/A")
                    lines.append(f"      - Low-pass Filter: {h_freq} Hz")
                if stage.epoch_time_window:
                    tmin = stage.epoch_time_window.get("tmin")
                    tmax = stage.epoch_time_window.get("tmax")
                    lines.append(f"      - Epoch Window: [{tmin}, {tmax}] s")
                if stage.processing_tasks:
                    lines.append(f"      - Tasks: {len(stage.processing_tasks)} steps")

            # Embedding-specific statistics
            if stage.model_name is not None:
                lines.append("    Embedding Statistics:")
                lines.append(f"      - Model: {stage.model_name}")
                if stage.batch_size:
                    lines.append(f"      - Batch Size: {stage.batch_size}")
                if stage.precision:
                    lines.append(f"      - Precision: {stage.precision}")
                if stage.num_images_processed:
                    total = sum(stage.num_images_processed.values())
                    lines.append(f"      - Images Processed: {total}")
                if stage.embedding_variants:
                    lines.append(
                        f"      - Variants: {', '.join(stage.embedding_variants)}"
                    )

            # Validation results
            if stage.validation_status is not None:
                lines.append("    Validation Results:")
                status_emoji = {
                    "passed": "✓",
                    "warning": "⚠",
                    "failed": "✗",
                }
                emoji = status_emoji.get(stage.validation_status, "?")
                lines.append(
                    f"      - Status: {emoji} {stage.validation_status.upper()}"
                )

                if stage.validation_checks_passed is not None:
                    lines.append(
                        f"      - Checks Passed: {stage.validation_checks_passed}"
                    )
                if (
                    stage.validation_warnings is not None
                    and stage.validation_warnings > 0
                ):
                    lines.append(f"      - Warnings: {stage.validation_warnings}")
                if (
                    stage.validation_checks_failed is not None
                    and stage.validation_checks_failed > 0
                ):
                    lines.append(
                        f"      - Checks Failed: {stage.validation_checks_failed}"
                    )

                # Show validation issues if any
                if stage.validation_issues and len(stage.validation_issues) > 0:
                    lines.append("      - Issues Detected:")
                    for issue in stage.validation_issues[:5]:  # Show first 5 issues
                        severity = issue.get("severity", "unknown")
                        check_name = issue.get("check_name", "Unknown Check")
                        message = issue.get("message", "No message")
                        lines.append(
                            f"        [{severity.upper()}] {check_name}: {message}"
                        )

                    if len(stage.validation_issues) > 5:  # noqa: PLR2004
                        lines.append(
                            f"        ... and {len(stage.validation_issues) - 5} more issues"
                        )

            if stage.files_produced:
                lines.append(f"    Files Produced: {len(stage.files_produced)}")
                for file in stage.files_produced[
                    :10
                ]:  # Show first 10 files only for avoiding overcluttering
                    lines.append(f"      - {file}")  # noqa: PERF401
                if len(stage.files_produced) > 10:  # noqa: PLR2004
                    lines.append(f"      ... and {len(stage.files_produced) - 10} more")

            if stage.error_log:
                lines.append("    Error Log:")
                for line in stage.error_log.split("\n"):
                    lines.append(f"      {line}")  # noqa: PERF401

        lines.append("")

        # System information
        lines.append("-" * 80)
        lines.append("SYSTEM INFORMATION")
        lines.append("-" * 80)
        for key, value in self.report.system_info.items():
            lines.append(f"{key.replace('_', ' ').title()}: {value}")
        lines.append("")

        lines.append("=" * 80)
        lines.append(f"Report generated at: {datetime.now().isoformat()}")  # noqa: DTZ005
        lines.append("=" * 80)

        return "\n".join(lines)

    def save_text_report(self, filename: str | None = None) -> Path:
        """Save the text report to a file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # noqa: DTZ005
            filename = f"process_report_{timestamp}.txt"

        report_path = self.reports_dir / filename
        report_text = self.generate_text_report()

        with open(report_path, "w") as f:  # noqa: PTH123
            f.write(report_text)

        return report_path

    def save_json_report(self, filename: str | None = None) -> Path:
        """Save the report as JSON for programmatic access."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # noqa: DTZ005
            filename = f"process_report_{timestamp}.json"

        report_path = self.reports_dir / filename

        # Convert report to dict
        report_dict = asdict(self.report)
        # Convert enums to strings
        for stage in report_dict["stages"]:
            stage["status"] = (
                stage["status"].value
                if isinstance(stage["status"], StageStatus)
                else stage["status"]
            )

        with open(report_path, "w") as f:  # noqa: PTH123
            json.dump(report_dict, f, indent=2)

        return report_path

    def generate_stage_text_report(self, stage: StageReport) -> str:  # noqa: PLR0912, PLR0915
        """Generate a human-readable text report for a single stage."""
        lines = []
        lines.append("=" * 80)
        lines.append(f"STAGE REPORT: {stage.name.upper()}")
        lines.append("=" * 80)
        lines.append("")

        lines.append(f"Status: {stage.status.value.upper()}")
        lines.append(f"Start Time: {stage.start_time or 'N/A'}")
        lines.append(f"End Time: {stage.end_time or 'N/A'}")
        if stage.elapsed_time is not None:
            lines.append(f"Elapsed Time: {self._format_time(stage.elapsed_time)}")  # type: ignore
        lines.append("")

        if stage.parameters:
            lines.append("-" * 80)
            lines.append("PARAMETERS")
            lines.append("-" * 80)
            for key, value in stage.parameters.items():
                lines.append(f"{key}: {value}")
            lines.append("")

        # Download-specific information
        if stage.requested_subjects is not None:
            lines.append("-" * 80)
            lines.append("DOWNLOAD STATISTICS")
            lines.append("-" * 80)
            lines.append(f"Requested Subjects: {stage.requested_subjects}")
            lines.append(f"Downloaded Subjects: {stage.downloaded_subjects or []}")
            lines.append(
                f"Successfully Downloaded: {len(stage.downloaded_subjects or [])}/{len(stage.requested_subjects or [])}"
            )
            lines.append(
                f"Source Data Downloaded: {'Yes' if stage.source_data_downloaded else 'No'}"
            )

            if stage.total_download_size is not None:
                lines.append(f"Total Download Size: {stage.total_download_size:.2f} GB")

            if stage.missing_files:
                lines.append(f"\nMissing Files/Issues ({len(stage.missing_files)}):")
                for missing in stage.missing_files[:20]:  # Limit to first 20
                    lines.append(f"  - {missing}")  # noqa: PERF401
                if len(stage.missing_files) > 20:  # noqa: PLR2004
                    lines.append(f"  ... and {len(stage.missing_files) - 20} more")
            else:
                lines.append("\nNo missing files detected.")
            lines.append("")

        # Preprocessing-specific information
        if stage.sampling_frequency is not None:
            lines.append("-" * 80)
            lines.append("PREPROCESSING STATISTICS")
            lines.append("-" * 80)

            if stage.original_sampling_frequency:
                lines.append(
                    f"Original Sampling Frequency: {stage.original_sampling_frequency} Hz"
                )
            lines.append(f"Target Sampling Frequency: {stage.sampling_frequency} Hz")

            if stage.num_channels:
                lines.append(f"Number of EEG Channels: {stage.num_channels}")

            if stage.num_sessions_processed:
                lines.append(
                    f"Sessions Processed per Subject: {stage.num_sessions_processed}"
                )

            if stage.processing_tasks:
                lines.append("\nProcessing Tasks Performed:")
                for task in stage.processing_tasks:
                    lines.append(f"  • {task}")  # noqa: PERF401

            if stage.filter_type or stage.filter_frequencies:
                lines.append("\nFiltering:")
                if stage.filter_type:
                    lines.append(f"  Filter Type: {stage.filter_type}")
                if stage.filter_frequencies:
                    l_freq = stage.filter_frequencies.get("l_freq", "None")
                    h_freq = stage.filter_frequencies.get("h_freq", "None")
                    lines.append(
                        f"  Low Frequency: {l_freq} Hz"
                        if l_freq != "None"
                        else "  Low Frequency: None (high-pass not applied)"
                    )
                    lines.append(
                        f"  High Frequency: {h_freq} Hz"
                        if h_freq != "None"
                        else "  High Frequency: None (low-pass not applied)"
                    )

            if stage.epoch_time_window:
                tmin = stage.epoch_time_window.get("tmin", "N/A")
                tmax = stage.epoch_time_window.get("tmax", "N/A")
                lines.append("\nEpoching:")
                lines.append(f"  Time Window: [{tmin}, {tmax}] seconds")
                lines.append(f"  Duration: {float(tmax) - float(tmin):.2f} seconds")

            if stage.baseline_correction:
                lines.append("\nBaseline Correction:")
                baseline_start = stage.baseline_correction.get("start", "N/A")
                baseline_end = stage.baseline_correction.get("end", "N/A")
                lines.append(
                    f"  Baseline Period: [{baseline_start}, {baseline_end}] seconds"
                )
                if baseline_start == "None":
                    lines.append(f"  (from start of epoch to time {baseline_end})")

            lines.append("")

        # Embedding-specific information
        if stage.model_name is not None:
            lines.append("-" * 80)
            lines.append("EMBEDDING GENERATION STATISTICS")
            lines.append("-" * 80)

            lines.append(f"Model: {stage.model_name}")

            if stage.batch_size:
                lines.append(f"Batch Size: {stage.batch_size}")

            if stage.precision:
                lines.append(f"Precision: {stage.precision}")

            if stage.embedding_dimension:
                lines.append("\nEmbedding Dimensions:")
                for variant, dim in stage.embedding_dimension.items():
                    lines.append(f"  {variant.title()}: {dim}")

            if stage.embedding_variants:
                lines.append("\nEmbedding Variants Generated:")
                for variant in stage.embedding_variants:
                    lines.append(f"  • {variant}")  # noqa: PERF401

            if stage.num_images_processed:
                lines.append("\nImages Processed:")
                total_images = sum(stage.num_images_processed.values())
                for partition, count in stage.num_images_processed.items():
                    lines.append(f"  {partition.title()}: {count} images")
                lines.append(f"  Total: {total_images} images")

            lines.append("")

        if stage.files_produced:
            lines.append("-" * 80)
            lines.append(f"FILES PRODUCED ({len(stage.files_produced)})")
            lines.append("-" * 80)
            for file in stage.files_produced:
                lines.append(f"  - {file}")  # noqa: PERF401
            lines.append("")

        if stage.error_log:
            lines.append("-" * 80)
            lines.append("ERROR LOG")
            lines.append("-" * 80)
            lines.append(stage.error_log)
            lines.append("")

        lines.append("=" * 80)
        lines.append(f"Report generated at: {datetime.now().isoformat()}")  # noqa: DTZ005
        lines.append("=" * 80)

        return "\n".join(lines)

    def save_stage_report_text(self, stage: StageReport, filename: str) -> Path:
        """Save a single stage report as text."""
        report_path = self.reports_dir / filename
        report_text = self.generate_stage_text_report(stage)

        with open(report_path, "w") as f:  # noqa: PTH123
            f.write(report_text)

        return report_path

    def save_stage_report_json(self, stage: StageReport, filename: str) -> Path:
        """Save a single stage report as JSON."""
        report_path = self.reports_dir / filename

        stage_dict = asdict(stage)
        stage_dict["status"] = stage.status.value

        with open(report_path, "w") as f:  # noqa: PTH123
            json.dump(stage_dict, f, indent=2)

        return report_path

    @staticmethod
    def _format_time(seconds: float) -> str:
        """Format time in seconds to human-readable format."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60

        if hours > 0:
            return f"{hours}h {minutes}m {secs:.2f}s"
        elif minutes > 0:
            return f"{minutes}m {secs:.2f}s"
        else:
            return f"{secs:.2f}s"
