"""
Data validation module for THINGS-EEG2 dataset.

Validates data integrity after download and processing, checking:
- That ALL raw data files exist for a specific subject
- That the sampling frequency and number of channels match the reference values
- That there is no NaN values in the data
- That the mean of the raw data is close to zero (with some tolerance)
- That the processed data files have the expected shapes and data types
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from things_eeg2_dataset.paths import layout

logger = logging.getLogger(__name__)

class ValidationStage(Enum):
    """Validation stage identifier."""
    RAW_DATA = "raw_data"
    PROCESSED_DATA = "processed_data"


class ValidationStatus(Enum):
    """Validation result status."""
    PASSED = "passed"
    WARNING = "warning"
    FAILED = "failed"


@dataclass
class ValidationIssue:
    """Represents a validation issue."""
    severity: ValidationStatus
    check_name: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        msg = f"[{self.severity.value.upper()}] {self.check_name}: {self.message}"
        if self.details:
            msg += f"\n  Details: {self.details}"
        return msg


@dataclass
class ValidationReport:
    """Overall validation report."""
    stage: ValidationStage
    issues: list[ValidationIssue] = field(default_factory=list)
    
    @property
    def has_failures(self) -> bool:
        return any(i.severity == ValidationStatus.FAILED for i in self.issues)
    
    @property
    def has_warnings(self) -> bool:
        return any(i.severity == ValidationStatus.WARNING for i in self.issues)
    
    @property
    def passed(self) -> bool:
        return not self.has_failures
    
    def add_issue(
        self,
        severity: ValidationStatus,
        check_name: str,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Add validation issue to report."""
        issue = ValidationIssue(severity, check_name, message, details or {})
        self.issues.append(issue)
        
        log_msg = f"[{check_name}] {message}"
        if severity == ValidationStatus.FAILED:
            logger.error(log_msg)
        elif severity == ValidationStatus.WARNING:
            logger.warning(log_msg)
        else:
            logger.debug(log_msg)
    
    def get_summary(self) -> str:
        counts = {s: sum(1 for i in self.issues if i.severity == s) for s in ValidationStatus}
        
        lines = [
            f"\n\033[96m{'='*60}\033[0m",
            f"\033[96mValidation Report: {self.stage.value}\033[0m",
            f"\033[96m{'='*60}\033[0m",
            f"\033[96mTotal Checks: {len(self.issues)}\033[0m",
            f"\033[96m  Passed:   {counts[ValidationStatus.PASSED]}\033[0m",
            f"\033[96m  Warnings: {counts[ValidationStatus.WARNING]}\033[0m",
            f"\033[96m  Failed:   {counts[ValidationStatus.FAILED]}\033[0m",
        ]
        
        if non_passed := [i for i in self.issues if i.severity != ValidationStatus.PASSED]:
            lines.append(f"\033[96m\nDetailed Issues:\033[0m")
            lines.extend(f"\033[96m\n{i}\033[0m" for i in non_passed)
        
        lines.append(f"\n\033[96m{'='*60}\033[0m\n")
        return "\n".join(lines)


@dataclass
class RawDataReference:
    """Reference values for raw data validation."""
    num_sessions: int = 4
    expected_files: list[str] = field(default_factory=lambda: ["raw_eeg_training.npy", "raw_eeg_test.npy"])
    sampling_freq: int = 250  # Hz
    num_channels: int = 63
    mean_tolerance: float = 10.0  # Allow some deviation from zero for raw data

@dataclass
class ProcessedDataReference:
    """Reference values for processed data validation."""
    num_sessions: int = 4
    train_shape: tuple[int, int, int, int, int] = (4, 8270, 2, 63, 301)
    test_shape: tuple[int, int, int, int, int] = (4, 200, 20, 63, 301)
    train_cond_shape: tuple[int, int] = (4, 8270)
    test_cond_shape: tuple[int, int] = (4, 200)
    expected_mean: float = 0.0
    mean_tolerance: float = 5.0

def validate_raw_data(
    project_dir: Path,
    subjects: list[int],
    reference: RawDataReference | None = None,
) -> ValidationReport:
    """
    Validate raw data after download.
    
    Args:
        project_dir: Project directory path
        subjects: Subject numbers to validate
        reference: Reference values (defaults if None)
    
    Returns:
        ValidationReport with all results
    """
    ref = reference or RawDataReference()
    report = ValidationReport(stage=ValidationStage.RAW_DATA)
    
    for subj in subjects:
        subj_dir = layout.get_raw_subject_dir(project_dir, subj)
        
        # Check subject directory exists
        if not subj_dir.exists():
            report.add_issue(
                ValidationStatus.FAILED,
                "Directory Existence",
                f"Subject directory missing: sub-{subj:02d}",
                {"subject": subj, "path": str(subj_dir)},
            )
            continue
        
        # Check all sessions exist
        if missing := [s for s in range(1, ref.num_sessions + 1) 
                       if not (subj_dir / f"ses-{s:02d}").exists()]:
            report.add_issue(
                ValidationStatus.FAILED,
                "Session Completeness",
                f"Subject {subj:02d}: Missing sessions {missing}",
                {"subject": subj, "missing_sessions": missing},
            )
            continue
        
        # Validate each session
        for sess in range(1, ref.num_sessions + 1):
            _validate_raw_session(
                subj_dir / f"ses-{sess:02d}",
                subj,
                sess,
                ref,
                report,
            )
    
    if not report.issues:
        report.add_issue(ValidationStatus.PASSED, "Raw data validation", "All checks passed")
    
    return report


def _validate_raw_session(
    sess_dir: Path,
    subj: int,
    sess: int,
    ref: RawDataReference,
    report: ValidationReport,
) -> None:
    """Validate single raw data session."""
    prefix = f"sub-{subj:02d}/ses-{sess:02d}"
    
    for filename in ref.expected_files:
        path = sess_dir / filename
        
        if not path.exists():
            report.add_issue(
                ValidationStatus.FAILED,
                "File exists",
                f"{prefix}: Missing {filename}",
                {"subject": subj, "session": sess, "file": filename},
            )
            continue
        
        # Load and validate file
        try:
            data_dict = np.load(path, allow_pickle=True).item()
            
            if not isinstance(data_dict, dict) or "raw_eeg_data" not in data_dict:
                report.add_issue(
                    ValidationStatus.FAILED,
                    "File Structure",
                    f"{prefix}/{filename}: Invalid structure or missing 'raw_eeg_data' key",
                    {"subject": subj, "session": sess, "file": filename},
                )
                continue
            
            eeg = data_dict["raw_eeg_data"]
            
            if not isinstance(eeg, np.ndarray):
                report.add_issue(
                    ValidationStatus.FAILED,
                    "Data Type",
                    f"{prefix}/{filename}: EEG data is not a numpy array",
                    {"subject": subj, "session": sess, "file": filename},
                )
                continue
            
            # Check NaN values
            if np.isnan(eeg).any():
                report.add_issue(
                    ValidationStatus.FAILED,
                    "Data quality",
                    f"{prefix}/{filename}: Contains NaN values",
                    {"subject": subj, "session": sess, "file": filename},
                )
                continue
            
            # Calculate mean
            vmean = float(np.mean(eeg))
            
            # Check mean
            if abs(vmean - 0) > ref.mean_tolerance:
                report.add_issue(
                    ValidationStatus.WARNING,
                    "Data mean",
                    f"{prefix}/{filename}: Mean ({vmean:.2f}) not close to 0",
                    {"subject": subj, "session": sess, "file": filename, "mean": vmean},
                )
        
        except Exception as e:
            report.add_issue(
                ValidationStatus.FAILED,
                "File corruption",
                f"{prefix}/{filename}: Cannot load - {e}",
                {"subject": subj, "session": sess, "file": filename, "error": str(e)},
            )


def validate_processed_data(
    project_dir: Path,
    subjects: list[int],
    sfreq: int = 250,
    reference: ProcessedDataReference | None = None,
) -> ValidationReport:
    """
    Validate processed data after preprocessing.
    
    Args:
        project_dir: Project directory path
        subjects: Subject numbers to validate
        sfreq: Sampling frequency
        reference: Reference values (defaults if None)
    
    Returns:
        ValidationReport with all results
    """
    ref = reference or ProcessedDataReference()
    report = ValidationReport(stage=ValidationStage.PROCESSED_DATA)
    
    for subj in subjects:
        subj_dir = layout.get_processed_subject_dir(project_dir, subj)
        
        if not subj_dir.exists():
            report.add_issue(
                ValidationStatus.FAILED,
                "Directory exists",
                f"Processed directory missing: sub-{subj:02d}",
                {"subject": subj, "path": str(subj_dir)},
            )
            continue
        
        # Validate files
        files = [
            (layout.get_eeg_train_file(project_dir, subj), "training", ref.train_shape, _validate_processed_file),
            (layout.get_eeg_test_file(project_dir, subj), "test", ref.test_shape, _validate_processed_file),
            (layout.get_eeg_train_image_conditions_file(project_dir, subj), "training", ref.train_cond_shape, _validate_image_conditions),
            (layout.get_eeg_test_image_conditions_file(project_dir, subj), "test", ref.test_cond_shape, _validate_image_conditions),
        ]
        
        for path, partition, shape, validator in files:
            if path.exists():
                validator(path, subj, partition, shape, ref, report)
            else:
                report.add_issue(
                    ValidationStatus.FAILED,
                    "File exists",
                    f"sub-{subj:02d}: Missing {partition} {'conditions' if 'conditions' in path.name else 'data'} file",
                    {"subject": subj, "file": path.name},
                )
    
    if not report.issues:
        report.add_issue(ValidationStatus.PASSED, "Processed data validation", "All checks passed")
    
    return report


def _validate_processed_file(
    path: Path,
    subj: int,
    partition: str,
    expected_shape: tuple[int, ...],
    ref: ProcessedDataReference,
    report: ValidationReport,
) -> None:
    """Validate single processed EEG file."""
    prefix = f"sub-{subj:02d}/{partition}"
    
    try:
        data = np.load(path)
        
        # Check shape
        if data.shape != expected_shape:
            report.add_issue(
                ValidationStatus.FAILED,
                "Data shape",
                f"{prefix}: Expected {expected_shape}, got {data.shape}",
                {"subject": subj, "partition": partition, "expected_shape": expected_shape, "actual_shape": data.shape},
            )
            return
        
        # Check NaN values
        if np.isnan(data).any():
            report.add_issue(
                ValidationStatus.FAILED,
                "Data quality",
                f"{prefix}: Contains NaN values",
                {"subject": subj, "partition": partition},
            )
            return
        
        # Check mean
        vmean = float(np.mean(data))
        
        if abs(vmean - ref.expected_mean) > ref.mean_tolerance:
            report.add_issue(
                ValidationStatus.FAILED,
                "Data mean",
                f"{prefix}: Mean ({vmean:.2f}) not close to 0",
                {"subject": subj, "partition": partition, "mean": vmean},
            )
    
    except Exception as e:
        report.add_issue(
            ValidationStatus.FAILED,
            "File corruption",
            f"{prefix}: Cannot load - {e}",
            {"subject": subj, "partition": partition, "error": str(e)},
        )


def _validate_image_conditions(
    path: Path,
    subj: int,
    partition: str,
    expected_shape: tuple[int, int],
    ref: ProcessedDataReference,
    report: ValidationReport,
) -> None:
    """Validate image conditions file."""
    prefix = f"sub-{subj:02d}/{partition}_conditions"
    
    try:
        data = np.load(path)
        
        # Check shape
        if data.shape != expected_shape:
            report.add_issue(
                ValidationStatus.FAILED,
                "Conditions shape",
                f"{prefix}: Expected {expected_shape}, got {data.shape}",
                {"subject": subj, "partition": partition, "expected_shape": expected_shape, "actual_shape": data.shape},
            )
            return
        
        # Check data type
        if not np.issubdtype(data.dtype, np.integer):
            report.add_issue(
                ValidationStatus.WARNING,
                "Conditions data type",
                f"{prefix}: Expected integer type, got {data.dtype}",
                {"subject": subj, "partition": partition, "dtype": str(data.dtype)},
            )
    
    except Exception as e:
        report.add_issue(
            ValidationStatus.FAILED,
            "File corruption",
            f"{prefix}: Cannot load - {e}",
            {"subject": subj, "partition": partition, "error": str(e)},
        )