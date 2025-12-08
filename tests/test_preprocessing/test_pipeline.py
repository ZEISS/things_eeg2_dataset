import logging
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add source to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from things_eeg2_dataset.processing.pipeline import (
    PipelineConfig,
    ProjectPaths,
    ThingsEEGPipeline,
)


def test_project_paths_custom_processed_name(tmp_path: Path) -> None:
    """Test that ProjectPaths handles custom processed directory names."""
    project_dir = tmp_path / "test_project"
    paths = ProjectPaths(project_dir, "custom_processed")

    # Verify path construction
    assert paths.processed == project_dir / "custom_processed"
    assert paths.raw_data == project_dir / "raw_data"


def test_make_structure_preserves_existing_content(tmp_path: Path) -> None:
    """Test that creating structure doesn't wipe existing files."""
    root = tmp_path / "test_project"
    root.mkdir()

    # Create a pre-existing file
    (root / "important.txt").write_text("keep me")

    paths = ProjectPaths(root, "processed")
    paths.make_structure()

    assert (root / "important.txt").read_text() == "keep me"


# =============================================================================
# Path Consistency Tests (Testing Argument Propagation)
# =============================================================================


@pytest.fixture
def mock_pipeline_config(tmp_path: Path) -> PipelineConfig:
    """Helper to create a base config."""
    return PipelineConfig(
        project_dir=tmp_path,
        subjects=[1],
        models=["test_model"],
        processed_dir_name="custom_processed",  # Key test parameter
        dry_run=False,
    )


@patch("things_eeg2_dataset.processing.pipeline.RawProcessor")
def test_processor_receives_correct_processed_dir(
    mock_raw_proc: MagicMock, mock_pipeline_config: PipelineConfig
) -> None:
    """Test that RawProcessor gets the custom dir name from the pipeline."""
    pipeline = ThingsEEGPipeline(mock_pipeline_config)

    # Run just the EEG step
    pipeline.step_process_eeg()

    # Verify RawProcessor initialization args
    call_kwargs = mock_raw_proc.call_args.kwargs
    assert call_kwargs["processed_dir_name"] == "custom_processed"
    assert call_kwargs["project_dir"] == str(mock_pipeline_config.project_dir)


@patch("things_eeg2_dataset.processing.pipeline.build_embedder")
def test_embedder_receives_correct_data_path(
    mock_build: MagicMock, mock_pipeline_config: PipelineConfig
) -> None:
    """Test that embedders look in the correct Image_set directory."""
    pipeline = ThingsEEGPipeline(mock_pipeline_config)

    pipeline.step_generate_embeddings()

    call_kwargs = mock_build.call_args.kwargs
    expected_path = mock_pipeline_config.project_dir / "Image_set"
    assert call_kwargs["data_path"] == str(expected_path)


# =============================================================================
# Main Integration Tests
# =============================================================================


def test_check_raw_data_missing_dirs(tmp_path: Path) -> None:
    """Test _check_raw_data fails when directories are missing."""
    cfg = PipelineConfig(tmp_path, [1], [], "proc")
    pipeline = ThingsEEGPipeline(cfg)

    assert pipeline._check_raw_data() is False


@patch("things_eeg2_dataset.processing.pipeline.setup_logging")
def test_check_raw_data_missing_subjects(
    mock_setup: MagicMock, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Test _check_raw_data identifies missing subjects."""

    # 1. Return a standard logger without resetting global config
    #    This preserves the pytest caplog handler
    mock_setup.return_value = logging.getLogger("pipeline")

    cfg = PipelineConfig(
        project_dir=tmp_path, subjects=[1, 2], models=[], processed_dir_name="proc"
    )
    pipeline = ThingsEEGPipeline(cfg)

    # Create structure for subject 1 only
    (tmp_path / "raw_data" / "sub-01").mkdir(parents=True)
    session = ["ses-01", "ses-02", "ses-03", "ses-04"]
    training_data_file = "raw_eeg_training.npy"
    for ses in session:
        ses_dir = tmp_path / "raw_data" / "sub-01" / ses
        ses_dir.mkdir(parents=True)
        (ses_dir / training_data_file).touch()

    assert pipeline._check_raw_data() is False

    # 2. Fix the assertion string
    # The code logs the list of integers, e.g., "[2]"
    # It does NOT log the string "sub-02"
    assert "Missing raw data" in caplog.text
    assert "[2]" in caplog.text
