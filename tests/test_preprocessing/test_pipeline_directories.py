import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from things_eeg2_dataset.processing.pipeline import (
    PipelineConfig,
    ProjectPaths,
    ThingsEEGPipeline,
)


def test_project_paths_definitions(tmp_path: Path) -> None:
    """Test that ProjectPaths defines correct paths."""
    root = tmp_path / "test_project"
    processed_name = "custom_processed"

    paths = ProjectPaths(root, processed_name)

    assert paths.root == root
    assert paths.processed == root / processed_name
    assert paths.raw_data == root / "raw_data"
    assert paths.images == root / "Image_set"
    assert paths.embeddings == root / "embeddings"


def test_make_structure_idempotent(tmp_path: Path) -> None:
    """Test that make_structure can be called multiple times."""
    root = tmp_path / "test_project"
    paths = ProjectPaths(root, "processed")

    paths.make_structure()
    # Should not crash
    paths.make_structure()

    assert paths.processed.exists()


# =============================================================================
# Pipeline Orchestration Tests (Testing ThingsEEGPipeline)
# =============================================================================


@pytest.fixture
def mock_pipeline(tmp_path: Path) -> ThingsEEGPipeline:
    """Fixture to create a pipeline instance with mocked dependencies."""
    config = PipelineConfig(
        project_dir=tmp_path,
        subjects=[1],
        models=["test_model"],
        processed_dir_name="test_processed",
    )
    return ThingsEEGPipeline(config)


def test_pipeline_initializes_correct_paths(
    mock_pipeline: ThingsEEGPipeline, tmp_path: Path
) -> None:
    """Test that the pipeline sets up paths correctly from config."""
    assert mock_pipeline.paths.processed == tmp_path / "test_processed"
    assert mock_pipeline.paths.root == tmp_path


# =============================================================================
# Error Handling Tests
# =============================================================================


def test_check_raw_data_fails_missing_files(mock_pipeline: ThingsEEGPipeline) -> None:
    """Test validation of raw data existence."""
    # We haven't created any files in tmp_path, so this should fail
    assert mock_pipeline._check_raw_data() is False


def test_check_raw_data_succeeds_with_files(
    mock_pipeline: ThingsEEGPipeline, tmp_path: Path
) -> None:
    """Test validation succeeds when files exist."""
    sub_dirs = [
        tmp_path / "raw_data" / "sub-01" / session
        for session in ["ses-01", "ses-02", "ses-03", "ses-04"]
    ]
    for sub_dir in sub_dirs:
        sub_dir.mkdir(parents=True)
        (sub_dir / "raw_eeg_training.npy").touch()

    assert mock_pipeline._check_raw_data() is True
