import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from things_eeg2_dataset.processing.pipeline import (
    ThingsEEGPipeline,
)

# =============================================================================
# Error Handling Tests
# =============================================================================


def test_check_raw_data_fails_missing_files(mock_pipeline: ThingsEEGPipeline) -> None:
    """Test validation of raw data existence."""
    # We haven't created any files in tmp_path, so this should fail
    assert mock_pipeline.validate_pipeline_inputs() is False


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

    assert mock_pipeline.validate_pipeline_inputs() is True
