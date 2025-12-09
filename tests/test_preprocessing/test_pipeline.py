import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add source to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from things_eeg2_dataset.processing.pipeline import (
    PipelineConfig,
    ThingsEEGPipeline,
)

# Path Consistency Tests (Testing Argument Propagation)
# =============================================================================


@pytest.fixture
def mock_pipeline_config(tmp_path: Path) -> PipelineConfig:
    """Helper to create a base config."""
    return PipelineConfig(
        project_dir=tmp_path,
        subjects=[1],
        models=["test_model"],
        processed_dir=Path("custom_processed"),  # Key test parameter
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
    assert call_kwargs["processed_dir"] == Path("custom_processed")
    assert call_kwargs["project_dir"] == mock_pipeline_config.project_dir
