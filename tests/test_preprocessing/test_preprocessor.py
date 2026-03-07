from pathlib import Path

import pytest

from things_eeg2_dataset.processing import (
    RawProcessor,
)


@pytest.fixture
def mock_project_dir(tmp_path: Path) -> Path:
    """Create a mock project directory."""
    (tmp_path / "processed").mkdir()
    (tmp_path / "images" / "train").mkdir(parents=True)
    (tmp_path / "images" / "test").mkdir(parents=True)
    return tmp_path


@pytest.fixture
def processor(mock_project_dir: Path) -> RawProcessor:
    """Return a RawProcessor instance for testing."""
    return RawProcessor(
        subjects=[1],
        project_dir=mock_project_dir,
        sfreq=250,
    )


def test_apply_mvnn_raises_error(processor: RawProcessor) -> None:
    """Test that apply_mvnn raises ValueError when called before epoching data."""
    with pytest.raises(ValueError, match="Epoched data not found"):
        processor.apply_mvnn()


def test_save_preprocessed_data_raises_error(processor: RawProcessor) -> None:
    """Test that save_preprocessed_data raises ValueError when called before MVNN whitening."""
    with pytest.raises(ValueError, match="Whitened data not found"):
        processor.save_preprocessed_data(1)
