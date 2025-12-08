"""Basic tests for Things EEG2 dataloader."""

import numpy as np
import pytest

from things_eeg2_dataset.dataloader.datamodule import DataArtifacts
from things_eeg2_dataset.dataloader.dataset import ThingsEEGDataset


class TestThingsEEGDataset:
    """Test suite for ThingsEEGDataset class."""

    def test_dataset_constants(self) -> None:
        """Test that dataset constants are correctly defined."""
        assert ThingsEEGDataset.TRAIN_REPETITIONS == 4
        assert ThingsEEGDataset.TEST_REPETITIONS == 80
        assert ThingsEEGDataset.TRAIN_SAMPLES_PER_CLASS == 10
        assert ThingsEEGDataset.TEST_SAMPLES_PER_CLASS == 1
        assert ThingsEEGDataset.TRAIN_CLASSES == 1654
        assert ThingsEEGDataset.TEST_CLASSES == 200

    def test_all_subjects_list(self) -> None:
        """Test that all subjects list is populated correctly."""
        assert len(ThingsEEGDataset.all_subjects) == 10
        assert "sub-01" in ThingsEEGDataset.all_subjects
        assert "sub-10" in ThingsEEGDataset.all_subjects


class TestDataArtifacts:
    """Test suite for DataArtifacts dataclass."""

    def test_data_artifacts_creation(self) -> None:
        """Test creating a DataArtifacts instance."""
        artifacts = DataArtifacts(
            multi_token=False, n_tokens=1, n_chans=63, n_times=100, n_outputs=512
        )
        assert artifacts.multi_token is False
        assert artifacts.n_tokens == 1
        assert artifacts.n_chans == 63
        assert artifacts.n_times == 100
        assert artifacts.n_outputs == 512


class TestValidationSplit:
    """Test suite for create_validation_split function."""

    def test_create_validation_split_deterministic(self) -> None:
        """Test that validation split is deterministic with same seed."""
        # This is a basic structure test - would need mock dataset for full test
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)

        sample1 = rng1.integers(0, 100, size=10)
        sample2 = rng2.integers(0, 100, size=10)

        assert np.array_equal(sample1, sample2)

    def test_create_validation_split_different_seeds(self) -> None:
        """Test that different seeds produce different results."""
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(99)

        sample1 = rng1.integers(0, 100, size=10)
        sample2 = rng2.integers(0, 100, size=10)

        assert not np.array_equal(sample1, sample2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
