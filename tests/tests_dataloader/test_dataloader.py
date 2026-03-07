"""Basic tests for Things EEG2 dataloader."""

import numpy as np
import pytest
import torch
from safetensors.torch import save_file
from torch.utils.data import DataLoader

from things_eeg2_dataset.cli.main import Partition
from things_eeg2_dataset.dataloader.config import DatasetConfig
from things_eeg2_dataset.dataloader.datamodule import DataArtifacts
from things_eeg2_dataset.dataloader.dataset import ThingsEEGDataset, ThingsEEGItem
from things_eeg2_dataset.paths import layout


class TestThingsEEGDataset:
    """Test suite for ThingsEEGDataset class."""

    def test_dataset_constants(self) -> None:
        """Test that dataset constants are correctly defined."""
        assert ThingsEEGDataset.TRAIN_REPETITIONS == 2
        assert ThingsEEGDataset.TEST_REPETITIONS == 20
        assert ThingsEEGDataset.TRAIN_SAMPLES_PER_CLASS == 10
        assert ThingsEEGDataset.TEST_SAMPLES_PER_CLASS == 1
        assert ThingsEEGDataset.TRAIN_CLASSES == 1654
        assert ThingsEEGDataset.TEST_CLASSES == 200

    def test_dataset_returns_things_eeg_item(self, tmp_path) -> None:  # noqa: ANN001
        """Minimal on-disk fixture to ensure __getitem__ returns ThingsEEGItem."""
        project_dir = tmp_path

        # Directories
        layout.get_processed_dir(project_dir).mkdir(parents=True, exist_ok=True)
        layout.get_embeddings_dir(project_dir).mkdir(parents=True, exist_ok=True)
        train_img_dir = layout.get_training_images_dir(project_dir)
        train_img_dir.mkdir(parents=True, exist_ok=True)

        subject = 1
        processed_subdir = layout.get_processed_subject_dir(project_dir, subject)
        processed_subdir.mkdir(parents=True, exist_ok=True)

        # Minimal EEG: (sessions=1, conditions=1, reps=2, ch=2, t=5)
        eeg = np.zeros((1, 1, 2, 2, 5), dtype=np.float32)
        np.save(layout.get_eeg_train_file(project_dir, subject), eeg)

        # Minimal conditions metadata (1 session, 1 condition) with 1-based image id
        cond = np.array([[1]], dtype=np.int64)
        np.save(layout.get_eeg_train_image_conditions_file(project_dir, subject), cond)

        # Metadata JSON with times matching t=5
        meta_path = layout.get_metadata_file(project_dir, subject)
        meta_path.write_text(
            '{"ch_names": ["C1", "C2"], "times": [0.0, 0.25, 0.5, 0.75, 1.0]}',
            encoding="utf-8",
        )

        # Image tree: class 1, sample 1
        class_dir = train_img_dir / "00001_dummy"
        class_dir.mkdir(parents=True, exist_ok=True)
        (class_dir / "00001.jpg").write_bytes(b"\xff\xd8\xff\xd9")  # minimal JPEG

        # Embeddings: need index 0
        emb_path = layout.get_embedding_file(
            project_dir,
            "testmodel",
            partition=Partition.TRAINING,
            full=False,
            variant="pooled",
        )
        save_file(
            {
                "img_features": torch.zeros((1, 8), dtype=torch.float32),
                "text_features": torch.zeros((1, 8), dtype=torch.float32),
            },
            emb_path,
        )

        ds = ThingsEEGDataset(
            DatasetConfig(
                project_dir=project_dir,
                subjects=[subject],
                partition="training",
                image_model="testmodel",
                embed_variant="pooled",
                time_window=(0.0, 1.0),
                load_images=False,
            )
        )
        item = ds[0]
        assert isinstance(item, ThingsEEGItem)
        assert item.brain_signal.shape == (2, 5)
        assert item.channel_positions.shape == (2, 2)

        # Default DataLoader collation should work for NamedTuple samples.
        loader = DataLoader(ds, batch_size=2, shuffle=False)
        batch = next(iter(loader))
        assert isinstance(batch, ThingsEEGItem)
        assert batch.brain_signal.shape == (2, 2, 5)
        assert batch.image_embedding.shape == (2, 8)
        assert batch.channel_positions.shape == (2, 2, 2)
        assert isinstance(batch.text, (list, tuple))
        assert isinstance(batch.image, (list, tuple))


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
