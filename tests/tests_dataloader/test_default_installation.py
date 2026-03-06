from __future__ import annotations

import pytest

from things_eeg2_dataset.cli.main import DEFAULT_PROJECT_DIR
from things_eeg2_dataset.dataloader import (
    DataModuleConfig,
    DatasetConfig,
    ThingsEEGDataModule,
    ThingsEEGDataset,
)
from things_eeg2_dataset.paths import layout


def _default_install_available(subject: int = 1) -> bool:
    project_dir = DEFAULT_PROJECT_DIR
    if not project_dir.exists():
        return False

    required = [
        layout.get_eeg_train_file(project_dir, subject),
        layout.get_eeg_test_file(project_dir, subject),
        layout.get_eeg_train_image_conditions_file(project_dir, subject),
        layout.get_eeg_test_image_conditions_file(project_dir, subject),
        layout.get_metadata_file(project_dir, subject),
        layout.get_training_images_dir(project_dir),
        layout.get_test_images_dir(project_dir),
    ]
    if any(not p.exists() for p in required):
        return False

    # Quick check that at least one class folder exists.
    train_dir = layout.get_training_images_dir(project_dir)
    if next(train_dir.glob("00001_*"), None) is None:
        return False
    return True


def _skip_if_missing() -> None:
    if not _default_install_available(subject=1):
        pytest.skip("Default dataset install not found at ~/things_eeg2; skipping integration test")


def test_default_install_dataset_poststim_window_loads() -> None:
    _skip_if_missing()

    ds = ThingsEEGDataset(
        DatasetConfig(
            project_dir=DEFAULT_PROJECT_DIR,
            subjects=[1],
            partition="training",
            use_image_embeddings=False,
            image_model=None,
            time_window=(0.0, 1.0),
        )
    )
    item = ds[0]
    # At 250 Hz, [0.0, 1.0] inclusive is 251 samples.
    assert item.brain_signal.shape[1] == 251


def test_default_install_dataset_with_prestim_window_loads() -> None:
    _skip_if_missing()

    ds = ThingsEEGDataset(
        DatasetConfig(
            project_dir=DEFAULT_PROJECT_DIR,
            subjects=[1],
            partition="training",
            use_image_embeddings=False,
            image_model=None,
            time_window=(-0.2, 1.0),
        )
    )
    item = ds[0]
    # At 250 Hz, [-0.2, 1.0] inclusive is 301 samples.
    assert item.brain_signal.shape[1] == 301


def test_default_install_datamodule_setup_and_batch() -> None:
    _skip_if_missing()

    dm = ThingsEEGDataModule(
        DataModuleConfig(
            project_dir=DEFAULT_PROJECT_DIR,
            subjects=[1],
            use_image_embeddings=False,
            image_model=None,
            time_window=(0.0, 1.0),
            batch_size=4,
            num_workers=0,
        )
    )
    dm.setup()
    batch = next(iter(dm.train_dataloader()))
    assert batch.brain_signal.shape[2] == 251
