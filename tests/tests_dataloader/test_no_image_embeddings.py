from __future__ import annotations

import numpy as np
import torch

from things_eeg2_dataset.dataloader.config import DatasetConfig
from things_eeg2_dataset.dataloader.dataset import ThingsEEGDataset
from things_eeg2_dataset.paths import layout


def test_dataset_runs_without_image_embeddings(tmp_path) -> None:
    project_dir = tmp_path
    subject = 1

    layout.get_processed_subject_dir(project_dir, subject).mkdir(
        parents=True, exist_ok=True
    )

    # Minimal EEG: (sessions=1, conditions=1, reps=2, ch=2, t=5)
    np.save(
        layout.get_eeg_train_file(project_dir, subject),
        np.zeros((1, 1, 2, 2, 5), dtype=np.float32),
    )
    np.save(
        layout.get_eeg_train_image_conditions_file(project_dir, subject),
        np.array([[1]], dtype=np.int64),
    )

    layout.get_metadata_file(project_dir, subject).write_text(
        '{"ch_names": ["C1", "C2"], "times": [0.0, 0.25, 0.5, 0.75, 1.0]}',
        encoding="utf-8",
    )

    # Minimal image tree
    img_root = layout.get_training_images_dir(project_dir)
    class_dir = img_root / "00001_dummy"
    class_dir.mkdir(parents=True, exist_ok=True)
    (class_dir / "00001.jpg").write_bytes(b"\xff\xd8\xff\xd9")

    ds = ThingsEEGDataset(
        DatasetConfig(
            project_dir=project_dir,
            subjects=[subject],
            partition="training",
            use_image_embeddings=False,
            image_model=None,
        )
    )
    item = ds[0]
    assert isinstance(item.image_embedding, torch.Tensor)
    assert item.image_embedding.numel() == 0
    assert item.channel_positions.shape == (2, 2)
