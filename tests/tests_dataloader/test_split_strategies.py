from __future__ import annotations

import numpy as np
import torch
from safetensors.torch import save_file

from things_eeg2_dataset.cli.main import Partition
from things_eeg2_dataset.dataloader import DataModuleConfig
from things_eeg2_dataset.dataloader.datamodule import ThingsEEGDataModule
from things_eeg2_dataset.paths import layout


def _make_min_project(project_dir) -> None:  # noqa: ANN001
    layout.get_processed_dir(project_dir).mkdir(parents=True, exist_ok=True)
    layout.get_embeddings_dir(project_dir).mkdir(parents=True, exist_ok=True)
    layout.get_training_images_dir(project_dir).mkdir(parents=True, exist_ok=True)
    layout.get_test_images_dir(project_dir).mkdir(parents=True, exist_ok=True)


def _write_subject_training(project_dir, subject: int, *, raw_ids: np.ndarray) -> None:  # noqa: ANN001
    """Write minimal processed training files.

    raw_ids: shape (sessions, conditions) with 1-based image ids.
    """

    layout.get_processed_subject_dir(project_dir, subject).mkdir(
        parents=True, exist_ok=True
    )

    # EEG shape: (sessions, conditions, reps=2, ch=2, t=5)
    eeg = np.zeros((raw_ids.shape[0], raw_ids.shape[1], 2, 2, 5), dtype=np.float32)
    np.save(layout.get_eeg_train_file(project_dir, subject), eeg)
    np.save(layout.get_eeg_train_image_conditions_file(project_dir, subject), raw_ids)

    meta_path = layout.get_metadata_file(project_dir, subject)
    meta_path.write_text(
        '{"ch_names": ["C1", "C2"], "times": [0.0, 0.25, 0.5, 0.75, 1.0]}',
        encoding="utf-8",
    )

    # Image folders for concept 0 (00001_*) and concept 1 (00002_*)
    for img_root in (
        layout.get_training_images_dir(project_dir),
        layout.get_test_images_dir(project_dir),
    ):
        class1 = img_root / "00001_dummy"
        class2 = img_root / "00002_dummy"
        class1.mkdir(parents=True, exist_ok=True)
        class2.mkdir(parents=True, exist_ok=True)
        (class1 / "00001.jpg").write_bytes(b"\xff\xd8\xff\xd9")
        (class2 / "00001.jpg").write_bytes(b"\xff\xd8\xff\xd9")


def _write_subject_test(project_dir, subject: int) -> None:  # noqa: ANN001
    layout.get_processed_subject_dir(project_dir, subject).mkdir(
        parents=True, exist_ok=True
    )
    # Minimal TEST EEG: (sessions=1, conditions=1, reps=1, ch=2, t=5)
    eeg = np.zeros((1, 1, 1, 2, 5), dtype=np.float32)
    np.save(layout.get_eeg_test_file(project_dir, subject), eeg)
    # One test condition (1-based)
    np.save(
        layout.get_eeg_test_image_conditions_file(project_dir, subject),
        np.array([[1]], dtype=np.int64),
    )


def _write_embeddings(project_dir) -> None:  # noqa: ANN001
    # Need index 10 (image id 11 -> 0-based 10)
    for part in (Partition.TRAINING, Partition.TEST):
        emb_path = layout.get_embedding_file(
            project_dir,
            "testmodel",
            partition=part,
            full=False,
            variant="pooled",
        )
        save_file(
            {
                "img_features": torch.zeros((11, 8), dtype=torch.float32),
                "text_features": torch.zeros((11, 8), dtype=torch.float32),
            },
            emb_path,
        )


def test_trial_split_disjoint_indices(tmp_path) -> None:  # noqa: ANN001
    project_dir = tmp_path
    _make_min_project(project_dir)
    _write_subject_training(project_dir, 1, raw_ids=np.array([[1, 11]], dtype=np.int64))
    _write_subject_test(project_dir, 1)
    _write_embeddings(project_dir)

    cfg = DataModuleConfig(
        project_dir=project_dir,
        subjects=[1],
        image_model="testmodel",
        embed_variant="pooled",
        num_workers=0,
        split_strategy="trial",
        val_units=2,
        retrieval_set_size=2,
    )
    dm = ThingsEEGDataModule(cfg)
    dm.setup()
    assert dm.train_dataset is not None
    assert dm.val_dataset is not None
    assert dm.train_eval_dataset is not None

    train_idx = set(dm.train_dataset.indices)
    val_idx = set(dm.val_dataset.indices)
    train_eval_idx = set(dm.train_eval_dataset.indices)
    assert train_idx.isdisjoint(val_idx)
    assert train_eval_idx.isdisjoint(val_idx)
    assert train_idx.isdisjoint(train_eval_idx)


def test_image_split_disjoint_image_ids(tmp_path) -> None:  # noqa: ANN001
    project_dir = tmp_path
    _make_min_project(project_dir)
    _write_subject_training(project_dir, 1, raw_ids=np.array([[1, 11]], dtype=np.int64))
    _write_subject_test(project_dir, 1)
    _write_embeddings(project_dir)

    cfg = DataModuleConfig(
        project_dir=project_dir,
        subjects=[1],
        image_model="testmodel",
        embed_variant="pooled",
        num_workers=0,
        split_strategy="image",
        val_units=1,
        retrieval_set_size=2,
        val_fraction=0.5,
    )
    dm = ThingsEEGDataModule(cfg)
    dm.setup()
    base = dm.train_dataset.dataset  # type: ignore
    image_ids = base.get_image_ids()

    train_image_ids = set(image_ids[list(dm.train_dataset.indices)])  # type: ignore
    val_image_ids = set(image_ids[list(dm.val_dataset.indices)])  # type: ignore
    assert train_image_ids.isdisjoint(val_image_ids)


def test_concept_split_disjoint_concepts(tmp_path) -> None:  # noqa: ANN001
    project_dir = tmp_path
    _make_min_project(project_dir)
    _write_subject_training(project_dir, 1, raw_ids=np.array([[1, 11]], dtype=np.int64))
    _write_subject_test(project_dir, 1)
    _write_embeddings(project_dir)

    cfg = DataModuleConfig(
        project_dir=project_dir,
        subjects=[1],
        image_model="testmodel",
        embed_variant="pooled",
        num_workers=0,
        split_strategy="concept",
        val_units=1,
        retrieval_set_size=2,
        val_fraction=0.5,
    )
    dm = ThingsEEGDataModule(cfg)
    dm.setup()
    base = dm.train_dataset.dataset  # type: ignore
    concept_ids = base.get_concept_ids()

    train_concepts = set(concept_ids[list(dm.train_dataset.indices)])  # type: ignore
    val_concepts = set(concept_ids[list(dm.val_dataset.indices)])  # type: ignore
    assert train_concepts.isdisjoint(val_concepts)
