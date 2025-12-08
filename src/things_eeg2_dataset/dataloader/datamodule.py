"""Data loading and feature bank utilities for EEG training."""

from __future__ import annotations

from argparse import Namespace
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import lightning as L
import numpy as np
from torch.utils.data import DataLoader, Subset

from things_eeg2_dataset.dataloader.dataset import ThingsEEGDataset


@dataclass
class DataArtifacts:
    multi_token: bool
    n_tokens: int | None
    n_chans: int
    n_times: int
    n_outputs: int


def create_validation_split(
    train_dataset: ThingsEEGDataset,
    val_classes: int | Sequence = 200,
    subject: str | None = None,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a validation split by selecting one sample per randomly chosen class.
    Args:
        train_dataset (ThingsEEGDataset | Subset): The training dataset to split.
        val_classes (int | np.ndarray): Number of classes to include in the validation set.
        subject (str | None): Subject string to anchor indexing arithmetic. Defaults to first subject in dataset.
        seed (int): RNG seed for reproducibility.
    Returns:
        train_subset (Subset): Subset of the training dataset excluding validation samples.
    """
    rng = np.random.default_rng(seed)
    ds = train_dataset

    # Pull constants
    TRAIN_CLASSES = ds.TRAIN_CLASSES
    S = ds.TRAIN_SAMPLES_PER_CLASS
    R = ds.TRAIN_REPETITIONS
    per_subject_len = TRAIN_CLASSES * S * R

    # Determine subject index
    subject = ds.included_subjects[0] if subject is None else subject
    subj_idx = ds.included_subjects.index(subject)

    # Determine validation classes
    if isinstance(val_classes, int):
        if val_classes > TRAIN_CLASSES:
            raise ValueError(
                f"Requested {val_classes} classes but only {TRAIN_CLASSES} available"
            )
        val_classes = rng.choice(
            np.arange(TRAIN_CLASSES), size=val_classes, replace=False
        )

    # Pick one sample per class (random) and a fixed repetition (0) for determinism
    if isinstance(val_classes, int):
        raise ValueError("val_classes should be an array of class indices here.")
    sample_choices = rng.integers(low=0, high=S, size=len(val_classes))
    rep_choice = 0  # could randomize this as well if desired
    val_indices: list[int] = []
    subj_offset = subj_idx * per_subject_len

    for i, class_id in enumerate(val_classes):
        sample_id = sample_choices[i]
        within_class_offset = sample_id * R + rep_choice
        local_index = class_id * (S * R) + within_class_offset
        global_index = subj_offset + local_index
        val_indices.append(global_index)

    val_idx = np.array(sorted(val_indices))
    return val_idx, val_classes


def _get_split_indices(
    train_dataset: ThingsEEGDataset,
    subjects: list[str],
    retrieval_set_size: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]:
    """
    Determines the indices for train, validation, and train-eval splits without creating subsets.

    This prevents modifying the dataset state during index selection, ensuring all splits
    are derived from the full original dataset, which avoids the bug of an empty train-eval set.

    Returns
    -------
    train_indices : np.ndarray
        Indices for the final training set.
    val_indices_per_subject : Dict[str, np.ndarray]
        A dictionary mapping each subject to their validation set indices.
    train_eval_indices : np.ndarray
        Indices for the train-eval set, disjoint from all validation sets.
    """
    val_indices_per_subject = {}
    all_val_indices = np.array([], dtype=int)

    # get random classes
    val_classes = np.random.default_rng(seed).choice(
        np.arange(train_dataset.TRAIN_CLASSES),
        size=retrieval_set_size * 2,
        replace=False,
    )
    train_eval_classes = val_classes[:retrieval_set_size]
    val_classes = val_classes[retrieval_set_size:]

    # 1. Determine validation indices for each subject from the full dataset
    for _, subject in enumerate(subjects):
        # Use a different seed per subject for validation splits to ensure diversity
        val_idx, _ = create_validation_split(
            train_dataset,
            val_classes=val_classes,
            subject=subject,
            seed=seed,
        )
        val_indices_per_subject[subject] = val_idx
        all_val_indices = np.union1d(all_val_indices, val_idx)

    # 2. Determine train-eval indices from the full dataset, excluding validation classes
    train_eval_indices, _ = create_validation_split(
        train_dataset,
        val_classes=train_eval_classes,
        seed=seed,
    )

    # 3. Determine final training indices by removing all reserved indices
    full_indices = np.arange(len(train_dataset))
    train_mask = ~np.isin(full_indices, all_val_indices)
    train_indices = full_indices[train_mask]

    return train_indices, val_indices_per_subject, train_eval_indices


def _create_subsets_from_indices(
    original_dataset: ThingsEEGDataset,
    train_indices: np.ndarray,
    val_indices_per_subject: dict[str, np.ndarray],
    train_eval_indices: np.ndarray,
) -> tuple[Subset, dict[str, Subset], Subset]:
    """Creates Subset objects from pre-calculated index arrays."""
    train_subset = Subset(original_dataset, train_indices)
    train_eval_subset = Subset(original_dataset, train_eval_indices)
    val_subsets = {
        subject: Subset(original_dataset, indices)
        for subject, indices in val_indices_per_subject.items()
    }
    return train_subset, val_subsets, train_eval_subset


class ThingsEEGDataModule(L.LightningDataModule):
    def __init__(self, args: Namespace, subject: str) -> None:
        super().__init__()
        self.args = args
        self.subject = subject
        self.train_dataset = None
        self.subj_val_datasets: dict[str, Subset[Any]] | None = None
        self.test_dataset: None | ThingsEEGDataset = None
        self.train_eval_dataset: None | Subset[Any] = (
            None  # small disjoint subset of training classes for overfitting monitoring
        )
        self.artifacts: DataArtifacts | None = None
        self.retrieval_set_size = self.args.retrieval_set_size
        self.batch_size = self.args.batch_size
        self.val_batch_size = self.retrieval_set_size
        if self.val_batch_size > self.batch_size:
            self.val_batch_size = 4

    def setup(self, stage: str | None = None) -> None:
        ds_args = dict(
            image_model=self.args.image_model,
            data_path=self.args.data_path,
            img_directory_training=self.args.img_directory_training,
            img_directory_test=self.args.img_directory_test,
            embeddings_dir=self.args.embeddings_dir,
            embed_stats_dir=self.args.stats_path
            if hasattr(self.args, "stats_path")
            else None,
            normalize_embed=self.args.normalize_embeddings
            if hasattr(self.args, "normalize_embeddings")
            else False,
            load_images=self.args.load_images
            if hasattr(self.args, "load_images")
            else False,
        )
        if self.args.across_subjects:
            if self.subject == "all":
                print("Loading data across all subjects...")
                train_dataset = ThingsEEGDataset(
                    **ds_args, exclude_subs=[], subjects=self.args.subjects, train=True
                )
                test_dataset = ThingsEEGDataset(
                    **ds_args,
                    exclude_subs=[],
                    subjects=self.args.subjects[:1],
                    train=False,
                )
            else:
                print(
                    f"Loading data across all subjects except {self.subject} for zero-shot testing..."
                )
                train_dataset = ThingsEEGDataset(
                    **ds_args,
                    exclude_subs=[self.subject],
                    subjects=self.args.subjects,
                    train=True,
                )
                test_dataset = ThingsEEGDataset(
                    **ds_args, exclude_subs=[], subjects=[self.subject], train=False
                )  # test subject zero-shot
        else:
            print(f"Loading data for subject {self.subject}...")
            train_dataset = ThingsEEGDataset(
                **ds_args, subjects=[self.subject], train=True
            )
            test_dataset = ThingsEEGDataset(
                **ds_args, subjects=[self.subject], train=False
            )

        train_indices, val_indices, train_eval_indices = _get_split_indices(
            train_dataset=train_dataset,
            subjects=train_dataset.included_subjects,
            retrieval_set_size=self.retrieval_set_size,
            seed=self.args.seed,
        )

        train_dataset, subj_val_datasets, train_eval_dataset = (
            _create_subsets_from_indices(
                original_dataset=train_dataset,
                train_indices=train_indices,
                val_indices_per_subject=val_indices,
                train_eval_indices=train_eval_indices,
            )
        )
        for subject, val_dataset in subj_val_datasets.items():
            if not len(val_dataset) == self.retrieval_set_size:
                print(
                    f"Warning: Val-dataset for subject {subject} has size {len(val_dataset)} instead of {self.retrieval_set_size}"
                )
                raise ValueError("Validation dataset size mismatch")
        if not len(train_eval_dataset) == self.retrieval_set_size:
            print(
                f"Warning: Train-eval dataset has size {len(train_eval_dataset)} instead of {self.retrieval_set_size}"
            )
            raise ValueError("Train-eval dataset size mismatch")

        # Inspect sample for dimensions
        sample = train_dataset[0]
        inputs = sample["brain_signal"]
        outputs = sample["embedding"]
        n_chans, n_times = inputs.shape[0], inputs.shape[1]
        multi_token = outputs.dim() > 1
        n_tokens = outputs.shape[0] if multi_token else None
        n_outputs = outputs.shape[-1]

        self.train_dataset = train_dataset
        self.subj_val_datasets = subj_val_datasets
        self.test_dataset = test_dataset
        self.train_eval_dataset = train_eval_dataset
        self.artifacts = DataArtifacts(
            multi_token=multi_token,
            n_tokens=n_tokens,
            n_chans=n_chans,
            n_times=n_times,
            n_outputs=n_outputs,
        )

    # Dataloaders
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.train_workers,
            drop_last=True,
            persistent_workers=True,
            pin_memory=True,
        )

    def val_dataloader(self) -> list[DataLoader]:
        """Return two validation dataloaders:
        1. Standard validation set (one sample per selected validation class).
        2. Train-eval subset: Subset of training data to monitor overfitting.
        """

        if self.subj_val_datasets is None or self.train_eval_dataset is None:
            raise ValueError("Validation datasets have not been set up yet.")
        main_val_loaders = [
            DataLoader(
                val_dataset,
                batch_size=self.val_batch_size,
                shuffle=False,
                num_workers=1,
                drop_last=True,
                persistent_workers=True,
                pin_memory=True,
            )
            for subject, val_dataset in self.subj_val_datasets.items()
        ]
        train_eval_loader = DataLoader(
            self.train_eval_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=1,
            drop_last=True,
            persistent_workers=True,
            pin_memory=True,
        )
        return [train_eval_loader, *main_val_loaders]

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=0,
            drop_last=True,
            persistent_workers=False,
            pin_memory=True,
        )
