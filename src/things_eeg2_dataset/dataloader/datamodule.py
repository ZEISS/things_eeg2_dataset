"""Lightning DataModule for THINGS-EEG2.

This module is intentionally config-driven (no argparse.Namespace) so it can be
instantiated with defaults for quick experimentation and reliable tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import lightning as L
import numpy as np
from torch.utils.data import DataLoader, Subset

from things_eeg2_dataset.cli.main import Partition
from things_eeg2_dataset.dataloader.config import DataModuleConfig, DatasetConfig
from things_eeg2_dataset.dataloader.dataset import ThingsEEGDataset, ThingsEEGItem


@dataclass
class DataArtifacts:
    multi_token: bool
    n_tokens: int | None
    n_chans: int
    n_times: int
    n_outputs: int


def _select_units_for_validation(
    unit_ids: np.ndarray,
    *,
    seed: int,
    val_fraction: float,
    retrieval_set_size: int,
    val_units: int | None,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    unique_units = np.unique(unit_ids)
    if unique_units.size == 0:
        raise ValueError("No units available for splitting")

    if val_units is None:
        n_units = int(max(1, int(unique_units.size * float(val_fraction))))
        n_units = min(int(retrieval_set_size), n_units)
    else:
        n_units = int(val_units)

    n_units = int(min(max(n_units, 1), unique_units.size - 1))
    return rng.choice(unique_units, size=n_units, replace=False)


def _split_indices(
    n: int, *, seed: int, val_size: int, train_eval_size: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = np.arange(n)
    rng.shuffle(indices)
    val_size = int(min(max(val_size, 1), n))
    train_eval_size = int(min(max(train_eval_size, 1), n - val_size))

    val_idx = np.sort(indices[:val_size])
    train_eval_idx = np.sort(indices[val_size : val_size + train_eval_size])
    train_idx = np.sort(indices[val_size + train_eval_size :])
    return train_idx, val_idx, train_eval_idx


def _split_by_units(
    unit_ids: np.ndarray,
    *,
    seed: int,
    val_fraction: float,
    retrieval_set_size: int,
    val_units: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    chosen_units = _select_units_for_validation(
        unit_ids,
        seed=seed,
        val_fraction=val_fraction,
        retrieval_set_size=retrieval_set_size,
        val_units=val_units,
    )
    val_mask = np.isin(unit_ids, chosen_units)
    val_idx = np.nonzero(val_mask)[0]
    train_pool = np.nonzero(~val_mask)[0]
    if train_pool.size == 0:
        raise ValueError("Validation split consumed all samples")

    rng = np.random.default_rng(seed)
    train_eval_size = int(
        min(max(1, min(retrieval_set_size, val_idx.size)), train_pool.size)
    )
    train_eval_idx = rng.choice(train_pool, size=train_eval_size, replace=False)

    train_mask = ~np.isin(train_pool, train_eval_idx)
    train_idx = train_pool[train_mask]
    return np.sort(train_idx), np.sort(val_idx), np.sort(train_eval_idx)


class ThingsEEGDataModule(L.LightningDataModule):
    def __init__(self, config: DataModuleConfig | None = None) -> None:
        super().__init__()
        self.cfg = config or DataModuleConfig()

        self.train_dataset: Subset[Any] | None = None
        self.val_dataset: Subset[Any] | None = None
        self.test_dataset: ThingsEEGDataset | None = None
        self.train_eval_dataset: Subset[Any] | None = None
        self.artifacts: DataArtifacts | None = None
        self.val_batch_size = min(
            self.cfg.retrieval_set_size, max(self.cfg.batch_size, 1)
        )

    def setup(self, stage: str | None = None) -> None:
        ds_common = dict(
            project_dir=self.cfg.project_dir,
            subjects=self.cfg.subjects,
            use_image_embeddings=self.cfg.use_image_embeddings,
            allow_missing_image_embeddings=self.cfg.allow_missing_image_embeddings,
            image_model=self.cfg.image_model,
            embed_variant=self.cfg.embed_variant,
            embed_full=self.cfg.embed_full,
            image_embeddings_dir=self.cfg.image_embeddings_dir,
            embed_stats_dir=self.cfg.embed_stats_dir,
            normalize_embed=self.cfg.normalize_embed,
            load_images=self.cfg.load_images,
            time_window=self.cfg.time_window,
        )

        full_train = ThingsEEGDataset(
            DatasetConfig(**ds_common, partition=Partition.TRAINING)  # type: ignore[arg-type]
        )
        self.test_dataset = ThingsEEGDataset(
            DatasetConfig(**ds_common, partition=Partition.TEST)  # type: ignore[arg-type]
        )

        n = len(full_train)

        if self.cfg.split_strategy == "trial":
            val_size = (
                int(self.cfg.val_units)
                if self.cfg.val_units is not None
                else min(
                    self.cfg.retrieval_set_size, max(1, int(n * self.cfg.val_fraction))
                )
            )
            train_eval_size = min(self.cfg.retrieval_set_size, max(1, int(val_size)))
            train_idx, val_idx, train_eval_idx = _split_indices(
                n,
                seed=self.cfg.seed,
                val_size=val_size,
                train_eval_size=train_eval_size,
            )
        elif self.cfg.split_strategy == "image":
            image_ids = full_train.get_image_ids()
            train_idx, val_idx, train_eval_idx = _split_by_units(
                image_ids,
                seed=self.cfg.seed,
                val_fraction=self.cfg.val_fraction,
                retrieval_set_size=self.cfg.retrieval_set_size,
                val_units=self.cfg.val_units,
            )
        elif self.cfg.split_strategy == "concept":
            concept_ids = full_train.get_concept_ids()
            train_idx, val_idx, train_eval_idx = _split_by_units(
                concept_ids,
                seed=self.cfg.seed,
                val_fraction=self.cfg.val_fraction,
                retrieval_set_size=self.cfg.retrieval_set_size,
                val_units=self.cfg.val_units,
            )
        else:
            raise ValueError(f"Unknown split_strategy: {self.cfg.split_strategy!r}")

        self.train_dataset = Subset(full_train, train_idx)
        self.val_dataset = Subset(full_train, val_idx)
        self.train_eval_dataset = Subset(full_train, train_eval_idx)

        # Inspect sample for dimensions
        sample_item: ThingsEEGItem = full_train[0]
        inputs = sample_item.brain_signal
        outputs = sample_item.image_embedding

        n_chans, n_times = inputs.shape[0], inputs.shape[1]
        if outputs.numel() == 0:
            multi_token = False
            n_tokens = None
            n_outputs = 0
        else:
            multi_token = outputs.dim() > 1
            n_tokens = outputs.shape[0] if multi_token else None
            n_outputs = outputs.shape[-1]

        self.artifacts = DataArtifacts(
            multi_token=multi_token,
            n_tokens=n_tokens,
            n_chans=n_chans,
            n_times=n_times,
            n_outputs=n_outputs,
        )

    # Dataloaders
    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("DataModule not set up. Call setup() first.")
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            drop_last=True,
            persistent_workers=self.cfg.num_workers > 0,
            pin_memory=True,
        )

    def val_dataloader(self) -> list[DataLoader]:
        if self.val_dataset is None or self.train_eval_dataset is None:
            raise ValueError("DataModule not set up. Call setup() first.")

        train_eval_loader = DataLoader(
            self.train_eval_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            drop_last=True,
            persistent_workers=self.cfg.num_workers > 0,
            pin_memory=True,
        )
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            drop_last=True,
            persistent_workers=self.cfg.num_workers > 0,
            pin_memory=True,
        )
        return [train_eval_loader, val_loader]

    def test_dataloader(self) -> DataLoader:
        if self.test_dataset is None:
            raise ValueError("DataModule not set up. Call setup() first.")
        return DataLoader(
            self.test_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=0,
            drop_last=True,
            persistent_workers=False,
            pin_memory=True,
        )
