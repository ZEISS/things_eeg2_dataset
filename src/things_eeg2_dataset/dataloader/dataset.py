from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

import logging

import mne
import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file
from torch.utils.data import Dataset

from things_eeg2_dataset.cli.main import Partition
from things_eeg2_dataset.dataloader.config import DatasetConfig
from things_eeg2_dataset.dataloader.sample_info import get_info_for_sample
from things_eeg2_dataset.paths import layout


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LoadedEmbeddingBank:
    img_features: torch.Tensor | None
    text_features: torch.Tensor | None


class ThingsEEGItem(NamedTuple):
    """Data item returned by `ThingsEEGDataset.__getitem__`."""
    brain_signal: torch.Tensor
    image_embedding: torch.Tensor
    subject: torch.Tensor
    image_id: torch.Tensor
    image_class: torch.Tensor
    sample_id: torch.Tensor
    repetition: torch.Tensor
    channel_positions: torch.Tensor
    text: str
    image: str | torch.Tensor


def _compute_channel_positions(ch_names: list[str], montage: str = "standard_1020") -> torch.Tensor:
    """Compute normalized 2D sensor positions for the given channel names.

    Arguments:
        ch_names: List of channel names as given in the metadata JSON. These must be resolvable by the specified montage.
        montage: Name of the standard MNE montage to use for channel position lookup (e.g. "standard_1020").

    Returns a float32 tensor of shape (n_channels, 2) in normalized [0, 1] coords.
    """

    montage = mne.channels.make_standard_montage(montage)
    ch_pos = montage.get_positions()["ch_pos"]

    missing = [c for c in ch_names if c not in ch_pos]
    if missing:
        raise ValueError(
            "Some channels are missing from the montage position lookup. "
            f"Missing: {missing}"
        )

    xy = np.stack([np.asarray(ch_pos[c], dtype=np.float32)[:2] for c in ch_names], axis=0)
    mins = xy.min(axis=0)
    maxs = xy.max(axis=0)
    denom = np.maximum(maxs - mins, 1e-8)
    xy = (xy - mins) / denom
    return torch.tensor(xy, dtype=torch.float32)


class ThingsEEGDataset(Dataset):
    """THINGS-EEG2 dataset.

    Uses the project directory structure defined in `things_eeg2_dataset.paths.layout`.
    """

    # Each sessions only shows half of the train image classes and half of the  train repetitions
    # and all of the test image classes but only a quarter of the test repetitions.
    # There are 10 train images per classes and 1 test image per class.
    # So each session has 1654*10/2=8270 or 200 images, and 4/2=2 or 80/4=20 repetitions per class.

    # Expected shapes produced by this repo's preprocessing pipeline.
    # Training: (4, 8270, 2, 64, 301)
    # Test:     (4, 200, 20, 64, 301)
    NUM_SESSIONS: int = 4
    TRAIN_REPETITIONS: int = 2
    TEST_REPETITIONS: int = 20
    TRAIN_CLASSES: int = 1654
    TEST_CLASSES: int = 200
    TRAIN_SAMPLES_PER_CLASS: int = 10
    TEST_SAMPLES_PER_CLASS: int = 1

    def __init__(self, config: DatasetConfig | None = None, **overrides: Any) -> None:
        cfg = config or DatasetConfig()
        if overrides:
            cfg = DatasetConfig(**{**cfg.__dict__, **overrides})
        self.cfg = cfg

        self.project_dir = cfg.project_dir
        self.subjects = list(cfg.subjects)
        self.partition: Partition = cfg.partition  # normalized in config
        self.load_images = cfg.load_images

        if not self.subjects:
            raise ValueError("subjects must not be empty")

        # Load EEG per subject
        self.eeg_data: list[np.ndarray] = []
        self.conditions: list[np.ndarray] = []

        part_str = "training" if self.partition is Partition.TRAINING else "test"
        for subj in self.subjects:
            if self.partition is Partition.TRAINING:
                eeg_file = layout.get_eeg_train_file(self.project_dir, subj)
                cond_file = layout.get_eeg_train_image_conditions_file(
                    self.project_dir, subj
                )
            else:
                eeg_file = layout.get_eeg_test_file(self.project_dir, subj)
                cond_file = layout.get_eeg_test_image_conditions_file(
                    self.project_dir, subj
                )

            if not eeg_file.exists():
                raise FileNotFoundError(
                    f"Missing processed EEG file for subject {subj} ({part_str}): {eeg_file}"
                )
            if not cond_file.exists():
                raise FileNotFoundError(
                    f"Missing image-conditions file for subject {subj} ({part_str}): {cond_file}"
                )

            self.eeg_data.append(np.load(eeg_file, mmap_mode="r"))
            self.conditions.append(np.load(cond_file))

        # Dimensions: (sessions, conditions, reps, channels, timepoints)
        self.num_sessions = int(self.eeg_data[0].shape[0])
        self.num_conditions = int(self.eeg_data[0].shape[1])
        self.num_reps = int(self.eeg_data[0].shape[2])
        self.num_ch = int(self.eeg_data[0].shape[3])
        self.num_t = int(self.eeg_data[0].shape[4])

        # Time mask (load from saved metadata JSON)
        meta_json = layout.get_metadata_file(self.project_dir, self.subjects[0])
        if not meta_json.exists():
            raise FileNotFoundError(
                f"Missing processed metadata JSON (needed for times/ch_names): {meta_json}"
            )

        with meta_json.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        times = torch.tensor(meta["times"], dtype=torch.float32)
        ch_names = list(meta.get("ch_names") or [])
        if not ch_names:
            raise ValueError(
                "Missing or empty 'ch_names' in metadata JSON. "
                f"Expected key 'ch_names' in: {meta_json}"
            )

        # Preprocessing keeps the full epoch including pre-stimulus. The saved times
        # vector must match the EEG time dimension.
        if times.numel() != self.num_t:
            raise ValueError(
                "Times vector length does not match EEG time dimension. "
                f"times={times.numel()} vs eeg={self.num_t}. "
                "This usually indicates stale processed files; re-run preprocessing."
            )

        start, end = cfg.time_window
        self.time_mask = (times >= float(start)) & (times <= float(end))
        if not torch.any(self.time_mask):
            raise ValueError(f"time_window {cfg.time_window} selects zero samples")

        # Channel positions (used by NeuroVision models; must be part of base dataset)
        self.ch_names = ch_names
        self.channel_positions = _compute_channel_positions(self.ch_names)

        # Embeddings
        bank = self._load_embeddings()
        self.image_embeddings = bank.img_features
        self.text_embeddings = bank.text_features

        # Length
        if self.partition is Partition.TRAINING:
            self._len = (
                len(self.subjects)
                * self.num_sessions
                * self.num_conditions
                * self.num_reps
            )
        else:
            self._len = len(self.subjects) * self.num_sessions * self.num_conditions

    def __len__(self) -> int:
        return self._len

    def get_image_ids(self) -> np.ndarray:
        """Return a vector of image condition indices for every sample index.

        Uses the already loaded `img_conditions_*` arrays (no filesystem access).
        The returned image ids are 0-based and match `ThingsEEGItem.image_id`.
        """

        per_subject: list[np.ndarray] = []
        for cond in self.conditions:
            # cond is 1-based image id as stored in preprocessing; convert to 0-based.
            base = cond.astype(np.int64) - 1
            if self.partition is Partition.TRAINING:
                # (sessions, conditions, reps) with reps fastest
                ids = np.repeat(base[..., None], self.num_reps, axis=2).reshape(-1)
            else:
                # (sessions, conditions)
                ids = base.reshape(-1)
            per_subject.append(ids)

        image_ids = np.concatenate(per_subject, axis=0)
        if image_ids.shape[0] != len(self):
            raise RuntimeError(
                f"get_image_ids produced length {image_ids.shape[0]} but dataset length is {len(self)}"
            )
        return image_ids

    def get_concept_ids(self) -> np.ndarray:
        """Return a vector of concept/class indices for every sample index."""
        return self.get_image_ids() // self.TRAIN_SAMPLES_PER_CLASS

    def _load_embeddings(self) -> LoadedEmbeddingBank:
        if not self.cfg.use_image_embeddings or not self.cfg.image_model:
            logger.info("Running without image embeddings (disabled by config).")
            return LoadedEmbeddingBank(img_features=None, text_features=None)

        emb_file = layout.get_embedding_file(
            self.project_dir,
            self.cfg.image_model,
            self.partition,
            full=self.cfg.embed_full,
            variant=self.cfg.embed_variant,
        )

        # Optional override of the embeddings directory.
        override_dir = self.cfg.image_embeddings_dir
        if override_dir is not None:
            emb_file = Path(override_dir).expanduser().resolve() / emb_file.name

        if not emb_file.exists():
            if self.cfg.allow_missing_image_embeddings:
                logger.info(
                    "Running without image embeddings (file not found: %s)", emb_file
                )
                return LoadedEmbeddingBank(img_features=None, text_features=None)
            raise FileNotFoundError(f"Embedding file not found: {emb_file}")

        saved = load_file(emb_file)
        img = saved["img_features"]
        txt = saved.get("text_features")

        # Optional normalization (stats generation is not guaranteed)
        if self.cfg.normalize_embed and self.cfg.embed_stats_dir is not None:
            stats_path = (
                Path(self.cfg.embed_stats_dir)
                / f"{emb_file.stem}_stats.safetensors"
            )
            if stats_path.exists():
                stats = load_file(stats_path)
                if "vis_mean" in stats and "vis_std" in stats:
                    img = (img - stats["vis_mean"]) / stats["vis_std"]

        return LoadedEmbeddingBank(img_features=img, text_features=txt)

    def __getitem__(self, index: int) -> ThingsEEGItem:
        # --- unravel index ---
        if self.partition is Partition.TRAINING:
            rep = index % self.num_reps
            index //= self.num_reps
        else:
            rep = -1

        data_idx = index % self.num_conditions
        index //= self.num_conditions

        session = index % self.num_sessions
        subj_idx = index // self.num_sessions

        eeg = self.eeg_data[subj_idx]
        subj = self.subjects[subj_idx]

        # --- EEG extraction ---
        if self.partition is Partition.TRAINING:
            trial = eeg[session, data_idx, rep]
        else:
            trial = eeg[session, data_idx].mean(axis=0)

        trial = trial[..., self.time_mask.numpy()]
        brain_signal = torch.from_numpy(trial).float()

        part_str = "training" if self.partition is Partition.TRAINING else "test"
        info = get_info_for_sample(
            project_dir=self.project_dir,
            subject=subj,
            session=session + 1,
            data_idx=data_idx,
            partition=part_str,
        )

        if self.image_embeddings is None:
            emb = torch.empty(0, dtype=torch.float32)
        else:
            emb = self.image_embeddings[info.image_condition_index]
            if isinstance(emb, torch.Tensor) and emb.dtype != torch.float32:
                emb = emb.float()

        if self.load_images:
            pil = Image.open(info.image_path).convert("RGB")
            # Return a Tensor so default collation can batch images.
            from torchvision.transforms.functional import to_tensor  # noqa: PLC0415

            image: str | torch.Tensor = to_tensor(pil)
        else:
            # Return a string path so default collation doesn't choke on Path objects.
            image = str(info.image_path)

        return ThingsEEGItem(
            brain_signal=brain_signal,
            image_embedding=emb,
            subject=torch.tensor(subj),
            image_id=torch.tensor(info.image_condition_index),
            image_class=torch.tensor(info.class_idx),
            sample_id=torch.tensor(info.sample_idx),
            repetition=torch.tensor(rep),
            channel_positions=self.channel_positions,
            text=info.class_name,
            image=image,
        )
