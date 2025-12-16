from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from things_eeg2_dataset.dataloader.sample_info import get_info_for_sample
from things_eeg2_dataset.paths import layout


class ThingsEEGDataset(Dataset):
    def __init__(  # noqa: PLR0913
        self,
        project_dir: Path,
        subjects: list[int],
        partition: str,
        image_model: str,
        embeddings_dir: Path,
        embed_stats_dir: Path | None = None,
        normalize_embed: bool = True,
        load_images: bool = False,
        time_window: tuple[float, float] = (0.0, 1.0),
    ) -> None:
        self.project_dir = project_dir.resolve()
        self.subjects = subjects
        self.partition = partition
        self.load_images = load_images

        # Load EEG per subject
        self.eeg_data: list[np.ndarray] = []
        self.metadata: list[np.ndarray] = []

        for subj in subjects:
            if partition == "training":
                eeg_file = layout.get_eeg_train_file(self.project_dir, subj)
                meta_file = layout.get_eeg_train_image_conditions_file(
                    self.project_dir, subj
                )
            else:
                eeg_file = layout.get_eeg_test_file(self.project_dir, subj)
                meta_file = layout.get_eeg_test_image_conditions_file(
                    self.project_dir, subj
                )

            self.eeg_data.append(np.load(eeg_file, mmap_mode="r"))
            self.metadata.append(np.load(meta_file))

        # Dimensions
        _, self.num_conditions, self.num_reps, self.num_ch, self.num_t = self.eeg_data[
            0
        ].shape
        self.num_sessions = self.eeg_data[0].shape[0]

        # Time mask
        times = np.load(
            layout.get_times_file(self.project_dir, subjects[0])  # type: ignore[attr-defined]
        )
        times = torch.from_numpy(times).float()
        start, end = time_window
        self.time_mask = (times >= start) & (times <= end)

        # Load embeddings
        self._load_embeddings(
            image_model, embeddings_dir, embed_stats_dir, normalize_embed
        )

        # Length
        if partition == "training":
            self._len = (
                len(subjects) * self.num_sessions * self.num_conditions * self.num_reps
            )
        else:
            self._len = len(subjects) * self.num_sessions * self.num_conditions

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, index: int) -> dict[str, Any]:
        # --- unravel index ---
        if self.partition == "training":
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
        if self.partition == "training":
            trial = eeg[session, data_idx, rep]
        else:
            trial = eeg[session, data_idx].mean(axis=0)

        trial = trial[..., self.time_mask.numpy()]
        brain_signal = torch.from_numpy(trial).float()

        # --- Resolve sample via metadata ---
        info = get_info_for_sample(
            project_dir=self.project_dir,
            subject=subj,
            session=session + 1,
            data_idx=data_idx,
            partition=self.partition,
        )

        emb = self.embeddings[info.image_condition_index]
        if self.emb_stats is not None:
            emb = (emb - self.emb_stats["vis_mean"]) / self.emb_stats["vis_std"]

        image = info.image_path
        if self.load_images:
            image = Image.open(image).convert("RGB")

        return {
            "brain_signal": brain_signal,
            "embedding": emb,
            "subject": torch.tensor(subj_idx),
            "image_id": torch.tensor(info.image_condition_index),
            "image_class": torch.tensor(info.class_idx),
            "sample_id": torch.tensor(info.sample_idx),
            "repetition": torch.tensor(rep),
            "text": info.class_name,
            "image": image,
        }
