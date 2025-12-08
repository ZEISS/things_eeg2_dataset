from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import mne
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset


@dataclass
class ThingsEEGItem:
    brain_signal: torch.Tensor  # (ch, t)
    embedding: torch.Tensor  # image embedding
    subject: int  # subject index
    image_id: int  # image index
    image_class: int  # class id
    sample_id: int  # sample within class
    repetition: int  # repetition id
    channel_positions: torch.Tensor  # (ch, 2) normalized positions
    text: str  # image caption
    image: Path | Tensor  # loaded image or path

    # Example: image_id = 1200
    # Then it follows that image_class = 1200 // 10 = 120
    # and sample_id = 1200 % 120 + 1 = 1


class ThingsEEGDataset(Dataset):
    # Constants / meta
    all_subjects: Final[list[str]] = [
        "sub-01",
        "sub-02",
        "sub-05",
        "sub-04",
        "sub-03",
        "sub-06",
        "sub-07",
        "sub-08",
        "sub-09",
        "sub-10",
    ]
    TRAIN_REPETITIONS: Final[int] = 4
    TEST_REPETITIONS: Final[int] = 80
    TRAIN_SAMPLES_PER_CLASS: Final[int] = 10
    TEST_SAMPLES_PER_CLASS: Final[int] = 1
    TRAIN_CLASSES: Final[int] = 1654
    TEST_CLASSES: Final[int] = 200

    def __init__(  # noqa: PLR0913
        self,
        image_model: str,
        data_path: str,
        img_directory_training: str,
        img_directory_test: str,
        embeddings_dir: str,
        embed_stats_dir: str | None = None,
        normalize_embed: bool = True,
        flat_embed: bool = False,
        subjects: list[str] | None = None,
        exclude_subs: list[str] | None = None,
        train: bool = True,
        time_window: list[float] | tuple = (0, 1.0),
        load_images: bool = False,
    ) -> None:
        self.image_model = image_model
        self.data_path = Path(data_path)
        self.img_directory_training = img_directory_training
        self.img_directory_test = img_directory_test
        self.embeddings_dir = embeddings_dir
        self.embed_stats_dir = embed_stats_dir
        self.normalize_embed = normalize_embed
        self.flat_embeddings = flat_embed
        self.train = train
        self.time_window = time_window
        self.load_images = load_images
        self.exclude_subs = exclude_subs or []
        self.subjects = self.all_subjects if subjects is None else subjects
        self.n_cls = self.TRAIN_CLASSES if train else self.TEST_CLASSES

        # Filter excluded subjects early
        self.included_subjects = [
            s for s in self.subjects if s not in self.exclude_subs
        ]
        if len(self.included_subjects) == 0:
            raise ValueError("No subjects left after applying exclusions.")

        # Validate provided subjects exist
        invalid = [s for s in self.included_subjects if s not in self.all_subjects]
        if invalid:
            raise ValueError(f"Unknown subjects requested: {invalid}")

        # Will be set after reading first subject file
        self.ch_names: list[str] | None = None
        self.times: torch.Tensor | None = None  # trimmed (post 50 offset)
        self._time_offset = 50  # replicate original [50:] trimming
        self.time_indices: torch.Tensor | None = None  # mask for time_window

        # Subject data handles: list of dicts with memmap / ndarray reference
        self.subject_data: list[dict[str, Any]] = []

        # index dataset - will be set after building index
        self.image_index_df: pd.DataFrame | None = None
        self.eeg_index_df: pd.DataFrame | None = None
        self._eeg_idx_arrays: dict[
            str, np.ndarray
        ] = {}  # column -> array for fast slicing

        # setup image transform
        self.tfm = T.Compose([T.ToTensor(), T.Normalize([0.5], [0.5])])

        # setup components
        self.setup()

    def setup(self) -> None:
        """Setup the dataset by preparing all necessary components."""
        # Prepare textual descriptions + image paths (shared across subjects)
        self.texts, self.img_paths = self._prepare_text_and_images()

        # Build subject array handles & indexing meta
        self._init_subject_handles()
        self._build_index_offsets()

        # Prepare eeg index (global index -> subject, local index, metadata)
        self._build_eeg_index()

        # Load embeddings (kept in shared memory for DataLoader workers)
        self._load_embeddings(self.normalize_embed)

        # Channel positions (after ch_names available)
        self.montage = "standard_1020"
        self.ch_pos = self._compute_channel_positions()

    def _prepare_text_and_images(self) -> tuple[list[str], list[str]]:  # noqa: PLR0912, PLR0915
        """Create text descriptions (one per class) and list all image paths.

        The folder naming convention is assumed consistent with the original
        implementation (class folders begin with an ID + '_' + description).
        """
        directory = (
            self.img_directory_training if self.train else self.img_directory_test
        )
        parent_dir = Path(directory).parent
        self.index_csv = (
            Path(parent_dir) / f"{'train' if self.train else 'test'}_image_index.csv"
        )

        if Path(self.index_csv).exists():
            try:
                df = pd.read_csv(self.index_csv)
                # Basic sanity checks (allow backward compatibility if new cols absent)
                required_cols = {
                    "image_index",
                    "class_id",
                    "class_folder",
                    "caption",
                    "image_filename",
                    "image_path",
                }
                if required_cols.issubset(df.columns):
                    # If new metadata columns missing, rebuild to enrich (one-time cost)
                    needed_new = {"sample_id", "base_row"}
                    if not needed_new.issubset(df.columns):
                        raise RuntimeError(
                            "Index CSV missing new metadata columns; rebuilding to enrich."
                        )
                    df = df.sort_values("image_index").reset_index(drop=True)
                    texts = (
                        df.drop_duplicates("class_id")
                        .sort_values("class_id")["caption"]
                        .tolist()
                    )
                    images = df.sort_values("image_index")["image_path"].tolist()
                    self.image_index_df = df
                    return texts, images
                else:
                    warnings.warn(
                        "Index CSV found but missing required columns. Rebuilding.",
                        stacklevel=2,
                    )
            except Exception as e:
                warnings.warn(
                    f"Failed loading existing (or enriching) index CSV ({e}). Rebuilding.",
                    stacklevel=2,
                )

        # Build fresh index
        dirnames = [d for d in Path(directory).iterdir() if d.is_dir()]
        dirnames.sort()

        class_meta = []
        texts = []
        for class_id, d in enumerate(dirnames):
            try:
                idx = d.name.index("_")
                desc = d.name[idx + 1 :]
            except ValueError:
                warnings.warn(f"Skipping folder without '_' in name: {d}", stacklevel=2)
                continue
            caption = f"This picture is {desc}"
            texts.append(caption)
            class_meta.append((class_id, d, caption))

        rows = []
        image_index = 0
        images = []
        for class_id, folder, caption in class_meta:
            folder_path = Path(directory) / folder
            if not folder_path.is_dir():
                continue
            file_list = [
                f
                for f in Path(folder_path).iterdir()
                if f.name.lower().endswith((".png", ".jpg", ".jpeg"))
            ]
            file_list.sort()
            for sample_id, fname in enumerate(file_list):
                fpath = Path(folder_path) / fname
                # base_row is the position used in EEG arrays (class_id * S + sample_id) for training
                if self.train:
                    base_row = class_id * self.TRAIN_SAMPLES_PER_CLASS + sample_id
                else:
                    # test set has one sample per class
                    base_row = class_id * self.TEST_SAMPLES_PER_CLASS + sample_id
                rows.append(
                    {
                        "image_index": image_index,
                        "class_id": class_id,
                        "class_folder": folder,
                        "caption": caption,
                        "image_filename": fname,
                        "image_path": fpath,
                        "sample_id": sample_id,
                        "base_row": base_row,
                    }
                )
                images.append(fpath)
                image_index += 1

        df = pd.DataFrame(rows)

        # Persist for future faster startup
        try:
            df.to_csv(self.index_csv, index=False)
        except Exception as e:
            warnings.warn(f"Could not write image index CSV: {e}", stacklevel=2)

        self.image_index_df = df
        return texts, images

    def _build_eeg_index(self) -> None:  # noqa: PLR0915
        """
        Precompute a global index table mapping dataset index -> all lookup metadata.

        Creates:
          self.eeg_index_df : DataFrame with columns
              ['global_index','subject','subject_pos','local_index',
               'class_id','sample_id','repetition','base_row',
               'img_index','text_index']
          self._eeg_idx_arrays : dict of column -> numpy array for O(1) fast slicing

        For training:
          local_index ordering matches arithmetic:
             class major -> sample -> repetition
        For test:
          One item per class; repetitions will be aggregated (repetition stored -1).
        """
        eeg_index_csv = Path(
            self.data_path, f"{'train' if self.train else 'test'}_eeg_index.csv"
        )

        required_cols = {
            "global_index",
            "subject",
            "subject_pos",
            "local_index",
            "class_id",
            "sample_id",
            "repetition",
            "base_row",
            "img_index",
            "text_index",
        }
        # Try load existing (skip if shape mismatch / subjects mismatch)
        if Path(eeg_index_csv).exists():
            try:
                df_old = pd.read_csv(eeg_index_csv)
                if (
                    required_cols.issubset(df_old.columns)
                    and len(df_old) == self._dataset_len
                ):
                    # quick subject consistency check
                    subj_set_in_file = set(df_old["subject"].unique())
                    if subj_set_in_file == set(self.included_subjects):
                        self.eeg_index_df = df_old
                        self._eeg_idx_arrays = {
                            c: df_old[c].to_numpy() for c in required_cols
                        }
                        return
            except Exception as e:
                warnings.warn(
                    f"Failed loading existing EEG index CSV ({e}); rebuilding.",
                    stacklevel=2,
                )

        records = []
        total_len_check = 0

        if self.train:
            C = self.TRAIN_CLASSES
            S = self.TRAIN_SAMPLES_PER_CLASS
            R = self.TRAIN_REPETITIONS

            class_ids = np.repeat(np.arange(C, dtype=np.int32), S * R)
            sample_ids_within_class = np.tile(
                np.repeat(np.arange(S, dtype=np.int32), R), C
            )
            rep_ids = np.tile(np.arange(R, dtype=np.int32), C * S)
            base_rows = class_ids * S + sample_ids_within_class
            local_indices = (
                class_ids * (S * R) + sample_ids_within_class * R + rep_ids
            )  # (C*S*R,)

            per_subject_len = C * S * R

            for s_pos, subj in enumerate(self.included_subjects):
                offset = self.subject_offsets[s_pos]
                global_indices = offset + local_indices
                subj_arr = np.full_like(class_ids, s_pos)
                subj_name_arr = np.array([subj] * per_subject_len)

                df_sub = pd.DataFrame(
                    {
                        "global_index": global_indices,
                        "subject": subj_name_arr,
                        "subject_pos": subj_arr,
                        "local_index": local_indices,
                        "class_id": class_ids,
                        "sample_id": sample_ids_within_class,
                        "repetition": rep_ids,
                        "base_row": base_rows,
                        "img_index": base_rows,
                        "text_index": class_ids,
                    }
                )
                records.append(df_sub)
                total_len_check += per_subject_len
        else:
            C = self.TEST_CLASSES
            S = self.TEST_SAMPLES_PER_CLASS  # == 1
            per_subject_len = C * S  # == C
            class_ids = np.arange(C, dtype=np.int32)
            sample_ids = np.zeros(C, dtype=np.int32)
            base_rows = class_ids  # S == 1
            local_indices = class_ids
            rep_ids = np.full(C, -1, dtype=np.int32)  # not used (averaged)

            for s_pos, subj in enumerate(self.included_subjects):
                offset = self.subject_offsets[s_pos]
                global_indices = offset + local_indices
                subj_arr = np.full(C, s_pos, dtype=np.int32)
                subj_name_arr = np.array([subj] * C)

                df_sub = pd.DataFrame(
                    {
                        "global_index": global_indices,
                        "subject": subj_name_arr,
                        "subject_pos": subj_arr,
                        "local_index": local_indices,
                        "class_id": class_ids,
                        "sample_id": sample_ids,
                        "repetition": rep_ids,
                        "base_row": base_rows,
                        "img_index": base_rows,
                        "text_index": class_ids,
                    }
                )
                records.append(df_sub)
                total_len_check += per_subject_len

        eeg_df = pd.concat(records, axis=0, ignore_index=True)
        # Sanity
        if total_len_check != self._dataset_len or eeg_df.shape[0] != self._dataset_len:
            raise RuntimeError("EEG index length mismatch after build.")

        # Sort by global_index just to be explicit
        eeg_df = eeg_df.sort_values("global_index").reset_index(drop=True)
        if not np.all(
            eeg_df["global_index"].to_numpy() == np.arange(self._dataset_len)
        ):
            raise RuntimeError("global_index column not continuous 0..N-1.")

        # Persist
        try:
            eeg_df.to_csv(eeg_index_csv, index=False)
        except Exception as e:
            warnings.warn(f"Could not write EEG index CSV: {e}", stacklevel=2)

        self.eeg_index_df = eeg_df
        self._eeg_idx_arrays = {c: eeg_df[c].to_numpy() for c in required_cols}

    def _load_embeddings(self, normalize_embed: bool) -> None:
        if self.train:
            feat_file = (
                Path(self.embeddings_dir) / f"{self.image_model}_features_train.pt"
            )
            stats_file = (
                Path(self.embed_stats_dir)
                / f"{self.image_model}_features_train_stats.pt"
                if self.embed_stats_dir
                else None
            )
        else:
            feat_file = (
                Path(self.embeddings_dir) / f"{self.image_model}_features_test.pt"
            )

            stats_file = (
                Path(self.embed_stats_dir)
                / f"{self.image_model}_features_test_stats.pt"
                if self.embed_stats_dir
                else None
            )

        if not Path(feat_file).exists():
            raise FileNotFoundError(f"Embedding feature file not found: {feat_file}")

        saved = torch.load(feat_file, weights_only=True)
        img_embeddings = saved["img_features"]
        text_embeddings = saved.get("text_features", None)

        if self.flat_embeddings:
            img_embeddings = img_embeddings.view(img_embeddings.size(0), -1)
            if text_embeddings is not None:
                text_embeddings = text_embeddings.view(text_embeddings.size(0), -1)

        self.emb_stats = None
        if normalize_embed and stats_file and Path(stats_file).exists():
            self.emb_stats = torch.load(stats_file)

        # Share memory so multiple workers don't duplicate
        self._shared_objects = {
            "embedding": img_embeddings.share_memory_(),
        }
        if text_embeddings is not None:
            self._shared_objects["text_embedding"] = text_embeddings.share_memory_()

    def _open_subject_file(
        self, subject: str
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Open a subject EEG file and return (data_array, times, ch_names).

        Tries memmap mode; falls back gracefully if not possible.
        """
        eeg_fname = (
            "train_processed_eeg_data.npy"
            if self.train
            else "test_processed_eeg_data.npy"
        )
        fpath = Path(self.data_path) / subject / eeg_fname
        if not fpath.exists():
            raise FileNotFoundError(f"Missing EEG file for {subject}: {fpath}")

        # load with memmap for eeg data
        preprocessed = np.load(fpath, mmap_mode="r", allow_pickle=True)
        ch_names = np.load(self.data_path / subject / "ch_names.npy", allow_pickle=True)
        times = np.load(self.data_path / subject / "times.npy", allow_pickle=True)

        # preprocessed expected shape:
        #   train: (C*S, R, ch, t)
        #   test:  (C*1, R, ch, t)
        return preprocessed, times, ch_names

    def _init_subject_handles(self) -> None:
        for subj in self.included_subjects:
            data_array, times_full, ch_names = self._open_subject_file(subj)

            # Initialize global times / channels from first subject
            if self.times is None:
                if len(times_full) <= self._time_offset:
                    raise ValueError("Times vector shorter than offset 50.")
                trimmed_times = times_full[self._time_offset :]
                self.times = torch.from_numpy(trimmed_times).float()
                # Accept list / np array of channel names
                self.ch_names = (
                    list(ch_names) if not isinstance(ch_names, list) else ch_names
                )
                # Prepare time window indices (boolean mask)
                start, end = self.time_window
                idx_mask = (self.times >= start) & (self.times <= end)
                if not torch.any(idx_mask):
                    raise ValueError(
                        f"Time window {self.time_window} selects zero samples."
                    )
                self.time_indices = idx_mask
            else:
                # Consistency checks
                if self.ch_names is None:
                    raise RuntimeError("Channel names should have been set already.")
                if list(ch_names) != list(self.ch_names):
                    raise ValueError(f"Channel mismatch for subject {subj}")
                comp_trim = times_full[self._time_offset :]
                if comp_trim.shape[0] != self.times.shape[0]:
                    raise ValueError(f"Time length mismatch for subject {subj}")

            handle = {
                "subject": subj,
                "data": data_array,  # memmap / ndarray
                "shape": data_array.shape,
            }
            self.subject_data.append(handle)

        # Derive per-subject length for indexing
        if self.train:
            # Expect shape (slasses * samples_per_class, repetitions, channels, timepoints)
            expected_first_dim = self.TRAIN_CLASSES * self.TRAIN_SAMPLES_PER_CLASS
            first_dim = self.subject_data[0]["shape"][0]
            if first_dim != expected_first_dim:
                warnings.warn(
                    f"First dim ({first_dim}) != expected ({expected_first_dim}) for training data. Proceeding anyway.",
                    stacklevel=2,
                )
            self.per_subject_len = (
                self.TRAIN_CLASSES
                * self.TRAIN_SAMPLES_PER_CLASS
                * self.TRAIN_REPETITIONS
            )
        else:
            # Test items are averaged across repetitions -> one item per class
            expected_first_dim = self.TEST_CLASSES * self.TEST_SAMPLES_PER_CLASS
            first_dim = self.subject_data[0]["shape"][0]
            if first_dim != expected_first_dim:
                warnings.warn(
                    f"First dim ({first_dim}) != expected ({expected_first_dim}) for test data. Proceeding anyway.",
                    stacklevel=2,
                )
            self.per_subject_len = self.TEST_CLASSES * self.TEST_SAMPLES_PER_CLASS

    def _build_index_offsets(self) -> None:
        # Prefix sums to quickly map global index -> subject
        self.subject_offsets = []  # starting global index per subject
        running = 0
        for _ in self.subject_data:
            self.subject_offsets.append(running)
            running += self.per_subject_len
        self._dataset_len = running

    def _compute_channel_positions(self) -> Tensor:
        montage = mne.channels.make_standard_montage(self.montage)
        ch_pos_dict = montage.get_positions()["ch_pos"]
        if self.ch_names is None:
            raise RuntimeError("Channel names not set before computing positions.")
        ch_list = [ch_pos_dict[c] for c in self.ch_names if c in ch_pos_dict]
        if len(ch_list) != len(self.ch_names):
            missing = [c for c in self.ch_names if c not in ch_pos_dict]
            raise ValueError(f"Channels missing in montage: {missing}")
        ch_pos = np.array(ch_list)[:, :2]
        # Normalize
        mins = ch_pos.min(axis=0)
        maxs = ch_pos.max(axis=0)
        ch_pos = (ch_pos - mins) / (maxs - mins)
        return torch.tensor(ch_pos, dtype=torch.float32)

    def _locate_subject(self, index: int) -> tuple[int, int]:
        # Binary search over offsets (list is small; linear acceptable)
        if index < 0 or index >= self._dataset_len:
            raise IndexError(index)
        # Since number of subjects small, simple loop
        for si, start in enumerate(self.subject_offsets):
            end = start + self.per_subject_len
            if start <= index < end:
                return si, index - start
        # Fallback (should never reach)
        raise RuntimeError("Failed to map index to subject")

    def __len__(self) -> int:
        return self._dataset_len

    def sub_str2idx(self, subject: str) -> int:
        return self.all_subjects.index(subject) if subject in self.all_subjects else -1

    def __getitem__(self, index: int) -> dict[str, Any]:
        A = self._eeg_idx_arrays
        subject_pos = int(A["subject_pos"][index])
        subj_name = self.subject_data[subject_pos]["subject"]
        subj_global_idx = self.sub_str2idx(subj_name)

        base_row = int(A["base_row"][index])
        class_id = int(A["class_id"][index])
        sample_id = int(A["sample_id"][index])
        rep_id = int(A["repetition"][index])
        img_index = int(A["img_index"][index])
        text_index = int(A["text_index"][index])

        data_arr = self.subject_data[subject_pos]["data"]

        if self.train:
            # Direct row + repetition
            trial_np = data_arr[base_row, rep_id]  # (ch, t)
        else:
            # Average over repetitions (base_row shape (R, ch, t))
            trial_np = data_arr[base_row].mean(axis=0)

        if self.time_indices is not None:
            trial_np = trial_np[..., self.time_indices.numpy()]

        brain_signal = torch.from_numpy(np.array(trial_np, copy=False)).float()

        img_features = self._shared_objects["embedding"][img_index]
        if self.emb_stats is not None:
            img_features = (img_features - self.emb_stats["vis_mean"]) / self.emb_stats[
                "vis_std"
            ]

        text = self.texts[text_index] if text_index < len(self.texts) else ""
        image = self.img_paths[img_index] if img_index < len(self.img_paths) else ""

        if self.load_images:
            # Load image into memory without any transforms
            # acceptable by dataloader
            image = Image.open(image).convert("RGB")
            image = self.tfm(image)

        return {
            "brain_signal": brain_signal,
            "embedding": img_features,
            "subject": torch.tensor(subj_global_idx, dtype=torch.long),
            "image_id": torch.tensor(img_index, dtype=torch.long),
            "image_class": torch.tensor(class_id, dtype=torch.long),
            "sample_id": torch.tensor(sample_id, dtype=torch.long),
            "repetition": torch.tensor(rep_id, dtype=torch.long),
            "channel_positions": self.ch_pos.clone(),
            "text": text,
            "image": image,
        }
