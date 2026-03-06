from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from things_eeg2_dataset.cli.main import DEFAULT_PROJECT_DIR, Partition


def _default_subjects() -> list[int]:
    return [1]


@dataclass(frozen=True)
class DatasetConfig:
    """Configuration for `ThingsEEGDataset`.

    All paths are optional and default to the standard on-disk structure rooted at
    `Path.home() / "things_eeg2"`.
    """

    project_dir: Path = field(default_factory=lambda: DEFAULT_PROJECT_DIR)
    subjects: list[int] = field(default_factory=_default_subjects)
    partition: Partition | Literal["training", "test"] = Partition.TRAINING

    # Embeddings
    use_image_embeddings: bool = True
    allow_missing_image_embeddings: bool = True
    image_model: str | None = "siglip2-base-patch16-224"
    embed_variant: str = "pooled"
    embed_full: bool = False
    # Preferred override name (post-rename)
    image_embeddings_dir: Path | None = None
    embed_stats_dir: Path | None = None
    normalize_embed: bool = False

    # Data loading
    load_images: bool = False
    time_window: tuple[float, float] = (0.0, 1.0)

    # channels (fix)
    teeg_channel_names: list[str] = ["Fp1", "Fp2", "AF7", "AF3", "AFz", "AF4", "AF8", "F7", "F5", "F3", "F1", "F2", "F4", "F6", "F8", "FT9", "FT7", "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6", "FT8", "FT10", "T7", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "T8", "TP9", "TP7", "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6", "TP8", "TP10", "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8", "PO7", "PO3", "POz", "PO4", "PO8", "O1", "Oz", "O2"],


    def __post_init__(self) -> None:
        object.__setattr__(
            self, "project_dir", Path(self.project_dir).expanduser().resolve()
        )

        part = self.partition
        if isinstance(part, str):
            if part not in {"training", "test"}:
                raise ValueError(
                    f"partition must be 'training' or 'test' when given as a string, got: {part!r}"
                )
            object.__setattr__(
                self, "partition", Partition.TRAINING if part == "training" else Partition.TEST
            )


@dataclass(frozen=True)
class DataModuleConfig:
    """Configuration for `ThingsEEGDataModule`."""

    project_dir: Path = field(default_factory=lambda: DEFAULT_PROJECT_DIR)
    subjects: list[int] = field(default_factory=_default_subjects)

    # Embeddings / dataset options
    use_image_embeddings: bool = True
    allow_missing_image_embeddings: bool = True
    image_model: str | None = "siglip2-base-patch16-224"
    embed_variant: str = "pooled"
    embed_full: bool = False
    # Optional overrides (match DatasetConfig names)
    image_embeddings_dir: Path | None = None
    embed_stats_dir: Path | None = None
    normalize_embed: bool = False
    load_images: bool = False
    time_window: tuple[float, float] = (0.0, 1.0)

    # Loader / split
    seed: int = 42
    batch_size: int = 32
    num_workers: int = 4
    retrieval_set_size: int = 200
    val_fraction: float = 0.02

    # Split strategy
    split_strategy: Literal["trial", "image", "concept"] = "trial"
    # Optional override for how many units go into validation.
    # - trial: number of trials
    # - image: number of image_ids
    # - concept: number of concepts/classes
    val_units: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "project_dir", Path(self.project_dir).expanduser().resolve()
        )
