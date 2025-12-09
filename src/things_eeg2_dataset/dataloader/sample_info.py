from dataclasses import dataclass
from pathlib import Path

import numpy as np

from things_eeg2_dataset.paths import layout


@dataclass(frozen=True)
class SampleInfo:
    """Resolved information for a specific sample."""

    image_condition_index: int  # 0-based image condition index
    class_idx: int  # 0-based class index
    sample_idx: int  # 0-based sample index within the class
    class_folder: Path
    image_path: Path
    class_name: str

    def __repr__(self) -> str:
        return (
            f"SampleInfo(\n"
            f"    image_condition_index={self.image_condition_index},\n"
            f"    class_idx={self.class_idx},\n"
            f"    sample_idx={self.sample_idx},\n"
            f"    class_folder='{self.class_folder}',\n"
            f"    image_path='{self.image_path}',\n"
            f"    class_name='{self.class_name}'\n"
            f")"
        )


def get_indices_from_metadata(
    session_id: int, data_idx: int, metadata_array: np.ndarray
) -> tuple[int, int, int]:
    """Calculates 0-based indices from the metadata array."""
    # Lookup 1-based Image ID
    raw_image_id = metadata_array[session_id - 1, data_idx]

    # Convert to 0-based indices
    image_condition_index = raw_image_id - 1
    class_idx = image_condition_index // 10
    sample_idx = image_condition_index % 10

    return image_condition_index, class_idx, sample_idx


def resolve_file_paths(
    image_dir: Path, class_idx: int, sample_idx: int
) -> tuple[Path, Path, str]:
    class_id_str = f"{class_idx + 1:05d}"
    class_folders = list(image_dir.glob(f"{class_id_str}_*"))

    if not class_folders:
        raise FileNotFoundError(f"No folder found for class ID prefix {class_id_str}")

    class_folder = class_folders[0]
    class_name = "_".join(class_folder.name.split("_")[1:])
    image_paths = sorted(list(class_folder.glob("*.jpg")))

    if sample_idx >= len(image_paths):
        raise IndexError(f"Sample index {sample_idx} out of range for {class_folder}")

    image_path = image_paths[sample_idx]

    return class_folder, image_path, class_name


def get_info_for_sample(
    project_dir: Path, subject: int, session: int, data_idx: int, partition: str
) -> SampleInfo:
    # Resolve project_dir
    project_dir = project_dir.resolve()
    if partition == "training":
        image_dir = layout.get_training_images_dir(project_dir)
        cond_file = layout.get_eeg_train_image_conditions_file(project_dir, subject)
    elif partition == "test":
        image_dir = layout.get_test_images_dir(project_dir)
        cond_file = layout.get_eeg_test_image_conditions_file(project_dir, subject)
    else:
        raise ValueError(f"Unknown partition: '{partition}'. Use 'training' or 'test'.")

    if not cond_file.exists():
        raise FileNotFoundError(f"Metadata file not found: {cond_file}")

    metadata = np.load(cond_file)

    image_condition_index, class_idx, sample_idx = get_indices_from_metadata(
        session, data_idx, metadata
    )

    class_folder, image_path, class_name = resolve_file_paths(
        image_dir, class_idx, sample_idx
    )

    return SampleInfo(
        image_condition_index=image_condition_index,
        class_idx=class_idx,
        sample_idx=sample_idx,
        class_folder=class_folder,
        image_path=image_path,
        class_name=class_name,
    )
