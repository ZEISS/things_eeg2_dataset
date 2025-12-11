from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataDirectoryLayout:
    """
    Defines the relative directory structure and file naming conventions.
    This class is 'stateless' regarding the actual project root.
    """

    # Directory Names
    raw_dir: str = "raw_data"
    source_dir: str = "source_data"
    images_dir: str = "Image_set"
    processed_dir: str = "processed"
    embeddings_dir: str = "embeddings"

    # Subdirectories
    train_imgs_dir: str = "training_images"
    test_imgs_dir: str = "test_images"

    # File Templates
    raw_zip_template: str = "sub-{subject:02d}.zip"
    raw_subdir_template: str = "sub-{subject:02d}"

    processed_subdir_template: str = "sub-{subject:02d}"

    # Shape: (sessions, conditions, repetitions, channels, timepoints) -> (4, 8270, 2, 64, 251)
    eeg_train_template: str = "preprocessed_eeg_training_sub-{subject:02d}.npy"
    # Shape: (sessions, conditions, repetitions, channels, timepoints) -> (4, 200, 20, 64, 251)
    eeg_test_template: str = "preprocessed_eeg_test_sub-{subject:02d}.npy"

    # Shape: (sessions, conditions) -> (4, 8270)
    train_image_conditions_template: str = (
        "img_conditions_training_sub-{subject:02d}.npy"
    )
    # Shape: (sessions, conditions) -> (4, 200)
    test_image_conditions_template: str = "img_conditions_test_sub-{subject:02d}.npy"

    embedding_template: str = "{model}_embeddings.pt"
    version_file: str = "DATA_VERSION.txt"

    def get_raw_dir(self, root: Path) -> Path:
        return root / self.raw_dir

    def get_source_dir(self, root: Path) -> Path:
        return root / self.source_dir

    def get_processed_dir(self, root: Path) -> Path:
        return root / self.processed_dir

    def get_raw_subject_dir(self, root: Path, subject: int) -> Path:
        return self.get_raw_dir(root) / self.raw_subdir_template.format(subject=subject)

    def get_processed_subject_dir(self, root: Path, subject: int) -> Path:
        return self.get_processed_dir(root) / self.raw_subdir_template.format(
            subject=subject
        )

    def get_images_dir(self, root: Path) -> Path:
        return root / self.images_dir

    def get_training_images_dir(self, root: Path) -> Path:
        return root / self.images_dir / self.train_imgs_dir

    def get_test_images_dir(self, root: Path) -> Path:
        return root / self.images_dir / self.test_imgs_dir

    def get_embeddings_dir(self, root: Path) -> Path:
        return root / self.embeddings_dir

    def get_eeg_train_file(self, root: Path, subject: int) -> Path:
        return (
            self.get_processed_dir(root)
            / self.processed_subdir_template.format(subject=subject)
            / self.eeg_train_template.format(subject=subject)
        )

    def get_eeg_test_file(self, root: Path, subject: int) -> Path:
        return (
            self.get_processed_dir(root)
            / self.processed_subdir_template.format(subject=subject)
            / self.eeg_test_template.format(subject=subject)
        )

    def get_eeg_train_image_conditions_file(self, root: Path, subject: int) -> Path:
        return (
            self.get_processed_dir(root)
            / self.processed_subdir_template.format(subject=subject)
            / self.train_image_conditions_template.format(subject=subject)
        )

    def get_eeg_test_image_conditions_file(self, root: Path, subject: int) -> Path:
        return (
            self.get_processed_dir(root)
            / self.processed_subdir_template.format(subject=subject)
            / self.test_image_conditions_template.format(subject=subject)
        )

    def get_metadata_file(self, root: Path, subject: int) -> Path:
        return (
            self.get_processed_dir(root)
            / self.processed_subdir_template.format(subject=subject)
            / f"meta_sub-{subject:02d}.json"
        )

    def get_embedding_file(self, root: Path, model_name: str) -> Path:
        return self.get_embeddings_dir(root) / self.embedding_template.format(
            model=model_name
        )

    def get_version_file(self, root: Path) -> Path:
        return root / self.version_file


# Global instance to be imported by other modules
layout = DataDirectoryLayout()
