"""RawDownloader class for downloading THINGS-EEG2 raw data."""

import logging
import shutil
import zipfile
from pathlib import Path
from typing import ClassVar, TypedDict

from osfclient import OSF

from .download_utils import download_from_gdrive

logger = logging.getLogger(__name__)

TOTAL_EXISTING_SUBJECTS = 10


class DownloadSummary(TypedDict):
    total_subjects: int
    subjects_to_download: list[int]  # Assuming subject_id is int
    subjects_existing: list[int]
    total_size_mb: int  # or float


class Downloader:
    """Download and manage THINGS-EEG2 raw data files.

    This class handles downloading raw EEG data from OpenNeuro,
    extracting ZIP archives, and validating data structure.

    Attributes:
        data_path: Path to store downloaded data
        subjects: List of subject IDs to download (1-10)
        force: Whether to re-download existing files
        dry_run: If True, only show what would be downloaded
        timeout: Network timeout in seconds
        max_retries: Maximum number of retry attempts
    """

    # Raw data from Google Drive
    RAW_DATA_GDRIVE_URLS: ClassVar[dict[int, str]] = {
        1: "https://drive.google.com/uc?id=1GCEoU_VFAnxwhX3wOXgzpcqdMkzK2j4d",
        2: "https://drive.google.com/uc?id=1fmzu5I_sP11zmARpG4up_inn8wbG4GQE",
        3: "https://drive.google.com/uc?id=1gKB-9AuueH9pfbT0hIKe0hstMuCbC9m4",
        4: "https://drive.google.com/uc?id=1hEJuZbw9EAXsdZk7G8Joif5V64-mrC3x",
        5: "https://drive.google.com/uc?id=19Q0s9oZdlxt1Ct0VuGVwCJVo8uXMnwuS",
        6: "https://drive.google.com/uc?id=1puOoIkZjWXCNWf3iIzYackAOFxmwqSH0",
        7: "https://drive.google.com/uc?id=1Z-FtP6kR02N-5G9p24mdfY12z9XUhUEB",
        8: "https://drive.google.com/uc?id=1mkOEFmoSyEZiIqa7fZ47Q00V0PDJxqjQ",
        9: "https://drive.google.com/uc?id=1NV9bL_M2jSlL8iZ2qI69azbxiW8Pptfb",
        10: "https://drive.google.com/uc?id=1f29e8A5Pr3Iu8el7aPkhJSRfd-rrAE0W",
    }

    # Raw data from OpenNeuro Figshare
    RAW_DATA_FIGSHARE_URLS: ClassVar[dict[int, str]] = {
        1: "https://plus.figshare.com/ndownloader/files/33244238",
        2: "https://plus.figshare.com/ndownloader/files/33247340",
        3: "https://plus.figshare.com/ndownloader/files/33247355",
        4: "https://plus.figshare.com/ndownloader/files/33247361",
        5: "https://plus.figshare.com/ndownloader/files/33247376",
        6: "https://plus.figshare.com/ndownloader/files/34404491",
        7: "https://plus.figshare.com/ndownloader/files/33247622",
        8: "https://plus.figshare.com/ndownloader/files/33247652",
        9: "https://plus.figshare.com/ndownloader/files/38916017",
        10: "https://plus.figshare.com/ndownloader/files/33247694",
    }

    # Source data from Google Drive
    SOURCE_DATA_GDRIVE_URLS: ClassVar[dict[int, str]] = {
        1: "https://drive.google.com/uc?id=1_uhLBexafzG79YhQqQ1XBoxARA0iaslK",
        2: "https://drive.google.com/uc?id=1vkcpzO2F-ZWSRDyOKEiAh60YvdOZAv71",
        3: "https://drive.google.com/uc?id=16gNXYBXcEIy-UPXN6pGuWZfmIEqG2iW0",
        4: "https://drive.google.com/uc?id=1UsgaLgAyfEvBXQlzL8DKGYfj2x_5mnB3",
        5: "https://drive.google.com/uc?id=1RvejTZ1KAwV31IMACoT2fuq2fWTsYJZt",
        6: "https://drive.google.com/uc?id=1lySVhBgPWM-Q91n8xTMdBqcxTpVSvs_y",
        7: "https://drive.google.com/uc?id=1RJr0m_JoS3683A8Ee_ncSsE4TDMAulD6",
        8: "https://drive.google.com/uc?id=1YxaTXZct7CJz6UrZT3IstMkFRPKx-xDY",
        9: "https://drive.google.com/uc?id=1ldDKfrJKY8DXRR8iZBr6oMH7HSmaCTgH",
        10: "https://drive.google.com/uc?id=16LIyat3CYlsgsDjVe1AXMDPa94fAo4bI",
    }

    OSF_THINGS_EEG2_PROJECT_ID: str = "Y63gw"
    # Expected file sizes (approximate, in bytes)
    SUBJECT_SIZE_MB = 10240  # ~10GB per subject
    METADATA_SIZE_KB = 100  # ~100KB for metadata

    def __init__(  # noqa: PLR0913
        self,
        project_dir: str | Path = "data/things-eeg2/",
        subjects: list[int] | None = None,
        overwrite: bool = False,
        dry_run: bool = False,
        timeout: int = 300,
        max_retries: int = 3,
    ) -> None:
        """Initialize the RawDownloader.

        Args:
            project_dir: Directory path to store downloaded data
            subjects: List of subject IDs (1-10). Default is all subjects.
            force: If True, re-download existing files
            dry_run: If True, only report what would be downloaded
            timeout: Network timeout in seconds
            max_retries: Maximum retry attempts for failed downloads

        Raises:
            ValueError: If subjects contains invalid IDs
        """
        self.project_dir = Path(project_dir)
        self.raw_dir = self.project_dir / "raw_data"
        self.source_dir = self.project_dir / "source_data"
        self.image_dir = self.project_dir / "Image_set"
        self.train_img_dir = self.image_dir / "training_images"
        self.test_img_dir = self.image_dir / "test_images"
        self.subjects = subjects if subjects is not None else list(range(1, 11))
        self.overwrite = overwrite
        self.dry_run = dry_run
        self.timeout = timeout
        self.max_retries = max_retries

        # Validate subject IDs
        invalid_subjects = [
            s for s in self.subjects if s < 1 or s > TOTAL_EXISTING_SUBJECTS
        ]
        if invalid_subjects:
            raise ValueError(
                f"Invalid subject IDs: {invalid_subjects}. "
                "Subject IDs must be between 1 and 10."
            )

        # Create data directory if it doesn't exist
        if not self.dry_run:
            self.raw_dir.mkdir(parents=True, exist_ok=True)
            self.source_dir.mkdir(parents=True, exist_ok=True)
            self.image_dir.mkdir(parents=True, exist_ok=True)
            self.train_img_dir.mkdir(parents=True, exist_ok=True)
            self.test_img_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Downloader initialized for subjects: {self.subjects}")
        logger.info(f"Raw data path: {self.raw_dir}")
        logger.info(f"Mode: {'DRY RUN' if self.dry_run else 'LIVE'}")

    def _check_if_exists(
        self, subject_id: int, source_dir: Path
    ) -> tuple[bool, bool, bool]:
        """Check if subject data exists and is valid.

        Args:
            subject_id: Subject ID to check

        Returns:
            Tuple of (zip_exists, extracted_exists, valid_structure)
        """
        subject_path = source_dir / f"sub-{subject_id:02d}"
        zip_path = source_dir / f"sub-{subject_id:02d}.zip"

        zip_exists = zip_path.exists()
        extracted_exists = subject_path.exists()
        valid_structure = False

        if extracted_exists:
            valid_structure = self._validate_subject_structure(subject_path)

        return zip_exists, extracted_exists, valid_structure

    def _validate_subject_structure(self, subject_path: Path) -> bool:
        """Validate that subject directory has expected structure.

        Expected structure:
        sub-XX/
            ses-01/
                raw_eeg_training.npy
                raw_eeg_test.npy
            ses-02/
                raw_eeg_training.npy
                raw_eeg_test.npy
            ses-03/
                raw_eeg_training.npy
                raw_eeg_test.npy
            ses-04/
                raw_eeg_training.npy
                raw_eeg_test.npy

        Args:
            subject_path: Path to subject directory

        Returns:
            True if structure is valid, False otherwise
        """
        required_files = [
            "ses-01/raw_eeg_training.npy",
            "ses-01/raw_eeg_test.npy",
            "ses-02/raw_eeg_training.npy",
            "ses-02/raw_eeg_test.npy",
            "ses-03/raw_eeg_training.npy",
            "ses-03/raw_eeg_test.npy",
            "ses-04/raw_eeg_training.npy",
            "ses-04/raw_eeg_test.npy",
        ]

        for rel_path in required_files:
            full_path = subject_path / rel_path
            if not full_path.exists():
                logger.debug(f"Missing required file: {full_path}")
                return False

        logger.debug(f"Valid structure confirmed: {subject_path}")
        return True

    def _extract_zip(
        self, zip_path: Path, extract_dir: Path, keep_zip: bool = False
    ) -> None:
        if self.dry_run:
            logger.info(f"[DRY RUN] Would extract {zip_path}")

        try:
            logger.info(f"Extracting {zip_path.name}...")

            if not zipfile.is_zipfile(zip_path):
                logger.error(f"File is not a valid ZIP archive: {zip_path}")

            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(extract_dir)

            logger.info(f"Successfully extracted {zip_path.name}")

            if not keep_zip:
                logger.info(f"Removing ZIP file: {zip_path}")
                zip_path.unlink()

        except zipfile.BadZipFile as e:
            logger.error(f"Corrupted ZIP file: {zip_path}")
            raise zipfile.BadZipFile(f"Corrupted ZIP: {zip_path.name}") from e

        except Exception as e:
            logger.error(f"Unexpected error extracting {zip_path}: {e}")
            raise RuntimeError(f"Extraction failed for {zip_path.name}") from e

    def download_images(self) -> bool:
        """Download image data. LICENSE.txt, test_images.zip, training_images.zip and image_metadata.npy.

        Returns:
            True if download succeeded, False otherwise
        """

        if not self.overwrite and any(self.train_img_dir.iterdir()):
            logger.info("Training images already exist, skipping download.")
            return True

        osf = OSF()
        project = osf.project(self.OSF_THINGS_EEG2_PROJECT_ID)
        storage = project.storage("osfstorage")

        try:
            for f in storage.files:
                fpath = self.image_dir / str(f.path).lstrip("/")
                logger.info(f"Processing image file: {fpath}")

                if not self.dry_run:
                    # Ensure parent directory exists
                    fpath.parent.mkdir(parents=True, exist_ok=True)
                    with fpath.open("wb") as out:
                        f.write_to(out)

        except Exception as e:
            logger.error(f"Error downloading image data: {e}")
            return False

        zip_files = ["test_images.zip", "training_images.zip"]
        for zip_fname in zip_files:
            zip_path = self.image_dir / zip_fname

            if not zip_path.exists():
                logger.error(f"Image ZIP file not found: {zip_path}")
                return False

        return True

    def download_subject(self, subject_id: int, url_dict: dict, base_dir: Path) -> bool:
        logger.info(f"Starting download for subject {subject_id:02d}")
        zip_exists, extracted_exists, valid_structure = self._check_if_exists(
            subject_id, base_dir
        )

        if valid_structure and not self.overwrite:
            logger.info(
                f"Subject {subject_id:02d} already exists with valid structure, skipping"
            )
            return True

        if self.dry_run:
            logger.info(
                f"[DRY RUN] Would download data for subject {subject_id:02d} in {base_dir}"
            )
            return True

        if self.overwrite and (zip_exists or extracted_exists):
            logger.info(
                f"Overwrite mode: removing existing data for subject {subject_id:02d}"
            )
            # Remove ZIP if exists
            zip_path = base_dir / f"sub-{subject_id:02d}.zip"
            if zip_path.exists():
                zip_path.unlink()

            # Remove extracted directory if exists
            subject_path = base_dir / f"sub-{subject_id:02d}"
            if subject_path.exists():
                shutil.rmtree(subject_path)

        if extracted_exists and not self.overwrite:
            logger.info(
                f"Extracted files already exist for subject {subject_id:02d}, skipping download"
            )
            return True

        if not zip_exists:
            url = url_dict[subject_id]
            zip_path = base_dir / f"sub-{subject_id:02d}.zip"
            download_from_gdrive(url, zip_path)
        logger.info(f"Successfully processed subject {subject_id:02d}")
        return True

    def download_raw_data(self) -> dict[int, bool]:
        results = {}

        for subject_id in self.subjects:
            success = self.download_subject(
                subject_id, self.RAW_DATA_GDRIVE_URLS, self.raw_dir
            )
            results[subject_id] = success

            if not success:
                logger.warning(
                    f"Subject {subject_id:02d} failed, continuing with next subject"
                )

        successful = sum(results.values())
        failed = len(results) - successful

        logger.info(f"Download complete: {successful} successful, {failed} failed")

        if failed > 0:
            failed_subjects = [sid for sid, success in results.items() if not success]
            logger.warning(f"Failed subjects: {failed_subjects}")

        return results

    def download_source_data(self) -> dict[int, bool]:
        logger.info(f"Starting download for {len(self.subjects)} subjects")

        results = {}

        for subject_id in self.subjects:
            success = self.download_subject(
                subject_id, self.SOURCE_DATA_GDRIVE_URLS, self.source_dir
            )
            results[subject_id] = success

            if not success:
                logger.warning(
                    f"Subject {subject_id:02d} failed, continuing with next subject"
                )

        successful = sum(results.values())
        failed = len(results) - successful

        logger.info(f"Download complete: {successful} successful, {failed} failed")

        if failed > 0:
            failed_subjects = [sid for sid, success in results.items() if not success]
            logger.warning(f"Failed subjects: {failed_subjects}")

        return results

    def print_summary(self) -> None:
        print("\n" + "=" * 70)
        print("THINGS-EEG2 Raw Data Download Summary")
        print("=" * 70)
        print(f"Subjects to download: {self.subjects}")
        print(f"Data path: {self.project_dir}")
        print(f"Mode: {'DRY RUN' if self.dry_run else 'LIVE'}")
        print(f"Force download: {self.overwrite}")
        print()
        print("=" * 70 + "\n")
