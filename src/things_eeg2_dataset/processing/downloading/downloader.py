"""RawDownloader class for downloading THINGS-EEG2 raw data."""

import logging
import shutil
import zipfile
from pathlib import Path
from typing import ClassVar, TypedDict

from osfclient import OSF

from things_eeg2_dataset.paths import layout

from .download_utils import download_from_gdrive, download_from_url

logger = logging.getLogger(__name__)

TOTAL_EXISTING_SUBJECTS = 10


class DownloadSummary(TypedDict):
    total_subjects: int
    subjects_to_download: list[int]
    subjects_existing: list[int]
    total_size_mb: int


class Downloader:
    """Download and manage THINGS-EEG2 raw data files.

    This class handles downloading raw EEG data and image stimuli

    Attributes:
        data_path: Path to store downloaded data
        subjects: List of subject IDs to download (1-10)
        overwrite: Whether to re-download existing files
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
    SUBJECT_SIZE_MB = 10240  # ~10GB per subject

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
            overwrite: If True, re-download existing files
            dry_run: If True, only report what would be downloaded
            timeout: Network timeout in seconds
            max_retries: Maximum retry attempts for failed downloads

        Raises:
            ValueError: If subjects contains invalid IDs
        """
        self.project_dir = Path(project_dir)
        self.raw_dir = layout.get_raw_dir(self.project_dir)
        self.source_dir = layout.get_source_dir(self.project_dir)
        self.image_dir = layout.get_images_dir(self.project_dir)
        self.train_img_dir = layout.get_training_images_dir(self.project_dir)
        self.test_img_dir = layout.get_test_images_dir(self.project_dir)
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
                f"Subject IDs must be between 1 and {TOTAL_EXISTING_SUBJECTS}."
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
            source_dir: Directory where subject data is stored

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
        """Extract a ZIP file to the specified directory.

        Args:
            zip_path: Path to the ZIP file
            extract_dir: Directory to extract contents into
            keep_zip: If True, keep the ZIP file after extraction
        """

        if self.dry_run:
            logger.info(f"[DRY RUN] Would extract {zip_path}")
            return

        try:
            logger.info(f"Extracting {zip_path.name}...")

            if not zipfile.is_zipfile(zip_path):
                logger.error(f"File is not a valid ZIP archive: {zip_path}")
                raise zipfile.BadZipFile(f"Invalid ZIP: {zip_path.name}")

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
        # Check if training images already exist (only if directory exists)
        if (
            not self.overwrite
            and self.train_img_dir.exists()
            and any(self.train_img_dir.iterdir())
        ):
            logger.info("Training images already exist, skipping download.")
            return True

        if self.dry_run:
            logger.info("[DRY RUN] Would download image data from OSF.")
            return True

        logger.info("Downloading image data from OSF...")

        osf = OSF()
        project = osf.project(self.OSF_THINGS_EEG2_PROJECT_ID)
        storage = project.storage("osfstorage")

        try:
            # Download all files from OSF (including image_metadata.npy)
            for f in storage.files:
                fpath = self.image_dir / str(f.path).lstrip("/")
                logger.info(f"Downloading: {fpath.name}")

                # Ensure parent directory exists
                fpath.parent.mkdir(parents=True, exist_ok=True)

                with fpath.open("wb") as out:
                    f.write_to(out)

            # Verify required files were downloaded
            required_files = [
                "test_images.zip",
                "training_images.zip",
                "image_metadata.npy",
            ]
            for fname in required_files:
                fpath = self.image_dir / fname
                if not fpath.exists():
                    logger.error(f"Required file not found: {fname}")
                    return False

            # Extract image ZIP files
            logger.info("Extracting image archives...")
            for zip_fname in ["test_images.zip", "training_images.zip"]:
                zip_path = self.image_dir / zip_fname
                self._extract_zip(zip_path, self.image_dir, keep_zip=False)

            logger.info("Image data downloaded successfully")
            return True

        except Exception as e:
            logger.error(f"Error downloading image data: {e}")
            return False

    def download_subject(self, subject_id: int, url_dict: dict, raw_dir: Path) -> bool:  # noqa: PLR0912
        """Download raw data for a specific subject.

        Args:
            subject_id: Subject ID to download
            url_dict: Dictionary mapping subject IDs to download URLs
            raw_dir: Directory to store raw data

        Returns:
            True if download succeeded, False otherwise
        """
        logger.info(f"Starting download for subject {subject_id:02d}")

        zip_exists, extracted_exists, valid_structure = self._check_if_exists(
            subject_id, raw_dir
        )

        if valid_structure and not self.overwrite:
            logger.info(
                f"Subject {subject_id:02d} already exists with valid structure, skipping"
            )
            return True

        if self.dry_run:
            logger.info(
                f"[DRY RUN] Would download data for subject {subject_id:02d} in {raw_dir}"
            )
            return True

        if self.overwrite and (zip_exists or extracted_exists):
            logger.info(
                f"Overwrite mode: removing existing data for subject {subject_id:02d}"
            )
            # Remove ZIP if exists
            zip_path = raw_dir / f"sub-{subject_id:02d}.zip"
            if zip_path.exists():
                zip_path.unlink()

            # Remove extracted directory if exists
            subject_path = raw_dir / f"sub-{subject_id:02d}"
            if subject_path.exists():
                shutil.rmtree(subject_path)

        if extracted_exists and not self.overwrite:
            logger.info(
                f"Extracted files already exist for subject {subject_id:02d}, skipping download"
            )
            return True

        if not zip_exists:
            url = url_dict[subject_id]
            zip_path = raw_dir / f"sub-{subject_id:02d}.zip"

            # Choose appropriate download method based on URL type
            if "figshare.com" in url:
                # Use urllib-based download for Figshare URLs
                success = download_from_url(
                    url=url,
                    dest_path=zip_path,
                    description=f"subject {subject_id:02d} data",
                    dry_run=self.dry_run,
                    max_retries=self.max_retries,
                )
                if not success:
                    logger.error(
                        f"Failed to download subject {subject_id:02d} from Figshare"
                    )
                    return False
            else:
                # Use gdown for Google Drive URLs
                try:
                    download_from_gdrive(url, zip_path)
                except Exception as e:
                    logger.error(
                        f"Failed to download subject {subject_id:02d} from Google Drive: {e}"
                    )
                    return False

        # Extract the ZIP file after download (or if it already existed)
        zip_path = raw_dir / f"sub-{subject_id:02d}.zip"
        if zip_path.exists() and not self.dry_run:
            subject_dir = raw_dir / f"sub-{subject_id:02d}"
            if not subject_dir.exists() or self.overwrite:
                self._extract_zip(zip_path, raw_dir, keep_zip=False)

        logger.info(f"Successfully processed subject {subject_id:02d}")
        return True

    def download_raw_data(self) -> dict[int, bool]:
        """Download raw EEG data for all specified subjects.

        Returns:
            Dictionary mapping subject IDs to download success status
        """
        logger.info(f"Starting download for {len(self.subjects)} subjects")
        results = {}

        for subject_id in self.subjects:
            # Try Figshare first
            success = self.download_subject(
                subject_id, self.RAW_DATA_FIGSHARE_URLS, self.raw_dir
            )

            # If not successful, try Google Drive
            if not success:
                logger.warning(
                    f"Figshare download failed for subject {subject_id:02d}, trying Google Drive"
                )
                success = self.download_subject(
                    subject_id, self.RAW_DATA_GDRIVE_URLS, self.raw_dir
                )

            results[subject_id] = success

            if not success:
                logger.warning(
                    f"Subject {subject_id:02d} failed from both sources, continuing with next subject"
                )

        successful = sum(results.values())
        failed = len(results) - successful

        logger.info(
            f"Raw data download complete: {successful} successful, {failed} failed"
        )

        if failed > 0:
            failed_subjects = [sid for sid, success in results.items() if not success]
            logger.warning(f"Failed subjects: {failed_subjects}")

        return results

    def download_source_data(self) -> dict[int, bool]:
        """ "Download source data from Google Drive

        Returns:
            Dictionary mapping subject IDs to download success status
        """
        logger.info("Downloading source data from Google Drive...")
        results = {}

        for subject_id in self.subjects:
            success = self.download_subject(
                subject_id, self.SOURCE_DATA_GDRIVE_URLS, self.source_dir
            )
            results[subject_id] = success

            if not success:
                logger.warning(
                    f"Source data download failed for subject {subject_id:02d}"
                )

        successful = sum(results.values())
        failed = len(results) - successful

        logger.info(
            f"Source data download complete: {successful} successful, {failed} failed"
        )

        if failed > 0:
            failed_subjects = [sid for sid, success in results.items() if not success]
            logger.warning(f"Failed subjects: {failed_subjects}")

        return results

    def print_manual_download_instructions(self, failed_subjects: list[int]) -> None:
        """Print manual instructions for failed subjects

        Args:
            failed_subjects: List of subject IDs that failed to download
        """
        print("\n" + "!" * 70)
        print("MANUAL DOWNLOAD REQUIRED")
        print("!" * 70)
        print("\nSome source data subjects failed to download from Google Drive.")
        print(f"Failed subjects: {failed_subjects}")
        print(
            "\nPlease download these subjects manually and place the extracted .zip files in the source data directory ./things_eeg2/source_data:"
        )
        print()

    def download_all(self) -> bool:
        """Download all data: raw EEG and images, then optionally source data.

        Returns:
            True if all requested downloads succeeded, False if raw/image downloads failed
        """
        logger.info("=" * 70)
        logger.info("Starting full THINGS-EEG2 data download")
        logger.info("=" * 70)

        # Download raw EEG data
        logger.info("\n --- Downloading raw EEG data... ---")
        raw_results = self.download_raw_data()
        raw_success = all(raw_results.values())

        # Download image data
        logger.info("\n --- Downloading image data... ---")
        image_success = self.download_images()

        # Check if initial downloads (raw + images) succeeded
        logger.info("\n" + "=" * 70)
        if not (raw_success and image_success):
            logger.error("✗ Initial download incomplete")
            if not raw_success:
                failed_subjects = [
                    sid for sid, success in raw_results.items() if not success
                ]
                logger.error(f"  Failed raw data subjects: {failed_subjects}")
            if not image_success:
                logger.error("  Failed to download image data")
            logger.error(
                "Cannot proceed to source data download without raw data and images."
            )
            logger.info("=" * 70)
            return False

        logger.info("✓ Raw data and image data downloaded successfully")
        logger.info("=" * 70)

        # Prompt user to continue with source data download
        if self.dry_run:
            logger.info("[DRY RUN] Would prompt user for source data download")
            logger.info("=" * 70)
            return True

        # Interactive prompt for live mode
        print()
        while True:
            user_input = (
                input(
                    "Do you want to proceed with downloading the source data? (y/n): "
                )
                .strip()
                .lower()
            )
            if user_input in ["y", "yes"]:
                break  # Continue to source data download
            elif user_input in ["n", "no"]:
                logger.info("Source data download skipped by user.")
                logger.info("=" * 70)
                return True  # Exit successfully without source data
            else:
                print("Please enter 'yes' or 'no'")

        # Download source data (only reached if user said "yes")
        logger.info("\n --- Downloading source data... ---")
        source_results = self.download_source_data()
        source_success = all(source_results.values())

        # Print final status
        logger.info("\n" + "=" * 70)
        if source_success:
            logger.info("✓ All data downloaded successfully (raw + images + source)")
        else:
            # Some subjects failed - provide manual download instructions
            failed_subjects = [
                sid for sid, success in source_results.items() if not success
            ]
            logger.warning("✗ Source data download incomplete")
            self.print_manual_download_instructions(failed_subjects)

        logger.info("=" * 70)
        return True

    def print_summary(self) -> None:
        """Print download configuration summary."""
        print("\n" + "=" * 70)
        print("THINGS-EEG2 Download Configuration")
        print("=" * 70)
        print(f"Subjects to download: {self.subjects}")
        print(f"Data path: {self.project_dir}")
        print(f"Mode: {'DRY RUN' if self.dry_run else 'LIVE'}")
        print(f"Overwrite existing: {self.overwrite}")
        print()
        print("=" * 70 + "\n")
