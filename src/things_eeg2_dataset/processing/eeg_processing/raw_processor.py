"""
RawProcessor class for modular EEG data preprocessing.

This class encapsulates the preprocessing pipeline for raw EEG data including:
- Channel selection and epoching
- Frequency downsampling
- Baseline correction
- Multivariate noise normalization (MVNN)
- Data sorting and reshaping

https://www.sciencedirect.com/science/article/pii/S1053811922008758
Refer to the code of Things-EEG2 but with a few differences.
Many thanks!
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .epoching import epoch
from .save import save_prepr
from .whiten import mvnn_whiten

logger = logging.getLogger(__name__)

TRAIN_IMG_CLASSES = 1654
TEST_IMG_CLASSES = 200
TRAIN_IMG_SAMPLES_PER_CLASS = 10
TEST_IMG_SAMPLES_PER_CLASS = 1
TRAIN_REPETITIONS = 4
TEST_REPETITIONS = 80


@dataclass
class Subject:
    """Class for keeping track of a subject's information."""

    sub_id: int
    dir_name: str


class RawProcessor:
    """
    A modular processor for raw EEG data preprocessing.

    This class provides a structured interface to preprocess raw EEG data through
    channel selection, epoching, frequency downsampling, baseline correction,
    multivariate noise normalization (MVNN), and data sorting.

    Parameters
    ----------
    subjects : List[int]
        List of subject numbers to process.
    project_dir : str
        Directory of the project folder containing raw data.
    sfreq : int, optional
        Downsampling frequency in Hz (default: 250).
    mvnn_dim : str, optional
        Whether to compute the MVNN covariance matrices for each time point
        ('time') or for each epoch/repetition ('epochs'). Default: 'epochs'.

    Attributes
    ----------
    subjects : List[int]
        List of subject numbers.
    n_ses : int
        Number of EEG sessions.
    sfreq : int
        Downsampling frequency.
    mvnn_dim : str
        MVNN dimension mode.
    project_dir : str
        Project directory path.
    epoched_test : list[np.ndarray] | None
        Epoched test data after processing.
    epoched_train : list[np.ndarray] | None
        Epoched training data after processing.
    whitened_test : list[np.ndarray] | None
        Whitened test data after MVNN.
    whitened_train : list[np.ndarray] | None
        Whitened training data after MVNN.
    ch_names : list[str] | None
        EEG channel names.
    times : np.ndarray | None
        EEG time points.
    img_conditions_train : list[np.ndarray] | None
        Image conditions for training data.
    """

    def __init__(
        self,
        subjects: list[int],
        project_dir: str,
        processed_dir_name: str = "processed",
        sfreq: int = 250,
        mvnn_dim: str = "epochs",
    ) -> None:
        """Initialize the RawProcessor with preprocessing parameters."""
        self.subjects = subjects
        self.number_of_sessions = 4
        self.sampling_frequency = sfreq
        self.mvnn_dim = mvnn_dim
        self.project_dir = Path(project_dir)
        self.processed_dir = self.project_dir / processed_dir_name
        self.train_img_dir = self.project_dir / "Image_set" / "training_images"
        self.test_img_dir = self.project_dir / "Image_set" / "test_images"

        # Initialize data containers
        self.epoched_test: list[np.ndarray] | None = None
        self.epoched_train: list[np.ndarray] | None = None
        self.whitened_test: list[np.ndarray] | None = None
        self.whitened_train: list[np.ndarray] | None = None
        self.ch_names: list[str] | None = None
        self.times: np.ndarray | None = None
        self.img_conditions_train: list[np.ndarray] | None = None
        self.img_conditions_test: list[np.ndarray] | None = None
        self.sub: int | None = None

        # Stimulus metadata for validation
        self.stimulus_metadata_train: dict[str, Any] | None = None
        self.stimulus_metadata_test: dict[str, Any] | None = None

    def _check_processed_data_exists(self) -> bool:
        for subject in self.subjects:
            if not self._check_processed_data_exists_for_subject(subject):
                return False
        return True

    def _check_processed_data_exists_for_subject(self, subject: int) -> bool:
        processed_path = self.processed_dir / f"sub-{subject:02d}/"
        return processed_path.exists() and len(list(processed_path.iterdir())) > 0

    def epoch_and_sort(self, sub: int) -> None:
        logger.info("\n=== Epoching and sorting data ===")

        # Process test partition
        (
            self.epoched_test,
            self.img_conditions_test,
            self.ch_names,
            self.times,
        ) = epoch(sub, self.project_dir, self.sampling_frequency, "test")

        # Process training partition
        (
            self.epoched_train,
            self.img_conditions_train,
            # Channel names and times are the same as for test data
            _,
            _,
        ) = epoch(sub, self.project_dir, self.sampling_frequency, "training")

    def apply_mvnn(self) -> None:
        if self.epoched_test is None or self.epoched_train is None:
            raise ValueError(
                "Epoched data not found. Please run epoch_and_sort() first."
            )

        logger.info("\n=== Applying Multivariate Noise Normalization ===")

        self.whitened_test, self.whitened_train = mvnn_whiten(
            4, self.mvnn_dim, self.epoched_test, self.epoched_train
        )

        # Clean up epoched data to save memory
        self.epoched_test = None
        self.epoched_train = None

    def save_preprocessed_data(self, sub: int) -> None:
        if self.whitened_test is None or self.whitened_train is None:
            raise ValueError("Whitened data not found. Please run apply_mvnn() first.")

        logger.info("\n=== Saving preprocessed data ===")

        # Make sure that ch_names and times are not None
        if (
            self.ch_names is None
            or self.times is None
            or self.img_conditions_train is None
            or self.img_conditions_test is None
        ):
            raise ValueError("Channel names, times, or image conditions are None.")

        save_prepr(
            sub,
            self.whitened_test,
            self.whitened_train,
            self.img_conditions_train,
            self.img_conditions_test,
            self.ch_names,
            self.times,
            processed_dir=self.processed_dir,
        )

        logger.info(f"Data saved to: {self.processed_dir}/sub-{int(sub):02d}/")

    def run(self, overwrite: bool = False, dry_run: bool = False) -> None:
        """
        Run the complete preprocessing pipeline.

        This method executes all preprocessing steps in sequence:
        1. Epoch and sort the data
        2. Apply multivariate noise normalization
        3. Save the preprocessed data

        This is a convenience method that calls all processing steps automatically.

        Parameters
        ----------
        force : bool, optional
            If True, forces reprocessing even if processed data exists (default: False).
        dry_run : bool, optional
            If True, only simulates the preprocessing without executing it (default: False).
        """
        logger.info(">>> EEG Data Preprocessing <<<")
        logger.info(f"\nSubjects:          {self.subjects}")
        logger.info(f"Number of sessions: {self.number_of_sessions}")
        logger.info(f"Sampling frequency: {self.sampling_frequency} Hz")
        logger.info(f"MVNN dimension:     {self.mvnn_dim}")
        logger.info(f"Project directory:  {self.project_dir}")

        for subject in self.subjects:
            logger.info(f"\n--- Processing subject: {subject} ---")
            # only if processed data doesn't already exist or force is True
            if not self._check_processed_data_exists() or overwrite:
                self.epoch_and_sort(subject)
                self.apply_mvnn()
                if not dry_run:
                    self.save_preprocessed_data(subject)
            else:
                logger.info(
                    f"\nProcessed data for subject {subject} already exists. Skipping..."
                )

        logger.info("Preprocessing completed successfully!")
