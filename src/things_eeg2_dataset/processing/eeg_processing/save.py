import json
import logging
from pathlib import Path

import numpy as np

from things_eeg2_dataset.paths import layout

logger = logging.getLogger(__name__)


def save_prepr(  # noqa: PLR0913
    subject: int,
    whitened_test: list[np.ndarray],
    whitened_train: list[np.ndarray],
    img_conditions_train: list[np.ndarray],
    img_conditions_test: list[np.ndarray],
    ch_names: list[str],
    times: np.ndarray,
    project_dir: Path,
) -> None:
    # Replace the above loop with concatenate for efficiency

    save_path = layout.get_processed_subject_dir(project_dir, subject)

    save_path.mkdir(parents=True, exist_ok=True)

    # Shape: (Number of sessions x Image conditions x EEG repetitions x EEG channels x EEG time points)
    # For all channels, this should be (4, 8270, 2, 64, 301) for the training data and (4, 200, 20, 64, 301) for the testing data
    test_eeg_data = np.array(whitened_test)
    np.save(layout.get_eeg_test_file(project_dir, subject), test_eeg_data)

    img_conditions_test = np.array(img_conditions_test)
    np.save(
        layout.get_eeg_test_image_conditions_file(project_dir, subject),
        img_conditions_test,
    )

    train_eeg_data = np.array(whitened_train)
    np.save(layout.get_eeg_train_file(project_dir, subject), train_eeg_data)

    img_conditions_train = np.array(img_conditions_train)
    np.save(
        layout.get_eeg_train_image_conditions_file(project_dir, subject),
        img_conditions_train,
    )

    # Save channel names and times as JSON files
    with layout.get_metadata_file(project_dir, subject).open("w") as f:
        meta_info = {"ch_names": ch_names, "times": times.tolist()}
        json.dump(meta_info, f)

    logger.info(f"Data saved to: {save_path}/")
