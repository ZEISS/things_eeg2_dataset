import json
from pathlib import Path

import numpy as np


def save_prepr(  # noqa: PLR0913
    subject: int,
    whitened_test: list[np.ndarray],
    whitened_train: list[np.ndarray],
    img_conditions_train: list[np.ndarray],
    img_conditions_test: list[np.ndarray],
    ch_names: list[str],
    times: np.ndarray,
    processed_dir: Path,
) -> None:
    # Replace the above loop with concatenate for efficiency

    TEST_DATA_NAME = "preprocessed_eeg_test.npy"
    TRAIN_DATA_NAME = "preprocessed_eeg_training.npy"
    TEST_IMG_COND_NAME = "img_conditions_test.npy"
    TRAIN_IMG_COND_NAME = "img_conditions_training.npy"

    save_path = Path(processed_dir) / f"sub-{format(subject, '02')}"

    save_path.mkdir(parents=True, exist_ok=True)

    # Shape: (Number of sessions x Image conditions x EEG repetitions x EEG channels x EEG time points)
    # For all channels, this should be (4, 8270, 2, 64, 251) for the training data and (4, 200, 20, 64, 251) for the testing data
    test_eeg_data = np.array(whitened_test)
    np.save(save_path / TEST_DATA_NAME, test_eeg_data)

    img_conditions_test = np.array(img_conditions_test)
    np.save(save_path / TEST_IMG_COND_NAME, img_conditions_test)

    train_eeg_data = np.array(whitened_train)
    np.save(save_path / TRAIN_DATA_NAME, train_eeg_data)

    img_conditions_train = np.array(img_conditions_train)
    np.save(save_path / TRAIN_IMG_COND_NAME, img_conditions_train)

    # Save channel names and times as JSON files
    with (save_path / "meta.json").open("w") as f:
        meta_info = {"ch_names": ch_names, "times": times.tolist()}
        json.dump(meta_info, f)
