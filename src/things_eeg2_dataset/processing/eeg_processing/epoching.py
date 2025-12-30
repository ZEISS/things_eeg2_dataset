import logging
import threading
from pathlib import Path

import mne
import numpy as np

ORIGINAL_SAMPLING_FREQUENCY = 1000  # Original EEG sampling frequency in Hz
STIM_CHANNEL = "stim"  # Name of the stimulus channel in the raw EEG data
NUM_SESSIONS = 4  # Number of EEG data collection sessions

mne.set_log_level("WARNING")

logger = logging.getLogger(__name__)


class PrefetchLoader:
    def __init__(self, paths: list[Path]) -> None:
        self.paths = paths
        self.results: list[dict | None] = [None] * len(paths)
        self.errors: list[Exception | None] = [None] * len(paths)
        self.threads: list[threading.Thread | None] = [None] * len(paths)

    def start(self, index: int) -> None:
        def load_fn(i: int, path: Path) -> None:
            try:
                self.results[i] = np.load(path, allow_pickle=True).item()
            except Exception as e:
                self.errors[i] = e

        t = threading.Thread(target=load_fn, args=(index, self.paths[index]))
        t.start()
        self.threads[index] = t

    def get(self, index: int) -> dict:
        t = self.threads[index]
        if not t:
            raise RuntimeError(f"Data for {self.paths[index]} was not started loading.")
        t.join()
        if self.errors[index] is not None:
            raise RuntimeError(f"Failed to load {self.paths[index]}") from self.errors[
                index
            ]

        result = self.results[index]

        if result is None:
            raise RuntimeError(f"No data loaded for {self.paths[index]}.")
        return result


def epoching(
    subject: int,
    project_dir: Path,
    sampling_frequency: int,
    data_part: str,
    use_decim: bool = True,
) -> tuple[np.ndarray, np.ndarray, list, np.ndarray]:
    """This function first converts the EEG data to MNE raw format, and
    performs channel selection, epoching, baseline correction and frequency
    downsampling.
    """
    chan_order = ["Fp1", "Fp2", "AF7", "AF3", "AFz", "AF4", "AF8", "F7", "F5", "F3", "F1", "F2", "F4", "F6", "F8", "FT9", "FT7", "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6", "FT8", "FT10", "T7", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "T8", "TP9", "TP7", "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6", "TP8", "TP10", "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8", "PO7", "PO3", "POz", "PO4", "PO8", "O1", "Oz", "O2"]  # fmt: skip

    epoched_data = []
    img_conditions = []

    # Build full paths for all sessions
    paths = [
        Path(project_dir)
        / "raw_data"
        / f"sub-{subject:02d}"
        / f"ses-{s + 1:02d}"
        / f"raw_eeg_{data_part}.npy"
        for s in range(NUM_SESSIONS)
    ]
    prefetch = PrefetchLoader(paths)

    # Start loading session 0
    prefetch.start(0)

    for s in range(NUM_SESSIONS):
        logger.info(f"  Processing session {s + 1}/{NUM_SESSIONS}...")

        # Wait for this session's data to finish loading
        eeg_data = prefetch.get(s)

        # Start loading next session in background
        if s + 1 < NUM_SESSIONS:
            prefetch.start(s + 1)

        # with open(Path(project_dir) / paths[s], "rb") as f:
        #     header = f.read()  # numpy header includes PICKLE prefix
        #     proto = header[header.find(b'\x80') + 1]  # pickle stream starts with 0x80 <proto>
        #     print("Protocol:", proto)       ch_names = eeg_data["ch_names"]

        ch_names = eeg_data["ch_names"]
        sfreq = eeg_data["sfreq"]
        ch_types = eeg_data["ch_types"]
        eeg_data = eeg_data["raw_eeg_data"]

        info = mne.create_info(ch_names, sfreq, ch_types)

        # Note: eeg_data is huge. MNE copies it by default.
        # If you run out of RAM, consider verbose=False or copy=False (if safe)
        raw = mne.io.RawArray(eeg_data, info)

        # Explicitly delete the huge numpy array to free RAM for the next load
        del eeg_data

        decim_factor = 1

        ### Get events, drop unused channels and reject target trials ###
        events = mne.find_events(raw, stim_channel="stim")
        # Check if we want to use decimation and if the math works (must be integer)
        if use_decim and (sfreq % sampling_frequency == 0):
            decim_factor = int(sfreq / sampling_frequency)

            # We filter at Nyquist / 1.5 (re_sfreq / 3.0) to be safe
            raw.filter(
                l_freq=None,
                h_freq=sampling_frequency / 3.0,
                n_jobs=-1,
                verbose=False,
            )

        elif sampling_frequency < ORIGINAL_SAMPLING_FREQUENCY:
            # Fallback to slow resampling if ratios don't match or use_decim is False
            stim_index = raw.info["ch_names"].index(STIM_CHANNEL)
            raw, events = raw.resample(
                sampling_frequency, events=events, n_jobs=-1, stim_picks=stim_index
            )

        raw.pick(chan_order)

        # Reject the target trials (event 99999)
        target_trial_id = 99999
        events = events[events[:, 2] != target_trial_id]

        ### Epoching, baseline correction and resampling ###
        # * [0, 1.0]
        epochs = mne.Epochs(
            raw,
            events,
            tmin=-0.2,
            tmax=1.0,
            baseline=(None, 0),
            preload=True,
            verbose=False,
            decim=decim_factor,
        )
        del raw

        ch_names = epochs.info["ch_names"]
        times = epochs.times

        ### Sort the data ###
        data = epochs.get_data()
        events = epochs.events[:, 2]
        img_cond = np.unique(events)

        del epochs

        # The number of repetitions differs between image conditions.
        # 20 and 2 are the mimimum available repetitions for test and training data, respectively.
        max_reps = 20 if data_part == "test" else 2
        sorted_data = np.zeros((len(img_cond), max_reps, data.shape[1], data.shape[2]))

        # Image conditions x EEG repetitions x EEG channels x EEG time points

        for i in range(len(img_cond)):
            # Find the indices of the selected image condition
            idx = np.where(events == img_cond[i])[0]

            # Remove all excess repetitions over max_reps
            # Randomly select only the max number of EEG repetitions
            sorted_data[i] = data[idx[:max_reps], :, :]

        del data

        # remove pre-stimulus period (200ms)
        pre_stim_samples = int(0.2 * sampling_frequency)
        sorted_data = sorted_data[:, :, :, pre_stim_samples:].astype(np.float32)

        epoched_data.append(sorted_data)
        img_conditions.append(img_cond)
        del sorted_data

    logger.debug(f"Epoched data shape: {[d.shape for d in epoched_data]}")

    return epoched_data, img_conditions, ch_names, times
