from pathlib import Path

import mne
import numpy as np

ORIGINAL_SAMPLING_FREQUENCY = 1000  # Original EEG sampling frequency in Hz
STIM_CHANNEL = "stim"  # Name of the stimulus channel in the raw EEG data
NUM_SESSIONS = 4  # Number of EEG data collection sessions

mne.set_log_level("WARNING")


def epoch(  # noqa: PLR0913
    subject: int,
    project_dir: Path,
    sampling_frequency: int,
    data_part: str,
    use_decim: bool = True,
    # TODO: Remove testing option  # noqa: FIX002
    testing: bool = False,
) -> tuple[np.ndarray, np.ndarray, list, np.ndarray]:
    """This function first converts the EEG data to MNE raw format, and
    performs channel selection, epoching, baseline correction and frequency
    downsampling.
    """
    chan_order = ["Fp1", "Fp2", "AF7", "AF3", "AFz", "AF4", "AF8", "F7", "F5", "F3", "F1", "F2", "F4", "F6", "F8", "FT9", "FT7", "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6", "FT8", "FT10", "T7", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "T8", "TP9", "TP7", "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6", "TP8", "TP10", "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8", "PO7", "PO3", "POz", "PO4", "PO8", "O1", "Oz", "O2"]  # fmt: skip

    epoched_data = []
    img_conditions = []

    for s in range(NUM_SESSIONS):
        eeg_dir = (
            Path("raw_data")
            / f"sub-{format(subject, '02')}"
            / f"ses-{format(s + 1, '02')}"
            / f"raw_eeg_{data_part}.npy"
        )
        eeg_data = np.load(Path(project_dir) / eeg_dir, allow_pickle=True).item()

        if not testing:
            ch_names = eeg_data["ch_names"]
            sfreq = eeg_data["sfreq"]
            ch_types = eeg_data["ch_types"]
            eeg_data = eeg_data["raw_eeg_data"]
        else:
            # Choose last two channels (to include stim channel)
            ch_names = eeg_data["ch_names"][-2:]
            sfreq = eeg_data["sfreq"]
            ch_types = eeg_data["ch_types"][-2:]
            eeg_data = eeg_data["raw_eeg_data"][-2:, :]

        info = mne.create_info(ch_names, sfreq, ch_types)
        raw = mne.io.RawArray(eeg_data, info)
        del eeg_data

        decim_factor = 1

        ### Get events, drop unused channels and reject target trials ###
        events = mne.find_events(raw, stim_channel="stim")
        # Check if we want to use decimation and if the math works (must be integer)
        if use_decim and (sfreq % sampling_frequency == 0):
            decim_factor = int(sfreq / sampling_frequency)

            # We filter at Nyquist / 1.5 (re_sfreq / 3.0) to be safe
            raw.filter(
                l_freq=None, h_freq=sampling_frequency / 3.0, n_jobs=-1, verbose=False
            )

        elif sampling_frequency < ORIGINAL_SAMPLING_FREQUENCY:
            # Fallback to slow resampling if ratios don't match or use_decim is False
            stim_index = raw.info["ch_names"].index(STIM_CHANNEL)
            raw, events = raw.resample(
                sampling_frequency, events=events, n_jobs=-1, stim_picks=stim_index
            )
        if not testing:
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
        sorted_data = sorted_data[:, :, :, pre_stim_samples:]

        epoched_data.append(sorted_data)
        img_conditions.append(img_cond)
        del sorted_data

    return epoched_data, img_conditions, ch_names, times
