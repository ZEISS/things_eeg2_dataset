import mne
import numpy as np
import scipy
from sklearn.discriminant_analysis import _cov
from tqdm import tqdm

mne.set_log_level("WARNING")


def mvnn_whiten(
    number_of_sessions: int,
    mvnn_dim: str,
    epoched_test: list[np.ndarray],
    epoched_train: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the covariance matrices of the EEG data (calculated for each
    time-point or epoch/repetitions of each image condition), and then average
    them across image conditions and data partitions. The inverse of the
    resulting averaged covariance matrix is used to whiten the EEG data
    (independently for each session).

    zero-score standardization also has well performance

    Parameters
    ----------
    number_of_sessions : int
            Number of EEG data collection sessions.
    mvnn_dim : int
            Dimension mode for MVNN ('time' or 'epochs').
    epoched_test : list of floats
            Epoched test EEG data.
    epoched_train : list of floats
            Epoched training EEG data.

    Returns
    -------
    whitened_test : list of float
            Whitened test EEG data.
    whitened_train : list of float
            Whitened training EEG data.

    """

    ### Loop across data collection sessions ###
    whitened_test = []
    whitened_train = []
    for s in range(number_of_sessions):
        session_data = [epoched_test[s], epoched_train[s]]

        ### Compute the covariance matrices ###
        # Data partitions covariance matrix of shape:
        # Data partitions x EEG channels x EEG channels
        sigma_part = np.empty(
            (len(session_data), session_data[0].shape[2], session_data[0].shape[2])
        )
        for p in range(sigma_part.shape[0]):
            # Image conditions covariance matrix of shape:
            # Image conditions x EEG channels x EEG channels
            sigma_cond = np.empty(
                (
                    session_data[p].shape[0],
                    session_data[0].shape[2],
                    session_data[0].shape[2],
                )
            )
            for i in tqdm(
                range(session_data[p].shape[0])
            ):  # iterating over Image conditions
                cond_data = session_data[p][
                    i
                ]  # EEG repetitions x EEG channels x EEG time points
                # Compute covariace matrices at each time point, and then
                # average across time points
                if mvnn_dim == "time":
                    sigma_cond[i] = np.mean()
                # Compute covariace matrices at each epoch (EEG repetition),
                # and then average across epochs/repetitions
                elif mvnn_dim == "epochs":
                    sigma_cond[i] = np.mean(
                        [
                            _cov(np.transpose(cond_data[e]), shrinkage="auto")
                            for e in range(cond_data.shape[0])
                        ],
                        axis=0,
                    )
            # Average the covariance matrices across image conditions
            sigma_part[p] = sigma_cond.mean(axis=0)
        # # Average the covariance matrices across image partitions
        # sigma_tot = sigma_part.mean(axis=0)
        # ? It seems not fair to use test data for mvnn, so we change to just use training data
        sigma_tot = sigma_part[1]
        # Compute the inverse of the covariance matrix
        sigma_inv = scipy.linalg.fractional_matrix_power(sigma_tot, -0.5)

        ### Whiten the data ###
        whitened_test.append(
            np.reshape(
                (
                    np.reshape(
                        session_data[0],
                        (-1, session_data[0].shape[2], session_data[0].shape[3]),
                    ).swapaxes(1, 2)
                    @ sigma_inv
                ).swapaxes(1, 2),
                session_data[0].shape,
            )
        )
        whitened_train.append(
            np.reshape(
                (
                    np.reshape(
                        session_data[1],
                        (-1, session_data[1].shape[2], session_data[1].shape[3]),
                    ).swapaxes(1, 2)
                    @ sigma_inv
                ).swapaxes(1, 2),
                session_data[1].shape,
            )
        )

    return whitened_test, whitened_train
