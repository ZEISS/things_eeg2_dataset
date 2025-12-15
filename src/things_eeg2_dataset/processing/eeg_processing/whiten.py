import logging
import multiprocessing as mp
from enum import Enum

import mne
import numpy as np
import scipy
from sklearn.discriminant_analysis import _cov

mne.set_log_level("WARNING")
logger = logging.getLogger(__name__)


class MVNNDim(str, Enum):
    TIME = "time"
    EPOCHS = "epochs"


def whiten_one_session(
    epoched_test: np.ndarray, epoched_train: np.ndarray, mvnn_dim: MVNNDim
) -> tuple[np.ndarray, np.ndarray]:
    whitened_test = []
    whitened_train = []
    session_data = [epoched_test, epoched_train]

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
        for i in range(session_data[p].shape[0]):  # iterating over Image conditions
            cond_data = session_data[p][
                i
            ]  # EEG repetitions x EEG channels x EEG time points
            # Compute covariace matrices at each time point, and then
            # average across time points
            if mvnn_dim == MVNNDim.TIME:
                sigma_cond[i] = np.mean()
            # Compute covariace matrices at each epoch (EEG repetition),
            # and then average across epochs/repetitions
            elif mvnn_dim == MVNNDim.EPOCHS:
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
    whitened_test = np.reshape(
        (
            np.reshape(
                session_data[0],
                (-1, session_data[0].shape[2], session_data[0].shape[3]),
            ).swapaxes(1, 2)
            @ sigma_inv
        ).swapaxes(1, 2),
        session_data[0].shape,
    )

    whitened_train = np.reshape(
        (
            np.reshape(
                session_data[1],
                (-1, session_data[1].shape[2], session_data[1].shape[3]),
            ).swapaxes(1, 2)
            @ sigma_inv
        ).swapaxes(1, 2),
        session_data[1].shape,
    )

    return whitened_test, whitened_train


# Define a helper to unpack arguments (necessary for pool.imap)
def _whiten_wrapper(
    args: tuple[np.ndarray, np.ndarray, MVNNDim],
) -> tuple[np.ndarray, np.ndarray]:
    return whiten_one_session(*args)


def mvnn_whiten(
    number_of_sessions: int,
    epoched_test: list[np.ndarray],
    epoched_train: list[np.ndarray],
    mvnn_dim: MVNNDim = MVNNDim.EPOCHS,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    tasks = [
        (epoched_test[s], epoched_train[s], mvnn_dim) for s in range(number_of_sessions)
    ]

    whitened_test = []
    whitened_train = []

    # Use imap to iterate over results in the MAIN process
    with mp.Pool(mp.cpu_count()) as pool:
        for i, result in enumerate(pool.imap(_whiten_wrapper, tasks)):
            whitened_test.append(result[0])
            whitened_train.append(result[1])
            logger.info(f"Whitening: Session {i + 1}/{number_of_sessions} completed.")

    return whitened_test, whitened_train
