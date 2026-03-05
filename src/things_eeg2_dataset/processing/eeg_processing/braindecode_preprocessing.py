from __future__ import annotations

from typing import Any

import mne


def apply_braindecode_preprocessing(
    raw: mne.io.BaseRaw,
    *,
    sfreq: int,
    l_freq: float | None = 0.5,
    h_freq: float | None = 40.0,
) -> mne.io.BaseRaw:
    """Apply a minimal Braindecode preprocessing pipeline (filter + resample).

    This is an optional alternative backend. Import is lazy so the core package
    does not require `braindecode` unless this path is used.
    """

    try:
        from braindecode.preprocessing import Preprocessor, preprocess  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "Braindecode backend requested but `braindecode` is not installed. "
            "Install it (e.g. `pip install braindecode`) or use backend='mne'."
        ) from e

    preprocessors: list[Any] = [
        Preprocessor("filter", l_freq=l_freq, h_freq=h_freq),
        Preprocessor("resample", sfreq=sfreq),
    ]
    preprocess(raw, preprocessors)
    return raw
