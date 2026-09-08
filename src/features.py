"""Phase 5 baseline #1: handcrafted signal features + logistic regression.

Simple statistics computed directly from the preprocessed ECG window,
without going through any image representation. This is the floor every
CV model needs to beat to justify its added complexity.
"""

from __future__ import annotations

import numpy as np
from scipy import signal as sp_signal
from scipy.stats import kurtosis, skew


def extract_features(window: np.ndarray, fs: int) -> np.ndarray:
    mean = np.mean(window)
    std = np.std(window)
    minimum = np.min(window)
    maximum = np.max(window)
    # skew/kurtosis divide by std**3 / std**4; a flat (zero-variance) window
    # -- e.g. the electrode-settling artifact found in c01 -- gives 0/0 = NaN.
    if std < 1e-8:
        sk, kurt = 0.0, 0.0
    else:
        sk = skew(window)
        kurt = kurtosis(window)
    zero_crossing_rate = np.mean(np.diff(np.sign(window)) != 0)

    # R-peak proxy: count threshold crossings above 3 std above the median,
    # a crude stand-in for heart-rate/R-R interval information without a
    # full QRS detector.
    peaks, _ = sp_signal.find_peaks(window, height=np.median(window) + 3 * std, distance=int(0.3 * fs))
    peak_count = len(peaks)
    if len(peaks) > 1:
        rr_intervals = np.diff(peaks) / fs
        rr_mean = np.mean(rr_intervals)
        rr_std = np.std(rr_intervals)
    else:
        rr_mean, rr_std = 0.0, 0.0

    return np.array(
        [mean, std, minimum, maximum, sk, kurt, zero_crossing_rate, peak_count, rr_mean, rr_std],
        dtype=np.float64,
    )


FEATURE_NAMES = [
    "mean", "std", "min", "max", "skew", "kurtosis",
    "zero_crossing_rate", "peak_count", "rr_mean", "rr_std",
]
