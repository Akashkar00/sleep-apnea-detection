import numpy as np

from src.features import FEATURE_NAMES, extract_features


def test_extract_features_shape_matches_names():
    signal = np.random.default_rng(0).normal(size=18000)
    features = extract_features(signal, fs=100)
    assert features.shape == (len(FEATURE_NAMES),)


def test_extract_features_finite_on_flat_signal():
    # A perfectly flat signal (like the c01-minute-1 artifact we found) must
    # not produce NaN/inf features that would break sklearn training.
    signal = np.zeros(18000)
    features = extract_features(signal, fs=100)
    assert np.all(np.isfinite(features))


def test_extract_features_peak_count_reasonable_for_synthetic_beats():
    fs = 100
    t = np.arange(18000) / fs
    # ~1 Hz spikes to mimic a simple heart rate
    signal = np.zeros_like(t)
    beat_times = np.arange(0, 180, 1.0)
    for bt in beat_times:
        idx = int(bt * fs)
        if idx < len(signal):
            signal[idx] = 10.0
    features = extract_features(signal, fs)
    peak_count = features[FEATURE_NAMES.index("peak_count")]
    assert 150 <= peak_count <= 190  # roughly one peak per second over 180s
