import numpy as np
import pytest

from src.data_loader import RecordData
from src.preprocessing import (
    HEAVY_CLIPPING_FRACTION,
    bandpass_filter,
    clip_extreme_artifacts,
    digital_extremes,
    extract_windows,
    remove_baseline_wander,
    robust_normalize,
)

FS = 100


def test_remove_baseline_wander_removes_slow_drift():
    t = np.arange(0, 60, 1 / FS)
    drift = 5 * np.sin(2 * np.pi * 0.05 * t)  # 0.05 Hz baseline wander
    ecg_like = np.sin(2 * np.pi * 1.2 * t)  # ~72 bpm-ish content
    signal = drift + ecg_like
    corrected = remove_baseline_wander(signal, FS)
    # Drift should be substantially attenuated relative to the raw signal's drift-driven range
    assert np.std(corrected) < np.std(signal)


def test_bandpass_filter_attenuates_out_of_band_noise():
    t = np.arange(0, 10, 1 / FS)
    low_freq = np.sin(2 * np.pi * 0.1 * t)  # below 0.5 Hz cutoff
    in_band = np.sin(2 * np.pi * 10 * t)  # well within 0.5-40 Hz
    signal = low_freq + in_band
    filtered = bandpass_filter(signal, FS, 0.5, 40.0)
    # In-band-only signal should pass through with similar energy
    in_band_only_filtered = bandpass_filter(in_band, FS, 0.5, 40.0)
    assert np.corrcoef(filtered, in_band_only_filtered)[0, 1] > 0.9


def test_robust_normalize_centers_and_scales():
    rng = np.random.default_rng(0)
    signal = rng.normal(loc=50, scale=10, size=1000)
    normed = robust_normalize(signal)
    assert abs(np.median(normed)) < 0.05
    q75, q25 = np.percentile(normed, [75, 25])
    assert abs((q75 - q25) - 1.0) < 0.05


def test_clip_extreme_artifacts_preserves_normal_peaks_but_clips_outliers():
    rng = np.random.default_rng(0)
    baseline = rng.normal(loc=0, scale=1.0, size=10000)
    signal = baseline.copy()
    signal[100] = 10_000  # a single, wildly implausible artifact spike
    clipped = clip_extreme_artifacts(signal, iqr_multiplier=20)
    q75, q25 = np.percentile(baseline, [75, 25])
    iqr = q75 - q25
    # The artifact must be pulled in...
    assert clipped[100] < 100
    # ...while ordinary samples are essentially untouched.
    untouched = np.delete(baseline, 100)
    untouched_clipped = np.delete(clipped, 100)
    assert np.allclose(untouched, untouched_clipped)


def test_digital_extremes_12_bit():
    lo, hi = digital_extremes(12)
    assert lo == -2048
    assert hi == 2047


def _make_synthetic_record(n_minutes: int, fs: int = FS) -> RecordData:
    n_samples = n_minutes * 60 * fs
    signal = np.zeros(n_samples)
    d_signal = np.zeros(n_samples, dtype=np.int64)
    ann_samples = np.arange(n_minutes) * 60 * fs
    annotations = ["N"] * n_minutes
    return RecordData(
        record_id="synthetic",
        fs=fs,
        signal=signal,
        annotations=annotations,
        d_signal=d_signal,
        adc_res=12,
        ann_samples=ann_samples,
    )


def _base_cfg():
    return {
        "data": {"window_minutes": 1, "context_minutes": [-1, 0, 1]},
    }


def test_extract_windows_rejects_edge_minutes_without_full_context():
    rec = _make_synthetic_record(n_minutes=5)
    windows = extract_windows(rec, _base_cfg(), rec.signal)
    # minute 0 has no minute -1 of context, last minute has no +1 -> both rejected
    assert windows[0].status == "rejected"
    assert windows[0].reason == "context_out_of_bounds"
    assert windows[-1].status == "rejected"
    assert windows[-1].reason == "context_out_of_bounds"
    # interior minutes should be accepted
    assert all(w.status == "ok" for w in windows[1:-1])


def test_extract_windows_rejects_invalid_label():
    rec = _make_synthetic_record(n_minutes=5)
    rec.annotations[2] = "X"
    windows = extract_windows(rec, _base_cfg(), rec.signal)
    assert windows[2].status == "rejected"
    assert windows[2].reason == "invalid_label:X"


def test_extract_windows_rejects_heavy_clipping():
    rec = _make_synthetic_record(n_minutes=5)
    lo, _ = digital_extremes(rec.adc_res)
    # Saturate more than HEAVY_CLIPPING_FRACTION of minute index 2's window
    window_len = 1 * 60 * FS
    context_span = 3 * window_len  # [-1, 0, 1] minutes
    start = rec.ann_samples[2] - window_len
    n_saturate = int(context_span * (HEAVY_CLIPPING_FRACTION + 0.01))
    rec.d_signal[start : start + n_saturate] = lo
    windows = extract_windows(rec, _base_cfg(), rec.signal)
    assert windows[2].status == "rejected"
    assert windows[2].reason == "heavy_clipping"


def test_extract_windows_accepted_window_has_correct_shape():
    rec = _make_synthetic_record(n_minutes=5)
    windows = extract_windows(rec, _base_cfg(), rec.signal)
    ok = [w for w in windows if w.status == "ok"][0]
    assert ok.samples.shape == (3 * 60 * FS,)  # 3 minutes of context at 100 Hz
