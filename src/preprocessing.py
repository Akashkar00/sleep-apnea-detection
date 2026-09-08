"""Phase 3: ECG signal preprocessing and windowing.

Pipeline per record:
1. Remove baseline wander (cascaded median-filter detrending).
2. Bandpass filter to retain QRS/rhythm content.
3. Robust per-record normalization (median/IQR), computed after filtering.
4. Extract one-minute center windows with configurable neighboring context.
   Windows that fall outside the signal, carry a non-A/N label, or are
   dominated by ADC saturation are rejected (not silently modified) and the
   reason is recorded rather than discarded.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, filtfilt, medfilt

from src.data_loader import RecordData

HEAVY_CLIPPING_FRACTION = 0.05  # reject a window if more than 5% of samples are saturated


def remove_baseline_wander(signal: np.ndarray, fs: int) -> np.ndarray:
    """Cascaded median-filter baseline removal (~200ms then ~600ms windows)."""
    def odd(n: int) -> int:
        return n if n % 2 == 1 else n + 1

    k1 = odd(max(1, round(0.2 * fs)))
    k2 = odd(max(1, round(0.6 * fs)))
    baseline = medfilt(medfilt(signal, kernel_size=k1), kernel_size=k2)
    return signal - baseline


def bandpass_filter(signal: np.ndarray, fs: int, low_hz: float, high_hz: float, order: int = 4) -> np.ndarray:
    nyquist = fs / 2
    b, a = butter(order, [low_hz / nyquist, high_hz / nyquist], btype="band")
    return filtfilt(b, a, signal)


def clip_extreme_artifacts(signal: np.ndarray, iqr_multiplier: float) -> np.ndarray:
    """Clip amplitude outliers (movement artifacts, filter ringing around
    saturation) to median +/- iqr_multiplier * IQR before normalization.

    Percentile-based clipping doesn't work here: normal R-peaks already
    recur in several percent of samples, so a percentile clip would cut into
    real QRS morphology. A generous IQR multiplier only touches genuinely
    extreme spikes (observed up to ~370 IQR in artifact-heavy records) while
    leaving normal R-peaks (typically within ~20 IQR) untouched.
    """
    median = np.median(signal)
    q75, q25 = np.percentile(signal, [75, 25])
    iqr = q75 - q25
    return np.clip(signal, median - iqr_multiplier * iqr, median + iqr_multiplier * iqr)


def robust_normalize(signal: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    median = np.median(signal)
    q75, q25 = np.percentile(signal, [75, 25])
    iqr = q75 - q25
    return (signal - median) / (iqr + eps)


def preprocess_signal(signal: np.ndarray, fs: int, cfg: dict) -> np.ndarray:
    detrended = remove_baseline_wander(signal, fs)
    filtered = bandpass_filter(
        detrended,
        fs,
        cfg["preprocessing"]["bandpass_low_hz"],
        cfg["preprocessing"]["bandpass_high_hz"],
    )
    clipped = clip_extreme_artifacts(filtered, cfg["preprocessing"]["clip_iqr_multiplier"])
    return robust_normalize(clipped)


@dataclass
class Window:
    record_id: str
    minute_index: int
    label: str | None
    status: str  # "ok" or "rejected"
    reason: str | None
    samples: np.ndarray | None = None
    clipped_fraction: float = 0.0


def digital_extremes(adc_res: int) -> tuple[int, int]:
    half = 2 ** (adc_res - 1)
    return -half, half - 1


def extract_windows(rec: RecordData, cfg: dict, clean_signal: np.ndarray) -> list[Window]:
    fs = rec.fs
    window_minutes = cfg["data"]["window_minutes"]
    context = cfg["data"]["context_minutes"]
    window_len = int(window_minutes * 60 * fs)
    lo_code, hi_code = digital_extremes(rec.adc_res)

    windows: list[Window] = []
    for i, (sample_i, symbol) in enumerate(zip(rec.ann_samples, rec.annotations)):
        if symbol not in ("A", "N"):
            windows.append(Window(rec.record_id, i, None, "rejected", f"invalid_label:{symbol}"))
            continue

        start = sample_i + min(context) * 60 * fs
        end = sample_i + max(context) * 60 * fs + window_len
        if start < 0 or end > len(clean_signal):
            windows.append(Window(rec.record_id, i, symbol, "rejected", "context_out_of_bounds"))
            continue

        d_segment = rec.d_signal[start:end]
        clipped_fraction = float(np.mean((d_segment <= lo_code) | (d_segment >= hi_code)))
        if clipped_fraction > HEAVY_CLIPPING_FRACTION:
            windows.append(
                Window(rec.record_id, i, symbol, "rejected", "heavy_clipping", clipped_fraction=clipped_fraction)
            )
            continue

        windows.append(
            Window(
                rec.record_id,
                i,
                symbol,
                "ok",
                None,
                samples=clean_signal[start:end].astype(np.float32),
                clipped_fraction=clipped_fraction,
            )
        )
    return windows
