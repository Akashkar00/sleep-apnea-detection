"""Phase 4: deterministic ECG-to-image representations.

Each function maps a 1D preprocessed ECG window to a fixed-size uint8 image
with no axes, labels, legends, or borders (nothing a model could use as a
shortcut instead of the actual signal). Intensity normalization uses fixed,
config-driven bounds rather than per-image min/max, so images are comparable
across records and splits.
"""

from __future__ import annotations

import numpy as np
import pywt
from PIL import Image, ImageDraw
from scipy.signal import stft


def _resize(array: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resize a uint8 2D array to (height, width) via bilinear interpolation."""
    height, width = size
    img = Image.fromarray(array)
    img = img.resize((width, height), resample=Image.BILINEAR)
    return np.array(img)


def _normalize_to_uint8(values: np.ndarray, clip_low: float, clip_high: float) -> np.ndarray:
    clipped = np.clip(values, clip_low, clip_high)
    scaled = (clipped - clip_low) / (clip_high - clip_low)
    return (scaled * 255).astype(np.uint8)


def render_waveform(signal: np.ndarray, size: tuple[int, int], amplitude_clip: float) -> np.ndarray:
    """Rasterize the waveform as a line plot with fixed time/amplitude axes."""
    height, width = size
    img = Image.new("L", (width, height), color=0)
    draw = ImageDraw.Draw(img)

    x_coords = np.linspace(0, width - 1, num=len(signal))
    clipped = np.clip(signal, -amplitude_clip, amplitude_clip)
    # amplitude_clip maps to y=0 (top), -amplitude_clip maps to y=height-1 (bottom)
    y_coords = (1 - (clipped + amplitude_clip) / (2 * amplitude_clip)) * (height - 1)

    points = list(zip(x_coords.tolist(), y_coords.tolist()))
    draw.line(points, fill=255, width=1)
    return np.array(img)


def compute_stft_image(
    signal: np.ndarray,
    fs: int,
    low_hz: float,
    high_hz: float,
    size: tuple[int, int],
    nperseg: int,
    db_clip: tuple[float, float],
) -> np.ndarray:
    freqs, _, Zxx = stft(signal, fs=fs, nperseg=nperseg)
    magnitude_db = 20 * np.log10(np.abs(Zxx) + 1e-10)

    band = (freqs >= low_hz) & (freqs <= high_hz)
    cropped = magnitude_db[band, :]
    cropped = np.flipud(cropped)  # low frequency at bottom, like a spectrogram convention

    image_u8 = _normalize_to_uint8(cropped, db_clip[0], db_clip[1])
    return _resize(image_u8, size)


def compute_cwt_image(
    signal: np.ndarray,
    fs: int,
    low_hz: float,
    high_hz: float,
    size: tuple[int, int],
    n_scales: int,
    wavelet: str,
    db_clip: tuple[float, float],
) -> np.ndarray:
    freqs_target = np.linspace(low_hz, high_hz, n_scales)
    scales = pywt.frequency2scale(wavelet, freqs_target / fs)
    coeffs, _ = pywt.cwt(signal, scales, wavelet, sampling_period=1 / fs)

    magnitude_db = 20 * np.log10(np.abs(coeffs) + 1e-10)
    cropped = np.flipud(magnitude_db)  # low frequency at bottom

    image_u8 = _normalize_to_uint8(cropped, db_clip[0], db_clip[1])
    return _resize(image_u8, size)


def generate_all_representations(signal: np.ndarray, fs: int, cfg: dict) -> dict[str, np.ndarray]:
    img_cfg = cfg["image"]
    size = tuple(img_cfg["size"])
    low_hz = cfg["preprocessing"]["bandpass_low_hz"]
    high_hz = cfg["preprocessing"]["bandpass_high_hz"]

    return {
        "waveform": render_waveform(signal, size, img_cfg["waveform_amplitude_clip"]),
        "stft": compute_stft_image(
            signal, fs, low_hz, high_hz, size, img_cfg["stft_nperseg"], tuple(img_cfg["stft_db_clip"])
        ),
        "cwt": compute_cwt_image(
            signal,
            fs,
            low_hz,
            high_hz,
            size,
            img_cfg["cwt_n_scales"],
            img_cfg["cwt_wavelet"],
            tuple(img_cfg["cwt_db_clip"]),
        ),
    }
