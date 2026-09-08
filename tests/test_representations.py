import numpy as np

from src.representations import (
    compute_cwt_image,
    compute_stft_image,
    generate_all_representations,
    render_waveform,
)

FS = 100
SIZE = (224, 224)


def _synthetic_signal(n_samples=18000):
    t = np.arange(n_samples) / FS
    return np.sin(2 * np.pi * 1.2 * t).astype(np.float32)


def test_render_waveform_shape_and_dtype():
    img = render_waveform(_synthetic_signal(), SIZE, amplitude_clip=5.0)
    assert img.shape == SIZE
    assert img.dtype == np.uint8


def test_render_waveform_deterministic():
    signal = _synthetic_signal()
    img1 = render_waveform(signal, SIZE, amplitude_clip=5.0)
    img2 = render_waveform(signal, SIZE, amplitude_clip=5.0)
    assert np.array_equal(img1, img2)


def test_compute_stft_image_shape_and_range():
    img = compute_stft_image(_synthetic_signal(), FS, 0.5, 40.0, SIZE, nperseg=256, db_clip=(-80, 0))
    assert img.shape == SIZE
    assert img.dtype == np.uint8
    assert img.min() >= 0 and img.max() <= 255


def test_compute_cwt_image_shape_and_range():
    img = compute_cwt_image(
        _synthetic_signal(), FS, 0.5, 40.0, SIZE, n_scales=64, wavelet="morl", db_clip=(-40, 40)
    )
    assert img.shape == SIZE
    assert img.dtype == np.uint8


def test_generate_all_representations_keys():
    cfg = {
        "image": {
            "size": list(SIZE),
            "waveform_amplitude_clip": 5.0,
            "stft_nperseg": 256,
            "stft_db_clip": [-80, 0],
            "cwt_n_scales": 64,
            "cwt_wavelet": "morl",
            "cwt_db_clip": [-40, 40],
        },
        "preprocessing": {"bandpass_low_hz": 0.5, "bandpass_high_hz": 40.0},
    }
    imgs = generate_all_representations(_synthetic_signal(), FS, cfg)
    assert set(imgs.keys()) == {"waveform", "stft", "cwt"}
    for img in imgs.values():
        assert img.shape == SIZE
        assert img.dtype == np.uint8
