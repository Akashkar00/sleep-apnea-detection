"""Phase 4: visual sanity check — render representative A/N examples plus a
known artifact case (heavy clipping) side by side for human inspection.

This is diagnostic tooling only; the rendered figure is for a person to look
at, not a training input (unlike the model-facing images in data/images/).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.data_loader import load_config
from src.representations import generate_all_representations

OUT_PATH = Path("reports/figures/sample_inspection.png")

# (record_id, index into that record's accepted-window array, human-readable note)
EXAMPLES = [
    ("a01", 100, "A-heavy record (a01), apnea minute"),
    ("c01", 100, "N-only record (c01), normal minute"),
    ("x30", 0, "heavy-clipping record (x30), first accepted window"),
]

INT_TO_LABEL = {0: "N", 1: "A"}


def main() -> None:
    cfg = load_config()
    fs = cfg["data"]["sampling_rate_hz"]

    fig, axes = plt.subplots(len(EXAMPLES), 4, figsize=(14, 3.5 * len(EXAMPLES)))

    for row, (record_id, idx, note) in enumerate(EXAMPLES):
        d = np.load(f"data/processed/{record_id}.npz")
        signal = d["X"][idx]
        label = INT_TO_LABEL[int(d["y"][idx])]
        imgs = generate_all_representations(signal, fs, cfg)

        axes[row, 0].plot(signal, linewidth=0.5)
        axes[row, 0].set_title(f"{record_id} #{idx} [{label}]\nraw normalized signal")
        axes[row, 0].set_xticks([])

        for col, rep in enumerate(["waveform", "stft", "cwt"], start=1):
            axes[row, col].imshow(imgs[rep], cmap="gray", aspect="auto")
            axes[row, col].set_title(rep)
            axes[row, col].axis("off")

        axes[row, 0].set_ylabel(note, fontsize=8)

    plt.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_PATH, dpi=120)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
