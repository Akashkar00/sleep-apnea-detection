"""Phase 4 driver: generate waveform/STFT/CWT images for every accepted window.

Reads data/processed/{record_id}.npz (produced by Phase 3) and writes:
- data/images/{record_id}.npz  (uint8 arrays: waveform, stft, cwt, y, minute_index)
- data/splits/image_manifest.csv (one row per record_id/minute_index/representation)
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from src.data_loader import all_base_records, load_config
from src.representations import generate_all_representations

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
IMAGES_DIR = PROJECT_ROOT / "data" / "images"
MANIFEST_PATH = PROJECT_ROOT / "data" / "splits" / "image_manifest.csv"

INT_TO_LABEL = {0: "N", 1: "A"}
REPRESENTATIONS = ["waveform", "stft", "cwt"]


def process_record(record_id: str, cfg: dict) -> None:
    out_path = IMAGES_DIR / f"{record_id}.npz"
    if out_path.exists():
        return  # already generated (resumable across interrupted runs)

    src_path = PROCESSED_DIR / f"{record_id}.npz"
    if not src_path.exists():
        return  # record had zero accepted windows in Phase 3

    d = np.load(src_path)
    X, y, minute_index = d["X"], d["y"], d["minute_index"]
    fs = cfg["data"]["sampling_rate_hz"]

    out = {rep: [] for rep in REPRESENTATIONS}
    for i in range(len(X)):
        imgs = generate_all_representations(X[i], fs, cfg)
        for rep in REPRESENTATIONS:
            out[rep].append(imgs[rep])

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        waveform=np.stack(out["waveform"]),
        stft=np.stack(out["stft"]),
        cwt=np.stack(out["cwt"]),
        y=y,
        minute_index=minute_index,
    )


def build_manifest(cfg: dict) -> None:
    """Rebuilt from whatever data/images/*.npz files currently exist, so it's
    always consistent even if generation was interrupted and resumed."""
    version = cfg["image"]["preprocessing_version"]
    manifest_rows: list[dict] = []
    for record_id in all_base_records(cfg):
        path = IMAGES_DIR / f"{record_id}.npz"
        if not path.exists():
            continue
        d = np.load(path)
        for i in range(len(d["y"])):
            label = INT_TO_LABEL[int(d["y"][i])]
            for rep in REPRESENTATIONS:
                manifest_rows.append(
                    {
                        "record_id": record_id,
                        "minute_index": int(d["minute_index"][i]),
                        "label": label,
                        "representation": rep,
                        "preprocessing_version": version,
                    }
                )

    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_PATH, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["record_id", "minute_index", "label", "representation", "preprocessing_version"]
        )
        writer.writeheader()
        writer.writerows(manifest_rows)


if __name__ == "__main__":
    cfg = load_config()
    records = all_base_records(cfg)
    for idx, record_id in enumerate(records, 1):
        if (IMAGES_DIR / f"{record_id}.npz").exists():
            print(f"[{idx}/{len(records)}] {record_id} already done, skipping", flush=True)
            continue
        print(f"[{idx}/{len(records)}] generating images for {record_id}...", flush=True)
        process_record(record_id, cfg)

    build_manifest(cfg)
    print(f"\nWrote {MANIFEST_PATH} and data/images/*.npz")
