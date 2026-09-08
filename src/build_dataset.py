"""Phase 3 driver: preprocess every base record and extract labeled windows.

Writes:
- data/processed/{record_id}.npz  (accepted windows: X, y, minute_index)
- data/splits/window_manifest.csv (every window, including rejected ones)
- reports/preprocessing_report.md
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_loader import all_base_records, load_config, load_record
from src.preprocessing import extract_windows, preprocess_signal
from src.splits import make_record_splits, save_splits

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MANIFEST_PATH = PROJECT_ROOT / "data" / "splits" / "window_manifest.csv"
REPORT_PATH = PROJECT_ROOT / "reports" / "preprocessing_report.md"

LABEL_TO_INT = {"N": 0, "A": 1}


def process_record(record_id: str, cfg: dict, manifest_rows: list[dict]) -> None:
    rec = load_record(record_id, cfg)
    clean_signal = preprocess_signal(rec.signal, rec.fs, cfg)
    windows = extract_windows(rec, cfg, clean_signal)

    accepted = [w for w in windows if w.status == "ok"]
    for w in windows:
        manifest_rows.append(
            {
                "record_id": w.record_id,
                "minute_index": w.minute_index,
                "label": w.label,
                "status": w.status,
                "reason": w.reason,
                "clipped_fraction": round(w.clipped_fraction, 4),
            }
        )

    if accepted:
        X = np.stack([w.samples for w in accepted])
        y = np.array([LABEL_TO_INT[w.label] for w in accepted], dtype=np.int64)
        minute_index = np.array([w.minute_index for w in accepted], dtype=np.int64)
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            PROCESSED_DIR / f"{record_id}.npz", X=X, y=y, minute_index=minute_index
        )


def write_report(manifest_df: pd.DataFrame, splits: dict[str, str]) -> None:
    manifest_df = manifest_df.copy()
    manifest_df["split"] = manifest_df["record_id"].map(splits)

    lines = ["# Preprocessing Report (Phase 3)\n"]
    lines.append(f"Total windows evaluated: {len(manifest_df)}\n")

    lines.append("\n## Rejection reasons\n")
    rejected = manifest_df[manifest_df["status"] == "rejected"]
    if rejected.empty:
        lines.append("None.\n")
    else:
        for reason, group in rejected.groupby(rejected["reason"].str.split(":").str[0]):
            lines.append(f"- {reason}: {len(group)}\n")

    lines.append("\n## Accepted windows by split and class\n")
    accepted = manifest_df[manifest_df["status"] == "ok"]
    counts = accepted.groupby(["split", "label"]).size().unstack(fill_value=0)
    lines.append("| split | " + " | ".join(str(c) for c in counts.columns) + " |\n")
    lines.append("|---" * (len(counts.columns) + 1) + "|\n")
    for split_name, row in counts.iterrows():
        lines.append(f"| {split_name} | " + " | ".join(str(v) for v in row) + " |\n")

    lines.append("\n## Records with no accepted windows\n")
    accepted_records = set(accepted["record_id"])
    all_records = set(manifest_df["record_id"])
    empty_records = sorted(all_records - accepted_records)
    lines.append(", ".join(empty_records) if empty_records else "None.")
    lines.append("\n")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("".join(lines))


if __name__ == "__main__":
    cfg = load_config()

    splits = make_record_splits(cfg)
    save_splits(splits)

    manifest_rows: list[dict] = []
    records = all_base_records(cfg)
    for idx, record_id in enumerate(records, 1):
        print(f"[{idx}/{len(records)}] processing {record_id}...")
        process_record(record_id, cfg, manifest_rows)

    manifest_df = pd.DataFrame(manifest_rows)
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    manifest_df.to_csv(MANIFEST_PATH, index=False)

    write_report(manifest_df, splits)
    print(f"\nWrote {MANIFEST_PATH}, {REPORT_PATH}, and data/processed/*.npz")
