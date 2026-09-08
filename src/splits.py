"""Record-level train/val/test split.

Validation is carved only out of the learning set (record-level, so no
minute-window ever appears in two splits). Split is stratified by each
record's apnea-minute fraction (quartile buckets) so validation isn't
accidentally dominated by the single-class records Phase 2 flagged.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_loader import learning_records, load_config, official_test_records

PROJECT_ROOT = Path(__file__).resolve().parent.parent
QUALITY_REPORT_CSV = PROJECT_ROOT / "reports" / "data_quality_summary.csv"
DEFAULT_SPLITS_PATH = PROJECT_ROOT / "data" / "splits" / "record_splits.csv"


def _apnea_fraction_by_record(cfg: dict) -> dict[str, float]:
    if not QUALITY_REPORT_CSV.exists():
        raise FileNotFoundError(
            f"{QUALITY_REPORT_CSV} not found — run `python -m src.quality_checks` first."
        )
    df = pd.read_csv(QUALITY_REPORT_CSV)
    df = df.set_index("record_id")
    total = df["n_apnea"] + df["n_normal"]
    return (df["n_apnea"] / total).to_dict()


def make_record_splits(cfg: dict, val_fraction: float = 0.2) -> dict[str, str]:
    rng = np.random.default_rng(cfg["seed"])
    learn = learning_records(cfg)
    frac_by_record = _apnea_fraction_by_record(cfg)

    fractions = np.array([frac_by_record[r] for r in learn])
    quartile_edges = np.quantile(fractions, [0.25, 0.5, 0.75])
    buckets = np.digitize(fractions, quartile_edges)

    splits: dict[str, str] = {}
    for bucket in np.unique(buckets):
        bucket_records = [r for r, b in zip(learn, buckets) if b == bucket]
        rng.shuffle(bucket_records)
        n_val = max(1, round(len(bucket_records) * val_fraction))
        for r in bucket_records[:n_val]:
            splits[r] = "val"
        for r in bucket_records[n_val:]:
            splits[r] = "train"

    for r in official_test_records(cfg):
        splits[r] = "test"

    return splits


def save_splits(splits: dict[str, str], path: Path = DEFAULT_SPLITS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["record_id", "split"])
        for record_id in sorted(splits):
            writer.writerow([record_id, splits[record_id]])


if __name__ == "__main__":
    cfg = load_config()
    splits = make_record_splits(cfg)
    save_splits(splits)
    n_train = sum(1 for v in splits.values() if v == "train")
    n_val = sum(1 for v in splits.values() if v == "val")
    n_test = sum(1 for v in splits.values() if v == "test")
    print(f"train={n_train} val={n_val} test={n_test} -> data/splits/record_splits.csv")
