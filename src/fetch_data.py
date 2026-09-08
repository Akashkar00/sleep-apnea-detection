"""Idempotent fetch/verify of the PhysioNet Apnea-ECG database.

Downloads only whatever is missing from ``data/raw`` and always re-verifies
the full expected file set against ``config.yaml``.
"""

from __future__ import annotations

from pathlib import Path

import wfdb

from src.data_loader import all_base_records, load_config

REQUIRED_EXTENSIONS = ["dat", "hea", "apn", "qrs"]


def fetch(cfg: dict) -> None:
    raw_dir = Path(cfg["data"]["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)

    missing_any = False
    for rec in all_base_records(cfg):
        for ext in REQUIRED_EXTENSIONS:
            if not (raw_dir / f"{rec}.{ext}").exists():
                missing_any = True
                break

    if missing_any:
        print("Some files missing; downloading apnea-ecg database...")
        wfdb.dl_database("apnea-ecg", dl_dir=str(raw_dir))
    else:
        print(f"All required files already present in {raw_dir}, skipping download.")


def verify(cfg: dict) -> list[str]:
    """Return a list of problems found (empty list = all good)."""
    raw_dir = Path(cfg["data"]["raw_dir"])
    problems = []
    for rec in all_base_records(cfg):
        for ext in REQUIRED_EXTENSIONS:
            p = raw_dir / f"{rec}.{ext}"
            if not p.exists():
                problems.append(f"missing {p}")
            elif p.stat().st_size == 0:
                problems.append(f"empty {p}")
    return problems


if __name__ == "__main__":
    cfg = load_config()
    fetch(cfg)
    problems = verify(cfg)
    if problems:
        print(f"\n{len(problems)} problem(s) found:")
        for p in problems:
            print(f"  - {p}")
        raise SystemExit(1)
    print(f"\nAll {len(all_base_records(cfg))} base records verified OK.")
