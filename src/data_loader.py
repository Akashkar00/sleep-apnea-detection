"""Record enumeration and WFDB loading helpers for the Apnea-ECG dataset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import wfdb
import yaml

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def learning_records(cfg: dict) -> list[str]:
    recs = []
    for prefix, (lo, hi) in cfg["data"]["learning_records"].items():
        recs += [f"{prefix}{i:02d}" for i in range(lo, hi + 1)]
    return recs


def official_test_records(cfg: dict) -> list[str]:
    recs = []
    for prefix, (lo, hi) in cfg["data"]["test_records"].items():
        recs += [f"{prefix}{i:02d}" for i in range(lo, hi + 1)]
    return recs


def all_base_records(cfg: dict) -> list[str]:
    """The 70 ECG records used for classification (excludes r/er variants)."""
    return learning_records(cfg) + official_test_records(cfg)


@dataclass
class RecordData:
    record_id: str
    fs: int
    signal: np.ndarray  # 1D ECG signal (physical units), single lead
    annotations: list[str]  # per-minute 'A'/'N' symbols, in order
    d_signal: np.ndarray  # 1D ECG signal (raw digital ADC codes)
    adc_res: int  # ADC resolution in bits, for clipping/saturation checks
    ann_samples: np.ndarray  # sample index of each annotation, aligned to `annotations`


def load_record(record_id: str, cfg: dict) -> RecordData:
    raw_dir = Path(cfg["data"]["raw_dir"])
    record = wfdb.rdrecord(str(raw_dir / record_id))
    d_record = wfdb.rdrecord(str(raw_dir / record_id), physical=False)
    ann = wfdb.rdann(str(raw_dir / record_id), extension="apn")
    return RecordData(
        record_id=record_id,
        fs=record.fs,
        signal=record.p_signal[:, 0],
        annotations=list(ann.symbol),
        d_signal=d_record.d_signal[:, 0],
        adc_res=d_record.adc_res[0],
        ann_samples=np.asarray(ann.sample),
    )
