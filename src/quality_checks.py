"""Phase 2: dataset quality checks for the Apnea-ECG database.

For every base record (a01-a20, b01-b05, c01-c10, x01-x35) this verifies
sampling rate, signal length, missing values, clipping, and per-minute
annotation alignment, then reports class counts / imbalance and documents
the excluded r/er duplicate-session records.

Writes reports/data_quality_summary.csv and reports/data_quality_report.md.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb

from src.data_loader import all_base_records, learning_records, load_config, official_test_records

EXPECTED_FS = 100
MISSING_SENTINEL = -32768  # WFDB format-16 "missing sample" code


def digital_extremes(adc_res: int) -> tuple[int, int]:
    """Min/max representable digital codes for a signed ADC of adc_res bits."""
    half = 2 ** (adc_res - 1)
    return -half, half - 1


def check_record(record_id: str, cfg: dict) -> dict:
    raw_dir = Path(cfg["data"]["raw_dir"])
    rec = wfdb.rdrecord(str(raw_dir / record_id), physical=False)
    ann = wfdb.rdann(str(raw_dir / record_id), extension="apn")

    fs = rec.fs
    n_samples = rec.sig_len
    duration_min = n_samples / fs / 60
    d_signal = rec.d_signal[:, 0]
    adc_res = rec.adc_res[0]

    lo, hi = digital_extremes(adc_res)
    n_missing = int(np.sum(d_signal == MISSING_SENTINEL))
    n_clipped = int(np.sum((d_signal <= lo) | (d_signal >= hi)))

    # Longest run of an identical consecutive value (flatline artifact check)
    changes = np.diff(d_signal) != 0
    if changes.size:
        run_lengths = np.diff(np.flatnonzero(np.concatenate(([True], changes, [True]))))
        longest_flat_run = int(run_lengths.max())
    else:
        longest_flat_run = n_samples

    symbols = list(ann.symbol)
    n_annotations = len(symbols)
    expected_minutes = int(duration_min)  # floor
    annotation_mismatch = abs(n_annotations - expected_minutes)

    n_apnea = symbols.count("A")
    n_normal = symbols.count("N")
    n_other = n_annotations - n_apnea - n_normal
    minority_frac = min(n_apnea, n_normal) / n_annotations if n_annotations else float("nan")

    return {
        "record_id": record_id,
        "fs": fs,
        "fs_ok": fs == EXPECTED_FS,
        "n_samples": n_samples,
        "duration_min": round(duration_min, 2),
        "n_missing": n_missing,
        "n_clipped": n_clipped,
        "longest_flat_run": longest_flat_run,
        "n_annotations": n_annotations,
        "expected_minutes": expected_minutes,
        "annotation_mismatch": annotation_mismatch,
        "n_apnea": n_apnea,
        "n_normal": n_normal,
        "n_other_symbol": n_other,
        "minority_class_frac": round(minority_frac, 4) if n_annotations else None,
    }


def run(cfg: dict) -> pd.DataFrame:
    rows = [check_record(r, cfg) for r in all_base_records(cfg)]
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, cfg: dict, imbalance_threshold: float = 0.10) -> str:
    learning = set(learning_records(cfg))
    test = set(official_test_records(cfg))

    fs_bad = df[~df["fs_ok"]]
    missing_bad = df[df["n_missing"] > 0]
    clipped_bad = df[df["n_clipped"] > 0]
    align_bad = df[df["annotation_mismatch"] > 1]
    imbalanced = df[df["minority_class_frac"] < imbalance_threshold]

    n_apnea_total = df["n_apnea"].sum()
    n_normal_total = df["n_normal"].sum()
    n_apnea_learn = df[df["record_id"].isin(learning)]["n_apnea"].sum()
    n_normal_learn = df[df["record_id"].isin(learning)]["n_normal"].sum()
    n_apnea_test = df[df["record_id"].isin(test)]["n_apnea"].sum()
    n_normal_test = df[df["record_id"].isin(test)]["n_normal"].sum()

    lines = []
    lines.append("# Data Quality Report — PhysioNet Apnea-ECG\n")
    lines.append(f"Records checked: {len(df)} (expected 70)\n")

    lines.append("## Sampling rate\n")
    lines.append(
        f"All records expected at {EXPECTED_FS} Hz. "
        f"{'All OK.' if fs_bad.empty else f'{len(fs_bad)} mismatch(es): ' + ', '.join(fs_bad['record_id'])}\n"
    )

    lines.append("## Missing values\n")
    if missing_bad.empty:
        lines.append("No missing-sample sentinel values found in any record.\n")
    else:
        lines.append("Records with missing-sample sentinels:\n")
        for _, r in missing_bad.iterrows():
            lines.append(f"- {r['record_id']}: {r['n_missing']} missing samples\n")

    lines.append("## Clipping / saturation\n")
    if clipped_bad.empty:
        lines.append("No samples at the digital ADC extremes in any record.\n")
    else:
        lines.append("Records with clipped/saturated samples:\n")
        for _, r in clipped_bad.iterrows():
            lines.append(f"- {r['record_id']}: {r['n_clipped']} clipped samples\n")

    lines.append("## Annotation alignment\n")
    lines.append(
        "Compares number of per-minute A/N annotations to floor(duration / 60s).\n"
    )
    if align_bad.empty:
        lines.append("All records match within 1 minute.\n")
    else:
        lines.append("Records with mismatch > 1 minute:\n")
        for _, r in align_bad.iterrows():
            lines.append(
                f"- {r['record_id']}: {r['n_annotations']} annotations vs "
                f"{r['expected_minutes']} expected minutes\n"
            )

    lines.append("## Class balance\n")
    lines.append(f"Overall: {n_apnea_total} apnea / {n_normal_total} normal minutes "
                  f"({n_apnea_total / (n_apnea_total + n_normal_total):.1%} apnea)\n")
    lines.append(f"Learning set: {n_apnea_learn} apnea / {n_normal_learn} normal "
                  f"({n_apnea_learn / (n_apnea_learn + n_normal_learn):.1%} apnea)\n")
    lines.append(f"Test set: {n_apnea_test} apnea / {n_normal_test} normal "
                  f"({n_apnea_test / (n_apnea_test + n_normal_test):.1%} apnea)\n")

    lines.append(f"\nRecords with minority class < {imbalance_threshold:.0%} "
                  f"(highly imbalanced):\n")
    if imbalanced.empty:
        lines.append("None.\n")
    else:
        for _, r in imbalanced.iterrows():
            lines.append(
                f"- {r['record_id']}: {r['n_apnea']} apnea / {r['n_normal']} normal "
                f"(minority frac {r['minority_class_frac']:.1%})\n"
            )

    lines.append("\n## Excluded duplicate-session records\n")
    lines.append(
        "The following are **not** independent ECG records — they carry "
        "respiration/SpO2 channels for the *same* recording sessions as "
        "a01-a04, b01, c01-c03. `a01er` etc. even reference `a01.dat` "
        "directly rather than owning separate signal data. They must never "
        "be used as additional training/validation/test samples, and must "
        "never appear in a split alongside their base record:\n\n"
    )
    for rec in cfg["data"]["excluded_related_records"]:
        lines.append(f"- {rec}\n")

    return "".join(lines)


if __name__ == "__main__":
    cfg = load_config()
    df = run(cfg)

    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    df.to_csv(reports_dir / "data_quality_summary.csv", index=False)

    report_md = summarize(df, cfg)
    (reports_dir / "data_quality_report.md").write_text(report_md)

    print(report_md)
    print(f"\nWrote reports/data_quality_summary.csv and reports/data_quality_report.md")
