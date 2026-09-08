"""Phase 5 baseline #1: logistic regression on handcrafted signal features.

Trains on data/processed/*.npz (Phase 3 windows, train split), evaluates on
val split, and writes reports/baseline_report.md. This is the floor the CNN
representations (Phase 5 remainder) need to beat.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, recall_score
from sklearn.preprocessing import StandardScaler

from src.data_loader import load_config
from src.datasets import load_record_splits
from src.features import FEATURE_NAMES, extract_features

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORT_PATH = PROJECT_ROOT / "reports" / "baseline_report.md"


def load_split(split: str, record_splits: dict[str, str], fs: int) -> tuple[np.ndarray, np.ndarray]:
    features, labels = [], []
    for record_id, s in record_splits.items():
        if s != split:
            continue
        path = PROCESSED_DIR / f"{record_id}.npz"
        if not path.exists():
            continue
        d = np.load(path)
        for i in range(len(d["y"])):
            features.append(extract_features(d["X"][i], fs))
            labels.append(d["y"][i])
    return np.stack(features), np.array(labels)


if __name__ == "__main__":
    cfg = load_config()
    fs = cfg["data"]["sampling_rate_hz"]
    record_splits = load_record_splits()

    print("Extracting features (train)...")
    X_train, y_train = load_split("train", record_splits, fs)
    print("Extracting features (val)...")
    X_val, y_val = load_split("val", record_splits, fs)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(X_train_scaled, y_train)

    val_proba = clf.predict_proba(X_val_scaled)[:, 1]
    val_pred = (val_proba >= 0.5).astype(int)

    pr_auc = average_precision_score(y_val, val_proba)
    sensitivity = recall_score(y_val, val_pred, pos_label=1)
    specificity = recall_score(y_val, val_pred, pos_label=0)
    balanced_acc = balanced_accuracy_score(y_val, val_pred)

    report = f"""# Baseline Report — Logistic Regression on Handcrafted Features (Phase 5, #1)

Train windows: {len(y_train)}, Val windows: {len(y_val)}

| Metric | Value |
|---|---|
| PR-AUC | {pr_auc:.4f} |
| Sensitivity (apnea recall) | {sensitivity:.4f} |
| Specificity | {specificity:.4f} |
| Balanced accuracy | {balanced_acc:.4f} |

Features: {", ".join(FEATURE_NAMES)}

This is the floor that CNN-on-images baselines need to beat to justify their
added complexity.
"""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report)
    print(report)
