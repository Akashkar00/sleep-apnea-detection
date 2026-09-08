"""Phase 5: train a vision model on one representation.

Usage: uv run python -m src.train --model small_cnn --representation waveform

Plain (unweighted) BCE loss is used deliberately — Phase 7 is where
class-imbalance handling (weighted loss vs. weighted sampling) gets compared
properly. Checkpoints by best validation PR-AUC.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from sklearn.metrics import average_precision_score, balanced_accuracy_score, recall_score
from torch.utils.data import DataLoader

from src.datasets import ApneaImageDataset
from src.models import MODEL_REGISTRY

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = PROJECT_ROOT / "models" / "checkpoints"
METRICS_DIR = PROJECT_ROOT / "reports" / "metrics"


def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    all_labels, all_probs = [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            probs = torch.sigmoid(logits)
            all_labels.append(labels.cpu())
            all_probs.append(probs.cpu())
    y_true = torch.cat(all_labels).numpy()
    y_prob = torch.cat(all_probs).numpy()
    y_pred = (y_prob >= 0.5).astype(int)

    return {
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "sensitivity": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "specificity": float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }


def train(model_name: str, representation: str, epochs: int, batch_size: int, lr: float) -> None:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"device: {device}")

    train_ds = ApneaImageDataset("train", representation)
    val_ds = ApneaImageDataset("val", representation)
    print(f"train windows: {len(train_ds)}, val windows: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = MODEL_REGISTRY[model_name]().to(device)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    criterion = torch.nn.BCEWithLogitsLoss()

    best_pr_auc = -1.0
    history = []
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = CHECKPOINT_DIR / f"{model_name}_{representation}.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        train_loss = running_loss / len(train_ds)
        val_metrics = evaluate(model, val_loader, device)
        elapsed = time.time() - t0
        print(
            f"epoch {epoch}/{epochs} train_loss={train_loss:.4f} "
            f"val_pr_auc={val_metrics['pr_auc']:.4f} val_balanced_acc={val_metrics['balanced_accuracy']:.4f} "
            f"({elapsed:.1f}s)"
        )
        history.append({"epoch": epoch, "train_loss": train_loss, **val_metrics})

        if val_metrics["pr_auc"] > best_pr_auc:
            best_pr_auc = val_metrics["pr_auc"]
            torch.save(model.state_dict(), checkpoint_path)

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    with open(METRICS_DIR / f"{model_name}_{representation}.json", "w") as f:
        json.dump({"model": model_name, "representation": representation, "history": history}, f, indent=2)

    print(f"\nBest val PR-AUC: {best_pr_auc:.4f} -> {checkpoint_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODEL_REGISTRY), default="small_cnn")
    parser.add_argument("--representation", choices=["waveform", "stft", "cwt"], required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    train(args.model, args.representation, args.epochs, args.batch_size, args.lr)
