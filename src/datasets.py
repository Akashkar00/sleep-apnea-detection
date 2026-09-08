"""Phase 5: PyTorch dataset over the generated images, with record-aware splits.

Loads directly from data/images/{record_id}.npz (Phase 4 output) rather than
individual files, so an epoch only pays the cost of a handful of npz reads
rather than tens of thousands of file opens.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMAGES_DIR = PROJECT_ROOT / "data" / "images"
SPLITS_PATH = PROJECT_ROOT / "data" / "splits" / "record_splits.csv"


def load_record_splits(path: Path = SPLITS_PATH) -> dict[str, str]:
    with open(path) as f:
        return {row["record_id"]: row["split"] for row in csv.DictReader(f)}


class ApneaImageDataset(Dataset):
    """One representation ('waveform', 'stft', or 'cwt') for a given split."""

    def __init__(self, split: str, representation: str, images_dir: Path = IMAGES_DIR):
        assert representation in ("waveform", "stft", "cwt")
        self.representation = representation
        record_splits = load_record_splits()
        record_ids = sorted(r for r, s in record_splits.items() if s == split)

        # Index maps a flat dataset index -> (record_id, position within that record's array)
        self._index: list[tuple[str, int]] = []
        self._images: dict[str, np.ndarray] = {}
        self._labels: dict[str, np.ndarray] = {}
        for record_id in record_ids:
            path = images_dir / f"{record_id}.npz"
            if not path.exists():
                continue
            # Materialize both arrays now. Indexing an open NpzFile decompresses
            # the whole array on every access, so holding the lazy handle would
            # re-inflate ~25 MB per __getitem__ and starve the GPU.
            with np.load(path) as d:
                self._images[record_id] = d[representation]
                self._labels[record_id] = d["y"]
            self._index.extend((record_id, i) for i in range(len(self._labels[record_id])))

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        record_id, i = self._index[idx]
        image = self._images[record_id][i].astype(np.float32) / 255.0
        label = float(self._labels[record_id][i])
        return torch.from_numpy(image).unsqueeze(0), torch.tensor(label, dtype=torch.float32)
