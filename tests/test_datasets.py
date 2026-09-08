import pytest
import torch

from src.datasets import ApneaImageDataset, load_record_splits


def test_load_record_splits_has_70_records():
    splits = load_record_splits()
    assert len(splits) == 70
    assert set(splits.values()) <= {"train", "val", "test"}


@pytest.mark.parametrize("representation", ["waveform", "stft", "cwt"])
def test_dataset_item_shape_and_dtype(representation):
    ds = ApneaImageDataset("val", representation)
    assert len(ds) > 0
    image, label = ds[0]
    assert image.shape == (1, 224, 224)
    assert image.dtype == torch.float32
    assert 0.0 <= image.min() and image.max() <= 1.0
    assert label.item() in (0.0, 1.0)


def test_train_val_test_datasets_are_disjoint_in_record_ids():
    splits = load_record_splits()
    train_records = {r for r, s in splits.items() if s == "train"}
    val_records = {r for r, s in splits.items() if s == "val"}
    test_records = {r for r, s in splits.items() if s == "test"}
    assert train_records.isdisjoint(val_records)
    assert train_records.isdisjoint(test_records)
    assert val_records.isdisjoint(test_records)
