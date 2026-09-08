import torch

from src.models import EfficientNetB0Transfer, SmallCNN


def test_small_cnn_forward_shape():
    model = SmallCNN()
    x = torch.rand(4, 1, 224, 224)
    out = model(x)
    assert out.shape == (4,)


def test_small_cnn_produces_finite_logits():
    model = SmallCNN()
    x = torch.rand(2, 1, 224, 224)
    out = model(x)
    assert torch.isfinite(out).all()


def test_efficientnet_b0_transfer_forward_shape():
    model = EfficientNetB0Transfer(freeze_backbone=True)
    x = torch.rand(2, 1, 224, 224)
    out = model(x)
    assert out.shape == (2,)


def test_efficientnet_b0_transfer_backbone_frozen_by_default():
    model = EfficientNetB0Transfer(freeze_backbone=True)
    frozen = [p for n, p in model.backbone.named_parameters() if not n.startswith("classifier.")]
    assert all(not p.requires_grad for p in frozen)
    assert all(p.requires_grad for p in model.backbone.classifier.parameters())


def test_efficientnet_b0_transfer_unfreeze():
    model = EfficientNetB0Transfer(freeze_backbone=True)
    model.unfreeze_backbone()
    assert all(p.requires_grad for p in model.backbone.parameters())
