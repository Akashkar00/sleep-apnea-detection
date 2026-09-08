"""Phase 5: vision model baselines, increasing in complexity.

All models take a single-channel 224x224 image and output one logit
(apnea probability via sigmoid). Multi-channel/fusion models are deferred
until the single-representation comparison (this phase) identifies which
representation(s) actually carry signal.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as tv_models


class SmallCNN(nn.Module):
    """A small custom 2D CNN trained from scratch — the Phase 5 baseline
    every larger model must beat before it's worth using."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 224 -> 112
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 112 -> 56
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 56 -> 28
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),  # -> (64, 1, 1)
        )
        self.classifier = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x).squeeze(1)


class EfficientNetB0Transfer(nn.Module):
    """ImageNet-pretrained EfficientNet-B0 adapted for 1-channel input and binary
    output. Chosen over a plain ResNet for its depthwise-separable inverted
    residuals and squeeze-excitation blocks, which give a far better
    accuracy-per-FLOP trade-off on a single consumer GPU. Backbone frozen by
    default — call unfreeze_backbone() to fine-tune once the frozen-backbone
    head has converged, per the plan's guidance to unfreeze cautiously with a
    smaller learning rate."""

    def __init__(self, freeze_backbone: bool = True):
        super().__init__()
        self.backbone = tv_models.efficientnet_b0(
            weights=tv_models.EfficientNet_B0_Weights.DEFAULT
        )

        # Replace the stem conv to accept 1 channel, averaging the pretrained
        # 3-channel weights rather than discarding the learned filters.
        old_conv = self.backbone.features[0][0]
        new_conv = nn.Conv2d(1, old_conv.out_channels, kernel_size=old_conv.kernel_size,
                              stride=old_conv.stride, padding=old_conv.padding, bias=False)
        with torch.no_grad():
            new_conv.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
        self.backbone.features[0][0] = new_conv

        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier[1] = nn.Linear(in_features, 1)

        if freeze_backbone:
            for name, param in self.backbone.named_parameters():
                if not name.startswith("classifier."):
                    param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x).squeeze(1)


MODEL_REGISTRY = {
    "small_cnn": SmallCNN,
    "efficientnet_b0": EfficientNetB0Transfer,
}
