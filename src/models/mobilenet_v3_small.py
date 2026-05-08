"""MobileNetV3-Small student model for EuroSAT.

Wraps torchvision.models.mobilenet_v3_small and replaces the final linear
layer of the classifier head for the target number of classes.
"""

import torch.nn as nn
import torchvision.models as tv


class MobileNetV3Small(nn.Module):
    def __init__(self, num_classes: int = 10, pretrained: bool = False):
        super().__init__()
        weights = tv.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        self.backbone = tv.mobilenet_v3_small(weights=weights)
        in_features = self.backbone.classifier[-1].in_features
        self.backbone.classifier[-1] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)
