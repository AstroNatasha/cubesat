"""ResNet18 teacher model for EuroSAT.

Wraps torchvision.models.resnet18 and replaces the final fully-connected
layer for the target number of classes.  Pretrained ImageNet weights can be
loaded via the config; when fine-tuning, all layers are trainable.
"""

import torch.nn as nn
import torchvision.models as tv


class ResNet18Teacher(nn.Module):
    def __init__(self, num_classes: int = 10, pretrained: bool = False):
        super().__init__()
        weights = tv.ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = tv.resnet18(weights=weights)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)
