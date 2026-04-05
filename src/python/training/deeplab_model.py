from __future__ import annotations

import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50


def build_deeplabv3(num_classes: int, pretrained_backbone: bool = False) -> nn.Module:
    """
    DeepLabV3 with ResNet-50 backbone (torchvision). Suitable for ONNX export at fixed resolution.
    """
    weights_backbone = None
    if pretrained_backbone:
        from torchvision.models import ResNet50_Weights

        weights_backbone = ResNet50_Weights.DEFAULT
    model = deeplabv3_resnet50(weights=None, weights_backbone=weights_backbone, num_classes=num_classes)
    return model
