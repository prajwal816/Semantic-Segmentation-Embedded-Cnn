from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F


class SegmentationAugment:
    def __init__(
        self,
        hflip_prob: float = 0.5,
        vflip_prob: float = 0.2,
        brightness: float = 0.15,
        contrast: float = 0.15,
    ) -> None:
        self.hflip_prob = hflip_prob
        self.vflip_prob = vflip_prob
        self.brightness = brightness
        self.contrast = contrast

    def __call__(self, image: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if torch.rand(1).item() < self.hflip_prob:
            image = torch.flip(image, dims=[2])
            mask = torch.flip(mask, dims=[1])
        if torch.rand(1).item() < self.vflip_prob:
            image = torch.flip(image, dims=[1])
            mask = torch.flip(mask, dims=[0])

        b = (torch.rand(1).item() * 2 - 1) * self.brightness
        image = (image + b).clamp(0.0, 1.0)
        c = 1.0 + (torch.rand(1).item() * 2 - 1) * self.contrast
        mean = image.mean(dim=(1, 2), keepdim=True)
        image = ((image - mean) * c + mean).clamp(0.0, 1.0)
        return image, mask


def resize_pair(
    image: torch.Tensor,
    mask: torch.Tensor,
    size: Tuple[int, int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """image: CHW, mask: HW -> resized to (H, W) = size."""
    h, w = size
    image = F.interpolate(image.unsqueeze(0), size=(h, w), mode="bilinear", align_corners=False).squeeze(0)
    mask = F.interpolate(mask.float().unsqueeze(0).unsqueeze(0), size=(h, w), mode="nearest").squeeze(0).squeeze(0).long()
    return image, mask
