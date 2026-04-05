from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def cross_entropy_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, target, ignore_index=-1)


def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Soft multi-class Dice loss on one-hot targets derived from integer mask.
    """
    probs = F.softmax(logits, dim=1)
    num_classes = logits.shape[1]
    target_oh = F.one_hot(target.clamp(min=0), num_classes=num_classes).permute(0, 3, 1, 2).float()
    dims = (0, 2, 3)
    intersection = (probs * target_oh).sum(dims)
    denom = probs.pow(2).sum(dims) + target_oh.pow(2).sum(dims)
    dice = (2 * intersection + eps) / (denom + eps)
    return 1.0 - dice.mean()


class CombinedSegmentationLoss(nn.Module):
    def __init__(self, ce_weight: float = 0.5, dice_weight: float = 0.5) -> None:
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = cross_entropy_loss(logits, target)
        di = dice_loss(logits, target)
        return self.ce_weight * ce + self.dice_weight * di
