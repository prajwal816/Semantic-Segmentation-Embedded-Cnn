from __future__ import annotations

import numpy as np
import torch


def mean_iou(pred: torch.Tensor, target: torch.Tensor, num_classes: int, ignore_index: int = -1) -> float:
    """
    pred/target: (N,H,W) int64 labels
    """
    pred = pred.view(-1)
    target = target.view(-1)
    mask = target != ignore_index
    pred = pred[mask]
    target = target[mask]
    ious = []
    for c in range(num_classes):
        p = pred == c
        t = target == c
        inter = (p & t).sum().item()
        union = (p | t).sum().item()
        if union == 0:
            continue
        ious.append(inter / union)
    if not ious:
        return 0.0
    return float(np.mean(ious))


def pixel_accuracy(pred: torch.Tensor, target: torch.Tensor, ignore_index: int = -1) -> float:
    pred = pred.view(-1)
    target = target.view(-1)
    mask = target != ignore_index
    correct = (pred[mask] == target[mask]).sum().item()
    total = mask.sum().item()
    return float(correct / max(total, 1))
