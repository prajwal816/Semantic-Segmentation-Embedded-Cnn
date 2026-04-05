import torch

from src.python.evaluation.metrics import mean_iou, pixel_accuracy


def test_perfect_miou():
    pred = torch.zeros(1, 16, 16, dtype=torch.long)
    target = torch.zeros(1, 16, 16, dtype=torch.long)
    pred[:, :, 8:] = 1
    target[:, :, 8:] = 1
    m = mean_iou(pred, target, num_classes=2)
    assert m > 0.99
