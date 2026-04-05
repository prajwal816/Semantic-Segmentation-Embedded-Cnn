import torch

from src.python.utils.losses import CombinedSegmentationLoss, cross_entropy_loss, dice_loss


def test_losses_finite():
    n, c, h, w = 2, 5, 32, 32
    logits = torch.randn(n, c, h, w)
    target = torch.randint(0, c, (n, h, w))
    ce = cross_entropy_loss(logits, target)
    di = dice_loss(logits, target)
    assert torch.isfinite(ce) and torch.isfinite(di)
    comb = CombinedSegmentationLoss()(logits, target)
    assert torch.isfinite(comb)
