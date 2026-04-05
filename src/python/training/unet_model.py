from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNet(nn.Module):
    """
    Compact U-Net for embedded segmentation. Export-friendly (fixed spatial size).
    """

    def __init__(self, in_channels: int = 3, num_classes: int = 5, base: int = 32) -> None:
        super().__init__()
        self.inc = DoubleConv(in_channels, base)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base, base * 2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base * 2, base * 4))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base * 4, base * 8))
        self.bot = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base * 8, base * 16))

        self.up1 = nn.ConvTranspose2d(base * 16, base * 8, 2, stride=2)
        self.conv1 = DoubleConv(base * 16, base * 8)
        self.up2 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.conv2 = DoubleConv(base * 8, base * 4)
        self.up3 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.conv3 = DoubleConv(base * 4, base * 2)
        self.up4 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.conv4 = DoubleConv(base * 2, base)
        self.outc = nn.Conv2d(base, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.bot(x4)

        u1 = self.up1(x5)
        u1 = self._concat_crop(u1, x4)
        u1 = self.conv1(u1)
        u2 = self.up2(u1)
        u2 = self._concat_crop(u2, x3)
        u2 = self.conv2(u2)
        u3 = self.up3(u2)
        u3 = self._concat_crop(u3, x2)
        u3 = self.conv3(u3)
        u4 = self.up4(u3)
        u4 = self._concat_crop(u4, x1)
        u4 = self.conv4(u4)
        return self.outc(u4)

    @staticmethod
    def _concat_crop(up: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        _, _, hu, wu = up.shape
        _, _, hs, ws = skip.shape
        y0 = max(0, (hs - hu) // 2)
        x0 = max(0, (ws - wu) // 2)
        skip = skip[:, :, y0 : y0 + hu, x0 : x0 + wu]
        return torch.cat([up, skip], dim=1)
