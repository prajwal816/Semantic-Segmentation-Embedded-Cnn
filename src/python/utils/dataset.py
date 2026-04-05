from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class SyntheticSemanticDataset(Dataset):
    """
    Deterministic pseudo-scene generator (~10K samples) for reproducible training
    without shipping a real dataset. Images are smooth multi-region layouts; labels
    are integer class maps suitable for cross-entropy training.
    """

    def __init__(
        self,
        num_samples: int,
        image_size: Tuple[int, int],
        num_classes: int,
        seed: int = 0,
        transform: Optional[Callable] = None,
    ) -> None:
        super().__init__()
        self.num_samples = num_samples
        self.h, self.w = image_size
        self.num_classes = num_classes
        self.seed = seed
        self.transform = transform

    def __len__(self) -> int:
        return self.num_samples

    def _sample_layout(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(self.seed + idx)
        y_grid, x_grid = np.mgrid[0 : self.h, 0 : self.w].astype(np.float32)
        x_grid /= max(self.w - 1, 1)
        y_grid /= max(self.h - 1, 1)

        k = self.num_classes
        centers = rng.uniform(0.1, 0.9, size=(k, 2)).astype(np.float32)
        scales = rng.uniform(0.12, 0.35, size=(k, 2)).astype(np.float32)
        logits = np.zeros((k, self.h, self.w), dtype=np.float32)
        for c in range(k):
            cx, cy = centers[c]
            sx, sy = scales[c]
            dist = (x_grid - cx) ** 2 / (sx**2 + 1e-6) + (y_grid - cy) ** 2 / (sy**2 + 1e-6)
            logits[c] = -dist + rng.normal(0.0, 0.05, size=(self.h, self.w)).astype(np.float32)

        mask = np.argmax(logits, axis=0).astype(np.int64)

        img = np.zeros((3, self.h, self.w), dtype=np.float32)
        palette = rng.uniform(0.15, 0.95, size=(k, 3)).astype(np.float32)
        for c in range(k):
            m = mask == c
            for ch in range(3):
                img[ch][m] = palette[c, ch]

        noise = rng.normal(0, 0.04, size=img.shape).astype(np.float32)
        img = np.clip(img + noise, 0.0, 1.0)
        return img, mask

    def __getitem__(self, idx: int):
        image_chw, mask_hw = self._sample_layout(idx)
        image = torch.from_numpy(image_chw)
        mask = torch.from_numpy(mask_hw)
        if self.transform is not None:
            image, mask = self.transform(image, mask)
        return image, mask.long()
