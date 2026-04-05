from __future__ import annotations

"""
Write PNG frames for TensorRT / Polygraphy INT8 calibration (RGB scene-like tensors on disk).
Uses the same synthetic layout generator as training for reproducibility without a camera.
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.python.utils.config import ensure_dir, resolve_path, set_seed  # noqa: E402
from src.python.utils.dataset import SyntheticSemanticDataset  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=str, default="data/calibration/frames")
    parser.add_argument("--num-frames", type=int, default=256)
    parser.add_argument("--size", type=str, default="256,256")
    parser.add_argument("--num-classes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    h, w = (int(x) for x in args.size.split(","))
    set_seed(args.seed)
    ds = SyntheticSemanticDataset(
        num_samples=args.num_frames,
        image_size=(h, w),
        num_classes=args.num_classes,
        seed=args.seed,
        transform=None,
    )
    out = ensure_dir(resolve_path(args.out_dir))
    for i in range(args.num_frames):
        img_chw, _ = ds[i]
        x = (img_chw * 255.0).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
        bgr = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(out / f"calib_{i:05d}.png"), bgr)
    print(f"Wrote {args.num_frames} PNGs to {out}")


if __name__ == "__main__":
    main()
