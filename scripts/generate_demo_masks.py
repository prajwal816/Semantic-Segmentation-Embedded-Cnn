from __future__ import annotations

"""
Writes example segmentation overlays to data/outputs/ using ONNXRuntime (no display required).
Run after training + export, or place any compatible ONNX at --onnx.
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.python.utils.config import ensure_dir, resolve_path  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=str, default="models/onnx/unet_scene_seg.onnx")
    parser.add_argument("--out-dir", type=str, default="data/outputs")
    args = parser.parse_args()

    onnx_path = resolve_path(args.onnx)
    if not onnx_path.is_file():
        print(f"Missing ONNX: {onnx_path}")
        sys.exit(2)

    out_dir = ensure_dir(resolve_path(args.out_dir))
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    _, c, h, w = inp.shape

    palette = np.array(
        [[0, 0, 0], [255, 64, 64], [64, 255, 64], [64, 128, 255], [255, 200, 64], [200, 64, 255]],
        dtype=np.uint8,
    )

    for i in range(4):
        rng = np.random.default_rng(100 + i)
        img = rng.uniform(0, 255, size=(h, w, 3)).astype(np.uint8)
        blob = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))[np.newaxis, ...]
        out = sess.run(None, {inp.name: blob})[0]
        mask = np.argmax(out, axis=1)[0].astype(np.int32)
        color = palette[np.clip(mask, 0, palette.shape[0] - 1)]
        blend = cv2.addWeighted(img, 0.55, cv2.cvtColor(color, cv2.COLOR_RGB2BGR), 0.45, 0)
        cv2.imwrite(str(out_dir / f"demo_frame_{i:02d}_input.png"), img)
        cv2.imwrite(str(out_dir / f"demo_frame_{i:02d}_overlay.png"), blend)
        cv2.imwrite(str(out_dir / f"demo_frame_{i:02d}_mask_id.png"), (mask * 50).astype(np.uint8))

    print(f"Wrote demo images to {out_dir}")


if __name__ == "__main__":
    main()
