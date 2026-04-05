from __future__ import annotations

"""
Writes example segmentation overlays to data/outputs/.
Prefer --onnx + ONNXRuntime; if onnxruntime is unavailable, use --checkpoint + --config (PyTorch).
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.python.training.deeplab_model import build_deeplabv3  # noqa: E402
from src.python.training.unet_model import UNet  # noqa: E402
from src.python.utils.config import ensure_dir, load_yaml, resolve_path  # noqa: E402


def _load_torch_model(cfg: dict, ckpt_path: Path, device: torch.device):
    blob = torch.load(ckpt_path, map_location=device, weights_only=False)
    name = blob.get("model_name") or cfg["model"]["name"]
    if name == "unet":
        m = cfg["model"]
        model = UNet(in_channels=m["in_channels"], num_classes=m["num_classes"], base=m["base_channels"])
    elif name == "deeplabv3":
        m = cfg["model"]
        model = build_deeplabv3(
            num_classes=m["num_classes"],
            pretrained_backbone=bool(m.get("pretrained_backbone", False)),
        )
    else:
        raise ValueError(name)
    model.load_state_dict(blob["model"], strict=True)
    model.to(device).eval()
    return model, name


def _forward(model, name: str, x: torch.Tensor) -> torch.Tensor:
    if name == "deeplabv3":
        return model(x)["out"]
    return model(x)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=str, default="")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--config", type=str, default="configs/train_unet.yaml")
    parser.add_argument("--out-dir", type=str, default="data/outputs")
    args = parser.parse_args()

    out_dir = ensure_dir(resolve_path(args.out_dir))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    palette = np.array(
        [[0, 0, 0], [255, 64, 64], [64, 255, 64], [64, 128, 255], [255, 200, 64], [200, 64, 255]],
        dtype=np.uint8,
    )

    use_ort = bool(args.onnx)
    if use_ort:
        try:
            import onnxruntime as ort
        except Exception as exc:  # noqa: BLE001
            print(f"ONNXRuntime unavailable ({exc}); use --checkpoint instead.")
            sys.exit(2)
        onnx_path = resolve_path(args.onnx)
        if not onnx_path.is_file():
            print(f"Missing ONNX: {onnx_path}")
            sys.exit(2)
        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        inp = sess.get_inputs()[0]
        _, c, h, w = inp.shape
        infer = lambda blob: sess.run(None, {inp.name: blob})[0]
    else:
        if not args.checkpoint:
            print("Provide --onnx PATH or --checkpoint PATH (with --config).")
            sys.exit(2)
        cfg = load_yaml(resolve_path(args.config))
        model, name = _load_torch_model(cfg, resolve_path(args.checkpoint), device)
        h, w = cfg["data"]["image_size"]
        c = 3

        def infer(blob: np.ndarray) -> np.ndarray:
            t = torch.from_numpy(blob).to(device)
            with torch.no_grad():
                y = _forward(model, name, t)
            return y.cpu().numpy()

    for i in range(4):
        rng = np.random.default_rng(100 + i)
        img = rng.uniform(0, 255, size=(h, w, 3)).astype(np.uint8)
        blob = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))[np.newaxis, ...]
        out = infer(blob)
        mask = np.argmax(out, axis=1)[0].astype(np.int32)
        color = palette[np.clip(mask, 0, palette.shape[0] - 1)]
        blend = cv2.addWeighted(img, 0.55, cv2.cvtColor(color, cv2.COLOR_RGB2BGR), 0.45, 0)
        cv2.imwrite(str(out_dir / f"demo_frame_{i:02d}_input.png"), img)
        cv2.imwrite(str(out_dir / f"demo_frame_{i:02d}_overlay.png"), blend)
        cv2.imwrite(str(out_dir / f"demo_frame_{i:02d}_mask_id.png"), (mask * 50).astype(np.uint8))

    print(f"Wrote demo images to {out_dir}")


if __name__ == "__main__":
    main()
