#!/usr/bin/env python3
"""
Create models/checkpoints/ci_unet.pt and models/onnx/ci_unet.onnx with matching weights,
then verify PyTorch vs ONNX reference evaluator (no ONNXRuntime required).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
import torch

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from src.python.training.unet_model import UNet  # noqa: E402
from src.python.utils.config import load_yaml, resolve_path  # noqa: E402


def main() -> None:
    cfg = load_yaml(resolve_path("configs/ci_export.yaml"))
    torch.manual_seed(0)
    m = UNet(
        in_channels=cfg["model"]["in_channels"],
        num_classes=cfg["model"]["num_classes"],
        base=cfg["model"]["base_channels"],
    )
    m.eval()
    _, _, ih, iw = cfg["export"]["input_size"]
    x = torch.randn(1, 3, ih, iw)

    ckpt_dir = resolve_path(cfg["training"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "ci_unet.pt"
    torch.save({"model": m.state_dict(), "model_name": "unet", "config": cfg}, ckpt_path)

    onnx_path = resolve_path(cfg["export"]["onnx_path"])
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        m,
        x,
        str(onnx_path),
        input_names=["input"],
        output_names=["logits"],
        opset_version=int(cfg["export"]["opset"]),
        do_constant_folding=True,
    )

    model_proto = onnx.load(str(onnx_path))
    onnx.checker.check_model(model_proto)

    from onnx.reference import ReferenceEvaluator

    with torch.no_grad():
        torch_out = m(x).cpu().numpy()
    ref_sess = ReferenceEvaluator(model_proto)
    ref_out = ref_sess.run(None, {"input": x.cpu().numpy()})[0]
    ok = np.allclose(torch_out, ref_out, rtol=1e-3, atol=1e-3)
    max_abs = float(np.max(np.abs(torch_out - ref_out)))
    print(f"ci_prepare_models: max_abs_diff={max_abs:.6e} reference_match={ok}")
    if not ok:
        raise SystemExit(1)
    print(f"Wrote {ckpt_path} and {onnx_path}")


if __name__ == "__main__":
    main()
