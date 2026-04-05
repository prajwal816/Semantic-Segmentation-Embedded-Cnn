from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import onnx
import torch

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.python.export.export_onnx import load_model_from_checkpoint  # noqa: E402
from src.python.utils.config import load_yaml, resolve_path, set_seed  # noqa: E402


def main() -> None:
    try:
        import onnxruntime as ort
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"ONNXRuntime import failed ({exc}). Install onnxruntime or use a Jetson/Linux env.") from exc

    parser = argparse.ArgumentParser(description="Compare PyTorch vs ONNXRuntime outputs.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--onnx", type=str, required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--atol", type=float, default=1e-3)
    args = parser.parse_args()

    cfg = load_yaml(resolve_path(args.config))
    set_seed(int(cfg["data"]["seed"]))
    device = torch.device(args.device or "cpu")

    onnx_path = resolve_path(args.onnx)
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)

    ckpt = resolve_path(args.checkpoint)
    torch_model, _ = load_model_from_checkpoint(ckpt, cfg, device)
    torch_model.to(device).eval()

    exp = cfg["export"]
    _, _, ih, iw = exp["input_size"]
    x = torch.randn(1, 3, ih, iw, device=device)

    with torch.no_grad():
        torch_out = torch_model(x).cpu().numpy()

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_out = sess.run(None, {"input": x.cpu().numpy()})[0]

    ok = np.allclose(torch_out, ort_out, rtol=args.rtol, atol=args.atol)
    max_abs = float(np.max(np.abs(torch_out - ort_out)))
    mean_abs = float(np.mean(np.abs(torch_out - ort_out)))
    print(f"max_abs_diff={max_abs:.6e} mean_abs_diff={mean_abs:.6e} match={ok}")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
