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


def _run_reference(onnx_model: onnx.ModelProto, feed: dict[str, np.ndarray]) -> np.ndarray:
    from onnx.reference import ReferenceEvaluator

    sess = ReferenceEvaluator(onnx_model)
    outs = sess.run(None, feed)
    return outs[0]


def _run_ort(onnx_path: Path, feed: dict[str, np.ndarray]) -> np.ndarray:
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    return sess.run(None, feed)[0]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare PyTorch vs ONNXRuntime or ONNX reference interpreter outputs."
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--onnx", type=str, required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument(
        "--backend",
        type=str,
        choices=("auto", "ort", "reference"),
        default="auto",
        help="auto: try ORT, fall back to onnx.reference (no DLL) on failure.",
    )
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

    feed = {"input": x.cpu().numpy()}
    backend_used = args.backend

    if args.backend == "reference":
        onnx_out = _run_reference(onnx_model, feed)
    elif args.backend == "ort":
        onnx_out = _run_ort(onnx_path, feed)
    else:
        try:
            onnx_out = _run_ort(onnx_path, feed)
            backend_used = "ort"
        except Exception as exc:  # noqa: BLE001
            print(f"[auto] ONNXRuntime unavailable ({exc}); using onnx.reference.")
            onnx_out = _run_reference(onnx_model, feed)
            backend_used = "reference"

    ok = np.allclose(torch_out, onnx_out, rtol=args.rtol, atol=args.atol)
    max_abs = float(np.max(np.abs(torch_out - onnx_out)))
    mean_abs = float(np.mean(np.abs(torch_out - onnx_out)))
    print(f"backend={backend_used} max_abs_diff={max_abs:.6e} mean_abs_diff={mean_abs:.6e} match={ok}")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
