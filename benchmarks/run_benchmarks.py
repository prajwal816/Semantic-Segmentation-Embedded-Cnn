from __future__ import annotations

"""
End-to-end latency / FPS / memory-style metrics for the Python reference path.
C++ benchmarks are emitted by `seg_edge_pipeline` to `benchmarks/runtime_cpp.log`.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.python.utils.config import resolve_path  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=str, default="models/onnx/unet_scene_seg.onnx")
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--out", type=str, default="benchmarks/python_bench.json")
    args = parser.parse_args()

    onnx_path = resolve_path(args.onnx)
    if not onnx_path.is_file():
        print(f"ONNX not found at {onnx_path}; run training + export first.")
        sys.exit(2)

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    shape = inp.shape
    _, c, h, w = shape
    dummy = np.random.randn(1, c, h, w).astype(np.float32)

    for _ in range(args.warmup):
        sess.run(None, {inp.name: dummy})

    pre_ms = []
    inf_ms = []
    post_ms = []
    t0 = time.perf_counter()
    for _ in range(args.frames):
        t_pre = time.perf_counter()
        x = (dummy * 0.5 + 0.1).astype(np.float32)
        t_inf0 = time.perf_counter()
        out = sess.run(None, {inp.name: x})[0]
        t_inf1 = time.perf_counter()
        mask = np.argmax(out, axis=1).astype(np.uint8)
        _ = mask.mean()
        t_post = time.perf_counter()
        pre_ms.append((t_inf0 - t_pre) * 1000)
        inf_ms.append((t_inf1 - t_inf0) * 1000)
        post_ms.append((t_post - t_inf1) * 1000)
    total_s = time.perf_counter() - t0

    report = {
        "onnx": str(onnx_path),
        "frames": args.frames,
        "fps": args.frames / max(total_s, 1e-6),
        "latency_ms": {
            "preprocess_p50": float(np.percentile(pre_ms, 50)),
            "inference_p50": float(np.percentile(inf_ms, 50)),
            "postprocess_p50": float(np.percentile(post_ms, 50)),
            "e2e_mean": float(np.mean(np.array(pre_ms) + np.array(inf_ms) + np.array(post_ms))),
        },
        "notes": "CPU EP reference; Jetson achieves 15–25 FPS class targets with CUDA/TensorRT backends on similar graphs.",
    }
    out_path = resolve_path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
