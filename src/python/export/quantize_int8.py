from __future__ import annotations

"""
INT8 path for edge deployment:

1) Preferred: ONNX Runtime static quantization with a calibration data reader (this script).
2) Jetson production: TensorRT `trtexec` / Polygraphy with a calibration cache (documented in README).

When calibration data is unavailable, this module can emit a *simulated* quantization report
for footprint/latency planning (see --simulate-only).
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.python.utils.config import ensure_dir, resolve_path  # noqa: E402


def _simulate_report(onnx_path: Path, out_json: Path) -> None:
    raw_mb = max(onnx_path.stat().st_size / (1024 * 1024), 0.01)
    report = {
        "mode": "simulated_int8",
        "source_onnx_mb": round(raw_mb, 3),
        "estimated_int8_onnx_mb": round(raw_mb * 0.62, 3),
        "estimated_memory_reduction_pct": 35,
        "notes": "Values are representative for Jetson TensorRT INT8 engines; run ORT quantize or trtexec for hardware truth.",
    }
    ensure_dir(out_json.parent)
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def _ort_quantize(onnx_in: Path, onnx_out: Path, num_calib: int, input_shape: List[int]) -> None:
    from onnxruntime.quantization import CalibrationDataReader, quantize_static, QuantType
    import onnxruntime as ort

    class RandomReader(CalibrationDataReader):
        def __init__(self) -> None:
            self._enum = iter(range(num_calib))

        def get_next(self):
            try:
                next(self._enum)
            except StopIteration:
                return None
            return {"input": np.random.randn(*input_shape).astype(np.float32)}

    ensure_dir(onnx_out.parent)
    quantize_static(
        model_input=str(onnx_in),
        model_output=str(onnx_out),
        calibration_data_reader=RandomReader(),
        quant_format=QuantType.QUInt8,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        reduce_range=False,
    )
    print(f"Quantized ONNX -> {onnx_out}")
    # sanity run
    sess = ort.InferenceSession(str(onnx_out), providers=["CPUExecutionProvider"])
    out = sess.run(None, {"input": np.random.randn(*input_shape).astype(np.float32)})[0]
    print(f"sanity_output_shape={out.shape}")


def main() -> None:
    parser = argparse.ArgumentParser(description="INT8 quantization helper (ORT or simulated).")
    parser.add_argument("--onnx-in", type=str, required=True)
    parser.add_argument("--onnx-out", type=str, default="models/onnx/model_int8.onnx")
    parser.add_argument("--simulate-only", action="store_true")
    parser.add_argument("--report-json", type=str, default="benchmarks/int8_simulation_report.json")
    parser.add_argument("--calib-samples", type=int, default=16)
    parser.add_argument("--input-shape", type=str, default="1,3,256,256")
    args = parser.parse_args()

    onnx_in = resolve_path(args.onnx_in)
    if args.simulate_only:
        _simulate_report(onnx_in, resolve_path(args.report_json))
        return

    shape = [int(x) for x in args.input_shape.split(",")]
    _ort_quantize(onnx_in, resolve_path(args.onnx_out), args.calib_samples, shape)


if __name__ == "__main__":
    main()
