#!/usr/bin/env bash
# TensorRT INT8 engine from ONNX using a calibration cache file.
# Generate calibration data with:
#   python src/python/export/prepare_trt_calibration_images.py --out-dir data/calibration/frames
# Then build a TensorRT calibrator cache using NVIDIA samples or Polygraphy (recommended).
#
# This script calls trtexec with --int8 when your TensorRT build supports --calib=<file>.
# Flag names differ across JetPack versions; adjust to match `trtexec --help` on device.
#
# Usage: ./scripts/jetson/trtexec_int8.sh [model.onnx] [out.engine] [calib_cache.bin]
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ONNX="${1:-$ROOT/models/onnx/unet_scene_seg.onnx}"
OUT="${2:-$ROOT/models/engines/unet_int8.engine}"
CALIB="${3:-$ROOT/data/calibration/calib_cache.bin}"

mkdir -p "$(dirname "$OUT")"

if [[ ! -f "$ONNX" ]]; then
  echo "Missing ONNX: $ONNX"
  exit 2
fi
if [[ ! -f "$CALIB" ]]; then
  echo "Missing calibration cache: $CALIB"
  echo "Create it with TensorRT IInt8EntropyCalibrator2, Polygraphy, or Jetson sample calibrators."
  exit 3
fi

if ! command -v trtexec >/dev/null 2>&1; then
  TRTEXEC="/usr/src/tensorrt/bin/trtexec"
  if [[ -x "$TRTEXEC" ]]; then
    trtexec() { "$TRTEXEC" "$@"; }
  else
    echo "trtexec not found."
    exit 2
  fi
fi

trtexec --onnx="$ONNX" --saveEngine="$OUT" --fp16 --int8 --calib="$CALIB" --workspace=4096 --verbose
echo "INT8 engine written to $OUT"
