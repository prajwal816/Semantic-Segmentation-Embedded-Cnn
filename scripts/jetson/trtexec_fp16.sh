#!/usr/bin/env bash
# Build a TensorRT FP16 engine from ONNX (Jetson / dGPU with trtexec on PATH).
# Usage: ./scripts/jetson/trtexec_fp16.sh [model.onnx] [out.engine]
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ONNX="${1:-$ROOT/models/onnx/unet_scene_seg.onnx}"
OUT="${2:-$ROOT/models/engines/unet_fp16.engine}"
mkdir -p "$(dirname "$OUT")"

if ! command -v trtexec >/dev/null 2>&1; then
  TRTEXEC="/usr/src/tensorrt/bin/trtexec"
  if [[ -x "$TRTEXEC" ]]; then
    trtexec() { "$TRTEXEC" "$@"; }
  else
    echo "trtexec not found. On Jetson, install TensorRT or add /usr/src/tensorrt/bin to PATH."
    exit 2
  fi
fi

trtexec --onnx="$ONNX" --saveEngine="$OUT" --fp16 --workspace=4096 --verbose
echo "Engine written to $OUT"
