#!/usr/bin/env bash
# Example: Polygraphy-based INT8 engine (install on Jetson/x86 with TensorRT).
#   pip install polygraphy
#
# Prepare calibration images first:
#   python src/python/export/prepare_trt_calibration_images.py --out-dir data/calibration/frames
#
# Then adapt the flags below to your Polygraphy version (`polygraphy convert --help`).
# Uncomment and edit when Polygraphy + TensorRT are available:
#
# set -euo pipefail
# ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# ONNX="${1:-$ROOT/models/onnx/unet_scene_seg.onnx}"
# OUT="${2:-$ROOT/models/engines/unet_poly_int8.engine}"
# polygraphy convert "$ONNX" -o "$OUT" --int8 --fp16

echo "Template only: enable commands in scripts/jetson/polygraphy_int8_example.sh after installing Polygraphy."
exit 0
