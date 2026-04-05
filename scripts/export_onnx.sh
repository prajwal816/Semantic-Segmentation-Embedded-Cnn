#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
CFG="${1:-configs/train_unet.yaml}"
CKPT="${2:-models/checkpoints/unet_best.pt}"
ONNX_PATH=$(python -c "import yaml,sys;print(yaml.safe_load(open(sys.argv[1]))['export']['onnx_path'])" "$CFG")
python src/python/export/export_onnx.py --config "$CFG" --checkpoint "$CKPT"
python src/python/export/validate_onnx.py --config "$CFG" --checkpoint "$CKPT" --onnx "$ONNX_PATH"
