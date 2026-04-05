param(
  [string]$Config = "configs/train_unet.yaml",
  [string]$Checkpoint = "models/checkpoints/unet_best.pt",
  [string]$Onnx = "models/onnx/unet_scene_seg.onnx"
)
$root = Split-Path -Parent $MyInvocation.MyCommand.Path | Split-Path -Parent
Set-Location $root
python src/python/export/export_onnx.py --config $Config --checkpoint $Checkpoint
python src/python/export/validate_onnx.py --config $Config --checkpoint $Checkpoint --onnx $Onnx --backend auto
