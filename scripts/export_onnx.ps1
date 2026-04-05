param(
  [string]$Config = "configs/train_unet.yaml",
  [string]$Checkpoint = "models/checkpoints/unet_best.pt"
)
$root = Split-Path -Parent $MyInvocation.MyCommand.Path | Split-Path -Parent
Set-Location $root
python src/python/export/export_onnx.py --config $Config --checkpoint $Checkpoint
python src/python/export/validate_onnx.py --config $Config --checkpoint $Checkpoint --onnx (python -c "import yaml;print(yaml.safe_load(open('$Config'))['export']['onnx_path'])")
