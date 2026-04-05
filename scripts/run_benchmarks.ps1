param([string]$Onnx = "models/onnx/unet_scene_seg.onnx")
$root = Split-Path -Parent $MyInvocation.MyCommand.Path | Split-Path -Parent
Set-Location $root
python benchmarks/run_benchmarks.py --onnx $Onnx
