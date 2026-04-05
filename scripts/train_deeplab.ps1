$root = Split-Path -Parent $MyInvocation.MyCommand.Path | Split-Path -Parent
Set-Location $root
python src/python/training/train.py --config configs/train_deeplab.yaml
