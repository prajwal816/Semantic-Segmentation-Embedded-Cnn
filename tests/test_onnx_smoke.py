import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]


def test_unet_onnx_export_writes_graph(tmp_path):
    """
    Run export in a subprocess so a broken onnxruntime install (common on some Windows
    hosts when PyTorch's ONNX stack probes ORT) does not take down the whole pytest process.
    """
    out = tmp_path / "mini.onnx"
    code = f"""
import sys
from pathlib import Path
import torch

sys.path.insert(0, {str(_ROOT.resolve())!r})
from src.python.training.unet_model import UNet

out = Path({str(out.resolve())!r})
m = UNet(in_channels=3, num_classes=4, base=16)
m.eval()
x = torch.randn(1, 3, 64, 64)
torch.onnx.export(
    m,
    x,
    str(out),
    input_names=["input"],
    output_names=["logits"],
    opset_version=12,
    do_constant_folding=True,
)
data = out.read_bytes()
assert len(data) > 2000
assert b"graph" in data or b"Graph" in data
"""
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(_ROOT),
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert proc.returncode == 0, proc.stdout + "\n" + proc.stderr
