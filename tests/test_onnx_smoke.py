import torch

from src.python.training.unet_model import UNet


def test_unet_onnx_export_writes_graph(tmp_path):
    torch.manual_seed(0)
    model = UNet(in_channels=3, num_classes=4, base=16)
    model.eval()
    x = torch.randn(1, 3, 64, 64)
    path = tmp_path / "mini.onnx"
    torch.onnx.export(
        model,
        x,
        str(path),
        input_names=["input"],
        output_names=["logits"],
        opset_version=12,
        do_constant_folding=True,
    )
    data = path.read_bytes()
    assert path.is_file() and len(data) > 2000
    assert b"graph" in data or b"Graph" in data
