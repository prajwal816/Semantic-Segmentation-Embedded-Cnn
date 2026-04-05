import numpy as np
import onnxruntime as ort
import pytest
import torch

from src.python.training.unet_model import UNet


def test_unet_onnx_roundtrip(tmp_path):
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
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    y = sess.run(None, {"input": x.numpy()})[0]
    assert y.shape[0] == 1 and y.shape[1] == 4
    assert np.isfinite(y).all()
