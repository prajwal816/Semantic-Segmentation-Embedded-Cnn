import numpy as np
import onnx
import pytest
import torch

from src.python.training.unet_model import UNet


def test_reference_evaluator_matches_torch(tmp_path):
    pytest.importorskip("onnx.reference")
    from onnx.reference import ReferenceEvaluator

    torch.manual_seed(1)
    m = UNet(in_channels=3, num_classes=4, base=16)
    m.eval()
    x = torch.randn(1, 3, 64, 64)
    path = tmp_path / "r.onnx"
    torch.onnx.export(
        m,
        x,
        str(path),
        input_names=["input"],
        output_names=["logits"],
        opset_version=12,
        do_constant_folding=True,
    )
    proto = onnx.load(str(path))
    ref = ReferenceEvaluator(proto)
    torch_out = m(x).detach().cpu().numpy()
    ref_out = ref.run(None, {"input": x.numpy()})[0]
    assert np.allclose(torch_out, ref_out, rtol=1e-3, atol=1e-3)
