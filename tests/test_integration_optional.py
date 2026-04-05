from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(
    not (_REPO / "models" / "onnx" / "unet_scene_seg.onnx").is_file(),
    reason="Full ONNX artifact not present; train+export first.",
)
def test_project_onnx_runs():
    pytest.importorskip("onnxruntime")
    import onnxruntime as ort

    p = _REPO / "models" / "onnx" / "unet_scene_seg.onnx"
    sess = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    shape = [int(d) if isinstance(d, int) else 1 for d in inp.shape]
    import numpy as np

    x = np.zeros(shape, dtype=np.float32)
    out = sess.run(None, {inp.name: x})[0]
    assert out.ndim == 4
