from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.python.training.deeplab_model import build_deeplabv3  # noqa: E402
from src.python.training.unet_model import UNet  # noqa: E402
from src.python.utils.config import ensure_dir, load_yaml, resolve_path, set_seed  # noqa: E402


class DeepLabOnnxWrapper(nn.Module):
    """Exports only the main segmentation logits."""

    def __init__(self, core: nn.Module) -> None:
        super().__init__()
        self.core = core

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.core(x)["out"]


def load_model_from_checkpoint(ckpt_path: Path, cfg: dict, device: torch.device) -> tuple[nn.Module, str]:
    blob = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_name = blob.get("model_name") or cfg["model"]["name"]
    if model_name == "unet":
        mcfg = cfg["model"]
        model = UNet(
            in_channels=mcfg["in_channels"],
            num_classes=mcfg["num_classes"],
            base=mcfg["base_channels"],
        )
    elif model_name == "deeplabv3":
        mcfg = cfg["model"]
        core = build_deeplabv3(
            num_classes=mcfg["num_classes"],
            pretrained_backbone=bool(mcfg.get("pretrained_backbone", False)),
        )
        model = DeepLabOnnxWrapper(core)
    else:
        raise ValueError(model_name)
    model.load_state_dict(blob["model"], strict=True)
    model.eval()
    return model, model_name


def main() -> None:
    parser = argparse.ArgumentParser(description="Export trained checkpoint to ONNX.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pt from training.")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    cfg = load_yaml(resolve_path(args.config))
    set_seed(int(cfg["data"]["seed"]))
    device = torch.device(args.device or "cpu")

    ckpt = resolve_path(args.checkpoint)
    model, _ = load_model_from_checkpoint(ckpt, cfg, device)
    model.to(device)

    exp = cfg["export"]
    opset = int(exp["opset"])
    _, _, ih, iw = exp["input_size"]
    dummy = torch.randn(1, 3, ih, iw, device=device)
    out_path = resolve_path(exp["onnx_path"])
    ensure_dir(out_path.parent)

    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        input_names=["input"],
        output_names=["logits"],
        opset_version=opset,
        do_constant_folding=True,
        dynamic_axes=None,
    )
    print(f"ONNX written to {out_path}")


if __name__ == "__main__":
    main()
