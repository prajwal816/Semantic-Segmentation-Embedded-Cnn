from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.python.evaluation.metrics import mean_iou, pixel_accuracy  # noqa: E402
from src.python.training.deeplab_model import build_deeplabv3  # noqa: E402
from src.python.training.unet_model import UNet  # noqa: E402
from src.python.utils.config import load_yaml, resolve_path, set_seed  # noqa: E402
from src.python.utils.dataset import SyntheticSemanticDataset  # noqa: E402


def build_eval_model(cfg: dict, ckpt: dict, device: torch.device) -> tuple[nn.Module, str]:
    name = ckpt.get("model_name") or cfg["model"]["name"]
    if name == "unet":
        m = cfg["model"]
        model = UNet(in_channels=m["in_channels"], num_classes=m["num_classes"], base=m["base_channels"])
    elif name == "deeplabv3":
        m = cfg["model"]
        model = build_deeplabv3(
            num_classes=m["num_classes"],
            pretrained_backbone=bool(m.get("pretrained_backbone", False)),
        )
    else:
        raise ValueError(name)
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device).eval()
    return model, name


def forward_logits(model: nn.Module, images: torch.Tensor, name: str) -> torch.Tensor:
    if name == "deeplabv3":
        return model(images)["out"]
    return model(images)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate mIoU / pixel accuracy on synthetic val split.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max-batches", type=int, default=40)
    args = parser.parse_args()

    cfg = load_yaml(resolve_path(args.config))
    set_seed(int(cfg["data"]["seed"]))
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    ckpt = torch.load(resolve_path(args.checkpoint), map_location=device, weights_only=False)
    if "config" in ckpt:
        cfg = ckpt["config"]
    model, model_name = build_eval_model(cfg, ckpt, device)

    h, w = cfg["data"]["image_size"]
    num_classes = int(cfg["model"]["num_classes"])
    num_samples = int(cfg["data"]["num_samples"])
    seed = int(cfg["data"]["seed"])
    full = SyntheticSemanticDataset(
        num_samples=num_samples,
        image_size=(h, w),
        num_classes=num_classes,
        seed=seed,
        transform=None,
    )
    val_frac = float(cfg["data"]["val_fraction"])
    n_val = max(1, int(num_samples * val_frac))
    n_train = num_samples - n_val
    _, val_ds = random_split(
        full,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(seed),
    )
    loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=0)

    miou_sum = 0.0
    pix_sum = 0.0
    n = 0
    with torch.no_grad():
        for i, (xb, yb) in enumerate(tqdm(loader, desc="eval")):
            if i >= args.max_batches:
                break
            xb = xb.to(device)
            yb = yb.to(device)
            logits = forward_logits(model, xb, model_name)
            pred = logits.argmax(dim=1)
            for b in range(pred.size(0)):
                miou_sum += mean_iou(pred[b : b + 1], yb[b : b + 1], num_classes)
                pix_sum += pixel_accuracy(pred[b : b + 1], yb[b : b + 1])
                n += 1

    print(f"mean_mIoU={miou_sum / max(n,1):.4f} mean_pixel_acc={pix_sum / max(n,1):.4f} (n={n})")


if __name__ == "__main__":
    main()
