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

from src.python.utils.augment import SegmentationAugment  # noqa: E402
from src.python.utils.config import ensure_dir, load_yaml, resolve_path, set_seed  # noqa: E402
from src.python.utils.dataset import SyntheticSemanticDataset  # noqa: E402
from src.python.utils.losses import CombinedSegmentationLoss, cross_entropy_loss, dice_loss  # noqa: E402
from src.python.training.deeplab_model import build_deeplabv3  # noqa: E402
from src.python.training.unet_model import UNet  # noqa: E402


def build_model(cfg: dict) -> nn.Module:
    name = cfg["model"]["name"]
    if name == "unet":
        m = cfg["model"]
        return UNet(
            in_channels=m["in_channels"],
            num_classes=m["num_classes"],
            base=m["base_channels"],
        )
    if name == "deeplabv3":
        m = cfg["model"]
        return build_deeplabv3(
            num_classes=m["num_classes"],
            pretrained_backbone=bool(m.get("pretrained_backbone", False)),
        )
    raise ValueError(f"Unknown model.name: {name}")


def forward_logits(model: nn.Module, images: torch.Tensor, name: str):
    if name == "deeplabv3":
        out = model(images)
        return out["out"]
    return model(images)


def build_criterion(cfg: dict) -> nn.Module:
    t = cfg["training"]
    mode = t.get("loss", "combined")
    if mode == "ce":

        class CEOnly(nn.Module):
            def forward(self, logits, target):
                return cross_entropy_loss(logits, target)

        return CEOnly()
    if mode == "dice":

        class DiceOnly(nn.Module):
            def forward(self, logits, target):
                return dice_loss(logits, target)

        return DiceOnly()
    return CombinedSegmentationLoss(
        ce_weight=float(t.get("ce_weight", 0.5)),
        dice_weight=float(t.get("dice_weight", 0.5)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train U-Net or DeepLabV3 on synthetic semantic data.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    parser.add_argument("--device", type=str, default=None, help="cuda | cpu (default: auto).")
    args = parser.parse_args()

    cfg_path = resolve_path(args.config)
    cfg = load_yaml(cfg_path)

    seed = int(cfg["data"]["seed"])
    set_seed(seed)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    h, w = cfg["data"]["image_size"]
    num_classes = int(cfg["model"]["num_classes"])
    num_samples = int(cfg["data"]["num_samples"])
    aug = SegmentationAugment(**cfg.get("augmentation", {}))
    full = SyntheticSemanticDataset(
        num_samples=num_samples,
        image_size=(h, w),
        num_classes=num_classes,
        seed=seed,
        transform=aug,
    )
    val_frac = float(cfg["data"]["val_fraction"])
    n_val = max(1, int(num_samples * val_frac))
    n_train = num_samples - n_val
    train_ds, val_ds = random_split(
        full,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(seed),
    )

    bs = int(cfg["training"]["batch_size"])
    nw = int(cfg["training"]["num_workers"])
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=nw, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=device.type == "cuda")

    model_name = cfg["model"]["name"]
    model = build_model(cfg).to(device)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["training"]["learning_rate"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )
    criterion = build_criterion(cfg).to(device)

    ckpt_dir = ensure_dir(resolve_path(cfg["training"]["checkpoint_dir"]))
    stem = f"{model_name}_best.pt"
    best_path = ckpt_dir / stem

    epochs = int(cfg["training"]["epochs"])
    log_every = int(cfg["training"]["log_interval"])

    best_val = float("inf")
    for epoch in range(epochs):
        model.train()
        running = 0.0
        pbar = tqdm(train_loader, desc=f"train {epoch+1}/{epochs}")
        for step, (xb, yb) in enumerate(pbar):
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logits = forward_logits(model, xb, model_name)
            loss = criterion(logits, yb)
            loss.backward()
            opt.step()
            running += float(loss.item())
            if (step + 1) % log_every == 0:
                pbar.set_postfix(loss=f"{running / log_every:.4f}")
                running = 0.0

        model.eval()
        vloss = 0.0
        n = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                logits = forward_logits(model, xb, model_name)
                loss = criterion(logits, yb)
                vloss += float(loss.item()) * xb.size(0)
                n += xb.size(0)
        vloss /= max(n, 1)
        print(f"epoch {epoch+1} val_loss={vloss:.4f}")
        if vloss < best_val:
            best_val = vloss
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": cfg,
                    "model_name": model_name,
                },
                best_path,
            )
            print(f"saved checkpoint -> {best_path}")

    print("Training complete.")


if __name__ == "__main__":
    main()
