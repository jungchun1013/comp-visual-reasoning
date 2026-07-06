"""Stage 1: Referential segmentation pretraining on CLEVR-Ref+.

Trains GCA + connector + seg_head to predict which patches correspond
to a referring expression. Standalone script — does not touch main pipeline.

Usage:
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python scripts/train_refseg.py \
        --backbone dinov2 --epochs 16

Stage 2 (VQA) uses the saved GCA/connector weights:
    python scripts/train.py +experiment=clevr_dinov2_decoder1l_scratch \
        training.resume=outputs/model/refseg_dinov2_s42/best.pt
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from model import CrossAttnViT
from data.clevr_ref import CLEVRRefDataset, clevr_ref_collate_fn

try:
    import wandb
except ImportError:
    wandb = None


BACKBONE_CONFIGS = {
    "dinov2": {
        "backbone_name": "vit_base_patch14_dinov2.lvd142m",
        "resolution": 336,
        "patch_size": 14,
        "feature_pool": "cls",
    },
    "siglip": {
        "backbone_name": "vit_base_patch16_siglip_256",
        "resolution": 256,
        "patch_size": 16,
        "feature_pool": "mean",
    },
    "sup": {
        "backbone_name": "vit_base_patch16_384.augreg_in21k_ft_in1k",
        "resolution": 384,
        "patch_size": 16,
        "feature_pool": "cls",
    },
}


def soft_cross_entropy(logits, targets):
    """Soft cross-entropy: L = -Σ y_i log p_i, averaged over batch."""
    probs = F.softmax(logits, dim=1)
    loss = -(targets * torch.log(probs + 1e-8)).sum(dim=1).mean()
    return loss


def compute_iou(logits, targets, threshold=0.5):
    """Compute mean IoU between predicted and GT patch masks."""
    preds = (F.softmax(logits, dim=1) > threshold).float()
    gt = (targets > 0).float()
    intersection = (preds * gt).sum(dim=1)
    union = ((preds + gt) > 0).float().sum(dim=1)
    iou = intersection / (union + 1e-8)
    return iou.mean().item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", type=str, default="dinov2",
                        choices=list(BACKBONE_CONFIGS.keys()))
    parser.add_argument("--data-root", type=str,
                        default="/home/jungchun/data/clevr/clevr_ref+_1.0")
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=100)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Model
    bcfg = BACKBONE_CONFIGS[args.backbone]
    cross_attn_layers = [1, 3, 5, 7, 9, 11]

    steervit = CrossAttnViT.from_config(
        bcfg["backbone_name"], device=device,
        cross_attn_layers=cross_attn_layers,
        resolution=bcfg["resolution"],
        feature_aggregation=bcfg["feature_pool"],
    )
    steervit.eval()  # freeze backbone

    # Trainable: GCA + connector + seg_head
    trainable = []
    for blk in steervit.vision_model.trunk.blocks:
        gca = getattr(blk, "gated_cross_attn", None)
        if gca is not None:
            for p in gca.parameters():
                p.requires_grad = True
                trainable.append(p)
    for p in steervit.connector.parameters():
        p.requires_grad = True
        trainable.append(p)
    for p in steervit.lin_seg_head.parameters():
        p.requires_grad = True
        trainable.append(p)

    n_trainable = sum(p.numel() for p in trainable)
    n_total = sum(p.numel() for p in steervit.parameters())
    print(f"Trainable: {n_trainable:,} / {n_total:,} ({100*n_trainable/n_total:.2f}%)")

    # Data
    transform = steervit.get_transforms()
    train_ds = CLEVRRefDataset(
        args.data_root, "train", transform,
        patch_size=bcfg["patch_size"], resolution=bcfg["resolution"])
    val_ds = CLEVRRefDataset(
        args.data_root, "val", transform,
        patch_size=bcfg["patch_size"], resolution=bcfg["resolution"])

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=8, collate_fn=clevr_ref_collate_fn,
        pin_memory=True, drop_last=True)
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=8, collate_fn=clevr_ref_collate_fn,
        pin_memory=True)

    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    # Optimizer
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.05)
    scaler = GradScaler("cuda")

    # Output
    run_name = f"refseg_{args.backbone}_s{args.seed}"
    out_dir = Path(f"outputs/model/{run_name}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # WandB
    if wandb is not None:
        wandb.init(project="comp-visual-reasoning", name=run_name,
                   group=f"refseg_{args.backbone}",
                   config=vars(args))

    best_iou = 0.0
    prefix = steervit.vision_model.trunk.num_prefix_tokens

    for epoch in range(args.epochs):
        t0 = time.time()

        # Train
        steervit.train()
        # Keep backbone frozen
        steervit.vision_model.eval()
        steervit.text_model.eval()

        total_loss = 0
        for step, batch in enumerate(train_loader):
            images = batch["image"].to(device)
            texts = batch["text"]
            patch_gt = batch["patch_mask_gt"].to(device)

            optimizer.zero_grad()
            with autocast(device_type="cuda", dtype=torch.bfloat16):
                feats = steervit.forward(images, texts)
                patch_feats = feats[:, prefix:, :]  # remove CLS/prefix
                logits = steervit.lin_seg_head(patch_feats).squeeze(-1)  # (B, n*n)
                loss = soft_cross_entropy(logits, patch_gt)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(trainable, 1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

            if (step + 1) % args.log_every == 0:
                avg_loss = total_loss / (step + 1)
                print(f"  Epoch {epoch} | Step {step+1} | Loss: {avg_loss:.4f}",
                      flush=True)

        elapsed = time.time() - t0
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch} | Train loss: {avg_loss:.4f} | Time: {elapsed:.1f}s")

        # Validate
        steervit.eval()
        val_loss, val_iou, val_count = 0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                texts = batch["text"]
                patch_gt = batch["patch_mask_gt"].to(device)

                with autocast(device_type="cuda", dtype=torch.bfloat16):
                    feats = steervit.forward(images, texts)
                    patch_feats = feats[:, prefix:, :]
                    logits = steervit.lin_seg_head(patch_feats).squeeze(-1)
                    loss = soft_cross_entropy(logits, patch_gt)

                val_loss += loss.item() * images.size(0)
                val_iou += compute_iou(logits.float(), patch_gt) * images.size(0)
                val_count += images.size(0)

        val_loss /= val_count
        val_iou /= val_count
        print(f"Epoch {epoch} | Val loss: {val_loss:.4f} | Val IoU: {val_iou:.4f}")

        # Save
        ckpt = {
            "epoch": epoch,
            "steervit_state": {
                k: v for k, v in steervit.state_dict().items()
                if any(p.data_ptr() == v.data_ptr() for p in trainable)
            },
            "val_loss": val_loss,
            "val_iou": val_iou,
            "backbone": args.backbone,
            "config": vars(args),
        }
        torch.save(ckpt, out_dir / "last.pt")

        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(ckpt, out_dir / "best.pt")
            print(f"  New best IoU: {best_iou:.4f}")

        if wandb is not None and wandb.run is not None:
            wandb.log({
                "train/loss": avg_loss,
                "val/loss": val_loss,
                "val/iou": val_iou,
                "epoch": epoch,
            })

        # JSON log
        log_entry = {"epoch": epoch, "train_loss": avg_loss,
                     "val_loss": val_loss, "val_iou": val_iou}
        with open(out_dir / "train_log.jsonl", "a") as f:
            f.write(json.dumps(log_entry) + "\n")

    print(f"\nDone. Best val IoU: {best_iou:.4f}")
    if wandb is not None and wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
