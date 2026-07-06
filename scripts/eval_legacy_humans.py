"""CLEVR-Humans eval + fine-tune for SteerViT checkpoints.

Supports both legacy and main codebase checkpoints.
Three freeze modes:
  all            — GCA + connector + decoder trainable
  gca_connector  — GCA + connector trainable, decoder frozen
  connector      — connector only trainable

Usage:
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python scripts/eval_legacy_humans.py \
        --checkpoint outputs/model/clevr_dinov2_decoder1l_scratch_s42/best.pt \
        --freeze-mode all --ft-lr 1e-4 --ft-epochs 16
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from model import CrossAttnViT
from data.clevr import CLEVRVQADataset, clevr_collate_fn, CLEVR_ANSWERS
from evaluator import evaluate_decoder, format_results


def build_vocab():
    vocab = {"<bos>": 0, "<eos>": 1, "<pad>": 2}
    for i, a in enumerate(CLEVR_ANSWERS):
        vocab[a] = i + 3
    return vocab


def answers_to_ids(answer_indices, vocab):
    B = answer_indices.size(0)
    bos, eos, pad = vocab["<bos>"], vocab["<eos>"], vocab["<pad>"]
    seqs = torch.full((B, 4), pad, dtype=torch.long, device=answer_indices.device)
    seqs[:, 0] = bos
    seqs[:, 1] = answer_indices + 3
    seqs[:, 2] = eos
    return seqs


def load_model(ckpt_path, device):
    """Unified loader (main or legacy format) — L3 dedupe, see src/model/checkpoint_io.py."""
    from model.checkpoint_io import load_any_checkpoint

    model, _steervit, transform, vocab, _task_type, meta = \
        load_any_checkpoint(ckpt_path, device)
    print(f"Loaded: {meta['name']} (epoch {meta['epoch']}, "
          f"val_acc={meta['val_acc']}, legacy={meta['legacy']})")
    return model, transform, vocab


def apply_freeze_mode(model, steervit, mode):
    """Apply freeze configuration. Returns trainable param count."""
    # Start: everything frozen
    for p in model.parameters():
        p.requires_grad = False

    # Always unfreeze connector
    if steervit.connector is not None:
        for p in steervit.connector.parameters():
            p.requires_grad = True

    if mode == "all":
        # GCA + connector + decoder
        for blk in steervit.vision_model.trunk.blocks:
            if hasattr(blk, 'gated_cross_attn') and blk.gated_cross_attn is not None:
                for p in blk.gated_cross_attn.parameters():
                    p.requires_grad = True
        for p in model.decoder.parameters():
            p.requires_grad = True
    elif mode == "gca_connector":
        # GCA + connector (decoder frozen)
        for blk in steervit.vision_model.trunk.blocks:
            if hasattr(blk, 'gated_cross_attn') and blk.gated_cross_attn is not None:
                for p in blk.gated_cross_attn.parameters():
                    p.requires_grad = True
    elif mode == "connector":
        # connector only (GCA + decoder frozen)
        pass

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Freeze mode: {mode} | Trainable: {trainable:,} / {total:,}")
    return trainable


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-root", type=str,
                        default="/home/jungchun/data/clevr")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--ft-epochs", type=int, default=4)
    parser.add_argument("--ft-lr", type=float, default=1e-4)
    parser.add_argument("--freeze-mode", type=str, default="all",
                        choices=["all", "gca_connector", "connector"])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    clevr_root = Path(args.data_root) / "CLEVR_v1.0"
    humans_root = Path(args.data_root) / "CLEVR-Humans"

    print(f"Loading: {args.checkpoint}")
    model, transform, vocab = load_model(args.checkpoint, device)
    steervit = model.steervit

    apply_freeze_mode(model, steervit, args.freeze_mode)

    # Val dataset
    val_ds = CLEVRVQADataset(
        str(clevr_root), "val", transform,
        questions_file=str(humans_root / "CLEVR-Humans-val.json"),
        image_dir=str(clevr_root / "images" / "val"))
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=8, collate_fn=clevr_collate_fn, pin_memory=True)

    # Zero-shot eval
    print("\n" + "=" * 60)
    print("CLEVR-Humans: Zero-shot")
    print("=" * 60)
    zs_results = evaluate_decoder(model, val_loader, device, vocab)
    print(f"Zero-shot: {zs_results['accuracy']:.4f} ({zs_results['total']})")
    print(format_results(zs_results))

    # Fine-tune
    print("\n" + "=" * 60)
    print(f"CLEVR-Humans: Fine-tuning (lr={args.ft_lr}, {args.ft_epochs} epochs)")
    print("=" * 60)

    train_ds = CLEVRVQADataset(
        str(clevr_root), "train", transform,
        questions_file=str(humans_root / "CLEVR-Humans-train.json"),
        image_dir=str(clevr_root / "images" / "train"))
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=8, collate_fn=clevr_collate_fn,
        pin_memory=True, drop_last=True)
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.ft_lr, weight_decay=0.05)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab["<pad>"])
    scaler = GradScaler("cuda")

    ckpt_name = Path(args.checkpoint).parent.name
    out_dir = Path(f"outputs/model/{ckpt_name}_humans_ft_{args.freeze_mode}")
    out_dir.mkdir(parents=True, exist_ok=True)
    best_acc = zs_results["accuracy"]
    best_epoch = -1
    last_acc = zs_results["accuracy"]

    for epoch in range(args.ft_epochs):
        model.train()
        total_loss, total_correct, total_tokens = 0, 0, 0
        for step, batch in enumerate(train_loader):
            images = batch["image"].to(device)
            questions = batch["question"]
            answers = batch["answer"].to(device)
            answer_ids = answers_to_ids(answers, vocab)

            optimizer.zero_grad()
            with autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(images, questions, answer_ids)
                targets = answer_ids[:, 1:]
                loss = criterion(
                    logits.reshape(-1, logits.size(-1)),
                    targets.reshape(-1))

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            scaler.step(optimizer)
            scaler.update()

            mask = targets != vocab["<pad>"]
            preds = logits.argmax(dim=-1)
            total_correct += (preds[mask] == targets[mask]).sum().item()
            total_tokens += mask.sum().item()
            total_loss += loss.item()

            if (step + 1) % 50 == 0:
                acc = total_correct / total_tokens if total_tokens > 0 else 0
                print(f"  Epoch {epoch} | Step {step+1} | "
                      f"Loss: {total_loss/(step+1):.4f} | Acc: {acc:.4f}",
                      flush=True)

        model.eval()
        val_results = evaluate_decoder(model, val_loader, device, vocab)
        val_acc = val_results["accuracy"]
        last_acc = val_acc
        print(f"Epoch {epoch} | Val acc: {val_acc:.4f}")
        print(format_results(val_results))

        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            torch.save({
                "epoch": epoch,
                "model_state_dict": {
                    k: v for k, v in model.state_dict().items()
                    if any(p.data_ptr() == v.data_ptr()
                           for p in model.parameters() if p.requires_grad)},
                "best_acc": best_acc,
            }, out_dir / "best.pt")
            print(f"  New best: {best_acc:.4f}")

    results = {"zero_shot": zs_results["accuracy"],
               "best_acc": best_acc, "best_epoch": best_epoch,
               "last_acc": last_acc, "last_epoch": args.ft_epochs - 1,
               "freeze_mode": args.freeze_mode, "lr": args.ft_lr}
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDone. Zero-shot: {zs_results['accuracy']:.4f}, "
          f"Best: {best_acc:.4f} (ep {best_epoch}), Last: {last_acc:.4f}")


if __name__ == "__main__":
    main()
