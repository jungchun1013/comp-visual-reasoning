"""CLOSURE eval + fine-tune for SteerViT checkpoints.

Train on CLOSURE val (7 types combined), eval on CLOSURE test (per-type breakdown).
Three freeze modes:
  all            — GCA + connector + decoder trainable
  gca_connector  — GCA + connector trainable, decoder frozen
  connector      — connector only trainable

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/eval_closure.py \
        --checkpoint outputs/model/clevr_dinov2_decoder1l_scratch_s42/best.pt \
        --freeze-mode all --ft-lr 1e-4 --ft-epochs 4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, ConcatDataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from model import CrossAttnViT
from data.clevr import CLEVRVQADataset, clevr_collate_fn, CLEVR_ANSWERS
from evaluator import evaluate_decoder, format_results

CLOSURE_TYPES = [
    "and_mat_spa",
    "compare_mat",
    "compare_mat_spa",
    "embed_mat_spa",
    "embed_spa_mat",
    "or_mat",
    "or_mat_spa",
]


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
    """Apply freeze configuration."""
    for p in model.parameters():
        p.requires_grad = False

    if steervit.connector is not None:
        for p in steervit.connector.parameters():
            p.requires_grad = True

    if mode == "all":
        for blk in steervit.vision_model.trunk.blocks:
            if hasattr(blk, 'gated_cross_attn') and blk.gated_cross_attn is not None:
                for p in blk.gated_cross_attn.parameters():
                    p.requires_grad = True
        for p in model.decoder.parameters():
            p.requires_grad = True
    elif mode == "gca_connector":
        for blk in steervit.vision_model.trunk.blocks:
            if hasattr(blk, 'gated_cross_attn') and blk.gated_cross_attn is not None:
                for p in blk.gated_cross_attn.parameters():
                    p.requires_grad = True
    elif mode == "connector":
        pass

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Freeze mode: {mode} | Trainable: {trainable:,} / {total:,}")


def build_closure_datasets(closure_root, clevr_image_dir, transform, split):
    """Build per-type datasets for a CLOSURE split (val or test)."""
    datasets = {}
    for ctype in CLOSURE_TYPES:
        qfile = closure_root / f"{ctype}_{split}.json"
        if qfile.exists():
            ds = CLEVRVQADataset(
                str(closure_root), split, transform,
                questions_file=str(qfile),
                image_dir=str(clevr_image_dir))
            datasets[ctype] = ds
    return datasets


def evaluate_per_type(model, type_datasets, device, vocab, batch_size):
    """Evaluate on each CLOSURE type separately. Returns overall + per-type acc."""
    total_correct, total_count = 0, 0
    per_type = {}

    for ctype, ds in type_datasets.items():
        loader = DataLoader(
            ds, batch_size=batch_size, shuffle=False,
            num_workers=4, collate_fn=clevr_collate_fn, pin_memory=True)
        results = evaluate_decoder(model, loader, device, vocab)
        acc = results["accuracy"]
        n = results["total"]
        per_type[ctype] = {"accuracy": acc, "total": n}
        total_correct += int(acc * n)
        total_count += n

    overall = total_correct / total_count if total_count > 0 else 0
    return overall, per_type


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
    closure_root = Path(args.data_root) / "CLOSURE"
    clevr_val_images = clevr_root / "images" / "val"

    print(f"Loading: {args.checkpoint}")
    model, transform, vocab = load_model(args.checkpoint, device)
    steervit = model.steervit

    apply_freeze_mode(model, steervit, args.freeze_mode)

    # Build test datasets (for eval)
    test_datasets = build_closure_datasets(closure_root, clevr_val_images, transform, "test")
    print(f"Test types: {list(test_datasets.keys())}")
    print(f"Test total: {sum(len(ds) for ds in test_datasets.values())}")

    # Build train datasets from val split (all types combined)
    train_type_datasets = build_closure_datasets(closure_root, clevr_val_images, transform, "val")
    train_ds = ConcatDataset(list(train_type_datasets.values()))
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=8, collate_fn=clevr_collate_fn,
        pin_memory=True, drop_last=True)
    print(f"Train (val split combined): {len(train_ds)}")

    # Zero-shot eval
    print("\n" + "=" * 60)
    print("CLOSURE: Zero-shot")
    print("=" * 60)
    model.eval()
    zs_overall, zs_per_type = evaluate_per_type(
        model, test_datasets, device, vocab, args.batch_size)
    print(f"Zero-shot overall: {zs_overall:.4f}")
    for ctype, info in zs_per_type.items():
        print(f"  {ctype:25s} {info['accuracy']:.4f} ({info['total']})")

    # Fine-tune
    print("\n" + "=" * 60)
    print(f"CLOSURE: Fine-tuning (lr={args.ft_lr}, {args.ft_epochs} epochs)")
    print("=" * 60)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.ft_lr, weight_decay=0.05)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab["<pad>"])
    scaler = GradScaler("cuda")

    ckpt_name = Path(args.checkpoint).parent.name
    out_dir = Path(f"outputs/model/{ckpt_name}_closure_ft_{args.freeze_mode}")
    out_dir.mkdir(parents=True, exist_ok=True)
    best_acc = zs_overall
    best_epoch = -1
    last_acc = zs_overall

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
        val_overall, val_per_type = evaluate_per_type(
            model, test_datasets, device, vocab, args.batch_size)
        last_acc = val_overall
        print(f"Epoch {epoch} | Test acc: {val_overall:.4f}")
        for ctype, info in val_per_type.items():
            print(f"  {ctype:25s} {info['accuracy']:.4f} ({info['total']})")

        if val_overall > best_acc:
            best_acc = val_overall
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

    results = {
        "zero_shot": {"overall": zs_overall, "per_type": zs_per_type},
        "best_acc": best_acc, "best_epoch": best_epoch,
        "last_acc": last_acc, "last_epoch": args.ft_epochs - 1,
        "freeze_mode": args.freeze_mode, "lr": args.ft_lr,
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDone. Zero-shot: {zs_overall:.4f}, "
          f"Best: {best_acc:.4f} (ep {best_epoch}), Last: {last_acc:.4f}")


if __name__ == "__main__":
    main()
