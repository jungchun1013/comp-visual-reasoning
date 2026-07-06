"""CLEVR-Math eval + fine-tune for SteerViT checkpoints.

Train on CLEVR-Math val, eval on CLEVR-Math test.
Three freeze modes:
  all            — GCA + connector + decoder trainable
  gca_connector  — GCA + connector trainable, decoder frozen
  connector      — connector only trainable

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/eval_clevr_math.py \
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-root", type=str,
                        default="/home/jungchun/data/clevr-math/data/clevr-math")
    parser.add_argument("--clevr-root", type=str,
                        default="/home/jungchun/data/clevr/CLEVR_v1.0")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--ft-epochs", type=int, default=4)
    parser.add_argument("--ft-lr", type=float, default=1e-4)
    parser.add_argument("--freeze-mode", type=str, default="all",
                        choices=["all", "gca_connector", "connector"])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data_root = Path(args.data_root)
    clevr_root = Path(args.clevr_root)
    test_image_dir = Path("/home/jungchun/data/clevr-math/data/clevr-math-test-images/CLEVR_v1.0/images/test")

    # CLEVR-Math answers are int, but CLEVRVQADataset expects str.
    # Preprocess JSON files to convert int answers to str.
    import tempfile, shutil
    proc_dir = Path(tempfile.mkdtemp(prefix="clevr_math_"))
    for split_name in ["val", "test"]:
        src = data_root / f"clevr-math-{split_name}.json"
        dst = proc_dir / f"clevr-math-{split_name}.json"
        with open(src) as f:
            raw = json.load(f)
        questions = raw if isinstance(raw, list) else raw.get("questions", [])
        for q in questions:
            if isinstance(q.get("answer"), int):
                q["answer"] = str(q["answer"])
        out = {"questions": questions} if isinstance(raw, dict) and "questions" in raw else raw
        if isinstance(raw, list):
            out = {"questions": questions}
        with open(dst, "w") as f:
            json.dump(out, f)
        print(f"Preprocessed {split_name}: {len(questions)} questions (answers → str)")
    data_root = proc_dir

    print(f"Loading: {args.checkpoint}")
    model, transform, vocab = load_model(args.checkpoint, device)
    steervit = model.steervit

    apply_freeze_mode(model, steervit, args.freeze_mode)

    # Test dataset (for eval)
    test_ds = CLEVRVQADataset(
        str(data_root), "test", transform,
        questions_file=str(data_root / "clevr-math-test.json"),
        image_dir=str(test_image_dir))
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=8, collate_fn=clevr_collate_fn, pin_memory=True)
    print(f"Test: {len(test_ds)}")

    # Train dataset (val split)
    train_ds = CLEVRVQADataset(
        str(data_root), "val", transform,
        questions_file=str(data_root / "clevr-math-val.json"),
        image_dir=str(clevr_root / "images" / "val"))
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=8, collate_fn=clevr_collate_fn,
        pin_memory=True, drop_last=True)
    print(f"Train (val split): {len(train_ds)}")

    # Zero-shot eval
    print("\n" + "=" * 60)
    print("CLEVR-Math: Zero-shot")
    print("=" * 60)
    model.eval()
    zs_results = evaluate_decoder(model, test_loader, device, vocab)
    print(f"Zero-shot: {zs_results['accuracy']:.4f} ({zs_results['total']})")
    print(format_results(zs_results))

    # Fine-tune
    print("\n" + "=" * 60)
    print(f"CLEVR-Math: Fine-tuning (lr={args.ft_lr}, {args.ft_epochs} epochs)")
    print("=" * 60)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.ft_lr, weight_decay=0.05)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab["<pad>"])
    scaler = GradScaler("cuda")

    ckpt_name = Path(args.checkpoint).parent.name
    out_dir = Path(f"outputs/model/{ckpt_name}_clevrmath_ft_{args.freeze_mode}")
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

            if (step + 1) % 200 == 0:
                acc = total_correct / total_tokens if total_tokens > 0 else 0
                print(f"  Epoch {epoch} | Step {step+1} | "
                      f"Loss: {total_loss/(step+1):.4f} | Acc: {acc:.4f}",
                      flush=True)

        model.eval()
        val_results = evaluate_decoder(model, test_loader, device, vocab)
        val_acc = val_results["accuracy"]
        last_acc = val_acc
        print(f"Epoch {epoch} | Test acc: {val_acc:.4f}")
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

    results = {
        "zero_shot": zs_results["accuracy"],
        "best_acc": best_acc, "best_epoch": best_epoch,
        "last_acc": last_acc, "last_epoch": args.ft_epochs - 1,
        "freeze_mode": args.freeze_mode, "lr": args.ft_lr,
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDone. Zero-shot: {zs_results['accuracy']:.4f}, "
          f"Best: {best_acc:.4f} (ep {best_epoch}), Last: {last_acc:.4f}")


if __name__ == "__main__":
    main()
