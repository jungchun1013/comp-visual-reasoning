"""CoGenT sample efficiency: fine-tune on NK valB samples, eval valA+valB.

Sweep N ∈ {1K, 5K, 10K, 20K, 30K, 50K}. Each run:
  - Reset to CoGenT-A checkpoint
  - Fine-tune 4 epochs on N random valB samples
  - Eval on remaining valB + full valA (150K)

Usage:
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python scripts/finetune_cogent_b.py \
        --checkpoint <cogent_A_checkpoint>
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import time
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from model import CrossAttnViT
from tasks.decoder import build_decoder_model, build_clevr_decoder_vocab
from data.clevr import CLEVRVQADataset, ANSWER_TO_IDX
from data.clevr_programs import coarse_question_type
from omegaconf import OmegaConf


def load_model(ckpt_path, data_root, device):
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    steervit = CrossAttnViT.from_config(
        "vit_base_patch14_dinov2.lvd142m", device=device,
        cross_attn_layers=[1, 3, 5, 7, 9, 11], resolution=336)
    if "steervit_trainable_state" in ckpt:
        steervit.load_state_dict(ckpt["steervit_trainable_state"], strict=False)

    vocab = build_clevr_decoder_vocab()
    cfg = OmegaConf.create({
        "model": {"backbone_name": "vit_base_patch14_dinov2.lvd142m",
                  "resolution": 336, "cross_attn_layers": [1, 3, 5, 7, 9, 11]},
        "task": {"type": "decoder", "decoder": {"d_model": 512, "nhead": 8,
                 "num_layers": 1, "max_len": 8}, "use_text_gca": False},
        "data": {"root": data_root},
    })
    model = build_decoder_model(steervit, cfg).to(device)
    if "decoder_state_dict" in ckpt:
        model.decoder.load_state_dict(ckpt["decoder_state_dict"], strict=False)
    transform = steervit.get_transforms()
    print(f"Loaded CoGenT-A model (epoch {ckpt.get('epoch', '?')})")
    return model, steervit, vocab, transform


def answers_to_decoder_ids(answer_indices, vocab):
    bos, eos, pad = vocab["<bos>"], vocab["<eos>"], vocab["<pad>"]
    inv = {v: k for k, v in ANSWER_TO_IDX.items()}
    B = answer_indices.size(0)
    max_len = 4
    ids = torch.full((B, max_len), pad, dtype=torch.long, device=answer_indices.device)
    ids[:, 0] = bos
    for i in range(B):
        ans = inv.get(answer_indices[i].item(), "")
        tokens = ans.split()
        for j, tok in enumerate(tokens):
            if j + 1 < max_len - 1:
                ids[i, j + 1] = vocab.get(tok, pad)
        eos_pos = min(len(tokens) + 1, max_len - 1)
        ids[i, eos_pos] = eos
    return ids


class EvalDatasetWrapper:
    def __init__(self, ds, indices):
        self.ds = ds
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        item = self.ds[real_idx]
        q = self.ds.questions[real_idx]
        item["answer_text"] = q["answer"]
        item["question_type"] = coarse_question_type(q.get("program", []))
        return item


def evaluate(model, dataloader, device):
    model.eval()
    by_type = defaultdict(lambda: {"correct": 0, "total": 0})
    total_correct, total = 0, 0

    for batch in dataloader:
        images = batch["image"].to(device)
        questions = batch["question"]
        gt_answers = batch["answer_text"]

        with torch.no_grad(), autocast(device_type="cuda", dtype=torch.bfloat16):
            preds = model.generate(images, questions)

        for pred, gt, qtype in zip(preds, gt_answers, batch["question_type"]):
            correct = pred.strip().lower() == gt.strip().lower()
            total_correct += int(correct)
            total += 1
            by_type[qtype]["correct"] += int(correct)
            by_type[qtype]["total"] += 1

    acc = total_correct / total if total > 0 else 0
    return acc, dict(by_type)


def train_collate(batch):
    return {
        "image": torch.stack([b["image"] for b in batch]),
        "question": [b["question"] for b in batch],
        "answer": torch.tensor([b["answer"] for b in batch], dtype=torch.long),
    }


def eval_collate(batch):
    return {
        "image": torch.stack([b["image"] for b in batch]),
        "question": [b["question"] for b in batch],
        "answer": torch.tensor([b["answer"] for b in batch], dtype=torch.long),
        "answer_text": [b["answer_text"] for b in batch],
        "question_type": [b["question_type"] for b in batch],
    }


def run_one(model, steervit, vocab, train_loader, evalB_loader, evalA_loader,
            device, lr, epochs, eval_every_epoch=True):
    """Fine-tune for N epochs. If eval_every_epoch=False, only eval after last epoch."""
    for p in steervit.vision_model.trunk.parameters():
        p.requires_grad = False
    for blk in steervit.vision_model.trunk.blocks:
        if hasattr(blk, 'gated_cross_attn') and blk.gated_cross_attn is not None:
            for p in blk.gated_cross_attn.parameters():
                p.requires_grad = True
    if steervit.connector is not None:
        for p in steervit.connector.parameters():
            p.requires_grad = True
    for p in model.decoder.parameters():
        p.requires_grad = True

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=0.05)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab["<pad>"])
    scaler = GradScaler()

    epoch_results = []
    for epoch in range(epochs):
        model.train()
        steervit.vision_model.trunk.eval()
        if steervit.text_model is not None:
            steervit.text_model.eval()

        epoch_loss, t0 = 0.0, time.time()
        for step, batch in enumerate(train_loader):
            images = batch["image"].to(device)
            questions = batch["question"]
            answer_indices = batch["answer"].to(device)
            answer_ids = answers_to_decoder_ids(answer_indices, vocab).to(device)

            optimizer.zero_grad()
            with autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(images, questions, answer_ids)
                loss = criterion(logits.reshape(-1, logits.size(-1)),
                                answer_ids[:, 1:].reshape(-1))

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        elapsed = time.time() - t0
        print(f"    Epoch {epoch+1} | Loss: {avg_loss:.4f} | {elapsed:.0f}s", flush=True)

        if eval_every_epoch or epoch == epochs - 1:
            acc_B, by_type_B = evaluate(model, evalB_loader, device)
            acc_A, by_type_A = evaluate(model, evalA_loader, device)
            print(f"    Epoch {epoch+1} | valB: {acc_B:.1%} | valA: {acc_A:.1%}")
            epoch_results.append({
                "epoch": epoch + 1, "loss": avg_loss,
                "valA": acc_A, "valB": acc_B,
                "valB_by_type": {qt: r for qt, r in by_type_B.items()},
            })

    return epoch_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-root", type=str,
                        default="/home/jungchun/data/clevr/CLEVR_CoGenT_v1.0")
    parser.add_argument("--n-list", type=str, default="1000,5000,10000,20000,30000,50000")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--eval-final-only", action="store_true",
                        help="Only eval after last epoch")
    args = parser.parse_args()

    n_list = [int(n) for n in args.n_list.split(",")]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"LR: {args.lr}, Epochs: {args.epochs}")
    print(f"N list: {n_list}")

    model, steervit, vocab, transform = load_model(
        args.checkpoint, args.data_root, device)

    # Save initial state for resetting
    init_model_state = copy.deepcopy(model.state_dict())
    init_steervit_state = copy.deepcopy(steervit.state_dict())

    output_dir = Path(args.output_dir) if args.output_dir else \
        Path("outputs/analysis/cogent_sample_efficiency")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load datasets
    valB_dataset = CLEVRVQADataset(args.data_root, "valB", transform)
    valA_dataset = CLEVRVQADataset(args.data_root, "valA", transform)
    print(f"valB: {len(valB_dataset)}, valA: {len(valA_dataset)}")

    # Split valB: first 100K = fixed eval, last 50K = train pool
    rng = random.Random(args.seed)
    all_valB_indices = list(range(len(valB_dataset)))
    rng.shuffle(all_valB_indices)
    eval_indices = all_valB_indices[:100000]
    train_pool = all_valB_indices[100000:]  # ~50K
    print(f"EvalB: {len(eval_indices)}, Train pool: {len(train_pool)}")

    # Full valA eval loader
    valA_loader = DataLoader(
        EvalDatasetWrapper(valA_dataset, list(range(len(valA_dataset)))),
        batch_size=256, shuffle=False, num_workers=4,
        collate_fn=eval_collate, pin_memory=True)

    # Fixed evalB loader (same 100K for all runs)
    evalB_loader = DataLoader(
        EvalDatasetWrapper(valB_dataset, eval_indices),
        batch_size=256, shuffle=False, num_workers=4,
        collate_fn=eval_collate, pin_memory=True)

    # Eval before fine-tune
    print("\n--- Before fine-tune ---")
    acc_B0, by_type_B0 = evaluate(model, evalB_loader, device)
    acc_A0, _ = evaluate(model, valA_loader, device)
    print(f"valA: {acc_A0:.1%}, valB: {acc_B0:.1%}")
    for qt in sorted(by_type_B0.keys()):
        r = by_type_B0[qt]
        print(f"  {qt:20s}: {r['correct']}/{r['total']} = {r['correct']/r['total']:.1%}")

    results = {"before": {"valA": acc_A0, "valB": acc_B0}, "runs": {}}

    for n_ft in n_list:
        print(f"\n{'='*60}")
        print(f"N = {n_ft} ({args.epochs} epochs, LR={args.lr})")
        print(f"{'='*60}")

        # Reset model
        model.load_state_dict(init_model_state)
        steervit.load_state_dict(init_steervit_state)

        # Sample N from train pool (50K)
        train_indices = train_pool[:n_ft]

        train_loader = DataLoader(
            Subset(valB_dataset, train_indices),
            batch_size=args.batch_size, shuffle=True, num_workers=4,
            collate_fn=train_collate, pin_memory=True)

        print(f"  Train: {len(train_indices)}, EvalB: {len(eval_indices)} (fixed)")

        epoch_results = run_one(model, steervit, vocab, train_loader,
                                evalB_loader, valA_loader, device,
                                args.lr, args.epochs,
                                eval_every_epoch=not args.eval_final_only)
        results["runs"][str(n_ft)] = epoch_results

    # Summary
    print(f"\n{'='*60}")
    print(f"Summary (best epoch per N)")
    print(f"{'='*60}")
    print(f"{'N':>8s}  {'valB':>8s}  {'valA':>8s}  {'ΔvalB':>8s}  {'ΔvalA':>8s}  {'ep':>3s}")
    print(f"{'before':>8s}  {acc_B0:>7.1%}  {acc_A0:>7.1%}")
    for n_ft in n_list:
        epochs = results["runs"][str(n_ft)]
        best = max(epochs, key=lambda e: e["valB"])
        dB = best["valB"] - acc_B0
        dA = best["valA"] - acc_A0
        print(f"{n_ft:>8d}  {best['valB']:>7.1%}  {best['valA']:>7.1%}  "
              f"{dB:>+7.1%}  {dA:>+7.1%}  {best['epoch']:>3d}")

    save_path = output_dir / "sample_efficiency.json"
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {save_path}")


if __name__ == "__main__":
    main()
