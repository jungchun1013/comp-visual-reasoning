"""Generalization evaluation: CLEVR standard, CLOSURE, CoGenT.

Evaluates a trained checkpoint on multiple benchmarks to test
compositional generalization.

Usage:
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python scripts/eval_generalization.py \
        --checkpoint outputs/model/clevr_siglip_decoder1l_scratch_s42/best.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from data.clevr import CLEVRVQADataset, clevr_collate_fn
from evaluator import evaluate_decoder, evaluate_classification, format_results
from analysis.run_log import tee_stdout
from omegaconf import OmegaConf


# ── Model loading (reuse from tsne_viz) ─────────────────────────────

def load_model(ckpt_path, device):
    """Load checkpoint, auto-detect type, return model + metadata.

    Thin wrapper over model.checkpoint_io.load_any_checkpoint, which handles
    main/legacy formats and forwards use_gate/condition_type/feature_aggregation
    from the saved config (the old inline loader dropped use_gate, breaking
    nogate checkpoints).
    """
    from model.checkpoint_io import load_any_checkpoint

    model, _steervit, transform, vocab, task_type, meta = \
        load_any_checkpoint(ckpt_path, device)
    best_acc = meta["best_acc"] if meta["best_acc"] is not None else meta["val_acc"]
    return model, transform, vocab, task_type, meta["epoch"], best_acc


# ── Ablation wrappers ───────────────────────────────────────────────

class TextOnlyWrapper(torch.nn.Module):
    """Zero visual features — model can only use question text."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def generate(self, images, questions, **kwargs):
        raw = self.model.module if hasattr(self.model, "module") else self.model
        if hasattr(raw, "steervit"):
            # GCA decoder: zero patches, keep text
            prefix = raw.steervit.vision_model.trunk.num_prefix_tokens
            text_feats = None
            if raw.decoder.use_text_gca and questions is not None:
                if raw.text_cache is not None:
                    text_feats, _, _ = raw.text_cache.encode_text(questions)
                else:
                    text_feats, _, _ = raw.steervit.encode_text(questions)
                text_feats = text_feats.detach()
            # Zero visual patches
            B = images.size(0)
            visual_dim = raw.steervit.visual_dim
            patches = torch.zeros(B, 576, visual_dim, device=images.device)
            token_ids = raw.decoder.generate(
                patches, bos_id=raw.vocab["<bos>"], eos_id=raw.vocab["<eos>"],
                text_feats=text_feats, max_len=4)
            results = []
            for seq in token_ids:
                words = []
                for t in seq:
                    w = raw.inv_vocab.get(t.item(), "")
                    if w == "<eos>":
                        break
                    if w not in ("<bos>", "<pad>"):
                        words.append(w)
                results.append(" ".join(words))
            return results
        else:
            # MoT: zero vision embeddings
            raw.zero_vision = True
            result = raw.generate(images, questions)
            raw.zero_vision = False
            return result


class ImageOnlyWrapper(torch.nn.Module):
    """Zero text features — model can only use image."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def generate(self, images, questions, **kwargs):
        raw = self.model.module if hasattr(self.model, "module") else self.model
        if hasattr(raw, "steervit"):
            # GCA decoder: no steering (null questions), zero text_feats
            feats = raw.steervit.forward(images, None)  # no GCA conditioning
            prefix = raw.steervit.vision_model.trunk.num_prefix_tokens
            patches = feats[:, prefix:, :]
            token_ids = raw.decoder.generate(
                patches, bos_id=raw.vocab["<bos>"], eos_id=raw.vocab["<eos>"],
                text_feats=None, max_len=4)
            results = []
            for seq in token_ids:
                words = []
                for t in seq:
                    w = raw.inv_vocab.get(t.item(), "")
                    if w == "<eos>":
                        break
                    if w not in ("<bos>", "<pad>"):
                        words.append(w)
                results.append(" ".join(words))
            return results
        else:
            # MoT: zero text embeddings
            dummy_qs = [""] * images.size(0)
            return raw.generate(images, dummy_qs)


# ── Evaluation helpers ───────────────────────────────────────────────

def finetune_clevr_humans(model, transform, vocab, task_type, device, args):
    """Fine-tune model on CLEVR-Humans train, eval on val."""
    from torch.amp import GradScaler, autocast
    from trainer import _answers_to_decoder_ids

    data_root = Path(args.data_root)
    clevr_root = data_root / "CLEVR_v1.0"
    humans_root = data_root / "CLEVR-Humans"
    model_name = Path(args.checkpoint).parent.name

    # Zero-shot eval first
    print("=" * 60)
    print("CLEVR-Humans: Zero-shot")
    print("=" * 60)
    val_ds = CLEVRVQADataset(
        str(clevr_root), "val", transform,
        questions_file=str(humans_root / "CLEVR-Humans-val.json"),
        image_dir=str(clevr_root / "images" / "val"))
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=8, collate_fn=clevr_collate_fn, pin_memory=True)
    zs_results = evaluate_decoder(model, val_loader, device, vocab)
    print(f"Zero-shot: {zs_results['accuracy']:.4f} ({zs_results['total']})")
    print()

    # Fine-tune
    print("=" * 60)
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
    criterion = torch.nn.CrossEntropyLoss(ignore_index=vocab["<pad>"])
    scaler = GradScaler("cuda")

    out_dir = Path(f"outputs/model/{model_name}_humans_ft")
    out_dir.mkdir(parents=True, exist_ok=True)
    best_acc = zs_results["accuracy"]

    for epoch in range(args.ft_epochs):
        model.train()
        total_loss, total_correct, total_tokens = 0, 0, 0
        for step, batch in enumerate(train_loader):
            images = batch["image"].to(device)
            questions = batch["question"]
            answers = batch["answer"].to(device)
            answer_ids = _answers_to_decoder_ids(answers, vocab)

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
        print(f"Epoch {epoch} | Val acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": {
                    k: v for k, v in model.state_dict().items()
                    if any(p.data_ptr() == v.data_ptr()
                           for p in model.parameters() if p.requires_grad)},
                "best_acc": best_acc,
            }, out_dir / "best.pt")
            print(f"  New best: {best_acc:.4f}")

    results = {"zero_shot": zs_results["accuracy"], "fine_tuned": best_acc}
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDone. Zero-shot: {zs_results['accuracy']:.4f}, "
          f"Fine-tuned best: {best_acc:.4f}")
    return results


def eval_dataset(model, dataset, device, vocab, task_type, batch_size=64):
    """Evaluate model on a dataset, return results dict."""
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=8, collate_fn=clevr_collate_fn,
        pin_memory=True,
    )
    if task_type in ("decoder", "mot"):
        return evaluate_decoder(model, loader, device, vocab)
    else:
        return evaluate_classification(model, loader, device)


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--data-root", type=str,
                        default="/home/jungchun/data/clevr")
    parser.add_argument("--skip-standard", action="store_true")
    parser.add_argument("--skip-closure", action="store_true")
    parser.add_argument("--skip-cogent", action="store_true")
    parser.add_argument("--skip-humans", action="store_true")
    parser.add_argument("--ablation", type=str, default=None,
                        choices=["text_only", "image_only"],
                        help="Ablation mode: zero one modality")
    parser.add_argument("--finetune", type=str, default=None,
                        choices=["clevr_humans"],
                        help="Fine-tune on dataset, then eval")
    parser.add_argument("--ft-lr", type=float, default=2e-5)
    parser.add_argument("--ft-epochs", type=int, default=10)
    args = parser.parse_args()
    tee_stdout(Path("outputs/analysis/generalization"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    print(f"Loading: {args.checkpoint}")
    model, transform, vocab, task_type, epoch, best_acc = \
        load_model(args.checkpoint, device)
    model_name = Path(args.checkpoint).parent.name
    print(f"Model: {model_name} (epoch {epoch}, best_acc={best_acc})")
    print(f"Task type: {task_type}")

    # Apply ablation wrapper
    if args.ablation == "text_only":
        model = TextOnlyWrapper(model)
        print("Ablation: TEXT-ONLY (visual features zeroed)")
    elif args.ablation == "image_only":
        model = ImageOnlyWrapper(model)
        print("Ablation: IMAGE-ONLY (text features zeroed)")
    print()

    # Fine-tune mode: run and exit
    if args.finetune == "clevr_humans":
        finetune_clevr_humans(model, transform, vocab, task_type, device, args)
        return

    data_root = Path(args.data_root)
    clevr_root = data_root / "CLEVR_v1.0"
    closure_root = data_root / "CLOSURE"
    cogent_root = data_root / "CLEVR_CoGenT_v1.0"

    results = {"model": model_name, "epoch": epoch, "best_acc": best_acc}

    # ── CLEVR Standard ──────────────────────────────────────────────
    if not args.skip_standard:
        print("=" * 60)
        print("CLEVR Standard (val)")
        print("=" * 60)
        ds = CLEVRVQADataset(str(clevr_root), "val", transform)
        r = eval_dataset(model, ds, device, vocab, task_type, args.batch_size)
        print(format_results(r))
        results["clevr_standard"] = r
        print()

    # ── CLEVR-CoGenT ────────────────────────────────────────────────
    if not args.skip_cogent and cogent_root.exists():
        print("=" * 60)
        print("CLEVR-CoGenT")
        print("=" * 60)

        cogent_results = {}
        for split in ["valA", "valB"]:
            q_file = cogent_root / "questions" / f"CLEVR_{split}_questions.json"
            img_dir = cogent_root / "images" / split
            if not q_file.exists():
                print(f"  {split}: skipped (not found)")
                continue
            ds = CLEVRVQADataset(
                str(cogent_root), split, transform,
                questions_file=str(q_file), image_dir=str(img_dir))
            r = eval_dataset(model, ds, device, vocab, task_type, args.batch_size)
            cogent_results[split] = r["accuracy"]
            print(f"  {split}: {r['accuracy']:.4f} ({r['total']})")

        if "valA" in cogent_results and "valB" in cogent_results:
            gap = cogent_results["valB"] - cogent_results["valA"]
            print(f"  Gap (B-A): {gap:+.4f}")
            cogent_results["gap"] = gap

        results["cogent"] = cogent_results
        print()

    # ── CLOSURE ─────────────────────────────────────────────────────
    if not args.skip_closure and closure_root.exists():
        print("=" * 60)
        print("CLOSURE")
        print("=" * 60)

        # CLOSURE uses CLEVR val images
        clevr_val_img_dir = clevr_root / "images" / "val"

        closure_types = [
            "and_mat_spa", "compare_mat", "compare_mat_spa",
            "embed_mat_spa", "embed_spa_mat", "or_mat", "or_mat_spa",
        ]

        closure_results = {}
        total_correct, total_count = 0, 0

        for ctype in closure_types:
            q_file = closure_root / f"{ctype}_val.json"
            if not q_file.exists():
                print(f"  {ctype}: skipped (not found)")
                continue
            ds = CLEVRVQADataset(
                str(clevr_root), "val", transform,
                questions_file=str(q_file), image_dir=str(clevr_val_img_dir))
            r = eval_dataset(model, ds, device, vocab, task_type, args.batch_size)
            closure_results[ctype] = r["accuracy"]
            total_correct += int(r["accuracy"] * r["total"])
            total_count += r["total"]
            print(f"  {ctype:20s} {r['accuracy']:.4f} ({r['total']})")

        if total_count > 0:
            overall = total_correct / total_count
            closure_results["overall"] = overall
            print(f"  {'Overall':20s} {overall:.4f} ({total_count})")

        results["closure"] = closure_results
        print()

    # ── CLEVR-Humans ──────────────────────────────────────────────
    humans_root = data_root / "CLEVR-Humans"
    if not args.skip_humans and humans_root.exists():
        print("=" * 60)
        print("CLEVR-Humans (val)")
        print("=" * 60)

        q_file = humans_root / "CLEVR-Humans-val.json"
        clevr_val_img = clevr_root / "images" / "val"
        ds = CLEVRVQADataset(
            str(clevr_root), "val", transform,
            questions_file=str(q_file), image_dir=str(clevr_val_img))
        r = eval_dataset(model, ds, device, vocab, task_type, args.batch_size)
        print(format_results(r))
        results["clevr_humans"] = r
        print()

    # ── Save results ────────────────────────────────────────────────
    out_dir = Path("outputs/analysis/generalization")
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.ablation}" if args.ablation else ""
    out_path = out_dir / f"{model_name}{suffix}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
