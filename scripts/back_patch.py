"""Back-patching experiment: re-inject later-layer GCA activations into earlier layers.

Tests whether later-layer GCA information helps if available earlier.
Runs on questions of a specified type that the model gets wrong.

Usage:
    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python scripts/back_patch.py \
        --checkpoint outputs/clevr_dinov2_decoder1l_scratch_s42/best.pt \
        --source-layer 11 --target-layer 7 \
        --qtype compare_integer --n-samples 50
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.amp import autocast
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from model import CrossAttnViT
from data.clevr import CLEVRVQADataset, clevr_collate_fn
from tasks.decoder import build_decoder_model, build_clevr_decoder_vocab
from omegaconf import OmegaConf


class GCAHook:
    """Hook to capture and optionally replace GCA output at a specific block."""

    def __init__(self, block_idx, mode="capture"):
        self.block_idx = block_idx
        self.mode = mode
        self.cached_output = None  # GCA contribution: tanh(gate) * CA(x, text)
        self._handle = None

    def hook_fn(self, module, input, output):
        if self.mode == "capture":
            x_before = input[0]
            self.cached_output = (output - x_before).detach().clone()
            return output
        elif self.mode == "inject":
            x_before = input[0]
            return x_before + self.cached_output

    def attach(self, model):
        blocks = model.steervit.vision_model.trunk.blocks
        gca = getattr(blocks[self.block_idx], "gated_cross_attn", None)
        if gca is None:
            raise ValueError(f"Block {self.block_idx} has no gated_cross_attn")
        self._handle = gca.register_forward_hook(self.hook_fn)

    def remove(self):
        if self._handle:
            self._handle.remove()


def find_samples(model, dataloader, device, vocab, qtype, n_correct=100, n_wrong=100):
    """Find both correct and wrong questions of the specified type."""
    model.eval()
    raw = model.module if hasattr(model, "module") else model
    inv_vocab = {v: k for k, v in vocab.items() if v >= 3}

    correct_indices, correct_info = [], []
    wrong_indices, wrong_info = [], []

    for batch_idx, batch in enumerate(dataloader):
        images = batch["image"].to(device)
        questions = batch["question"]
        gt_indices = batch["answer"]
        qtypes = batch["question_type"]

        with torch.no_grad(), autocast(device_type="cuda", dtype=torch.bfloat16):
            pred_texts = raw.generate(images, questions)

        gt_texts = [inv_vocab.get(i.item() + 3, "?") for i in gt_indices]

        for i, (pred, gt, qt) in enumerate(zip(pred_texts, gt_texts, qtypes)):
            if qt != qtype:
                continue
            global_idx = batch_idx * dataloader.batch_size + i
            info = {"idx": global_idx, "question": questions[i], "gt": gt, "pred": pred}

            if pred.strip().lower() == gt.strip().lower():
                if len(correct_indices) < n_correct:
                    correct_indices.append(global_idx)
                    correct_info.append(info)
            else:
                if len(wrong_indices) < n_wrong:
                    wrong_indices.append(global_idx)
                    wrong_info.append(info)

            if len(correct_indices) >= n_correct and len(wrong_indices) >= n_wrong:
                return correct_indices, correct_info, wrong_indices, wrong_info

    return correct_indices, correct_info, wrong_indices, wrong_info

    return wrong_indices, wrong_info


def run_patching(model, dataset, indices, device, vocab, source_layer, target_layer):
    """Run back-patching: inject source_layer GCA activation at target_layer."""
    model.eval()
    raw = model.module if hasattr(model, "module") else model
    inv_vocab = {v: k for k, v in vocab.items() if v >= 3}

    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=1, shuffle=False, collate_fn=clevr_collate_fn)

    source_hook = GCAHook(source_layer, mode="capture")
    target_hook = GCAHook(target_layer, mode="inject")

    results = []

    for batch in loader:
        images = batch["image"].to(device)
        questions = batch["question"]
        gt_idx = batch["answer"]
        gt_text = inv_vocab.get(gt_idx[0].item() + 3, "?")

        # Step 1: Clean run — capture source layer GCA output
        source_hook.mode = "capture"
        source_hook.attach(raw)
        with torch.no_grad(), autocast(device_type="cuda", dtype=torch.bfloat16):
            clean_pred = raw.generate(images, questions)[0]
        source_hook.remove()

        # Step 2: Patched run — inject source activation at target layer
        target_hook.cached_output = source_hook.cached_output
        target_hook.mode = "inject"
        target_hook.attach(raw)
        with torch.no_grad(), autocast(device_type="cuda", dtype=torch.bfloat16):
            patched_pred = raw.generate(images, questions)[0]
        target_hook.remove()

        clean_correct = clean_pred.strip().lower() == gt_text.strip().lower()
        patched_correct = patched_pred.strip().lower() == gt_text.strip().lower()

        results.append({
            "question": questions[0],
            "gt": gt_text,
            "clean_pred": clean_pred,
            "patched_pred": patched_pred,
            "clean_correct": clean_correct,
            "patched_correct": patched_correct,
            "flipped": not clean_correct and patched_correct,
        })

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--source-layer", type=int, default=11, help="Layer to capture GCA from")
    parser.add_argument("--target-layer", type=int, default=7, help="Layer to inject GCA into")
    parser.add_argument("--qtype", type=str, default="compare_integer",
                        choices=["compare_integer", "count", "exist", "query_attribute", "equal_attribute"],
                        help="Question type to evaluate")
    parser.add_argument("--n-samples", type=int, default=50)
    parser.add_argument("--data-root", type=str, default="/home/jungchun/data/clevr/CLEVR_v1.0")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Back-patching: L{args.source_layer} GCA → L{args.target_layer}")
    print(f"Question type: {args.qtype}")

    # Load checkpoint config
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = OmegaConf.create(ckpt["config"])

    # Build model
    cross_attn_layers = list(cfg.model.cross_attn_layers)
    steervit = CrossAttnViT.from_config(
        cfg.model.backbone_name, device=device,
        cross_attn_layers=cross_attn_layers,
        resolution=cfg.model.resolution,
    )
    vocab = build_clevr_decoder_vocab()
    model_cfg = OmegaConf.create({
        "model": cfg.model,
        "task": cfg.task,
        "data": cfg.data,
    })
    model = build_decoder_model(steervit, model_cfg).to(device)

    # Load weights
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    print(f"Loaded checkpoint (epoch {ckpt.get('epoch', '?')})")

    # Load data
    transform = steervit.get_transforms()
    dataset = CLEVRVQADataset(args.data_root, "val", transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        collate_fn=clevr_collate_fn, num_workers=4)

    # Find correct + wrong samples
    n_each = args.n_samples // 2
    print(f"\nFinding {n_each} correct + {n_each} wrong '{args.qtype}' questions...")
    correct_indices, correct_info, wrong_indices, wrong_info = find_samples(
        model, loader, device, vocab, qtype=args.qtype,
        n_correct=n_each, n_wrong=n_each,
    )
    print(f"Found {len(correct_indices)} correct, {len(wrong_indices)} wrong")

    if not correct_indices and not wrong_indices:
        print("No samples found!")
        return

    # Run patching on wrong samples
    print(f"\nPatching {len(wrong_indices)} WRONG samples (L{args.source_layer} → L{args.target_layer})...")
    wrong_results = run_patching(
        model, dataset, wrong_indices, device, vocab,
        source_layer=args.source_layer, target_layer=args.target_layer,
    )

    # Run patching on correct samples
    print(f"Patching {len(correct_indices)} CORRECT samples (L{args.source_layer} → L{args.target_layer})...")
    correct_results = run_patching(
        model, dataset, correct_indices, device, vocab,
        source_layer=args.source_layer, target_layer=args.target_layer,
    )

    # Report
    n_wrong_flipped = sum(r["flipped"] for r in wrong_results)  # wrong → correct
    n_correct_broken = sum(r["clean_correct"] and not r["patched_correct"] for r in correct_results)  # correct → wrong

    print(f"\n{'='*60}")
    print(f"BACK-PATCHING: L{args.source_layer} → L{args.target_layer} | qtype={args.qtype}")
    print(f"{'='*60}")
    print(f"\n  WRONG samples ({len(wrong_results)}):")
    print(f"    Fixed (wrong → correct): {n_wrong_flipped}/{len(wrong_results)} ({n_wrong_flipped/max(len(wrong_results),1)*100:.1f}%)")
    print(f"    Still wrong:             {len(wrong_results) - n_wrong_flipped}/{len(wrong_results)}")
    print(f"\n  CORRECT samples ({len(correct_results)}):")
    print(f"    Broken (correct → wrong): {n_correct_broken}/{len(correct_results)} ({n_correct_broken/max(len(correct_results),1)*100:.1f}%)")
    print(f"    Still correct:            {len(correct_results) - n_correct_broken}/{len(correct_results)}")
    print(f"\n  NET EFFECT: +{n_wrong_flipped} fixed, -{n_correct_broken} broken = net {n_wrong_flipped - n_correct_broken:+d}")
    print()

    # Show examples
    flipped = [r for r in wrong_results if r["flipped"]]
    if flipped:
        print("Examples fixed (wrong → correct):")
        for r in flipped[:5]:
            print(f"  Q: {r['question'][:80]}")
            print(f"     GT={r['gt']}, Clean={r['clean_pred']}, Patched={r['patched_pred']}")
        print()

    broken = [r for r in correct_results if r["clean_correct"] and not r["patched_correct"]]
    if broken:
        print("Examples broken (correct → wrong):")
        for r in broken[:5]:
            print(f"  Q: {r['question'][:80]}")
            print(f"     GT={r['gt']}, Clean={r['clean_pred']}, Patched={r['patched_pred']}")

    # Save
    output_path = Path(args.checkpoint).parent / f"back_patch_L{args.source_layer}_to_L{args.target_layer}_{args.qtype}.json"
    with open(output_path, "w") as f:
        json.dump({
            "config": {
                "source_layer": args.source_layer,
                "target_layer": args.target_layer,
                "qtype": args.qtype,
                "n_correct": len(correct_results),
                "n_wrong": len(wrong_results),
                "checkpoint": args.checkpoint,
            },
            "summary": {
                "wrong_fixed": n_wrong_flipped,
                "wrong_total": len(wrong_results),
                "fix_rate": n_wrong_flipped / max(len(wrong_results), 1),
                "correct_broken": n_correct_broken,
                "correct_total": len(correct_results),
                "break_rate": n_correct_broken / max(len(correct_results), 1),
                "net_effect": n_wrong_flipped - n_correct_broken,
            },
            "wrong_results": wrong_results,
            "correct_results": correct_results,
        }, f, indent=2)
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
