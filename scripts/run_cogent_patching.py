"""Head-level activation patching for CoGenT before/after comparison.

Runs two experiments on a given checkpoint:
1. Described attr: per-head denoising for color/material/size/shape corruptions
2. Query attr: per-head denoising for What color→What shape swaps

Both use HeadPatcher (SA + GCA head-level patching).

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/run_cogent_patching.py \
        --checkpoint outputs/model/cogent_dinov2_decoder1l_scratch_s42/best.pt \
        --output-dir outputs/analysis/cogent_patching/before_ft
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from model import CrossAttnViT
from data.clevr import CLEVRVQADataset
from tasks.decoder import build_clevr_decoder_vocab, VQADecoder, DecoderModel
from analysis.patching_utils import HeadPatcher
from analysis.patching_sampling import (
    CATEGORIES, build_corruption_index, collect_corruption_samples,
)

# ── Query attr detection (from legacy run_query_attr_patching.py) ──

QUERY_PATTERNS = [
    (re.compile(r"\bwhat is the (color|shape|material|size) of\b", re.I), 1),
    (re.compile(r"\bwhat (color|shape|material|size) is\b", re.I), 1),
    (re.compile(r"\bhow big is\b", re.I), None),
    (re.compile(r"\bis the (color|shape|material|size) of\b", re.I), 1),
]
ALL_ATTRS = ["color", "shape", "material", "size"]
SWAP_TEMPLATES = {
    "what is the {attr} of": {
        "pattern": re.compile(r"\bwhat is the (color|shape|material|size) of\b", re.I),
        "replace": lambda m, new: f"What is the {new} of",
    },
    "what {attr} is": {
        "pattern": re.compile(r"\bwhat (color|shape|material|size) is\b", re.I),
        "replace": lambda m, new: f"What {new} is",
    },
    "how big is": {
        "pattern": re.compile(r"\bhow big is\b", re.I),
        "replace": lambda m, new: f"What {new} is",
    },
    "is the {attr} of": {
        "pattern": re.compile(r"\bis the (color|shape|material|size) of\b", re.I),
        "replace": lambda m, new: f"Is the {new} of",
    },
}


def detect_query_attr(question):
    for pattern, group in QUERY_PATTERNS:
        m = pattern.search(question)
        if m:
            return "size" if group is None else m.group(group).lower()
    return None


def swap_query_attr(question, original_attr, new_attr):
    for tmpl in SWAP_TEMPLATES.values():
        m = tmpl["pattern"].search(question)
        if m:
            replacement = tmpl["replace"](m, new_attr)
            return question[:m.start()] + replacement + question[m.end():]
    return None


def generate_query_attr_corruptions(question):
    orig_attr = detect_query_attr(question)
    if orig_attr is None:
        return []
    results = []
    for new_attr in ALL_ATTRS:
        if new_attr == orig_attr:
            continue
        corrupted = swap_query_attr(question, orig_attr, new_attr)
        if corrupted and corrupted != question:
            results.append({
                "original_attr": orig_attr,
                "new_attr": new_attr,
                "corrupted_question": corrupted,
            })
    return results


# ── Model loading ──

def load_model(ckpt_path, device):
    """Thin wrapper over model.checkpoint_io.load_any_checkpoint (L3 dedupe)."""
    from model.checkpoint_io import load_any_checkpoint

    model, steervit, transform, vocab, _task_type, meta = \
        load_any_checkpoint(ckpt_path, device)
    decoder = model.decoder
    decoder._head_type = "decoder"
    decoder._feature_pool = "cls"
    decoder._vocab_offset = 3
    print(f"Loaded: {meta['name']} (epoch {meta['epoch']})")
    return steervit, decoder, vocab, transform


# ── Exp 1: Described attr head patching ──
# Ported from legacy run_headwise_by_type.py

def run_headwise_category(patcher, samples, device):
    """Per-head SA + GCA denoising on collected samples.

    Returns:
        sa_maps: list of (n_layers, sa_heads) arrays
        gca_maps: list of (n_gca, gca_heads) arrays
    """
    sa_maps, gca_maps = [], []

    for i, (image, question, corrupt_q, answer_idx) in enumerate(samples):
        token_id = patcher.to_token_id(answer_idx)
        img_batch = image.unsqueeze(0).to(device)

        sa_hm, _, _ = patcher.run_sa_denoising(
            img_batch, [question], img_batch, [corrupt_q], token_id, device)
        gca_hm, _, _ = patcher.run_gca_denoising(
            img_batch, [question], img_batch, [corrupt_q], token_id, device)

        sa_maps.append(sa_hm)
        gca_maps.append(gca_hm)

        if (i + 1) % 10 == 0:
            print(f"    {i+1}/{len(samples)}", flush=True)

    return sa_maps, gca_maps


def run_described_attr(patcher, dataset, num_samples, device):
    """Described attr head-level patching: fine_attribute group only."""
    eligible_index = build_corruption_index(dataset)

    stats = {
        "gca_layers": patcher.gca_layers,
        "sa_num_heads": patcher.sa_num_heads,
        "gca_num_heads": patcher.gca_num_heads,
        "num_samples_per_category": num_samples,
    }

    # Only fine_attribute categories
    fine_categories = [(g, k) for g, k in CATEGORIES if g == "fine_attribute"]

    for group, category_key in fine_categories:
        print(f"\n  --- {category_key} ---", flush=True)
        samples = collect_corruption_samples(dataset, eligible_index,
                                             category_key, num_samples)
        print(f"  Sampled {len(samples)}/{num_samples}", flush=True)
        if not samples:
            continue

        sa_maps, gca_maps = run_headwise_category(patcher, samples, device)

        key = f"fine_attribute_denoising"
        if key not in stats:
            stats[key] = {}
        stats[key][category_key] = {
            "sa_mean": np.mean(sa_maps, axis=0).tolist(),
            "gca_mean": np.mean(gca_maps, axis=0).tolist(),
            "sa_std": np.std(sa_maps, axis=0).tolist(),
            "gca_std": np.std(gca_maps, axis=0).tolist(),
            "n": len(samples),
        }

    return stats


# ── Exp 2: Query attr head patching ──
# Ported from legacy run_query_attr_patching.py

def run_query_attr(patcher, dataset, num_samples, device):
    """Query attr head-level patching: 30 samples per attr, 3 swaps averaged."""
    VOCAB_OFFSET = 3
    sub_sa, sub_gca = {}, {}
    sub_counts = {a: 0 for a in ALL_ATTRS}

    for idx in range(len(dataset)):
        if all(sub_counts[a] >= num_samples for a in ALL_ATTRS):
            break

        sample = dataset[idx]
        question = sample["question"]
        answer_idx = sample["answer"]
        correct_token_id = answer_idx + VOCAB_OFFSET

        corrs = generate_query_attr_corruptions(question)
        if not corrs:
            continue

        orig_attr = corrs[0]["original_attr"]
        if sub_counts[orig_attr] >= num_samples:
            continue

        img_batch = sample["image"].unsqueeze(0).to(device)

        # Average across all 3 swap targets
        sa_accum, gca_accum = [], []
        for corr in corrs:
            sa_hm, _, _ = patcher.run_sa_denoising(
                img_batch, [question],
                img_batch, [corr["corrupted_question"]],
                correct_token_id, device,
            )
            gca_hm, _, _ = patcher.run_gca_denoising(
                img_batch, [question],
                img_batch, [corr["corrupted_question"]],
                correct_token_id, device,
            )
            sa_accum.append(sa_hm)
            gca_accum.append(gca_hm)

        if orig_attr not in sub_sa:
            sub_sa[orig_attr] = []
            sub_gca[orig_attr] = []
        sub_sa[orig_attr].append(np.mean(sa_accum, axis=0))
        sub_gca[orig_attr].append(np.mean(gca_accum, axis=0))
        sub_counts[orig_attr] += 1

        total = sum(sub_counts.values())
        if total % 10 == 0:
            counts_str = " ".join(f"{a}={sub_counts[a]}" for a in ALL_ATTRS)
            print(f"  {counts_str}", flush=True)

    # Build stats
    stats = {
        "gca_layers": patcher.gca_layers,
        "sa_num_heads": patcher.sa_num_heads,
        "gca_num_heads": patcher.gca_num_heads,
        "counts": sub_counts,
    }
    for attr in ALL_ATTRS:
        if attr in sub_sa:
            stats[f"{attr}_sa"] = np.mean(sub_sa[attr], axis=0).tolist()
            stats[f"{attr}_gca"] = np.mean(sub_gca[attr], axis=0).tolist()

    # Print summary
    for attr in ALL_ATTRS:
        if f"{attr}_sa" not in stats:
            continue
        sa = np.array(stats[f"{attr}_sa"])
        gca = np.array(stats[f"{attr}_gca"])
        print(f"\n  query={attr} (n={sub_counts[attr]})")
        flat_sa = sorted([(sa[l, h], l, h) for l in range(sa.shape[0]) for h in range(sa.shape[1])],
                         key=lambda x: abs(x[0]), reverse=True)
        print(f"    SA top-5:  " + "  ".join(f"L{l}.H{h}={v:+.2f}" for v, l, h in flat_sa[:5]))
        gca_layers = stats["gca_layers"]
        flat_gca = sorted([(gca[gi, h], gca_layers[gi], h)
                           for gi in range(gca.shape[0]) for h in range(gca.shape[1])],
                          key=lambda x: abs(x[0]), reverse=True)
        print(f"    GCA top-5: " + "  ".join(f"L{l}.H{h}={v:+.2f}" for v, l, h in flat_gca[:5]))

    return stats


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--data-root", type=str, default="/home/jungchun/data/clevr/CLEVR_v1.0")
    p.add_argument("--questions-file", type=str, default=None,
                   help="Override questions JSON (e.g. CoGenT valA/valB)")
    p.add_argument("--image-dir", type=str, default=None,
                   help="Override image directory")
    p.add_argument("--num-described-samples", type=int, default=50)
    p.add_argument("--num-query-samples", type=int, default=30)
    p.add_argument("--output-dir", type=str, required=True)
    args = p.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    steervit, decoder, vocab, transform = load_model(args.checkpoint, device)
    patcher = HeadPatcher(steervit, decoder, vocab)
    print(f"GCA layers: {patcher.gca_layers}")
    print(f"SA heads: {patcher.sa_num_heads}, GCA heads: {patcher.gca_num_heads}")

    dataset = CLEVRVQADataset(
        args.data_root, "val", transform,
        questions_file=args.questions_file,
        image_dir=args.image_dir)

    # ── Exp 1: Described attr ──
    print("\n" + "=" * 60)
    print("EXP 1: DESCRIBED ATTR HEAD PATCHING")
    print("=" * 60)
    described_stats = run_described_attr(patcher, dataset, args.num_described_samples, device)
    with open(output_dir / "described_attr_stats.json", "w") as f:
        json.dump(described_stats, f, indent=2)
    print(f"\nSaved: {output_dir}/described_attr_stats.json")

    # ── Exp 2: Query attr ──
    print("\n" + "=" * 60)
    print("EXP 2: QUERY ATTR HEAD PATCHING")
    print("=" * 60)
    query_stats = run_query_attr(patcher, dataset, args.num_query_samples, device)
    with open(output_dir / "query_attr_stats.json", "w") as f:
        json.dump(query_stats, f, indent=2)
    print(f"\nSaved: {output_dir}/query_attr_stats.json")

    print("\nDone.")


if __name__ == "__main__":
    main()
