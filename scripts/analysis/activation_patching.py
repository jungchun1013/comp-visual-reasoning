"""Per-head activation patching: headwise SA + GCA denoising per corruption category.

Reference: SteerViT-legacy/exp_vqa/analysis/dinov2_gca/patching/run_headwise_by_type.py

For each corruption category, independently collect N samples, run per-head
patching (restore one clean head in corrupt run), produce SA + GCA heatmaps.

Categories:
  Fine Attribute:       color, material, size, shape
  Fine Attribute Query: what_color, what_material, what_size, what_shape

Output format compatible with legacy headwise_by_type_stats.json.

Usage:
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python scripts/analysis/activation_patching.py \
        --checkpoint outputs/model/clevr_dinov2_decoder1l_scratch_s42/best.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from model import CrossAttnViT
from data.clevr import CLEVRVQADataset
from tasks.decoder import build_clevr_decoder_vocab, VQADecoder, DecoderModel
from analysis.patching_utils import HeadPatcher
from analysis.run_log import tee_stdout
from analysis.patching_sampling import (
    CATEGORIES, build_corruption_index, collect_corruption_samples,
    collect_anchor_swap_samples,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


# ── Model loading (auto-detect legacy / main format) ──────────────

def load_model(ckpt_path, device):
    """Thin wrapper over model.checkpoint_io.load_any_checkpoint (L3 dedupe)."""
    from model.checkpoint_io import load_any_checkpoint

    model, steervit, transform, vocab, _task_type, meta = \
        load_any_checkpoint(ckpt_path, device)
    decoder = model.decoder
    decoder._head_type = "decoder"
    decoder._feature_pool = "cls"
    decoder._vocab_offset = 3
    print(f"Loaded: {meta['name']}")
    return steervit, decoder, vocab, transform


# ── Compute (same structure as legacy run_headwise_by_type.py) ────

def run_headwise_category(patcher, samples, device, direction="denoising",
                          visual_mode=False):
    """Run per-head SA + GCA patching on collected samples.

    For each sample:
      1. Call patcher.run_sa_denoising → (n_layers, sa_heads)
      2. Call patcher.run_gca_denoising → (n_gca, gca_heads)

    If visual_mode: samples are (clean_img, corrupt_img, question, answer_idx).
    Else (text mode): samples are (image, question, corrupt_question, answer_idx).

    Returns: sa_maps, gca_maps (lists of per-sample heatmaps)
    """
    sa_maps, gca_maps = [], []

    for i, sample in enumerate(samples):
        if visual_mode:
            clean_img, corrupt_img, question, answer_idx = sample
            token_id = patcher.to_token_id(answer_idx)
            clean_batch = clean_img.unsqueeze(0).to(device)
            corrupt_batch = corrupt_img.unsqueeze(0).to(device)
            # Visual corruption: different images, same question
            if direction == "denoising":
                sa_hm, _, _ = patcher.run_sa_denoising(
                    clean_batch, [question], corrupt_batch, [question], token_id, device)
                gca_hm, _, _ = patcher.run_gca_denoising(
                    clean_batch, [question], corrupt_batch, [question], token_id, device)
            else:
                sa_hm, _, _ = patcher.run_sa_denoising(
                    corrupt_batch, [question], clean_batch, [question], token_id, device)
                gca_hm, _, _ = patcher.run_gca_denoising(
                    corrupt_batch, [question], clean_batch, [question], token_id, device)
        else:
            image, question, corrupt_q, answer_idx = sample
            token_id = patcher.to_token_id(answer_idx)
            img_batch = image.unsqueeze(0).to(device)
            # Text corruption: same image, different questions
            if direction == "denoising":
                sa_hm, _, _ = patcher.run_sa_denoising(
                    img_batch, [question], img_batch, [corrupt_q], token_id, device)
                gca_hm, _, _ = patcher.run_gca_denoising(
                    img_batch, [question], img_batch, [corrupt_q], token_id, device)
            else:
                sa_hm, _, _ = patcher.run_sa_denoising(
                    img_batch, [corrupt_q], img_batch, [question], token_id, device)
                gca_hm, _, _ = patcher.run_gca_denoising(
                    img_batch, [corrupt_q], img_batch, [question], token_id, device)

        sa_maps.append(sa_hm)
        gca_maps.append(gca_hm)

        if (i + 1) % 10 == 0:
            print(f"    {i+1}/{len(samples)}", flush=True)

    return sa_maps, gca_maps


def compute(patcher, dataset, args, device):
    """For each category × direction, run per-head patching.

    Output format matches legacy headwise_by_type_stats.json.
    """
    corruption_index = build_corruption_index(dataset)

    stats = {
        "gca_layers": patcher.gca_layers,
        "sa_num_heads": patcher.sa_num_heads,
        "gca_num_heads": patcher.gca_num_heads,
        "num_samples_per_category": args.num_samples,
    }

    run_groups = set(args.groups.split(",")) if args.groups != "all" else {
        "fine_attribute", "fine_attribute_query"}
    run_directions = args.directions.split(",")

    for group, category_key in CATEGORIES:
        if group not in run_groups:
            continue

        print(f"\n--- {group}/{category_key} ---", flush=True)
        if group == "anchor_swap":
            samples = collect_anchor_swap_samples(dataset, args.num_samples)
        else:
            samples = collect_corruption_samples(
                dataset, corruption_index, category_key, args.num_samples)
        print(f"  Sampled {len(samples)}/{args.num_samples}", flush=True)
        if not samples:
            continue

        for direction in run_directions:
            print(f"  {direction}...", flush=True)
            sa_maps, gca_maps = run_headwise_category(
                patcher, samples, device, direction)

            key = f"{group}_{direction}"
            if key not in stats:
                stats[key] = {}
            stats[key][category_key] = {
                "sa_mean": np.mean(sa_maps, axis=0).tolist(),
                "gca_mean": np.mean(gca_maps, axis=0).tolist(),
                "sa_std": np.std(sa_maps, axis=0).tolist(),
                "gca_std": np.std(gca_maps, axis=0).tolist(),
                "n": len(samples),
            }

        # Incremental save
        output_dir = Path(args.output_dir)
        stats_path = output_dir / "headwise_by_type_stats.json"
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"  Saved: {stats_path}", flush=True)

    return stats


# ── Plot (same layout as legacy) ─────────────────────────────────

def plot(stats, output_dir):
    """Generate interleaved SA+GCA heatmaps per group × direction."""
    gca_layers = stats["gca_layers"]
    sa_heads = stats["sa_num_heads"]
    gca_heads = stats["gca_num_heads"]
    gca_set = set(gca_layers)
    max_heads = max(sa_heads, gca_heads)

    # Build column layout: interleave GCA and SA
    col_labels, col_is_gca = [], []
    for l in range(12):
        if l in gca_set:
            col_labels.append(f"GCA{l}")
            col_is_gca.append(True)
        col_labels.append(f"SA{l}")
        col_is_gca.append(False)
    n_cols = len(col_labels)

    def build_combined(sa_hm, gca_hm):
        combined = np.full((n_cols, max_heads), np.nan)
        gca_idx, ci = 0, 0
        for l in range(12):
            if l in gca_set:
                combined[ci, :gca_heads] = gca_hm[gca_idx]
                gca_idx += 1
                ci += 1
            combined[ci, :sa_heads] = sa_hm[l]
            ci += 1
        return combined

    _ORDER = {
        "fine_attribute": ["color", "material", "size", "shape"],
        "fine_attribute_query": ["what_color", "what_material", "what_size", "what_shape"],
        "anchor_swap": ["anchor_swap"],
    }

    for group in _ORDER:
        for direction in ("denoising", "noising"):
            key = f"{group}_{direction}"
            if key not in stats:
                continue

            available = stats[key]
            categories = [c for c in _ORDER[group] if c in available]
            if not categories:
                continue

            n = len(categories)
            cols = min(n, 2)
            rows = (n + cols - 1) // cols
            fig, axes = plt.subplots(rows, cols, figsize=(10 * cols, 5.5 * rows),
                                     constrained_layout=True)
            if rows == 1 and cols == 1:
                axes = np.array([[axes]])
            elif rows == 1:
                axes = axes[np.newaxis, :]
            elif cols == 1:
                axes = axes[:, np.newaxis]

            all_combined = {}
            for cat in categories:
                d = available[cat]
                all_combined[cat] = build_combined(
                    np.array(d["sa_mean"]), np.array(d["gca_mean"]))
            vmax = max(np.nanmax(np.abs(c)) for c in all_combined.values())

            for i, cat in enumerate(categories):
                ax = axes[i // cols, i % cols]
                hm = all_combined[cat]
                masked = np.ma.array(hm, mask=np.isnan(hm))
                im = ax.imshow(masked.T, aspect="auto", cmap="RdBu_r",
                               vmin=-vmax, vmax=vmax, origin="lower")
                ax.set_xlabel("Module")
                ax.set_ylabel("Head Index")
                n_samples = available[cat]["n"]
                ax.set_title(f"{cat} (n={n_samples})")
                ax.set_xticks(range(n_cols))
                ax.set_xticklabels(col_labels, rotation=60, ha="right", fontsize=7)
                ax.set_yticks(range(max_heads))
                for ci_idx in range(n_cols):
                    if col_is_gca[ci_idx]:
                        rect = Rectangle((ci_idx - 0.5, -0.5), 1, max_heads,
                                         linewidth=1.5, edgecolor="red",
                                         facecolor="none", linestyle="--")
                        ax.add_patch(rect)
                plt.colorbar(im, ax=ax, label=r"$\Delta$ logit")

            for i in range(len(categories), rows * cols):
                axes[i // cols, i % cols].axis("off")

            direction_label = "Denoising" if direction == "denoising" else "Noising"
            fig.suptitle(f"Per-Head {direction_label} — {group}")
            out_path = Path(output_dir) / f"headwise_{group}_{direction}.png"
            fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
            print(f"Saved: {out_path}")
            plt.close(fig)


# ── Summary ──────────────────────────────────────────────────────

def print_top_heads(stats):
    """Print top heads per category."""
    gca_layers = stats["gca_layers"]

    for group_dir_key, data in stats.items():
        if not isinstance(data, dict) or "gca_layers" in group_dir_key:
            continue
        # Skip metadata keys
        if group_dir_key in ("gca_layers", "sa_num_heads", "gca_num_heads",
                              "num_samples_per_category"):
            continue

        print(f"\n=== {group_dir_key} ===")
        for cat, cat_data in data.items():
            sa = np.array(cat_data["sa_mean"])
            gca = np.array(cat_data["gca_mean"])
            heads = []
            for l in range(sa.shape[0]):
                for h in range(sa.shape[1]):
                    heads.append(("SA", l, h, sa[l, h]))
            for gi, l in enumerate(gca_layers):
                for h in range(gca.shape[1]):
                    heads.append(("GCA", l, h, gca[gi, h]))
            heads.sort(key=lambda x: abs(x[3]), reverse=True)
            top5 = heads[:5]
            top_str = "  ".join(f"{t}_L{l}_H{h}={v:+.3f}" for t, l, h, v in top5)
            print(f"  {cat} (n={cat_data['n']}): {top_str}")


# ── Main ──────────────────────────────────────────────────────────

def main():
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-root", type=str,
                        default="/home/jungchun/data/clevr/CLEVR_v1.0")
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--groups", type=str, default="fine_attribute,fine_attribute_query",
                        help="Comma-separated: fine_attribute,fine_attribute_query (or 'all')")
    parser.add_argument("--directions", type=str, default="denoising",
                        help="Comma-separated: denoising,noising")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--replot", action="store_true",
                        help="Only re-plot from existing stats JSON")
    parser.add_argument("--compute-only", action="store_true",
                        help="Only compute, skip plotting")
    parser.add_argument("--visual-pairs", type=str, default=None,
                        help="Path to visual corruption pairs.json. "
                             "When set, runs visual corruption patching instead of text.")
    args = parser.parse_args()

    ckpt_dir = Path(args.checkpoint).parent
    model_name = ckpt_dir.name.replace("_s42", "")
    output_dir = Path(args.output_dir) if args.output_dir else \
        Path("outputs/analysis/activation_patching") / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir = str(output_dir)
    tee_stdout(output_dir)

    stats_path = output_dir / "headwise_by_type_stats.json"

    if not args.replot:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        steervit, decoder, vocab, transform = load_model(args.checkpoint, device)
        patcher = HeadPatcher(steervit, decoder, vocab)
        print(f"SA: {patcher.sa_num_heads} heads, GCA: {patcher.gca_num_heads} heads")
        print(f"GCA layers: {patcher.gca_layers}")

        dataset = CLEVRVQADataset(args.data_root, "val", transform)

        if args.visual_pairs:
            # Visual corruption mode
            from analysis.patching_sampling import collect_visual_corruption_samples
            print(f"\nVisual corruption mode: {args.visual_pairs}")
            samples = collect_visual_corruption_samples(
                dataset, args.visual_pairs, args.num_samples, transform)
            print(f"Loaded {len(samples)} visual corruption samples")

            stats = {
                "gca_layers": patcher.gca_layers,
                "sa_num_heads": patcher.sa_num_heads,
                "gca_num_heads": patcher.gca_num_heads,
                "num_samples_per_category": len(samples),
                "corruption_source": "visual",
            }
            # Determine corruption type from pairs JSON filename
            import json as _json
            with open(args.visual_pairs) as _f:
                _pairs = _json.load(_f)
            ctype = _pairs[0].get("corruption_type", "color") if _pairs else "color"

            for direction in args.directions.split(","):
                print(f"\n--- visual_{ctype} {direction} ---", flush=True)
                sa_maps, gca_maps = run_headwise_category(
                    patcher, samples, device, direction, visual_mode=True)
                key = f"visual_{ctype}_{direction}"
                stats[key] = {
                    ctype: {
                        "sa_mean": np.mean(sa_maps, axis=0).tolist(),
                        "gca_mean": np.mean(gca_maps, axis=0).tolist(),
                        "sa_std": np.std(sa_maps, axis=0).tolist(),
                        "gca_std": np.std(gca_maps, axis=0).tolist(),
                        "n": len(samples),
                    }
                }
            output_dir = Path(args.output_dir) if args.output_dir else \
                Path("outputs/analysis/activation_patching") / model_name
            output_dir.mkdir(parents=True, exist_ok=True)
            stats_path = output_dir / f"visual_{ctype}_stats.json"
        else:
            stats = compute(patcher, dataset, args, device)

        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"\nSaved: {stats_path}")

    if not args.compute_only:
        if args.replot:
            with open(stats_path) as f:
                stats = json.load(f)
        plot(stats, output_dir)

    print_top_heads(stats)
    print("\nDone.")


# torch import at top level for load_model
import torch

if __name__ == "__main__":
    main()
