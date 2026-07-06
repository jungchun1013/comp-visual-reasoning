"""Retrieval visualization: query image + top-K per GCA layer.

Reads .npz cache from tsne_steered.py and performs cosine similarity
retrieval at each layer.

Layout: rows = GCA layers, cols = Query | Top-1 | Top-2 | Top-3

Usage:
    PYTHONPATH=src python scripts/analysis/retrieval_viz.py \
        --cache outputs/analysis/tsne/clevr_siglip_decoder1l_scratch/cache_q0.npz
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from analysis.plot_style import PLOT_STYLE, apply_style

# Intentional overrides vs PLOT_STYLE: the image-grid retrieval figure uses
# smaller text throughout (12/14/16 vs 14/16/18/20) so labels fit the thumbnails.
S = dict(PLOT_STYLE, tick_labelsize=12, label_fontsize=14, legend_fontsize=14,
         subplot_title_fontsize=16, suptitle_fontsize=16)


def load_image(path, width=150):
    """Load image, resize keeping aspect ratio."""
    img = Image.open(path).convert("RGB")
    w, h = img.size
    new_h = int(h * width / w)
    img = img.resize((width, new_h), Image.LANCZOS)
    return np.array(img)


def retrieve_top_k(query_feat, db_feats, k=3):
    """Cosine similarity retrieval."""
    q = F.normalize(torch.from_numpy(query_feat).unsqueeze(0).float(), dim=-1)
    db = F.normalize(torch.from_numpy(db_feats).float(), dim=-1)
    sim = (q @ db.T).squeeze(0)
    scores, indices = sim.topk(k)
    return indices.tolist(), scores.tolist()


def plot_grid(q_img, question, answer, layers, topk_images, topk_scores,
              top_k, out_path, gca_layers=None):
    """Plot retrieval grid: rows=layers, cols=Query|Top-1|...|Top-K."""
    n_layers = len(layers)
    if gca_layers is None:
        gca_layers = set(layers)

    fig = plt.figure(figsize=(2.2 * (top_k + 1) + 1, n_layers * 1.6 + 2.0))
    gs = gridspec.GridSpec(
        n_layers + 1, 2 + top_k,
        figure=fig,
        width_ratios=[1, 0.12] + [1] * top_k,
        hspace=0.08, wspace=0.08,
        top=0.90, bottom=0.01, left=0.08, right=0.98,
    )

    # Title
    q_short = question[:65] + "..." if len(question) > 65 else question
    fig.suptitle(
        f"\"{q_short}\" \u2192 {answer}",
        fontsize=S["suptitle_fontsize"], fontweight="bold", y=0.97, va="top",
    )

    # Column headers
    ax_qh = fig.add_subplot(gs[0, 0])
    ax_qh.text(0.5, 0.5, "Query", ha="center", va="center",
               fontsize=S["label_fontsize"], fontweight="bold")
    ax_qh.axis("off")
    fig.add_subplot(gs[0, 1]).axis("off")
    for ri in range(top_k):
        ax_rh = fig.add_subplot(gs[0, 2 + ri])
        ax_rh.text(0.5, 0.5, f"Top-{ri+1}", ha="center", va="center",
                   fontsize=S["label_fontsize"], fontweight="bold")
        ax_rh.axis("off")

    # Image rows
    for row_i, l in enumerate(layers):
        row = row_i + 1
        is_gca = l in gca_layers

        # Query image
        ax_q = fig.add_subplot(gs[row, 0])
        if q_img is not None:
            ax_q.imshow(q_img)
        ax_q.axis("off")
        ax_q.text(-0.2, 0.5, f"L{l}", ha="right", va="center",
                  transform=ax_q.transAxes,
                  fontsize=S["label_fontsize"], fontweight="bold",
                  color="black" if is_gca else "#888888")

        # Gap
        fig.add_subplot(gs[row, 1]).axis("off")

        # Top-K
        imgs = topk_images.get(l, [])
        scores = topk_scores.get(l, [])
        for ci in range(top_k):
            ax_r = fig.add_subplot(gs[row, 2 + ci])
            if ci < len(imgs):
                ax_r.imshow(imgs[ci])
            ax_r.axis("off")
            if ci < len(scores):
                ax_r.text(0.5, -0.05, f"{scores[ci]:.3f}",
                          ha="center", va="top", transform=ax_r.transAxes,
                          fontsize=S["tick_labelsize"] - 2, color="gray")
            if is_gca:
                for spine in ax_r.spines.values():
                    spine.set_visible(True)
                    spine.set_color("#1f77b4")
                    spine.set_linewidth(2)

    fig.savefig(str(out_path), dpi=S["dpi"], bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", type=str, required=True,
                   help="Path to cache .npz from tsne_steered.py")
    p.add_argument("--data-root", type=str,
                   default="/home/jungchun/data/clevr/CLEVR_v1.0")
    p.add_argument("--output", type=str, default=None)
    p.add_argument("--top-k", type=int, default=3)
    args = p.parse_args()

    apply_style()

    cache_path = Path(args.cache)
    cached = np.load(str(cache_path), allow_pickle=True)

    gca_layers = cached["gca_layers"].tolist()
    db_indices = cached["db_indices"].tolist()
    question = str(cached["question"])
    answer = str(cached["answer"])

    # Load CLEVR questions for image filenames
    q_path = Path(args.data_root) / "questions" / "CLEVR_val_questions.json"
    with open(q_path) as f:
        all_questions = json.load(f)["questions"]
    image_dir = Path(args.data_root) / "images" / "val"

    # Find query image
    query_fname = None
    for q in all_questions:
        if q["question"] == question:
            query_fname = q["image_filename"]
            break
    q_img = load_image(image_dir / query_fname) if query_fname else None

    # Retrieve + load images per layer
    topk_images = {}
    topk_scores = {}
    for l in gca_layers:
        query_feat = cached[f"query_feat_{l}"]
        db_feats = cached[f"feat_{l}"]
        indices, scores = retrieve_top_k(query_feat, db_feats, args.top_k)
        topk_scores[l] = scores
        topk_images[l] = []
        for db_local_idx in indices:
            dataset_idx = db_indices[db_local_idx]
            fname = all_questions[dataset_idx]["image_filename"]
            topk_images[l].append(load_image(image_dir / fname))

    if args.output:
        out_path = Path(args.output)
    else:
        out_path = cache_path.parent / f"retrieval_{cache_path.stem.replace('cache_', '')}.png"

    plot_grid(q_img, question, answer, gca_layers, topk_images, topk_scores,
              args.top_k, out_path, gca_layers=set(gca_layers))


if __name__ == "__main__":
    main()
