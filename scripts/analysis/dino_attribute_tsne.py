"""DINOv2 single-object attribute t-SNE visualization.

Extracts CLS features from pretrained DINOv2 (no GCA, no fine-tune)
for rendered single-object CLEVR images, then plots t-SNE colored by
each of the 4 attributes (color, material, shape, size).

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/analysis/dino_attribute_tsne.py \
        --image-dir outputs/analysis/single_objects/images \
        --metadata outputs/analysis/single_objects/metadata.json \
        --output outputs/analysis/single_objects/dino_attribute_tsne.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import timm
import torch
from PIL import Image
from sklearn.manifold import TSNE
from timm.data import resolve_data_config
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from analysis.plot_style import PLOT_STYLE, apply_style

# ── Plot style ──────────────────────────────────────────────────

# Intentional override vs PLOT_STYLE: smaller legend (14 vs 16) for the
# dense per-attribute scatter legends (matches tsne_viz.py).
S = dict(PLOT_STYLE, legend_fontsize=14)


# ── CLEVR attribute colors for plotting ─────────────────────────

# Use actual CLEVR RGB values for the color attribute plot
CLEVR_PLOT_COLORS = {
    "gray":   (87/255, 87/255, 87/255),
    "red":    (173/255, 35/255, 35/255),
    "blue":   (42/255, 75/255, 215/255),
    "green":  (29/255, 105/255, 20/255),
    "brown":  (129/255, 74/255, 25/255),
    "purple": (129/255, 38/255, 192/255),
    "cyan":   (41/255, 208/255, 208/255),
    "yellow": (255/255, 238/255, 51/255),
}

_tab10 = plt.cm.tab10.colors
MATERIAL_COLORS = {"metal": _tab10[0], "rubber": _tab10[1]}
SHAPE_COLORS = {"cube": _tab10[0], "sphere": _tab10[1], "cylinder": _tab10[2]}
SIZE_COLORS = {"large": _tab10[0], "small": _tab10[1]}


# ── Feature extraction ──────────────────────────────────────────

def extract_features(image_dir: Path, filenames: list[str],
                     batch_size: int = 64, device: str = "cuda"):
    """Extract CLS features from pretrained DINOv2."""
    model = timm.create_model("vit_base_patch14_dinov2.lvd142m",
                              pretrained=True, num_classes=0)
    model = model.eval().to(device)

    data_config = resolve_data_config({}, model=model)
    img_size = data_config["input_size"][-1]  # native resolution (518 for DINOv2)
    transform = transforms.Compose([
        transforms.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        transforms.Resize((img_size, img_size),
                          interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=data_config["mean"], std=data_config["std"]),
    ])

    all_feats = []
    for start in range(0, len(filenames), batch_size):
        batch_files = filenames[start:start + batch_size]
        imgs = []
        for fname in batch_files:
            img = Image.open(image_dir / fname)
            imgs.append(transform(img))
        batch = torch.stack(imgs).to(device)

        with torch.no_grad(), torch.amp.autocast("cuda"):
            feats = model(batch)  # (B, 768)
        all_feats.append(feats.cpu().numpy())

        if (start // batch_size) % 10 == 0:
            print(f"  [{start + len(batch_files)}/{len(filenames)}] extracted")

    return np.concatenate(all_feats, axis=0)


# ── Plotting ────────────────────────────────────────────────────

def plot_attribute_tsne(emb: np.ndarray, metadata: list[dict],
                        output_path: Path):
    """Draw 2×2 t-SNE subplots, each colored by one attribute."""
    apply_style()

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.ravel()

    attr_configs = [
        ("color",    CLEVR_PLOT_COLORS, ["gray", "red", "blue", "green",
                                          "brown", "purple", "cyan", "yellow"]),
        ("shape",    SHAPE_COLORS,      ["cube", "sphere", "cylinder"]),
        ("material", MATERIAL_COLORS,   ["metal", "rubber"]),
        ("size",     SIZE_COLORS,       ["large", "small"]),
    ]

    labels = {attr: [m[attr] for m in metadata]
              for attr, _, _ in attr_configs}

    for ax, (attr, cmap, order) in zip(axes, attr_configs):
        for val in order:
            mask = np.array([l == val for l in labels[attr]])
            ax.scatter(emb[mask, 0], emb[mask, 1],
                       c=[cmap[val]], s=8, alpha=0.6, label=val,
                       edgecolors="none", rasterized=True)

        ax.set_title(attr.capitalize(), fontsize=S["subplot_title_fontsize"])
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)

        ax.legend(loc="best", fontsize=S["legend_fontsize"],
                  frameon=False, markerscale=3)

    fig.suptitle("DINOv2 (pretrained, no fine-tune) — Single Object Representations",
                 fontsize=S["subplot_title_fontsize"] + 2)
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=S["dpi"], bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close(fig)


# ── Main ────────────────────────────────────────────────────────

def main():
    apply_style()

    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--output", default="outputs/analysis/single_objects/dino_attribute_tsne.png")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--perplexity", type=float, default=30)
    parser.add_argument("--tsne-seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load metadata
    with open(args.metadata) as f:
        metadata = json.load(f)
    filenames = [m["filename"] for m in metadata]
    print(f"Loaded {len(metadata)} entries from {args.metadata}")

    # Extract features
    print("Extracting DINOv2 CLS features...")
    feats = extract_features(image_dir, filenames,
                             batch_size=args.batch_size, device=args.device)
    print(f"Features shape: {feats.shape}")

    # Cache features
    feat_path = output_path.with_suffix(".npy")
    np.save(str(feat_path), feats)
    print(f"Cached features: {feat_path}")

    # t-SNE
    print(f"Running t-SNE (perplexity={args.perplexity})...")
    tsne = TSNE(n_components=2, perplexity=args.perplexity,
                max_iter=1000, random_state=args.tsne_seed)
    emb = tsne.fit_transform(feats)
    print(f"t-SNE done. Embedding shape: {emb.shape}")

    # Plot
    plot_attribute_tsne(emb, metadata, output_path)


if __name__ == "__main__":
    main()
