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

from analysis.run_log import tee_stdout  # noqa: E402
from analysis.plot_style import (
    S, apply_style, ATTR_VALUE_COLORS, ATTR_VALUE_ORDER, style_tsne_ax,
    make_tsne_grid, finish_tsne_grid, TSNE_STYLE,
)


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

    from matplotlib.lines import Line2D

    # cell=4 (vs default 2.8): the shared bottom legend carries all 15
    # attribute values and needs the grid width
    fig, axes = make_tsne_grid(4, ncols=2, cell=4)

    attrs = ["color", "shape", "material", "size"]
    labels = {attr: [m[attr] for m in metadata] for attr in attrs}

    legend_handles = []
    for ax, attr in zip(axes, attrs):
        cmap = ATTR_VALUE_COLORS[attr]
        for val in ATTR_VALUE_ORDER[attr]:
            mask = np.array([l == val for l in labels[attr]])
            ax.scatter(emb[mask, 0], emb[mask, 1],
                       c=[cmap[val]], s=TSNE_STYLE["bg_size"], alpha=0.7,
                       edgecolors="none", rasterized=True)
            legend_handles.append(Line2D([0], [0], marker="o", color="w",
                                  markerfacecolor=cmap[val], markersize=7,
                                  label=val))

        ax.set_title(attr.capitalize(), fontsize=S["subplot_title_fontsize"])
        style_tsne_ax(ax)

    finish_tsne_grid(
        fig, legend_handles,
        suptitle="DINOv2 (pretrained, no fine-tune) — single objects",
        ncol=5)
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
    parser.add_argument("--features", default=None,
                        help="Cached feature .npy — skip extraction, replot only")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--perplexity", type=float, default=30)
    parser.add_argument("--tsne-seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tee_stdout(output_path.parent)

    # Load metadata
    with open(args.metadata) as f:
        metadata = json.load(f)
    filenames = [m["filename"] for m in metadata]
    print(f"Loaded {len(metadata)} entries from {args.metadata}")

    if args.features:
        feats = np.load(args.features)
        print(f"Loaded cached features: {args.features} {feats.shape}")
        assert len(feats) == len(metadata), "features/metadata length mismatch"
    else:
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
