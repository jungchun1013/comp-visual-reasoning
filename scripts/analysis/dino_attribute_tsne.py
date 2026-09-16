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

# same glyphs as tsne_single_object.py
SHAPE_MARKERS = {"sphere": "o", "cube": "s", "cylinder": "^"}


# tab20 dark/light pair start index per CLEVR color value
# (dark = tab20[i], light = tab20[i+1]; yellow -> olive, the nearest tab20 hue)
TAB20_PAIR = {"blue": 0, "green": 4, "red": 6, "purple": 8, "brown": 10,
              "gray": 14, "yellow": 16, "cyan": 18}

# Object colours as rendered by CLEVR (properties.json RGB / 255); used by the
# "clevr" palette so the hue on the plot is the hue in the image.
CLEVR_RGB = {"gray": (87, 87, 87), "red": (173, 35, 35), "blue": (42, 75, 215),
             "green": (29, 105, 20), "brown": (129, 74, 25),
             "purple": (129, 38, 192), "cyan": (41, 208, 208),
             "yellow": (255, 238, 51)}
PALETTES = ("tab20", "tab20light", "clevr", "tab10")


def face_color(m, palette="tab20"):
    """Hue = colour value, shade = material (metal full, rubber lightened)."""
    import matplotlib.pyplot as plt
    if palette == "tab20":
        t20 = plt.get_cmap("tab20").colors
        i = TAB20_PAIR[m["color"]]
        return t20[i] if m["material"] == "metal" else t20[i + 1]
    if palette == "tab20light":   # light member for both materials (pre-2026-07-08 look)
        t20 = plt.get_cmap("tab20").colors
        return t20[TAB20_PAIR[m["color"]] + 1]
    if palette == "clevr":
        base = tuple(v / 255 for v in CLEVR_RGB[m["color"]])
    elif palette == "tab10":
        base = ATTR_VALUE_COLORS["color"][m["color"]][:3]
    else:
        raise ValueError(palette)
    if m["material"] == "metal":
        return base
    return tuple(0.55 * c + 0.45 for c in base)   # blend 45 % towards white


def attribute_legend_handles():
    """Shared 13-handle legend for the 4-channel attribute encoding
    (hue=color, shade=material, glyph=shape, size=size)."""
    from matplotlib.lines import Line2D
    t20 = plt.get_cmap("tab20").colors
    handles = [Line2D([0], [0], marker="o", color="w",
                      markerfacecolor=t20[TAB20_PAIR[v]], markersize=7,
                      label=v)
               for v in ATTR_VALUE_ORDER["color"]]
    handles += [Line2D([0], [0], marker=mk, color="w",
                       markerfacecolor=TSNE_STYLE["gray"], markersize=7,
                       label=sh) for sh, mk in SHAPE_MARKERS.items()]
    handles += [Line2D([0], [0], marker="o", color="w",
                       markerfacecolor=TSNE_STYLE["gray"],
                       markersize=5 if sz == "small" else 10, label=sz)
                for sz in ATTR_VALUE_ORDER["size"]]
    handles += [Line2D([0], [0], marker="o", color="w",
                       markerfacecolor=t20[0] if mat == "metal" else t20[1],
                       markersize=7,
                       label=f"{mat} ({'dark' if mat == 'metal' else 'light'})")
                for mat in ATTR_VALUE_ORDER["material"]]
    return handles


def plot_attribute_tsne_combined(emb: np.ndarray, metadata: list[dict],
                                 output_path: Path):
    """Single-panel t-SNE with all four attributes on one scatter:
    hue=color, shade=material (metal dark / rubber light),
    marker glyph=shape, point size=size."""
    apply_style()

    from matplotlib.lines import Line2D
    t20 = plt.get_cmap("tab20").colors

    def face(m):
        i = TAB20_PAIR[m["color"]]
        return t20[i] if m["material"] == "metal" else t20[i + 1]

    # 2x the per-panel cell of the 2x2 variant: one panel carries the
    # full point budget plus a 13-handle legend
    fig, axes = make_tsne_grid(1, ncols=1, cell=4 * 2)
    ax = axes[0]

    size_pt = {"small": TSNE_STYLE["mid_size"], "large": 45}

    for shape, marker in SHAPE_MARKERS.items():
        idx = np.array([i for i, m in enumerate(metadata)
                        if m["shape"] == shape])
        sub = [metadata[i] for i in idx]
        # thin uniform outline for marker crispness — material is carried by
        # the tab20 shade, not the edge
        ax.scatter(emb[idx, 0], emb[idx, 1],
                   c=[face(m) for m in sub],
                   s=[size_pt[m["size"]] for m in sub],
                   marker=marker, edgecolors="black", linewidths=0.5,
                   rasterized=True)
    style_tsne_ax(ax)

    handles = attribute_legend_handles()

    finish_tsne_grid(
        fig, handles,
        suptitle="ViT backbone t-SNE — DINOv2, single objects",
        ncol=5)
    fig.savefig(str(output_path), dpi=S["dpi"], bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close(fig)


def subsample_per_combo(metadata: list[dict], n_per_combo: int, seed: int):
    """Pick n_per_combo position-variants of each (color, shape, material,
    size) combination; returns sorted indices."""
    groups = {}
    for i, m in enumerate(metadata):
        groups.setdefault(
            (m["color"], m["shape"], m["material"], m["size"]), []).append(i)
    rng = np.random.RandomState(seed)
    keep = []
    for key in sorted(groups):
        idx = groups[key]
        keep += list(rng.choice(idx, min(n_per_combo, len(idx)),
                                replace=False))
    return sorted(keep)


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
    parser.add_argument("--combined", action="store_true",
                        help="Single panel, all 4 attributes as visual channels")
    parser.add_argument("--per-combo", type=int, default=None,
                        help="Subsample N position-variants per attribute "
                             "combination before t-SNE (e.g. 3 -> 288 points)")
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

    if args.per_combo:
        keep = subsample_per_combo(metadata, args.per_combo, args.tsne_seed)
        feats = feats[keep]
        metadata = [metadata[i] for i in keep]
        print(f"Subsampled to {len(keep)} points "
              f"({args.per_combo} per attribute combination)")

    # t-SNE
    print(f"Running t-SNE (perplexity={args.perplexity})...")
    tsne = TSNE(n_components=2, perplexity=args.perplexity,
                max_iter=1000, random_state=args.tsne_seed)
    emb = tsne.fit_transform(feats)
    print(f"t-SNE done. Embedding shape: {emb.shape}")

    # Plot
    if args.combined:
        plot_attribute_tsne_combined(emb, metadata, output_path)
    else:
        plot_attribute_tsne(emb, metadata, output_path)


if __name__ == "__main__":
    main()
