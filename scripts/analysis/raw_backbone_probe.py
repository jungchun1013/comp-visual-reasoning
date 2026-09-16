#!/usr/bin/env python
"""Raw-backbone per-object patch-token probe on MULTI-object scenes (E8, v2 §A1).

Claim A1.2 (docs/paper_v2_outline.md): the pretrained substrate encodes attributes
at the PER-OBJECT level in multi-object scenes, before any language conditioning.

Method: build the backbone with fresh (zero-gated) GCA — tanh(0)=0, so the forward
pass equals the pure pretrained ViT (verified src/model/crossattention.py:95,106) —
and no question conditioning. For each scene object, pool the 3×3 patch-token
neighborhood at its pixel_coords, per block; probe each attribute with 5-fold
logistic regression per block.

Expected: high per-object decodability across mid/late blocks for DINOv2/SigLIP/sup,
weaker for MAE (substrate-quality claim A3.1). Contrast with the trained-model probe
(outputs/analysis/linear_probe/multi_object) and the selection gap shown by E7.

Usage (from main/):
  PYTHONPATH=src <interpreter> scripts/analysis/raw_backbone_probe.py \
      --backbone vit_base_patch14_dinov2.lvd142m --n-scenes 300
Output: outputs/analysis/raw_backbone_probe/<backbone>/probe_results.json + .png
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from analysis.run_log import tee_stdout

ATTRS = ["color", "material", "shape", "size"]

# (dir name under output_root, display label) — order fixed for the combined plot
BACKBONES = [
    ("vit_base_patch14_dinov2.lvd142m", "DINOv2"),
    ("vit_base_patch16_siglip_224", "SigLIP"),
    ("vit_base_patch16_224.augreg_in21k", "Sup-ViT"),
    ("vit_base_patch16_224.mae", "MAE"),
]


def plot_combined(output_root: Path):
    """Replot-only: one 1x4 figure (panel per attribute, curve per backbone)
    from the cached probe_results.json of all four backbones. No GPU."""
    from analysis.plot_style import (apply_style, PLOT_STYLE, S, line_kwargs,
                                     save_with_legend)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.cm as cm
    import matplotlib.pyplot as plt

    results = {}
    for dirname, label in BACKBONES:
        path = output_root / dirname / "probe_results.json"
        results[label] = json.loads(path.read_text())["per_block"]

    apply_style()
    w, h = PLOT_STYLE["subplot_size"]
    fig, axes = plt.subplots(1, len(ATTRS), figsize=(w * len(ATTRS), h),
                             sharey=True)
    colors = [cm.tab10(i) for i in range(len(BACKBONES))]
    for ax, attr in zip(axes, ATTRS):
        for (_, label), color in zip(BACKBONES, colors):
            blocks = sorted(results[label], key=int)
            ax.plot([int(b) for b in blocks],
                    [results[label][b][attr] for b in blocks],
                    **line_kwargs(label=label, color=color))
        ax.set_title(attr.capitalize(), fontsize=S["subplot_title_fontsize"])
        ax.set_xlabel("block")
    axes[0].set_ylabel("per-object probe accuracy (5-fold)")
    fig.suptitle("ViT backbone probing — multi-object scenes",
                 fontsize=S["suptitle_fontsize"])
    out_path = output_root / "combined_probe.png"
    save_with_legend(fig, str(out_path), ax_for_legend=axes[0])
    print(f"Saved: {out_path}")


def extract_blockwise(steervit, images):
    """Per-block patch tokens (prefix stripped): list of (B, P, D)."""
    trunk = steervit.vision_model.trunk
    prefix = trunk.num_prefix_tokens
    outs = []
    hooks = []
    for blk in trunk.blocks:
        def fn(mod, inp, out, _o=outs):
            o = out[0] if isinstance(out, tuple) else out
            _o.append(o.detach())
        hooks.append(blk.register_forward_hook(fn))
    with torch.no_grad():
        steervit.forward(images, None)
    for h in hooks:
        h.remove()
    n_blocks = len(trunk.blocks)
    per_block = outs[-n_blocks:]  # guard against re-entrant calls
    return [o[:, prefix:, :].float().cpu() for o in per_block]


def object_patch_feature(feats, px, py, img_w, img_h, grid):
    """Mean of the 3x3 patch neighborhood around pixel (px,py). feats: (P,D)."""
    gx = min(grid - 1, max(0, int(px / img_w * grid)))
    gy = min(grid - 1, max(0, int(py / img_h * grid)))
    idxs = [
        yy * grid + xx
        for yy in range(max(0, gy - 1), min(grid, gy + 2))
        for xx in range(max(0, gx - 1), min(grid, gx + 2))
    ]
    return feats[idxs].mean(0).numpy()


# Pooled-readout probe on the 1-/2-object controlled datasets (multi-object
# hallucination): per block, mean over ALL patch tokens -> PCA(50) -> logistic
# probe of the target object's attributes. Same protocol as the trained-model
# object-count probe, so the two are directly comparable.
# Datasets come from --n1-dir/--n2-dir; the default is the paired-render set
# (render_single_objects.py --paired): identical target placement within each
# n1/n2 pair, 96 types x 5, distractor >=2 attributes away, sizes free.


def _pooled_extract(backbone, entries, data_dir, device, batch_size):
    from model import CrossAttnViT
    steervit = CrossAttnViT.from_config(
        backbone, device=device, cross_attn_layers=[1, 3, 5, 7, 9, 11],
        resolution=336, pretrained=True)
    steervit.eval()
    tf = steervit.get_transforms()
    img_dir = Path(data_dir) / "images"
    feats = []
    for start in range(0, len(entries), batch_size):
        batch = [tf(Image.open(img_dir / e["filename"]).convert("RGB"))
                 for e in entries[start:start + batch_size]]
        per_block = extract_blockwise(steervit, torch.stack(batch).to(device))
        feats.append(torch.stack([b.mean(1) for b in per_block], 1))
        print(f"  [{min(start + batch_size, len(entries))}/{len(entries)}]",
              flush=True)
    del steervit
    return torch.cat(feats).numpy().astype(np.float16)  # (N, 12, D)


def run_pooled(args, device):
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import LabelEncoder

    out_dir = Path(args.output_root) / args.pooled_name
    out_dir.mkdir(parents=True, exist_ok=True)
    tee_stdout(out_dir)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from tsne_patch_level import load_entries

    datasets = {"n1": args.n1_dir, "n2": args.n2_dir}
    entries = {ds: load_entries(path) for ds, path in datasets.items()}
    for ds, es in entries.items():
        print(f"{ds}: {len(es)} scenes")

    backbones = [b for b in BACKBONES if args.only in (None, b[1])]
    for dirname, label in backbones:
        for ds, path in datasets.items():
            cache = out_dir / f"feats_{label}_{ds}.npz"
            if cache.exists():
                print(f"cache hit: {cache}")
                continue
            if args.replot_pooled:
                raise SystemExit(f"--replot-pooled but missing cache {cache}")
            print(f"extracting {label} / {ds} on {device} ...")
            F = _pooled_extract(dirname, entries[ds], path, device,
                                args.batch_size)
            tmp = cache.with_suffix(".tmp.npz")
            np.savez_compressed(tmp, feats=F)
            tmp.rename(cache)  # atomic: concurrent readers never see partials
            print(f"Cached: {cache} {F.shape}")

    if args.only:
        return  # extraction worker; probing runs once all caches exist

    results_path = out_dir / "probe_results.json"
    if results_path.exists():
        results = json.loads(results_path.read_text())
        print(f"loaded {results_path}")
    else:
        results = {}
        for _, label in BACKBONES:
            results[label] = {}
            for ds in datasets:
                F = np.load(out_dir / f"feats_{label}_{ds}.npz")["feats"]
                F = F.astype(np.float32)
                res_ds = {}
                for li in range(F.shape[1]):
                    X, row = F[:, li], {}
                    for a in ATTRS:
                        y = LabelEncoder().fit_transform(
                            [e[a] for e in entries[ds]])
                        if len(set(y)) < 2:  # n2 targets are all large
                            row[a] = None
                            continue
                        skf = StratifiedKFold(5, shuffle=True,
                                              random_state=args.seed)
                        accs = [make_pipeline(
                                    PCA(50, random_state=args.seed),
                                    LogisticRegression(max_iter=500))
                                .fit(X[tr], y[tr]).score(X[te], y[te])
                                for tr, te in skf.split(X, y)]
                        row[a] = float(np.mean(accs))
                    res_ds[str(li)] = row
                    print(f"{label} {ds} block {li:2d}: "
                          + "  ".join(f"{a} {row[a]:.3f}" for a in ATTRS
                                      if row[a] is not None),
                          flush=True)
                results[label][ds] = res_ds
        results_path.write_text(json.dumps(results, indent=2))
        print(f"Wrote {results_path}")

    plot_pooled_probe(results, out_dir)
    plot_pooled_tsne(entries, out_dir, args.seed)


def plot_pooled_probe(results, out_dir):
    from analysis.plot_style import (apply_style, PLOT_STYLE, S, line_kwargs,
                                     save_with_legend)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.cm as cm
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    apply_style()
    w, h = PLOT_STYLE["subplot_size"]
    fig, axes = plt.subplots(1, len(ATTRS), figsize=(w * len(ATTRS), h),
                             sharey=True)
    colors = [cm.tab10(i) for i in range(len(BACKBONES))]
    for ax, a in zip(axes, ATTRS):
        for (_, label), color in zip(BACKBONES, colors):
            for ds, style in (("n1", {}),
                              ("n2", {"linestyle": "--", "linewidth": 1,
                                      "alpha": 0.5, "marker": ""})):
                blocks = sorted(results[label][ds], key=int)
                vals = [results[label][ds][b][a] for b in blocks]
                if any(v is None for v in vals):  # attr constant in this ds
                    continue
                kw = line_kwargs(color=color)
                kw.update(style)
                ax.plot([int(b) for b in blocks], vals, **kw)
        ax.set_title(a.capitalize(), fontsize=S["subplot_title_fontsize"])
        ax.set_xlabel("block")
    axes[0].set_ylabel("pooled probe accuracy (5-fold)")
    fig.suptitle("ViT backbone probing — pooled features, 1 vs 2 objects",
                 fontsize=S["suptitle_fontsize"])
    handles = [Line2D([0], [0], color=c, lw=2, label=l)
               for (_, l), c in zip(BACKBONES, colors)]
    handles += [Line2D([0], [0], color="k", lw=2, label="1 object"),
                Line2D([0], [0], color="k", lw=1, ls="--", alpha=0.6,
                       label="2 objects (target)")]
    save_with_legend(fig, str(out_dir / "pooled_probe.png"), handles=handles,
                     labels=[hd.get_label() for hd in handles])
    print(f"Saved: {out_dir / 'pooled_probe.png'}")


def plot_pooled_tsne(entries, out_dir, seed):
    from sklearn.manifold import TSNE
    from analysis.plot_style import (apply_style, S, TSNE_STYLE,
                                     make_tsne_grid, style_tsne_ax,
                                     finish_tsne_grid)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from dino_attribute_tsne import (TAB20_PAIR, SHAPE_MARKERS,
                                     subsample_per_combo,
                                     attribute_legend_handles)

    apply_style()
    dinov2 = BACKBONES[0][1]
    feats = {ds: np.load(out_dir / f"feats_{dinov2}_{ds}.npz")["feats"]
             [:, 11].astype(np.float32) for ds in entries}
    # Both panels balanced: 5 per object type (96 x 5 = 480 points each)
    sel = {ds: subsample_per_combo(es, 5, seed) for ds, es in entries.items()}

    fig, axes = make_tsne_grid(2, ncols=2, cell=8)
    titles = {"n1": "1 object", "n2": "2 objects (colored by target)"}
    for ax, ds in zip(axes, entries):
        idx = sel[ds]
        emb = TSNE(n_components=2, perplexity=30, max_iter=1000,
                   random_state=seed).fit_transform(feats[ds][idx])
        _pooled_scatter(ax, emb, [entries[ds][i] for i in idx])
        ax.set_title(titles[ds], fontsize=S["subplot_title_fontsize"])
        style_tsne_ax(ax)
    finish_tsne_grid(
        fig, attribute_legend_handles(),
        suptitle="ViT backbone t-SNE — DINOv2 pooled features, block 11",
        ncol=5)
    fig.savefig(str(out_dir / "pooled_tsne.png"), dpi=S["dpi"],
                bbox_inches="tight")
    print(f"Saved: {out_dir / 'pooled_tsne.png'}")
    plt.close(fig)


def _pooled_scatter(ax, emb, sub, edgecolor="black", palette="tab20",
                    small_size=None, large_size=45):
    """Shared t-SNE panel encoding: hue = target color, shade = material
    (dark = metal, light = rubber), marker = shape, point size = size."""
    from analysis.plot_style import TSNE_STYLE
    from dino_attribute_tsne import SHAPE_MARKERS, face_color

    size_pt = {"small": TSNE_STYLE["mid_size"] if small_size is None else small_size,
               "large": large_size}
    for shape, marker in SHAPE_MARKERS.items():
        ii = [k for k, m in enumerate(sub) if m["shape"] == shape]
        ax.scatter(emb[ii, 0], emb[ii, 1],
                   c=[face_color(sub[k], palette) for k in ii],
                   s=[size_pt[sub[k]["size"]] for k in ii],
                   marker=marker, edgecolors=edgecolor, linewidths=0.5,
                   rasterized=True)


def plot_tsne_grid(args):
    """Replot-only 3x2 t-SNE grid: rows = raw DINOv2 backbone / trained model
    +CA color query / +CA shape query; columns = 1 object / 2 objects.
    Block-11 mean-pooled features throughout, shared visual encoding. Raw
    features come from this run's pooled caches; CA features from four
    tsne_single_object.py npz files (attrs.json expected next to each)."""
    from sklearn.manifold import TSNE
    from analysis.plot_style import (apply_style, S, make_tsne_grid,
                                     style_tsne_ax, finish_tsne_grid)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from tsne_patch_level import load_entries
    from dino_attribute_tsne import subsample_per_combo, attribute_legend_handles

    apply_style()
    out_dir = Path(args.output_root) / args.pooled_name
    dinov2 = BACKBONES[0][1]
    panels = []
    for ds, data_dir, title in (("n1", args.n1_dir, "1 object — ViT backbone"),
                                ("n2", args.n2_dir, "2 objects — ViT backbone")):
        F = np.load(out_dir / f"feats_{dinov2}_{ds}.npz")["feats"][:, 11]
        panels.append((title, F.astype(np.float32), load_entries(data_dir)))
    for path, title in zip(args.tsne_grid,
                           ("1 object — +CA, color query",
                            "2 objects — +CA, color query (referring)",
                            "1 object — +CA, shape query",
                            "2 objects — +CA, shape query (referring)")):
        path = Path(path)
        F = np.load(path)["11"].astype(np.float32)
        with open(path.parent / "attrs.json") as f:
            attrs = json.load(f)
        panels.append((title, F, attrs))

    fig, axes = make_tsne_grid(6, ncols=2, cell=8)
    for ax, (title, F, sub_entries) in zip(axes, panels):
        idx = subsample_per_combo(sub_entries, 5, args.seed)
        emb = TSNE(n_components=2, perplexity=30, max_iter=1000,
                   random_state=args.seed).fit_transform(F[idx])
        _pooled_scatter(ax, emb, [sub_entries[i] for i in idx])
        ax.set_title(title, fontsize=S["subplot_title_fontsize"])
        style_tsne_ax(ax)
    finish_tsne_grid(
        fig, attribute_legend_handles(),
        suptitle="Pooled t-SNE, block 11 — 1 vs 2 objects, "
                 "raw backbone vs language-conditioned",
        ncol=5)
    out_path = out_dir / "tsne_grid.png"
    fig.savefig(str(out_path), dpi=S["dpi"], bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="vit_base_patch14_dinov2.lvd142m")
    ap.add_argument("--resolution", type=int, default=336)
    ap.add_argument("--clevr-root",
                    default=os.environ.get("CLEVR_ROOT",
                                           "/home/jungchun/data/clevr/CLEVR_v1.0"))
    ap.add_argument("--n-scenes", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output-root", default="outputs/analysis/raw_backbone_probe")
    ap.add_argument("--combined", action="store_true",
                    help="Replot-only: combined 4-backbone figure from cached "
                         "probe_results.json; skips extraction entirely")
    ap.add_argument("--pooled", action="store_true",
                    help="Pooled-readout probe + t-SNE on the 1-/2-object "
                         "controlled datasets (multi-object hallucination)")
    ap.add_argument("--replot-pooled", action="store_true",
                    help="Pooled figures from caches only; no extraction")
    ap.add_argument("--device", default=None,
                    help="cpu/cuda override for --pooled extraction")
    ap.add_argument("--only", default=None,
                    help="Pooled mode: extract only this backbone label "
                         "(parallel worker), skip probing")
    ap.add_argument("--n1-dir", default="data/clevr_object_count/n1",
                    help="Pooled mode: 1-object dataset dir")
    ap.add_argument("--n2-dir", default="data/clevr_object_count/n2",
                    help="Pooled mode: 2-object dataset dir")
    ap.add_argument("--pooled-name", default="pooled_n1n2_v2",
                    help="Output subdir under --output-root for pooled mode")
    ap.add_argument("--tsne-grid", nargs=4,
                    metavar=("CA_COLOR_N1", "CA_COLOR_N2",
                             "CA_SHAPE_N1", "CA_SHAPE_N2"),
                    default=None,
                    help="Replot-only: 3x2 t-SNE grid (rows: raw backbone / "
                         "CA color query / CA shape query; cols: 1 vs 2 "
                         "objects) from this run's DINOv2 pooled caches plus "
                         "four trained-model feats npz files "
                         "(tsne_single_object.py output; attrs.json expected "
                         "next to each)")
    args = ap.parse_args()
    if args.combined:
        tee_stdout(Path(args.output_root))
        plot_combined(Path(args.output_root))
        return
    if args.tsne_grid:
        plot_tsne_grid(args)
        return
    if args.pooled or args.replot_pooled:
        device = torch.device(args.device
                              or ("cuda" if torch.cuda.is_available()
                                  else "cpu"))
        run_pooled(args, device)
        return
    tee_stdout(Path(args.output_root) / args.backbone.replace("/", "_"))

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import LabelEncoder

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from model import CrossAttnViT
    steervit = CrossAttnViT.from_config(
        args.backbone, device=device, cross_attn_layers=[1, 3, 5, 7, 9, 11],
        resolution=args.resolution, pretrained=True)
    steervit.eval()
    tf = steervit.get_transforms()
    grid = steervit.image_size[0] // steervit.patch_size
    print(f"Backbone {args.backbone} grid {grid}x{grid} device {device}")

    root = Path(args.clevr_root)
    scenes = json.loads((root / "scenes" / "CLEVR_val_scenes.json").read_text())["scenes"]
    rng = np.random.RandomState(args.seed)
    scenes = [scenes[i] for i in rng.choice(len(scenes), args.n_scenes, replace=False)]

    # collect per-object features per block
    feats_per_block = None
    labels = {a: [] for a in ATTRS}
    img_dir = root / "images" / "val"
    batch_imgs, batch_scenes = [], []

    def flush():
        nonlocal feats_per_block
        if not batch_imgs:
            return
        images = torch.stack(batch_imgs).to(device)
        per_block = extract_blockwise(steervit, images)
        if feats_per_block is None:
            feats_per_block = [[] for _ in per_block]
        w, h = 480, 320  # CLEVR native render size (pixel_coords space)
        for bi, sc in enumerate(batch_scenes):
            for obj in sc["objects"]:
                px, py = obj["pixel_coords"][0], obj["pixel_coords"][1]
                for li, fb in enumerate(per_block):
                    feats_per_block[li].append(
                        object_patch_feature(fb[bi], px, py, w, h, grid))
                for a in ATTRS:
                    labels[a].append(obj[a])
        batch_imgs.clear()
        batch_scenes.clear()

    for sc in scenes:
        img = Image.open(img_dir / sc["image_filename"]).convert("RGB")
        batch_imgs.append(tf(img))
        batch_scenes.append(sc)
        if len(batch_imgs) == args.batch_size:
            flush()
    flush()

    n_obj = len(labels["color"])
    print(f"Collected {n_obj} objects from {len(scenes)} scenes")

    results = {"backbone": args.backbone, "n_scenes": len(scenes),
               "n_objects": n_obj, "per_block": {}}
    for li, X in enumerate(feats_per_block):
        X = np.stack(X)
        row = {}
        for a in ATTRS:
            y = LabelEncoder().fit_transform(labels[a])
            skf = StratifiedKFold(5, shuffle=True, random_state=args.seed)
            accs = [LogisticRegression(max_iter=500).fit(X[tr], y[tr]).score(X[te], y[te])
                    for tr, te in skf.split(X, y)]
            row[a] = float(np.mean(accs))
        results["per_block"][str(li)] = row
        print(f"block {li:2d}: " + "  ".join(f"{a} {row[a]:.3f}" for a in ATTRS))

    out_dir = Path(args.output_root) / args.backbone.replace("/", "_")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "probe_results.json").write_text(json.dumps(results, indent=2))

    from analysis.plot_style import apply_style, PLOT_STYLE, line_kwargs, save_with_legend
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    apply_style()
    fig, ax = plt.subplots(figsize=PLOT_STYLE["subplot_size"])
    blocks = sorted(results["per_block"], key=int)
    for a in ATTRS:
        ax.plot([int(b) for b in blocks],
                [results["per_block"][b][a] for b in blocks],
                **line_kwargs(label=a))
    ax.set_xlabel("block")
    ax.set_ylabel("per-object probe accuracy (5-fold)")
    ax.set_title(f"Raw {args.backbone.split('.')[0]} — multi-object scenes")
    save_with_legend(fig, str(out_dir / "raw_backbone_probe.png"))
    print(f"Wrote {out_dir}/probe_results.json + raw_backbone_probe.png")


if __name__ == "__main__":
    main()
