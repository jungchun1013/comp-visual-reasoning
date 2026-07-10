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
    args = ap.parse_args()
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
