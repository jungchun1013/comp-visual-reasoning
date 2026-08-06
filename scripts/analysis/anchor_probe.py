#!/usr/bin/env python
"""Anchor probe (experiment 二b): decode scene-derived anchor labels from the
query image's OWN steered per-layer features.

New file (reason): no existing script probes the query image's steered per-layer
features against anchor labels. `linear_probe.py` probes DB-image labels;
`raw_backbone_probe.py` probes the un-conditioned raw backbone. Here the forward
IS the anchoring question on its own image — the same steered feature the
conditional RSA uses — and the label is the anchor object's spatial quadrant or
its described attribute value.

Readout (定錨問句): after the anchor-attribute structure has dissipated, is the
anchor quadrant still decodable at L10-11? A quadrant curve that survives late
while the attribute curves fade = position kept as an index after the attribute
description is consumed.

Method: 5-fold logistic regression per layer (same estimator as
raw_backbone_probe.py). Features cached to features.npz; --replot re-probes the
cache and re-renders without loading the model (project convention).

Usage (from main/):
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src <interp> \
    scripts/analysis/anchor_probe.py \
      --checkpoint outputs/model/clevr_dinov2_decoder1l_scratch_s42/best.pt
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))  # sibling scripts (conditional_rsa)

from analysis.run_log import tee_stdout

ATTRS = ["color", "shape", "material", "size"]
LABELS = ["quadrant"] + ATTRS
FRAME_W, FRAME_H = 480, 320  # CLEVR native render size (pixel_coords space)


def _probe(X, y, seed):
    """5-fold logistic-regression accuracy on (X, y). Drops classes with <5
    members; returns (mean_acc | None, n_used)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import LabelEncoder

    cnt = Counter(y.tolist())
    keep = {c for c, k in cnt.items() if k >= 5}
    mask = np.array([v in keep for v in y])
    Xf, yf = X[mask], y[mask]
    if len(set(yf.tolist())) < 2 or len(yf) < 10:
        return None, int(mask.sum())
    ye = LabelEncoder().fit_transform(yf)
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    accs = [LogisticRegression(max_iter=500).fit(Xf[tr], ye[tr]).score(Xf[te], ye[te])
            for tr, te in skf.split(Xf, ye)]
    return float(np.mean(accs)), int(mask.sum())


def run_probes(feats, quadrant, attr_arrays, seed):
    """feats: (num_layers, n, D). Returns per-layer results dict."""
    num_layers = feats.shape[0]
    per_layer = {}
    for l in range(num_layers):
        X = feats[l]
        row = {}
        acc, n = _probe(X, quadrant.astype(str), seed)
        row["quadrant"] = {"acc": acc, "n": n}
        for a in ATTRS:
            y = attr_arrays[a]
            valid = y != ""
            acc, n = _probe(X[valid], y[valid], seed) if valid.sum() >= 10 else (None, int(valid.sum()))
            row[a] = {"acc": acc, "n": n}
        per_layer[str(l)] = row
        print("layer {:2d}: ".format(l)
              + "  ".join(f"{k} {row[k]['acc']:.3f}" if row[k]["acc"] is not None
                          else f"{k} --" for k in LABELS), flush=True)
    return per_layer


def plot(results, out_path):
    from analysis.plot_style import apply_style, PLOT_STYLE, line_kwargs, save_with_legend
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    apply_style()
    fig, ax = plt.subplots(figsize=PLOT_STYLE["subplot_size"])
    layers = sorted(results["per_layer"], key=int)
    xs = [int(l) for l in layers]
    for label in LABELS:
        ys = [results["per_layer"][l][label]["acc"] for l in layers]
        ys = [np.nan if v is None else v for v in ys]
        ax.plot(xs, ys, **line_kwargs(label=label))
    for l in [1, 3, 5, 7, 9, 11]:
        ax.axvline(x=l, color="gray", linestyle="--", linewidth=0.8, alpha=0.15)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Anchor probe accuracy (5-fold)")
    save_with_legend(fig, out_path)
    print(f"Saved: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--data-root", type=str,
                    default="/home/jungchun/data/clevr/CLEVR_v1.0")
    ap.add_argument("--n-queries", type=int, default=800)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--output-dir", type=str,
                    default="outputs/analysis/anchor_probe/clevr_dinov2_decoder1l_scratch")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--replot", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tee_stdout(out_dir)

    if args.replot:
        cache = np.load(out_dir / "features.npz", allow_pickle=True)
        feats = cache["feats"]
        quadrant = cache["quadrant"]
        attr_arrays = {a: cache[a] for a in ATTRS}
        per_layer = run_probes(feats, quadrant, attr_arrays, args.seed)
        results = {"checkpoint": None, "n_queries": int(feats.shape[1]),
                   "num_layers": int(feats.shape[0]), "labels": LABELS,
                   "per_layer": per_layer}
        (out_dir / "probe_results.json").write_text(json.dumps(results, indent=2))
        plot(results, str(out_dir / "anchor_probe.png"))
        return

    import torch
    from data.clevr import CLEVRVQADataset
    from data.clevr_sampling import build_family_index, sample_queries
    from data.clevr_programs import find_anchor
    from conditional_rsa import load_model, AllLayerRetriever

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model, steervit, vocab = load_model(args.checkpoint, device)
    transform = steervit.get_transforms()
    retriever = AllLayerRetriever(steervit)
    num_layers = retriever.num_layers

    dataset = CLEVRVQADataset(args.data_root, "val", transform)
    with open(Path(args.data_root) / "scenes" / "CLEVR_val_scenes.json") as f:
        scenes = {s["image_filename"]: s for s in json.load(f)["scenes"]}

    # Gather same+spatial queries with a resolvable anchor.
    rng = random.Random(args.seed)
    index = build_family_index(dataset)
    cats = ["attr_query_same", "attr_query_spatial"]
    per = args.n_queries // len(cats)
    recs = []
    for cat in cats:
        for q in sample_queries(dataset, index, cat, n_total=per, rng=rng, scenes=scenes):
            fn = dataset.questions[q["dataset_idx"]]["image_filename"]
            anchor_obj, described = find_anchor(scenes[fn]["objects"], q["program"])
            if not anchor_obj:
                continue
            pc = anchor_obj.get("pixel_coords")
            if not pc:
                continue
            quad = int(pc[0] >= FRAME_W / 2) + 2 * int(pc[1] >= FRAME_H / 2)
            recs.append({"dataset_idx": q["dataset_idx"], "question": q["question"],
                         "quadrant": quad,
                         "attrs": {a: described.get(a) for a in ATTRS}})
    print(f"Usable queries with anchor: {len(recs)}", flush=True)

    # Steered per-layer mean-pooled patch features on each query's own image+q.
    feats_per_layer = {l: [] for l in range(num_layers)}
    for start in range(0, len(recs), args.batch_size):
        batch = recs[start:start + args.batch_size]
        imgs = torch.stack([dataset[r["dataset_idx"]]["image"] for r in batch]).to(device)
        qs = [r["question"] for r in batch]
        f = retriever.extract(imgs, qs)
        for l in range(num_layers):
            feats_per_layer[l].append(f[l])
        if start % (args.batch_size * 5) == 0:
            print(f"  extracted {start + len(batch)}/{len(recs)}", flush=True)

    feats = np.stack([np.concatenate([t.numpy() for t in feats_per_layer[l]])
                      for l in range(num_layers)])  # (L, n, D)
    quadrant = np.array([r["quadrant"] for r in recs])
    attr_arrays = {a: np.array([r["attrs"][a] or "" for r in recs]) for a in ATTRS}
    dataset_idx = np.array([r["dataset_idx"] for r in recs])

    np.savez(out_dir / "features.npz", feats=feats, quadrant=quadrant,
             dataset_idx=dataset_idx, **attr_arrays)
    print(f"Cached features.npz {feats.shape}", flush=True)

    per_layer = run_probes(feats, quadrant, attr_arrays, args.seed)
    results = {"checkpoint": str(args.checkpoint), "n_queries": len(recs),
               "num_layers": num_layers, "labels": LABELS, "per_layer": per_layer}
    (out_dir / "probe_results.json").write_text(json.dumps(results, indent=2))
    plot(results, str(out_dir / "anchor_probe.png"))


if __name__ == "__main__":
    main()
