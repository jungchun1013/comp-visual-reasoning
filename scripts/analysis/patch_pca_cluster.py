"""Patch-token PCA + per-image KMeans on the paired object-count renders — X19.

Hypothesis: a patch containing an object carries the local background
representation plus an additive, object-specific vector. X16's patch-level
t-SNE showed per-object patch clusters but t-SNE is distance-based, nonlinear,
and independently fit per panel — it cannot test additivity. Two analyses on
the paired dataset data/clevr_object_count (480 pairs, 96 combos x 5 positions,
target placement identical across n1/n2):

  A. PCA: one global fit per GCA layer over ALL patch tokens of the selected
     n1+n2 images (n1/n2 panels share the linear coordinate frame — the point
     of PCA over per-panel t-SNE). Additivity quantified in full 768-d space
     via offset = mean(object patches) - mean(background patches):
     within-combo-across-position vs between-combo cosine, top-1 SVD share,
     and n1-vs-n2 same-pair target-offset cosine.
  B. KMeans: 5 random pairs, k=2 on the n1 image / k=3 on the n2 image per
     GCA layer; clusters Hungarian-matched to segmentation owner masks;
     ARI + per-object IoU; overlays with target cluster red / distractor
     cluster blue at alpha 0.3.

Extraction and mask machinery are imported from tsne_patch_level (X16), not
duplicated. Condition: noca only (frozen backbone => the ViT backbone's own
patch representation). Phases as X16: --masks-only (CPU, inspect
masks_debug.png) -> extraction (cached npz, refuses to re-extract) -> --replot.

Run from main/ (CPU on purpose — leave the single GPU to training runs):
  PYTHONPATH=src CUDA_VISIBLE_DEVICES= <interpreter> \
      scripts/analysis/patch_pca_cluster.py [--masks-only|--replot]
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from PIL import Image as PILImage
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from analysis.plot_style import (
    apply_style, ATTR_VALUE_COLORS, TSNE_STYLE, S, GCA_LAYERS,
    make_tsne_grid, style_tsne_ax, finish_tsne_grid,
)
from analysis.run_log import tee_stdout
from tsne_single_object import SHAPE_MARKERS
from tsne_patch_level import (
    load_entries, build_masks, save_masks_debug, extract_condition, load_cached,
)

# tab10 red/blue — cluster-identity overlay colors (not attribute colors)
BACKBONE_LABELS = {"dinov2": "DINOv2", "siglip": "SigLIP", "sup": "Sup-ViT", "mae": "MAE"}
CLUSTER_RGB = {"target": (0.839, 0.153, 0.157), "distractor": (0.121, 0.467, 0.706)}


# ---------------------------------------------------------------------------
# Pair selection (deterministic under --seed; recorded in labels.json)
# ---------------------------------------------------------------------------

def combo_key(e):
    return (e["color"], e["shape"], e["material"], e["size"])


def select_pairs(n1_entries, n2_entries, rng, n_combos, n_cluster):
    """PCA set: n_combos full combos x all 5 positions (round-robin shapes,
    prefer unused colors). Cluster set: n_cluster uniform-random pairs.
    Both exclude gray and target-colored distractors — the hue-based
    segmentation (X16) cannot separate same-hue or desaturated objects."""
    groups = {}
    for i, e in enumerate(n1_entries):
        groups.setdefault(combo_key(e), []).append(i)
    assert all(len(v) == 5 for v in groups.values()), "expected 5 positions/combo"

    def seg_ok(i):
        t, ds = n2_entries[i], n2_entries[i]["distractors"]
        return all(d["color"] not in (t["color"], "gray") for d in ds)

    eligible = [c for c, idxs in groups.items()
                if c[0] != "gray" and all(seg_ok(i) for i in idxs)]
    by_shape = {s: [c for c in eligible if c[1] == s] for s in SHAPE_MARKERS}
    for v in by_shape.values():
        rng.shuffle(v)
    shapes = list(SHAPE_MARKERS)
    combos, used_colors = [], set()
    for k in range(n_combos):
        pool = by_shape[shapes[k % len(shapes)]]
        pick = next((c for c in pool if c[0] not in used_colors), pool[0])
        pool.remove(pick)
        used_colors.add(pick[0])
        combos.append(pick)
    pca_pairs = [i for c in combos for i in sorted(groups[c])]

    ok = [i for i in range(len(n2_entries))
          if n2_entries[i]["color"] != "gray" and seg_ok(i)]
    cluster_pairs = sorted(int(i) for i in rng.choice(ok, n_cluster, replace=False))
    return pca_pairs, cluster_pairs, combos


# ---------------------------------------------------------------------------
# Experiment A — PCA + additive-offset statistics
# ---------------------------------------------------------------------------

def fit_layer_pcas(feats_by_subset, pca_flags):
    """One PCA(2) per layer, fit on ALL patches of the PCA-set images of BOTH
    subsets (n1 U n2) so every panel shares one linear frame."""
    pcas = {}
    for layer in GCA_LAYERS:
        X = np.concatenate([
            feats[layer][flags].reshape(-1, feats[layer].shape[-1])
            for feats, flags in zip(feats_by_subset, pca_flags)
        ]).astype(np.float32)
        pcas[layer] = PCA(n_components=2, random_state=0).fit(X)
    return pcas


def plot_patch_pca(feats, owner, labels, pcas, out_path, *, suptitle,
                   bg_per_image=100, seed=42):
    """PCA scatter grid over GCA layers, X16 marker conventions (bg gray,
    object patches CLEVR color x shape marker)."""
    rows = []  # (img, patch, owner, shape, color)
    for b, rec in enumerate(labels):
        if not rec["in_pca_set"]:
            continue
        obj_attrs = [(rec["target"]["shape"], rec["target"]["color"])] + \
                    [(d["shape"], d["color"]) for d in rec["distractors"]]
        obj_idx = np.nonzero(owner[b] > 0)[0]
        bg_idx = np.nonzero(owner[b] == 0)[0]
        if bg_per_image is not None and len(bg_idx) > bg_per_image:
            bg_idx = np.random.RandomState(seed + b).choice(
                bg_idx, bg_per_image, replace=False)
        for p in obj_idx:
            sh, co = obj_attrs[owner[b, p] - 1]
            rows.append((b, p, owner[b, p], sh, co))
        for p in bg_idx:
            rows.append((b, p, 0, "", ""))

    own = np.array([r[2] for r in rows])
    shp = np.array([r[3] for r in rows])
    col = np.array([r[4] for r in rows])

    fig, axes = make_tsne_grid(len(GCA_LAYERS), ncols=3)
    for ax_i, layer in enumerate(GCA_LAYERS):
        ax = axes[ax_i]
        X = np.stack([feats[layer][b, p] for b, p, *_ in rows]).astype(np.float32)
        emb = pcas[layer].transform(X)
        bg = own == 0
        ax.scatter(emb[bg, 0], emb[bg, 1], c=[TSNE_STYLE["gray"]],
                   s=TSNE_STYLE["bg_size"], marker="o", edgecolors="none",
                   rasterized=True)
        for shape, marker in SHAPE_MARKERS.items():
            for color in ATTR_VALUE_COLORS["color"]:
                m = (shp == shape) & (col == color) & (own > 0)
                if not m.any():
                    continue
                ax.scatter(emb[m, 0], emb[m, 1],
                           c=[ATTR_VALUE_COLORS["color"][color]], marker=marker,
                           s=TSNE_STYLE["mid_size"], edgecolors="none",
                           rasterized=True)
        evr = pcas[layer].explained_variance_ratio_
        ax.set_title(f"L{layer} (PC1+2 {evr.sum():.0%})",
                     fontsize=S["subplot_title_fontsize"])
        style_tsne_ax(ax)
    for ax in axes[len(GCA_LAYERS):]:
        ax.axis("off")

    handles = [Line2D([0], [0], marker="o", color="w",
                      markerfacecolor=ATTR_VALUE_COLORS["color"][c],
                      markersize=7, label=c)
               for c in sorted(set(col[own > 0]))]
    handles += [Line2D([0], [0], marker=m, color="w",
                       markerfacecolor=TSNE_STYLE["gray"], markersize=7,
                       label=s) for s, m in SHAPE_MARKERS.items()
                if s in set(shp[own > 0])]
    handles.append(Line2D([0], [0], marker="o", color="w",
                          markerfacecolor=TSNE_STYLE["gray"], markersize=5,
                          label="background"))
    finish_tsne_grid(fig, handles, suptitle=suptitle, ncol=5)
    fig.savefig(out_path, dpi=S["dpi"], bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_patch_pca_single(feats, owner, labels, b, grid, out_path, *, suptitle,
                          image_path=None):
    """One image, ALL patches, per-layer PCA fit on that image's own tokens.
    Left column (both rows): the scene itself with the patch-owner masks
    overlaid (target red / distractor blue, alpha 0.3, same blend as the
    cluster overlays). Top row: owner/attribute coloring. Bottom row: same
    embedding colored by the patch's row in the 2D grid — diagnoses
    positional banding."""
    rec = labels[b]
    obj_attrs = [(rec["target"]["shape"], rec["target"]["color"])] + \
                [(d["shape"], d["color"]) for d in rec["distractors"]]
    rows_idx = np.arange(grid * grid) // grid
    n = len(GCA_LAYERS)

    fig = plt.figure(figsize=(2.6 * n + 4.0, 5.6))
    gs = fig.add_gridspec(2, n + 1, width_ratios=[1.7] + [1] * n)
    ax_img = fig.add_subplot(gs[:, 0])
    axes = np.array([[fig.add_subplot(gs[r, c + 1]) for c in range(n)]
                     for r in range(2)])
    if image_path is not None:
        img = PILImage.open(image_path).convert("RGB")
        overlay = np.asarray(img, dtype=np.float32) / 255.0
        w, h = img.size
        up = np.asarray(PILImage.fromarray(
            owner[b].reshape(grid, grid).astype(np.uint8)).resize(
            (w, h), PILImage.NEAREST))
        for oid, role in ((1, "target"), (2, "distractor")):
            sel = up == oid
            overlay[sel] = 0.7 * overlay[sel] + 0.3 * np.array(CLUSTER_RGB[role])
        ax_img.imshow(overlay)
        ax_img.set_title(f"{rec['filename']} ({grid}×{grid} patches)",
                         fontsize=9)
    ax_img.set_axis_off()
    last = None
    for c, layer in enumerate(GCA_LAYERS):
        X = feats[layer][b].astype(np.float32)
        pca = PCA(n_components=2, random_state=0).fit(X)
        emb = pca.transform(X)
        ax = axes[0, c]
        bg = owner[b] == 0
        ax.scatter(emb[bg, 0], emb[bg, 1], c=[TSNE_STYLE["gray"]],
                   s=TSNE_STYLE["bg_size"], marker="o", edgecolors="none",
                   rasterized=True)
        for oi, (sh, co) in enumerate(obj_attrs):
            m = owner[b] == oi + 1
            if m.any():
                ax.scatter(emb[m, 0], emb[m, 1],
                           c=[ATTR_VALUE_COLORS["color"][co]],
                           marker=SHAPE_MARKERS[sh], s=TSNE_STYLE["mid_size"],
                           edgecolors="none", rasterized=True)
        ax.set_title(f"L{layer} ({pca.explained_variance_ratio_.sum():.0%})",
                     fontsize=9)
        style_tsne_ax(ax)
        ax = axes[1, c]
        last = ax.scatter(emb[:, 0], emb[:, 1], c=rows_idx, cmap="viridis",
                          s=TSNE_STYLE["bg_size"], edgecolors="none",
                          rasterized=True)
        style_tsne_ax(ax)
    cb = fig.colorbar(last, ax=axes[1, :].tolist(), shrink=0.8, pad=0.01)
    cb.set_label("patch row (top→bottom)", fontsize=8)
    handles = [Line2D([0], [0], marker=SHAPE_MARKERS[sh], color="w",
                      markerfacecolor=ATTR_VALUE_COLORS["color"][co],
                      markersize=7, label=f"{co} {sh}")
               for sh, co in obj_attrs]
    handles.append(Line2D([0], [0], marker="o", color="w",
                          markerfacecolor=TSNE_STYLE["gray"], markersize=5,
                          label="background"))
    fig.legend(handles=handles, loc="lower center",
               ncol=len(handles), fontsize=8, frameon=False)
    fig.suptitle(suptitle, fontsize=12)
    fig.savefig(out_path, dpi=S["dpi"], bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def _cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def _pair_cosines(offs, combos, same_combo):
    return [_cos(offs[i], offs[j])
            for i, j in itertools.combinations(range(len(offs)), 2)
            if (combos[i] == combos[j]) == same_combo]


def _summ(vals):
    return {"mean": float(np.mean(vals)), "std": float(np.std(vals)),
            "n": len(vals)}


def offset_statistics(n1_feats, n1_owner, n2_feats, n2_owner, labels):
    """768-d additivity stats per layer over the PCA-set images (paired)."""
    idx = [b for b, rec in enumerate(labels) if rec["in_pca_set"]]
    combos = ["-".join(combo_key(labels[b]["target"])) for b in idx]
    stats = {}
    for layer in GCA_LAYERS:
        def off(feats, owner, b, oid):
            f = feats[layer][b].astype(np.float32)
            sel = owner[b] == oid
            if not sel.any():
                return None
            return f[sel].mean(0) - f[owner[b] == 0].mean(0)

        o_n1 = np.stack([off(n1_feats, n1_owner, b, 1) for b in idx])
        o_n2t = np.stack([off(n2_feats, n2_owner, b, 1) for b in idx])
        o_n2d = [off(n2_feats, n2_owner, b, 2) for b in idx]

        grand = o_n1.mean(0)
        resid = o_n1 - grand
        sv = np.linalg.svd(o_n1, compute_uv=False)
        stats[f"L{layer}"] = {
            "n1_within_combo_cos": _summ(_pair_cosines(o_n1, combos, True)),
            "n1_between_combo_cos": _summ(_pair_cosines(o_n1, combos, False)),
            "n1_resid_within_combo_cos": _summ(_pair_cosines(resid, combos, True)),
            "n1_resid_between_combo_cos": _summ(_pair_cosines(resid, combos, False)),
            "n1_svd_top1_share": float(sv[0] ** 2 / (sv ** 2).sum()),
            "n1_offset_norm": _summ([float(np.linalg.norm(o)) for o in o_n1]),
            "n1_grand_mean_norm": float(np.linalg.norm(grand)),
            "n1_vs_n2_target_same_pair_cos": _summ(
                [_cos(a, b) for a, b in zip(o_n1, o_n2t)]),
            "n2_target_vs_distractor_cos": _summ(
                [_cos(t, d) for t, d in zip(o_n2t, o_n2d) if d is not None]),
        }
        s = stats[f"L{layer}"]
        print(f"L{layer}: within {s['n1_within_combo_cos']['mean']:.3f} vs "
              f"between {s['n1_between_combo_cos']['mean']:.3f} | resid "
              f"{s['n1_resid_within_combo_cos']['mean']:.3f} vs "
              f"{s['n1_resid_between_combo_cos']['mean']:.3f} | top1 "
              f"{s['n1_svd_top1_share']:.3f} | n1~n2(target) "
              f"{s['n1_vs_n2_target_same_pair_cos']['mean']:.3f}")
    return stats


# ---------------------------------------------------------------------------
# Experiment B — per-image KMeans + overlays + metrics
# ---------------------------------------------------------------------------

def cluster_image(tokens, owner, k, seed):
    """KMeans on one image's (576, D) tokens; Hungarian match to owner ids.
    Returns (assignment mapped into owner-id space, ARI, {oid: IoU})."""
    labels_km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(
        tokens.astype(np.float32))
    M = np.zeros((k, k))
    for c in range(k):
        for o in range(k):
            M[c, o] = np.sum((labels_km == c) & (owner == o))
    ci, oi = linear_sum_assignment(-M)
    mapping = dict(zip(ci.tolist(), oi.tolist()))
    mapped = np.array([mapping[c] for c in labels_km], dtype=np.int8)
    ious = {}
    for oid in range(1, k):
        inter = np.sum((mapped == oid) & (owner == oid))
        union = np.sum((mapped == oid) | (owner == oid))
        ious[oid] = float(inter / union) if union else None
    # foreground = union of non-background clusters vs union of objects:
    # high fg-IoU with low per-object IoU = objects found but not separated
    fg_inter = np.sum((mapped > 0) & (owner > 0))
    fg_union = np.sum((mapped > 0) | (owner > 0))
    ious["fg"] = float(fg_inter / fg_union) if fg_union else None
    return mapped, float(adjusted_rand_score(owner, labels_km)), ious


def plot_cluster_overlays(images, mappeds, recs, grid, out_path, *, suptitle):
    """Rows = images, cols = GCA layers; target cluster red / distractor
    cluster blue at alpha 0.3 on the original render."""
    n = len(images)
    fig, axes = plt.subplots(n, len(GCA_LAYERS),
                             figsize=(3.2 * len(GCA_LAYERS), 2.2 * n))
    axes = np.atleast_2d(axes)
    for r in range(n):
        img = np.asarray(images[r], dtype=np.float32) / 255.0
        w, h = images[r].size
        for c, layer in enumerate(GCA_LAYERS):
            ax = axes[r, c]
            ax.axis("off")
            up = np.asarray(PILImage.fromarray(
                mappeds[r][layer].reshape(grid, grid).astype(np.uint8)).resize(
                (w, h), PILImage.NEAREST))
            overlay = img.copy()
            for oid, role in ((1, "target"), (2, "distractor")):
                sel = up == oid
                overlay[sel] = (0.7 * overlay[sel]
                                + 0.3 * np.array(CLUSTER_RGB[role]))
            ax.imshow(overlay)
            if r == 0:
                ax.set_title(f"L{layer}", fontsize=S["subplot_title_fontsize"])
        t = recs[r]["target"]
        d = recs[r]["distractors"]
        desc = f'{t["color"]} {t["shape"]}' + \
               (f' + {d[0]["color"]} {d[0]["shape"]}' if d else "")
        axes[r, 0].text(-0.06, 0.5, desc, transform=axes[r, 0].transAxes,
                        rotation=90, va="center", ha="center", fontsize=8)
    fig.suptitle(suptitle, fontsize=S["suptitle_fontsize"]
                 if "suptitle_fontsize" in S else 12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=S["dpi"], bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_cluster_metrics(records, out_path, *, suptitle):
    """Mean IoU / ARI vs layer; individual pairs as faint points."""
    fig, (ax_iou, ax_ari) = plt.subplots(1, 2, figsize=(9.0, 3.6))
    fig.suptitle(suptitle, fontsize=11)
    series = [("n1", "iou_target", "target (1-object, k=2)",
               CLUSTER_RGB["target"], "--"),
              ("n2", "iou_target", "target (2-object, k=3)",
               CLUSTER_RGB["target"], "-"),
              ("n2", "iou_distractor", "distractor (2-object, k=3)",
               CLUSTER_RGB["distractor"], "-"),
              ("n2", "iou_foreground", "foreground union (2-object)",
               (0.3, 0.3, 0.3), ":")]
    for subset, field, label, color, ls in series:
        means = []
        for layer in GCA_LAYERS:
            vals = [r[field] for r in records
                    if r["subset"] == subset and r["layer"] == layer
                    and r.get(field) is not None]
            means.append(np.mean(vals))
            ax_iou.scatter([layer] * len(vals), vals, s=8, color=color,
                           alpha=0.25, edgecolors="none")
        ax_iou.plot(GCA_LAYERS, means, ls, color=color, marker="o",
                    markersize=4, label=label)
    for subset, color, ls in (("n1", "0.35", "--"), ("n2", "0.1", "-")):
        means = [np.mean([r["ari"] for r in records
                          if r["subset"] == subset and r["layer"] == layer])
                 for layer in GCA_LAYERS]
        ax_ari.plot(GCA_LAYERS, means, ls, color=color, marker="o",
                    markersize=4,
                    label=f"{subset} (k={2 if subset == 'n1' else 3})")
    for ax, ylab in ((ax_iou, "IoU (cluster vs owner mask)"),
                     (ax_ari, "Adjusted Rand Index")):
        ax.set_xlabel("GCA layer")
        ax.set_ylabel(ylab)
        ax.set_xticks(GCA_LAYERS)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=S["dpi"], bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def background_templates(feats, owner):
    """Per-layer (P, D) template: mean token at each patch position over the
    images where that position is background. Instantiates the additive
    hypothesis — token = background(position) + object vector — so residual
    tokens should isolate the objects."""
    templates = {}
    for layer in GCA_LAYERS:
        f = feats[layer].astype(np.float32)          # (B, P, D)
        bg = (owner == 0)[..., None].astype(np.float32)
        templates[layer] = (f * bg).sum(0) / np.maximum(bg.sum(0), 1.0)
    return templates


def run_clustering(subsets, out_dir, args, variant="raw"):
    """subsets: {name: (feats, owner, labels, data_dir)}. variant 'raw'
    clusters tokens as-is; 'bgsub' subtracts the per-position background
    template first. Returns records."""
    records = []
    for name, (feats, owner, labels, data_dir) in subsets.items():
        templates = background_templates(feats, owner) if variant == "bgsub" \
            else None
        rows = [(b, rec) for b, rec in enumerate(labels) if rec["in_cluster_set"]]
        k = 2 if name == "n1" else 3
        images, mappeds, recs = [], [], []
        for b, rec in rows:
            images.append(PILImage.open(
                Path(data_dir) / "images" / rec["filename"]).convert("RGB"))
            recs.append(rec)
            per_layer = {}
            for layer in GCA_LAYERS:
                tokens = feats[layer][b].astype(np.float32)
                if templates is not None:
                    tokens = tokens - templates[layer]
                mapped, ari, ious = cluster_image(tokens, owner[b], k, args.seed)
                per_layer[layer] = mapped
                records.append({
                    "subset": name, "variant": variant,
                    "pair_index": rec["pair_index"],
                    "layer": layer, "k": k, "ari": ari,
                    "iou_target": ious.get(1),
                    "iou_distractor": ious.get(2),
                    "iou_foreground": ious.get("fg"),
                    "n_owner_target": int((owner[b] == 1).sum()),
                    "n_owner_distractor": int((owner[b] == 2).sum()),
                    "cluster_sizes": [int((mapped == i).sum()) for i in range(k)],
                })
            mappeds.append(per_layer)
        tag = "" if variant == "raw" else "_bgsub"
        note = "" if variant == "raw" else ", background template subtracted"
        plot_cluster_overlays(
            images, mappeds, recs, args.grid,
            out_dir / f"cluster_overlay_{name}{tag}.png",
            suptitle=f"{args.model_label} — patch-token KMeans (k={k}) — {name}, no cross-attention"
                     f"{note}; target cluster red, distractor cluster blue "
                     "(alpha 0.3)")
    return records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint",
                    default="outputs/model/clevr_dinov2_decoder1l_scratch_s42/best.pt")
    ap.add_argument("--n1-dir", default="data/clevr_object_count/n1")
    ap.add_argument("--n2-dir", default="data/clevr_object_count/n2")
    ap.add_argument("--out-dir", default="outputs/analysis/patch_pca_cluster")
    ap.add_argument("--masks-only", action="store_true",
                    help="selection + segmentation + masks_debug, no model")
    ap.add_argument("--replot", action="store_true",
                    help="all analyses from cached npz, no model")
    ap.add_argument("--model-label", default=None,
                    help="backbone name printed first in every figure title "
                         "(default: inferred from --checkpoint; pass it "
                         "explicitly with --replot)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-combos", type=int, default=6)
    ap.add_argument("--n-cluster-pairs", type=int, default=5)
    ap.add_argument("--bg-per-image", type=int, default=100)
    ap.add_argument("--grid", type=int, default=24)
    ap.add_argument("--resolution", type=int, default=336)
    ap.add_argument("--coverage-thresh", type=float, default=0.2)
    ap.add_argument("--sat-thresh", type=float, default=0.18)
    ap.add_argument("--hue-thresh", type=float, default=0.17)
    args = ap.parse_args()

    apply_style()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tee_stdout(out_dir)
    if args.model_label is None:
        stem = Path(args.checkpoint).parent.name
        args.model_label = next(
            (v for k, v in BACKBONE_LABELS.items() if f"_{k}_" in f"_{stem}_"),
            stem)
    label = args.model_label
    print(f"args: {vars(args)}")

    dirs = {"n1": args.n1_dir, "n2": args.n2_dir}
    if not args.replot:
        n1_entries = load_entries(args.n1_dir)
        n2_entries = load_entries(args.n2_dir)
        rng = np.random.RandomState(args.seed)
        pca_pairs, cluster_pairs, combos = select_pairs(
            n1_entries, n2_entries, rng, args.n_combos, args.n_cluster_pairs)
        chosen = pca_pairs + [i for i in cluster_pairs if i not in pca_pairs]
        print(f"PCA combos ({len(combos)}): {['-'.join(c) for c in combos]}")
        print(f"PCA pairs ({len(pca_pairs)}): {pca_pairs}")
        print(f"Cluster pairs ({len(cluster_pairs)}): {cluster_pairs}")
        for i in chosen:  # paired-render invariant
            assert all(n1_entries[i][a] == n2_entries[i][a]
                       for a in ("filename", "color", "shape", "material",
                                 "size", "x", "y")), f"pair {i} mismatch"

        state = {}
        for name, entries in (("n1", n1_entries), ("n2", n2_entries)):
            sub = out_dir / name
            sub.mkdir(parents=True, exist_ok=True)
            print(f"\n[{name}] building masks for {len(chosen)} images ...")
            images, owners, _ = build_masks(entries, chosen, dirs[name], args)
            labels = []
            for kx, i in enumerate(chosen):
                e = entries[i]
                labels.append({
                    "pair_index": i, "filename": e["filename"],
                    "target": {a: e[a] for a in
                               ("color", "shape", "material", "size")},
                    "position": {a: e[a] for a in ("x", "y", "rotation")},
                    "distractors": e["distractors"],
                    "in_pca_set": i in pca_pairs,
                    "in_cluster_set": i in cluster_pairs,
                    "n_target_patches": int((owners[kx] == 1).sum()),
                    "n_distractor_patches": int((owners[kx] == 2).sum()),
                })
            with open(sub / "labels.json", "w") as f:
                json.dump(labels, f, indent=1)
            save_masks_debug(images, owners, entries, chosen, args.grid,
                             sub / "masks_debug.png")
            if not args.masks_only:
                extract_condition(sub, "noca", images, owners, None, args, state)
        if args.masks_only:
            print("\n--masks-only: inspect n1/n2 masks_debug.png, then rerun.")
            return

    # ---- analyses (from cache; --replot enters here directly) ----
    subsets = {}
    for name in ("n1", "n2"):
        feats, owner, labels = load_cached(out_dir / name, "noca")
        subsets[name] = (feats, owner, labels, dirs[name])

    # A. PCA
    pca_flags = [np.array([rec["in_pca_set"] for rec in subsets[n][2]])
                 for n in ("n1", "n2")]
    pcas = fit_layer_pcas([subsets[n][0] for n in ("n1", "n2")], pca_flags)
    n_pca = int(pca_flags[0].sum())
    for name, tag, other in (("n1", "1-object", "2"), ("n2", "2-object", "1")):
        feats, owner, labels, _ = subsets[name]
        plot_patch_pca(
            feats, owner, labels, pcas, out_dir / f"pca_{name}.png",
            suptitle=f"{label} — patch-token PCA — {n_pca}×{tag} scenes (fit shared "
                     f"with the {other}-object set), no cross-attention",
            bg_per_image=args.bg_per_image, seed=args.seed)

    # single-image PCA (first cluster-set pair): all patches of ONE image,
    # per-layer own fit + positional-banding diagnostic row
    for name, tag in (("n1", "1-object"), ("n2", "2-object")):
        feats, owner, labels, data_dir = subsets[name]
        b = next(i for i, rec in enumerate(labels) if rec["in_cluster_set"])
        plot_patch_pca_single(
            feats, owner, labels, b, args.grid,
            out_dir / f"pca_single_{name}.png",
            suptitle=f"{label} — patch-token PCA — one {tag} image "
                     f"({labels[b]['filename']}), all patches, per-layer fit; "
                     "bottom row colored by patch row",
            image_path=Path(data_dir) / "images" / labels[b]["filename"])

    print("\nAdditive-offset statistics (768-d):")
    stats = offset_statistics(subsets["n1"][0], subsets["n1"][1],
                              subsets["n2"][0], subsets["n2"][1],
                              subsets["n1"][2])
    with open(out_dir / "offset_stats.json", "w") as f:
        json.dump(stats, f, indent=1)
    print(f"Saved: {out_dir / 'offset_stats.json'}")

    # B. KMeans (raw tokens = the specified experiment; bgsub = the additive
    # hypothesis' prediction: residuals after removing background(position)
    # should make the objects clusterable)
    records = []
    for variant in ("raw", "bgsub"):
        recs = run_clustering(subsets, out_dir, args, variant=variant)
        records += recs
        tag = "" if variant == "raw" else "_bgsub"
        plot_cluster_metrics(
            recs, out_dir / f"cluster_metrics{tag}.png",
            suptitle=f"{label} — patch-token KMeans vs owner masks "
                     f"({'raw tokens' if variant == 'raw' else 'background template subtracted'})")
        print(f"\nClustering summary [{variant}] (mean over cluster-set pairs):")
        for layer in GCA_LAYERS:
            parts = [f"L{layer}"]
            for subset, field in (("n1", "iou_target"), ("n2", "iou_target"),
                                  ("n2", "iou_distractor"),
                                  ("n2", "iou_foreground")):
                vals = [r[field] for r in recs
                        if r["subset"] == subset and r["layer"] == layer
                        and r.get(field) is not None]
                parts.append(f"{subset} {field.split('_')[1]} "
                             f"{np.mean(vals):.3f}")
            print("  ".join(parts))
    with open(out_dir / "cluster_metrics.json", "w") as f:
        json.dump(records, f, indent=1)


if __name__ == "__main__":
    main()
