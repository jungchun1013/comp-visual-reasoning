"""Patch-level t-SNE of individual (unpooled) ViT patch tokens — X16.

Every existing t-SNE/probe mean-pools patch tokens; this script embeds single
patch tokens per GCA layer to test whether attribute organization exists at the
patch level, whether it survives two objects, and whether Grounding (CA with a
shape-referring question) reorganizes the referent's patches.

Four figures on clevr_dinov2_decoder1l_scratch_s42 (24x24 patch grid),
t-SNE grid over GCA layers [1,3,5,7,9,11]:
  1. tsne_patch_noca_img0.png  — one 2-object image, all 576 patches
  2. single_object_10/tsne_patch_noca.png — 10 x 1-object overlay
  3. two_object_10/tsne_patch_noca.png    — 10 x 2-object overlay
  4. two_object_10/tsne_patch_ca_refshape.png — same 10 images, CA
     "What color is the {target.shape}?", referent patches edged

Object->patch masks come from pixel-color segmentation (two-object metadata has
no pixel_coords). Phases: --masks-only (CPU, inspect masks_debug.png before any
GPU spend) -> extraction (GPU, cached npz; refuses to re-extract) -> --replot.

Run from main/:
  PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 <interpreter> \
      scripts/analysis/tsne_patch_level.py [--masks-only|--replot]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.amp import autocast
from PIL import Image as PILImage
from scipy import ndimage
from sklearn.manifold import TSNE
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
from tsne_single_object import SingleObjRetriever, SHAPE_MARKERS

# CLEVR base colors, from SteerViT-legacy/tools/clevr-dataset-gen/
# image_generation/data/properties.json (render-time palette).
CLEVR_RGB = {
    "gray": (87, 87, 87), "red": (173, 35, 35), "blue": (42, 75, 215),
    "green": (29, 105, 20), "brown": (129, 74, 25), "purple": (129, 38, 192),
    "cyan": (41, 208, 208), "yellow": (255, 238, 51),
}


class PatchRetriever(SingleObjRetriever):
    """SingleObjRetriever without the mean-pool: per-layer (B, P, D) tokens."""

    @torch.no_grad()
    def extract_all(self, images, question_text=None):
        layer_out = {}
        hooks = []
        for idx, blk in enumerate(self.blocks):
            def make_hook(li):
                def fn(mod, inp, out):
                    o = out[0] if isinstance(out, tuple) else out
                    layer_out[li] = o.detach()
                return fn
            hooks.append(blk.register_forward_hook(make_hook(idx)))

        if question_text is None:
            texts = None
        elif isinstance(question_text, str):
            texts = [question_text] * images.shape[0]
        else:
            texts = list(question_text)
        if images.is_cuda:
            with autocast(device_type="cuda", dtype=torch.bfloat16):
                self.steervit.forward(images, texts)
        else:
            self.steervit.forward(images, texts)

        for h in hooks:
            h.remove()

        feats = {}
        for l in range(self.num_layers):
            normed = self.norm(layer_out[l].float())
            feats[l] = normed[:, self.prefix:, :].cpu().numpy().astype(np.float16)
        return feats


# ---------------------------------------------------------------------------
# Stimulus selection (deterministic under --seed; chosen indices go to labels.json)
# ---------------------------------------------------------------------------

def load_entries(data_dir):
    """Entry dicts without loading images: {filename, color, shape, material,
    size, distractors:[...]} for both scenes.json and metadata.json formats."""
    data_dir = Path(data_dir)
    if (data_dir / "metadata.json").exists():
        with open(data_dir / "metadata.json") as f:
            raw = json.load(f)
        return [dict(e, distractors=e.get("distractors", [])) for e in raw]
    with open(data_dir / "scenes.json") as f:
        scenes = json.load(f)["scenes"]
    return [dict(s["objects"][0], filename=s["image_filename"], distractors=[])
            for s in scenes]


def select_single_stimuli(entries, rng, n=10):
    """Round-robin the 3 shapes (4/3/3), prefer unused colors, exclude gray."""
    pools = {s: [i for i, e in enumerate(entries)
                 if e["shape"] == s and e["color"] != "gray"]
             for s in SHAPE_MARKERS}
    for p in pools.values():
        rng.shuffle(p)
    shapes = list(SHAPE_MARKERS)
    chosen, used_colors = [], set()
    for k in range(n):
        pool = pools[shapes[k % len(shapes)]]
        pick = next((i for i in pool if entries[i]["color"] not in used_colors),
                    pool[0])
        pool.remove(pick)
        used_colors.add(entries[pick]["color"])
        chosen.append(pick)
    return chosen


def select_two_stimuli(entries, rng, n=10):
    """Target/distractor differ in shape AND color, neither gray; greedily
    maximize distinct (target.color, distractor.color) pairs."""
    eligible = [i for i, e in enumerate(entries)
                if e["distractors"]
                and e["shape"] != e["distractors"][0]["shape"]
                and e["color"] != e["distractors"][0]["color"]
                and e["color"] != "gray"
                and e["distractors"][0]["color"] != "gray"]
    rng.shuffle(eligible)
    chosen, seen_pairs = [], set()
    for i in eligible:
        pair = (entries[i]["color"], entries[i]["distractors"][0]["color"])
        if pair not in seen_pairs:
            chosen.append(i)
            seen_pairs.add(pair)
        if len(chosen) == n:
            return chosen
    chosen += [i for i in eligible if i not in chosen][:n - len(chosen)]
    return chosen


# ---------------------------------------------------------------------------
# Pixel-color segmentation -> patch owner masks
# ---------------------------------------------------------------------------

def _hue(rgb):
    from matplotlib.colors import rgb_to_hsv
    return rgb_to_hsv(np.asarray(rgb, dtype=np.float32).reshape(-1, 3))[..., 0]


def _hue_dist(h1, h2):
    d = np.abs(h1 - h2)
    return np.minimum(d, 1.0 - d)


def object_pixel_masks(img, obj_colors, sat_thresh, hue_thresh):
    """Per-object boolean pixel masks at native resolution.

    Floor/backdrop is desaturated gray -> saturation gate. Foreground pixels
    are assigned to the nearest scene object color by hue: the renders are dim
    (ambient light pulls chromaticity toward gray) but hue is stable under
    that mixture; specular highlights are desaturated and fall to the sat
    gate, leaving holes that binary_closing fills."""
    rgb = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    sat = (rgb.max(-1) - rgb.min(-1)) / (rgb.max(-1) + 1e-6)
    pix_hue = _hue(rgb).reshape(rgb.shape[:2])
    dists = np.stack([_hue_dist(pix_hue, _hue(np.array(CLEVR_RGB[c]) / 255.0)[0])
                      for c in obj_colors])
    valid = (sat > sat_thresh) & (dists.min(0) <= hue_thresh)
    nearest = dists.argmin(0)
    masks = []
    for i in range(len(obj_colors)):
        m = valid & (nearest == i)
        m = ndimage.binary_closing(m, structure=np.ones((3, 3)))
        m = ndimage.binary_opening(m, structure=np.ones((3, 3)))
        lab, nl = ndimage.label(m)
        if nl > 1:
            sizes = ndimage.sum(m, lab, range(1, nl + 1))
            m = lab == (1 + int(np.argmax(sizes)))
        masks.append(m)
    return masks


def patch_owner_from_masks(masks, grid, resolution, coverage_thresh):
    """(grid*grid,) int8 owner: 0 bg, 1 target, 2 distractor. Masks go through
    the same non-aspect-preserving squash as the model input (Resize
    (res,res)), then per-patch coverage (clevr_ref._patchify_mask pattern)."""
    ps = resolution // grid
    covs = []
    for m in masks:
        pm = PILImage.fromarray(m.astype(np.uint8) * 255).resize(
            (resolution, resolution), PILImage.NEAREST)
        pm = np.asarray(pm) > 127
        covs.append(pm.reshape(grid, ps, grid, ps).mean(axis=(1, 3)).reshape(-1))
    covs = np.stack(covs)
    owner = np.zeros(grid * grid, dtype=np.int8)
    keep = covs.max(0) >= coverage_thresh
    owner[keep] = covs.argmax(0)[keep] + 1
    for i in range(len(masks)):
        if not (owner == i + 1).any():
            # tiny/distant object below the coverage threshold everywhere:
            # keep its single best patch (among patches it wins)
            won = np.nonzero(covs.argmax(0) == i)[0]
            if len(won) and covs[i, won].max() > 0:
                owner[won[covs[i, won].argmax()]] = i + 1
    return owner


def build_masks(entries, chosen, data_dir, args):
    """Per chosen image: PIL image, pixel masks, patch owner + sanity checks."""
    images, owners, pixel_masks = [], [], []
    for i in chosen:
        e = entries[i]
        img = PILImage.open(Path(data_dir) / "images" / e["filename"]).convert("RGB")
        obj_colors = [e["color"]] + [d["color"] for d in e["distractors"]]
        masks = object_pixel_masks(img, obj_colors, args.sat_thresh,
                                   args.hue_thresh)
        owner = patch_owner_from_masks(masks, args.grid, args.resolution,
                                       args.coverage_thresh)
        rgb = np.asarray(img, dtype=np.float32)
        for oi, (m, c) in enumerate(zip(masks, obj_colors)):
            n_patch = int((owner == oi + 1).sum())
            assert m.any() and n_patch > 0, \
                f"{e['filename']}: empty mask for object {oi} ({c})"
            if n_patch < 4 or n_patch > 150:
                print(f"  WARNING {e['filename']} obj{oi} ({c}): "
                      f"{n_patch} patches (expect 4-150)")
            # hue check against the full chromatic palette (chromaticity is
            # unusable here: dim renders drift toward gray)
            mean_hue = _hue(rgb[m].mean(0) / 255.0)[0]
            pal_d = {k: float(_hue_dist(np.array([mean_hue]),
                                        _hue(np.array(v) / 255.0)[0])[0])
                     for k, v in CLEVR_RGB.items() if k != "gray"}
            assert min(pal_d, key=pal_d.get) == c, \
                f"{e['filename']}: mask mean hue nearest to " \
                f"{min(pal_d, key=pal_d.get)}, expected {c}"
        images.append(img)
        owners.append(owner)
        pixel_masks.append(masks)
    return images, np.stack(owners), pixel_masks


def save_masks_debug(images, owners, entries, chosen, grid, out_path):
    n = len(images)
    fig, axes = plt.subplots((n + 4) // 5, 5,
                             figsize=(3.2 * 5, 2.4 * ((n + 4) // 5)))
    axes = np.atleast_1d(axes).flatten()
    for k, ax in enumerate(axes):
        ax.axis("off")
        if k >= n:
            continue
        img = np.asarray(images[k], dtype=np.float32) / 255.0
        e = entries[chosen[k]]
        obj_colors = [e["color"]] + [d["color"] for d in e["distractors"]]
        w, h = images[k].size
        up = np.asarray(PILImage.fromarray(
            owners[k].reshape(grid, grid).astype(np.uint8)).resize(
            (w, h), PILImage.NEAREST))
        overlay = img.copy()
        for oi, c in enumerate(obj_colors):
            sel = up == oi + 1
            overlay[sel] = 0.6 * overlay[sel] + 0.4 * np.array(CLEVR_RGB[c]) / 255.0
        ax.imshow(overlay)
        counts = [(owners[k] == oi + 1).sum() for oi in range(len(obj_colors))]
        ax.set_title(f"{e['filename']} {'/'.join(obj_colors)} {counts}",
                     fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=S["dpi"], bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_patch_tsne(feats, owner, entries, chosen, layers, out_path, *,
                    suptitle, image_subset=None, referent_edges=False,
                    referent_owner=1, bg_per_image=100, perplexity=30.0,
                    seed=42):
    """Generic t-SNE grid over layers. feats: {layer: (B,P,D)}; owner: (B,P).
    image_subset=None uses all images with bg subsampling; a list of image
    indices with bg_per_image=None keeps every patch (Fig 1)."""
    B, P = owner.shape
    imgs = list(range(B)) if image_subset is None else list(image_subset)

    rows = []  # (img, patch, owner, shape, color)
    for b in imgs:
        e = entries[chosen[b]]
        obj_attrs = [(e["shape"], e["color"])] + \
                    [(d["shape"], d["color"]) for d in e["distractors"]]
        obj_idx = np.nonzero(owner[b] > 0)[0]
        bg_idx = np.nonzero(owner[b] == 0)[0]
        if bg_per_image is not None and len(bg_idx) > bg_per_image:
            # deterministic given (owner, seed) -> shared across conditions
            bg_idx = np.random.RandomState(seed + b).choice(
                bg_idx, bg_per_image, replace=False)
        for p in obj_idx:
            sh, co = obj_attrs[owner[b, p] - 1]
            rows.append((b, p, owner[b, p], sh, co))
        for p in bg_idx:
            rows.append((b, p, 0, None, None))

    X_all = {l: np.stack([feats[l][b, p] for b, p, *_ in rows]).astype(np.float32)
             for l in layers}
    own = np.array([r[2] for r in rows])
    shp = np.array([r[3] if r[3] else "" for r in rows])
    col = np.array([r[4] if r[4] else "" for r in rows])

    fig, axes = make_tsne_grid(len(layers), ncols=3)
    for ax_i, layer in enumerate(layers):
        ax = axes[ax_i]
        n = len(rows)
        perp = min(perplexity, (n - 1) // 3)
        emb = TSNE(n_components=2, perplexity=perp,
                   random_state=seed).fit_transform(X_all[layer])
        bg = own == 0
        ax.scatter(emb[bg, 0], emb[bg, 1], c=[TSNE_STYLE["gray"]],
                   s=TSNE_STYLE["bg_size"], marker="o", edgecolors="none",
                   rasterized=True)
        for shape, marker in SHAPE_MARKERS.items():
            for color in ATTR_VALUE_COLORS["color"]:
                # referent drawn last, on top
                for ow in (3 - referent_owner, referent_owner):
                    m = (shp == shape) & (col == color) & (own == ow)
                    if not m.any():
                        continue
                    kw = dict(c=[ATTR_VALUE_COLORS["color"][color]],
                              marker=marker, rasterized=True)
                    if referent_edges and ow == referent_owner:
                        kw.update(s=TSNE_STYLE["hi_size"], edgecolors="black",
                                  linewidths=TSNE_STYLE["edge_width"])
                    else:
                        kw.update(s=TSNE_STYLE["mid_size"], edgecolors="none")
                    ax.scatter(emb[m, 0], emb[m, 1], **kw)
        ax.set_title(f"L{layer}", fontsize=S["subplot_title_fontsize"])
        style_tsne_ax(ax)
    for ax in axes[len(layers):]:
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
    if referent_edges:
        handles.append(Line2D([0], [0], marker="o", color="w",
                              markerfacecolor=TSNE_STYLE["gray"], markersize=8,
                              markeredgecolor="black", markeredgewidth=1.0,
                              label="referent (edge)"))
    finish_tsne_grid(fig, handles, suptitle=suptitle, ncol=5)
    fig.savefig(out_path, dpi=S["dpi"], bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def referring_question(entry):
    return f"What color is the {entry['shape']}?"


def referring_question_distractor(entry):
    """Refer to the small/distant distractor -> low-salience referent."""
    return f"What color is the {entry['distractors'][0]['shape']}?"


def prepare_subset(name, data_dir, select_fn, args, out_dir):
    sub_dir = out_dir / name
    sub_dir.mkdir(parents=True, exist_ok=True)
    entries = load_entries(data_dir)
    chosen = select_fn(entries, np.random.RandomState(args.seed))
    print(f"\n[{name}] {len(chosen)} stimuli from {data_dir}: {chosen}")
    images, owners, _ = build_masks(entries, chosen, data_dir, args)
    labels = []
    for k, i in enumerate(chosen):
        e = entries[i]
        rec = {"dataset_index": i, "filename": e["filename"],
               "target": {a: e[a] for a in ("color", "shape", "material", "size")},
               "distractors": e["distractors"],
               "n_target_patches": int((owners[k] == 1).sum()),
               "n_distractor_patches": int((owners[k] == 2).sum())}
        if e["distractors"]:
            assert e["shape"] != e["distractors"][0]["shape"]
            rec["question"] = referring_question(e)
            print(f'  "{rec["question"]}" -> target={e["color"]} {e["shape"]}'
                  f' ({rec["n_target_patches"]}p), distractor='
                  f'{e["distractors"][0]["color"]} {e["distractors"][0]["shape"]}'
                  f' ({rec["n_distractor_patches"]}p)')
        else:
            print(f'  {e["filename"]}: {e["color"]} {e["shape"]}'
                  f' ({rec["n_target_patches"]}p)')
        labels.append(rec)
    with open(sub_dir / "labels.json", "w") as f:
        json.dump(labels, f, indent=1)
    save_masks_debug(images, owners, entries, chosen, args.grid,
                     sub_dir / "masks_debug.png")
    return sub_dir, entries, chosen, images, owners


def extract_condition(sub_dir, condition, images, owners, questions, args, state):
    npz = sub_dir / f"feats_{condition}.npz"
    if npz.exists():
        print(f"Cached features exist, not re-extracting: {npz}")
        return
    if state.get("retriever") is None:
        from model.checkpoint_io import load_any_checkpoint
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading checkpoint on {device}: {args.checkpoint}")
        model, steervit, transform, vocab, task_type, meta = \
            load_any_checkpoint(args.checkpoint, device)
        assert len(steervit.vision_model.trunk.blocks) == 12
        state.update(retriever=PatchRetriever(steervit), transform=transform,
                     device=device)
    retr, tf, device = state["retriever"], state["transform"], state["device"]
    batch = torch.stack([tf(im) for im in images]).to(device)
    print(f"Extracting {condition}: {len(images)} images ...")
    feats = retr.extract_all(batch, questions)
    P = feats[GCA_LAYERS[0]].shape[1]
    assert P == args.grid ** 2, f"P={P}, expected {args.grid ** 2}"
    np.savez(npz, owner=owners,
             **{str(l): feats[l] for l in GCA_LAYERS})
    print(f"Saved: {npz}")


def load_cached(sub_dir, condition):
    npz = sub_dir / f"feats_{condition}.npz"
    data = np.load(npz)
    feats = {l: data[str(l)] for l in GCA_LAYERS}
    with open(sub_dir / "labels.json") as f:
        labels = json.load(f)
    return feats, data["owner"], labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint",
                    default="outputs/model/clevr_dinov2_decoder1l_scratch_s42/best.pt")
    ap.add_argument("--single-dir", default="data/clevr_single_object_v3")
    ap.add_argument("--two-dir", default="data/clevr_two_object_v2")
    ap.add_argument("--out-dir", default="outputs/analysis/tsne/patch_level")
    ap.add_argument("--masks-only", action="store_true",
                    help="stimulus selection + segmentation + masks_debug, no model")
    ap.add_argument("--replot", action="store_true",
                    help="regenerate all figures from cached npz, no model/GPU")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--perplexity", type=float, default=30.0)
    ap.add_argument("--bg-per-image", type=int, default=100)
    ap.add_argument("--grid", type=int, default=24)
    ap.add_argument("--resolution", type=int, default=336)
    ap.add_argument("--coverage-thresh", type=float, default=0.2)
    ap.add_argument("--sat-thresh", type=float, default=0.18)
    ap.add_argument("--hue-thresh", type=float, default=0.17,
                    help="max circular hue distance (0-0.5) for assignment")
    args = ap.parse_args()

    apply_style()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tee_stdout(out_dir)
    print(f"args: {vars(args)}")

    single_dir = out_dir / "single_object_10"
    two_dir = out_dir / "two_object_10"

    if not args.replot:
        state = {}
        s_dir, s_entries, s_chosen, s_images, s_owners = prepare_subset(
            "single_object_10", args.single_dir, select_single_stimuli,
            args, out_dir)
        t_dir, t_entries, t_chosen, t_images, t_owners = prepare_subset(
            "two_object_10", args.two_dir, select_two_stimuli, args, out_dir)
        if args.masks_only:
            print("\n--masks-only: inspect masks_debug.png, then rerun for "
                  "extraction.")
            return
        extract_condition(s_dir, "noca", s_images, s_owners, None, args, state)
        extract_condition(t_dir, "noca", t_images, t_owners, None, args, state)
        questions = [referring_question(t_entries[i]) for i in t_chosen]
        extract_condition(t_dir, "ca_refshape", t_images, t_owners, questions,
                          args, state)
        questions_d = [referring_question_distractor(t_entries[i])
                       for i in t_chosen]
        extract_condition(t_dir, "ca_refshape_distractor", t_images, t_owners,
                          questions_d, args, state)

    # Figures (from cache; --replot enters here directly)
    kw = dict(layers=GCA_LAYERS, perplexity=args.perplexity, seed=args.seed)
    feats, owner, labels = load_cached(single_dir, "noca")
    chosen = [r["dataset_index"] for r in labels]
    entries = {i: dict(r["target"], filename=r["filename"],
                       distractors=r["distractors"]) for i, r in zip(chosen, labels)}
    plot_patch_tsne(feats, owner, entries, chosen,
                    out_path=single_dir / "tsne_patch_noca.png",
                    suptitle="Patch-level t-SNE — 10×1-object, no cross-attention",
                    bg_per_image=args.bg_per_image, **kw)

    feats, owner, labels = load_cached(two_dir, "noca")
    chosen = [r["dataset_index"] for r in labels]
    entries = {i: dict(r["target"], filename=r["filename"],
                       distractors=r["distractors"]) for i, r in zip(chosen, labels)}
    plot_patch_tsne(feats, owner, entries, chosen,
                    out_path=two_dir / "tsne_patch_noca_img0.png",
                    suptitle="Patch-level t-SNE — one 2-object image, "
                             "no cross-attention",
                    image_subset=[0], bg_per_image=None, **kw)
    plot_patch_tsne(feats, owner, entries, chosen,
                    out_path=two_dir / "tsne_patch_noca.png",
                    suptitle="Patch-level t-SNE — 10×2-object, no cross-attention",
                    bg_per_image=args.bg_per_image, **kw)

    feats, owner, labels = load_cached(two_dir, "ca_refshape")
    plot_patch_tsne(feats, owner, entries, chosen,
                    out_path=two_dir / "tsne_patch_ca_refshape.png",
                    suptitle='Patch-level t-SNE — 10×2-object, CA "What color '
                             'is the {shape}?" (referent edged)',
                    referent_edges=True, bg_per_image=args.bg_per_image, **kw)
    plot_patch_tsne(feats, owner, entries, chosen,
                    out_path=two_dir / "tsne_patch_ca_refshape_img0.png",
                    suptitle='Patch-level t-SNE — one 2-object image, CA '
                             '"What color is the {shape}?" (referent edged)',
                    image_subset=[0], bg_per_image=None,
                    referent_edges=True, **kw)

    feats, owner, labels = load_cached(two_dir, "ca_refshape_distractor")
    plot_patch_tsne(feats, owner, entries, chosen,
                    out_path=two_dir /
                    "tsne_patch_ca_refshape_distractor_img0.png",
                    suptitle='Patch-level t-SNE — one 2-object image, CA '
                             '"What color is the {distractor.shape}?" '
                             '(small referent edged)',
                    image_subset=[0], bg_per_image=None,
                    referent_edges=True, referent_owner=2, **kw)


if __name__ == "__main__":
    main()
