"""Grounding cluster causal test — manipulation experiment.

Tests whether the grounding sub-clustering within the answer-match group
is causally important or merely coincidental.

Two manipulations:
  1. grounding: Shift grounding-match images toward non-grounding-answer-match
     images. Destroys grounding sub-cluster while preserving answer cluster.
  2. answer:    Shift all answer-match images toward non-answer images at the
     last layer. Positive control — should destroy answer signal.

Metrics:
  - Conditional RSA (Spearman ρ) before and after manipulation
  - Retrieval accuracy (nearest-neighbor answer match)

Usage:
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python scripts/analysis/grounding_manipulation.py \
        --checkpoint outputs/model/clevr_dinov2_decoder1l_scratch_s42/best.pt
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.distance import pdist, cdist
from scipy.stats import spearmanr, wilcoxon
from sklearn.manifold import TSNE
from torch.amp import autocast

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from data.clevr import CLEVRVQADataset
from data.clevr_sampling import RETRIEVAL_CATEGORIES, build_family_index, sample_queries
from data.clevr_conditions import (
    COND_COLORS, FAMILY_SUBCATEGORY, SPATIAL_SUBCATEGORIES,
    extract_query_info, label_db_image, ALL_COND_NAMES,
)
from omegaconf import OmegaConf

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from analysis.plot_style import PLOT_STYLE, apply_style


# ── Reuse model loading & feature extraction from conditional_rsa ─

def load_model(ckpt_path, device):
    from model import CrossAttnViT
    from tasks.decoder import build_decoder_model, build_clevr_decoder_vocab

    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    cfg = OmegaConf.create(ckpt["config"])
    cross_attn_layers = list(cfg.model.cross_attn_layers)
    pretrained = cfg.model.get("pretrained", True)
    condition_type = cfg.model.get("condition_type", "gca")
    steervit = CrossAttnViT.from_config(
        cfg.model.backbone_name, device=device,
        cross_attn_layers=cross_attn_layers,
        resolution=cfg.model.resolution,
        pretrained=pretrained,
        condition_type=condition_type,
    )
    vocab = build_clevr_decoder_vocab()
    model_cfg = OmegaConf.create({"model": cfg.model, "task": cfg.task, "data": cfg.data})
    model = build_decoder_model(steervit, model_cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()
    print(f"Loaded (epoch {ckpt.get('epoch', '?')})")
    return model, steervit, vocab


class AllLayerRetriever:
    def __init__(self, steervit):
        self.steervit = steervit
        self.blocks = steervit.vision_model.trunk.blocks
        self.norm = steervit.vision_model.trunk.norm
        self.prefix = steervit.vision_model.trunk.num_prefix_tokens
        self.num_layers = len(self.blocks)

    @torch.no_grad()
    def extract(self, images, questions):
        layer_out = {}
        hooks = []
        for idx, blk in enumerate(self.blocks):
            def make_hook(li):
                def fn(module, inp, output):
                    out = output[0] if isinstance(output, tuple) else output
                    layer_out[li] = out.detach()
                return fn
            hooks.append(blk.register_forward_hook(make_hook(idx)))

        with autocast(device_type="cuda", dtype=torch.bfloat16):
            self.steervit.forward(images, questions)

        for h in hooks:
            h.remove()

        feats = {}
        for l in range(self.num_layers):
            normed = self.norm(layer_out[l].float())
            patches = normed[:, self.prefix:, :]
            feats[l] = patches.mean(dim=1).cpu()
        return feats

    def build_steered_database(self, dataset, question, device, db_indices,
                               batch_size=32):
        num_layers = self.num_layers
        all_feats = {l: [] for l in range(num_layers)}
        for start in range(0, len(db_indices), batch_size):
            end = min(start + batch_size, len(db_indices))
            batch_imgs = torch.stack(
                [dataset[db_indices[i]]["image"] for i in range(start, end)]
            ).to(device)
            bs = end - start
            q = [question] * bs if question else None
            feats = self.extract(batch_imgs, q)
            for l in range(num_layers):
                all_feats[l].append(feats[l])
        for l in range(num_layers):
            all_feats[l] = torch.cat(all_feats[l], dim=0)
        return all_feats


def build_db_pool(dataset, scenes, exclude_fname, min_obj, max_obj, num_db):
    seen = set()
    db_fnames, db_indices = [], []
    for idx in range(len(dataset)):
        fname = dataset.questions[idx]["image_filename"]
        if fname in seen or fname == exclude_fname:
            continue
        scene = scenes.get(fname)
        if scene is None:
            continue
        n_obj = len(scene["objects"])
        if n_obj < min_obj or n_obj > max_obj:
            continue
        seen.add(fname)
        db_fnames.append(fname)
        db_indices.append(idx)
        if len(db_indices) >= num_db:
            break
    return db_fnames, db_indices


def build_binary_rdm(labels):
    n = len(labels)
    rdm = np.empty(n * (n - 1) // 2)
    k = 0
    for i in range(n):
        for j in range(i + 1, n):
            rdm[k] = 0.0 if (labels[i] == labels[j]) else 1.0
            k += 1
    return rdm


# ── Manipulation logic ───────────────────────────────────────────

# Direct conditions: C0=extraction, C1=binding, C2=grounding, C3=position, C4=answer
COND_BINDING = 1
COND_GROUNDING = 2
COND_ANSWER = 4

# v2 naming (docs/legacy-reference.md §1.1): the 3-stage "binding -> object
# grounding -> answer matching" pipeline is now 2-stage "Binding -> Retrieval";
# "Grounding" names the whole language-conditioning mechanism, never a stage.
# manip_type values ("grounding"/"answer"/"random") and JSON keys are
# unchanged for schema compat -- this only maps them to figure-visible text.
MANIP_DISPLAY_NAME = {
    "grounding": "Object match",
    "answer": "Retrieval",
    "random": "Random",
}


def apply_grounding_manipulation(feats_np, cond_labels):
    """Shift grounding-match answer images toward non-grounding answer images.

    Groups:
      G   = {i : C2=True AND C4=True}   (grounding + answer)
      A\\G = {i : C4=True AND C2=False}  (answer only)
      O   = {i : C4=False}              (non-answer)

    Manipulation: for each i in G, r_i' = r_i + (centroid(A\\G) - centroid(G))

    Returns (manipulated_feats, group_info).
    """
    answer_mask = cond_labels[:, COND_ANSWER]
    grounding_mask = cond_labels[:, COND_GROUNDING]

    g_mask = answer_mask & grounding_mask         # G
    ag_mask = answer_mask & ~grounding_mask        # A\G

    n_g = g_mask.sum()
    n_ag = ag_mask.sum()

    if n_g < 2 or n_ag < 2:
        return feats_np, {"n_g": int(n_g), "n_ag": int(n_ag), "skipped": True}

    centroid_g = feats_np[g_mask].mean(axis=0)
    centroid_ag = feats_np[ag_mask].mean(axis=0)
    delta = centroid_ag - centroid_g

    manipulated = feats_np.copy()
    manipulated[g_mask] = manipulated[g_mask] + delta

    return manipulated, {
        "n_g": int(n_g), "n_ag": int(n_ag),
        "n_other": int((~answer_mask).sum()),
        "delta_norm": float(np.linalg.norm(delta)),
        "skipped": False,
    }


def apply_random_manipulation(feats_np, cond_labels, rng=None):
    """Random-direction control: same group, same magnitude, random direction.

    Applies to the same grounding group G = {C2=True ∧ C4=True},
    with a random vector of the same norm as the grounding δ.
    """
    if rng is None:
        rng = np.random.RandomState(42)

    answer_mask = cond_labels[:, COND_ANSWER]
    grounding_mask = cond_labels[:, COND_GROUNDING]

    g_mask = answer_mask & grounding_mask
    ag_mask = answer_mask & ~grounding_mask

    n_g = g_mask.sum()
    n_ag = ag_mask.sum()

    if n_g < 2 or n_ag < 2:
        return feats_np, {"n_g": int(n_g), "n_ag": int(n_ag), "skipped": True}

    centroid_g = feats_np[g_mask].mean(axis=0)
    centroid_ag = feats_np[ag_mask].mean(axis=0)
    real_delta = centroid_ag - centroid_g
    target_norm = np.linalg.norm(real_delta)

    rand_dir = rng.randn(feats_np.shape[1]).astype(feats_np.dtype)
    rand_dir = rand_dir / np.linalg.norm(rand_dir) * target_norm

    manipulated = feats_np.copy()
    manipulated[g_mask] = manipulated[g_mask] + rand_dir

    return manipulated, {
        "n_g": int(n_g), "n_ag": int(n_ag),
        "delta_norm": float(target_norm),
        "skipped": False,
    }


def apply_answer_manipulation(feats_np, cond_labels):
    """Shift all answer-match images toward non-answer images (positive control).

    Groups:
      A = {i : C4=True}
      O = {i : C4=False}

    Manipulation: for each i in A, r_i' = r_i + (centroid(O) - centroid(A))

    Returns (manipulated_feats, group_info).
    """
    answer_mask = cond_labels[:, COND_ANSWER]

    a_mask = answer_mask
    o_mask = ~answer_mask

    n_a = a_mask.sum()
    n_o = o_mask.sum()

    if n_a < 2 or n_o < 2:
        return feats_np, {"n_a": int(n_a), "n_o": int(n_o), "skipped": True}

    centroid_a = feats_np[a_mask].mean(axis=0)
    centroid_o = feats_np[o_mask].mean(axis=0)
    delta = centroid_o - centroid_a

    manipulated = feats_np.copy()
    manipulated[a_mask] = manipulated[a_mask] + delta

    return manipulated, {
        "n_a": int(n_a), "n_o": int(n_o),
        "delta_norm": float(np.linalg.norm(delta)),
        "skipped": False,
    }


# ── Delta computation & hooks for val acc ──────────────────────

def compute_delta(feats_np, cond_labels, manip_type, rng=None):
    """Extract manipulation delta vector from DB pool features."""
    answer_mask = cond_labels[:, COND_ANSWER]
    grounding_mask = cond_labels[:, COND_GROUNDING]

    if manip_type == "grounding":
        g_mask = answer_mask & grounding_mask
        ag_mask = answer_mask & ~grounding_mask
        if g_mask.sum() < 2 or ag_mask.sum() < 2:
            return None
        return feats_np[ag_mask].mean(axis=0) - feats_np[g_mask].mean(axis=0)
    elif manip_type == "random":
        if rng is None:
            rng = np.random.RandomState(42)
        g_mask = answer_mask & grounding_mask
        ag_mask = answer_mask & ~grounding_mask
        if g_mask.sum() < 2 or ag_mask.sum() < 2:
            return None
        real_delta = feats_np[ag_mask].mean(axis=0) - feats_np[g_mask].mean(axis=0)
        target_norm = np.linalg.norm(real_delta)
        rand_dir = rng.randn(feats_np.shape[1]).astype(feats_np.dtype)
        return rand_dir / np.linalg.norm(rand_dir) * target_norm
    elif manip_type == "answer":
        a_mask = answer_mask
        o_mask = ~answer_mask
        if a_mask.sum() < 2 or o_mask.sum() < 2:
            return None
        return feats_np[o_mask].mean(axis=0) - feats_np[a_mask].mean(axis=0)
    return None


def make_manip_hook(delta_tensor, prefix_tokens):
    """Forward hook that adds delta to all patch tokens at a ViT block."""
    def hook_fn(module, input, output):
        out = output[0] if isinstance(output, tuple) else output
        out = out.clone()
        out[:, prefix_tokens:, :] = out[:, prefix_tokens:, :] + delta_tensor
        if isinstance(output, tuple):
            return (out,) + output[1:]
        return out
    return hook_fn


# ── Evaluation metrics ──────────────────────────────────────────

def compute_conditional_rsa(feats_np, cond_labels, binding_mask):
    """Compute Grounding|Binding and Answer|Binding RSA on binding subset."""
    subset = np.where(binding_mask)[0]
    if len(subset) < 5:
        return None, None

    sub_feats = feats_np[subset]
    neural_rdm = pdist(sub_feats, metric="correlation")
    if np.isnan(neural_rdm).all():
        return None, None

    grounding_labels = cond_labels[subset, COND_GROUNDING]
    answer_labels = cond_labels[subset, COND_ANSWER]

    rho_grounding = rho_answer = None

    n_true = grounding_labels.sum()
    n_false = len(grounding_labels) - n_true
    if n_true >= 2 and n_false >= 2:
        hyp = build_binary_rdm(grounding_labels)
        r, _ = spearmanr(neural_rdm, hyp)
        rho_grounding = float(r)

    n_true = answer_labels.sum()
    n_false = len(answer_labels) - n_true
    if n_true >= 2 and n_false >= 2:
        hyp = build_binary_rdm(answer_labels)
        r, _ = spearmanr(neural_rdm, hyp)
        rho_answer = float(r)

    return rho_grounding, rho_answer


def compute_retrieval_accuracy(query_feat, db_feats, answer_labels):
    """1-NN retrieval accuracy: does the nearest DB image have the same answer?"""
    dists = cdist(query_feat.reshape(1, -1), db_feats, metric="cosine")[0]
    nn_idx = np.argmin(dists)
    return bool(answer_labels[nn_idx])


# ── Plot ─────────────────────────────────────────────────────────

def plot_results(results, output_dir):
    """Bar chart comparing RSA before/after manipulation across layers."""
    layers = results["layers_tested"]
    manip_type = results["manipulation"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax_idx, metric in enumerate(["grounding_rsa", "answer_rsa"]):
        ax = axes[ax_idx]

        before_key = f"{metric}_before"
        after_key = f"{metric}_after"

        before_vals = [results["per_layer"].get(str(l), {}).get(before_key, float("nan"))
                       for l in layers]
        after_vals = [results["per_layer"].get(str(l), {}).get(after_key, float("nan"))
                      for l in layers]

        x = np.arange(len(layers))
        width = 0.35

        ax.bar(x - width / 2, before_vals, width, label="Before", color="steelblue", alpha=0.8)
        ax.bar(x + width / 2, after_vals, width, label="After", color="coral", alpha=0.8)

        ax.set_xlabel("Manipulation Layer")
        ax.set_ylabel("Spearman ρ")
        label = "Object match|Binding" if "grounding" in metric else "Retrieval|Binding"
        ax.set_title(f"{label} RSA")
        ax.set_xticks(x)
        ax.set_xticklabels([str(l) for l in layers])
        ax.legend()
        ax.axhline(y=0, color="gray", linewidth=0.5)

    # Intentional: smaller suptitle than PLOT_STYLE (14 vs 20) for wide 2-panel bar chart
    fig.suptitle(f"Manipulation: {MANIP_DISPLAY_NAME.get(manip_type, manip_type)}",
                 fontsize=14)
    fig.tight_layout()
    out = output_dir / f"manipulation_{manip_type}.png"
    fig.savefig(str(out), dpi=PLOT_STYLE["dpi"], bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def plot_retrieval(results, output_dir):
    """Bar chart for retrieval accuracy before/after."""
    layers = results["layers_tested"]
    manip_type = results["manipulation"]

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))

    before_vals = [results["per_layer"].get(str(l), {}).get("retrieval_acc_before", float("nan"))
                   for l in layers]
    after_vals = [results["per_layer"].get(str(l), {}).get("retrieval_acc_after", float("nan"))
                  for l in layers]

    x = np.arange(len(layers))
    width = 0.35

    ax.bar(x - width / 2, before_vals, width, label="Before", color="steelblue", alpha=0.8)
    ax.bar(x + width / 2, after_vals, width, label="After", color="coral", alpha=0.8)

    ax.set_xlabel("Manipulation Layer")
    ax.set_ylabel("Retrieval Accuracy")
    ax.set_title(f"1-NN Retrieval Accuracy — "
                 f"{MANIP_DISPLAY_NAME.get(manip_type, manip_type)} manipulation")
    ax.set_xticks(x)
    ax.set_xticklabels([str(l) for l in layers])
    ax.legend()
    ax.set_ylim(0, 1)

    fig.tight_layout()
    out = output_dir / f"retrieval_{manip_type}.png"
    fig.savefig(str(out), dpi=PLOT_STYLE["dpi"], bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ── t-SNE visualization ────────────────────────────────────────

_tab10 = plt.cm.tab10.colors

# Intentional overrides vs PLOT_STYLE: smaller legend (14 vs 16) and
# suptitle (18 vs 20) for the t-SNE grids (matches tsne_viz.py).
S_TSNE = dict(PLOT_STYLE, legend_fontsize=14, suptitle_fontsize=18)

FILL_COLORS = [
    np.array(_tab10[0][:3]),   # blue  — feature binding
    np.array(_tab10[1][:3]),   # orange — object grounding
]
ANSWER_MATCH_COLOR = np.array(_tab10[3][:3])  # red


def _plot_one_tsne_panel(ax, emb, cond_labels, title_str):
    """Plot one t-SNE panel with condition coloring."""
    N = len(emb)
    gray = np.array([0.75, 0.75, 0.75])
    colors = np.tile(gray, (N, 1))

    binding = cond_labels[:, COND_BINDING]
    grounding = cond_labels[:, COND_GROUNDING]
    answer = cond_labels[:, COND_ANSWER]

    colors[binding] = FILL_COLORS[0]       # blue = binding
    colors[grounding] = FILL_COLORS[1]     # orange = grounding

    no_answer = ~answer
    ax.scatter(emb[no_answer, 0], emb[no_answer, 1],
               c=colors[no_answer], s=10, edgecolors="none",
               alpha=0.6, rasterized=True)
    ax.scatter(emb[answer, 0], emb[answer, 1],
               c=colors[answer], s=18,
               edgecolors=ANSWER_MATCH_COLOR, linewidths=1.2,
               alpha=0.8, rasterized=True)

    ax.set_title(title_str, fontsize=S_TSNE["subplot_title_fontsize"])
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_box_aspect(1)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)


def _extract_final_with_hook(retriever, dataset, question, device, indices,
                             hook_layer, delta_tensor, batch_size=32):
    """Extract final-layer features with manipulation hook at a specific layer.

    The hook adds delta to patch tokens at block[hook_layer]. Downstream layers
    process the modified representation, so the final-layer output reflects the
    propagated effect — same as what drives val acc.
    """
    blocks = retriever.blocks
    prefix = retriever.prefix
    last_layer = retriever.num_layers - 1

    all_feats = []
    for start in range(0, len(indices), batch_size):
        end = min(start + batch_size, len(indices))
        batch_imgs = torch.stack(
            [dataset[indices[i]]["image"] for i in range(start, end)]
        ).to(device)

        handle = blocks[hook_layer].register_forward_hook(
            make_manip_hook(delta_tensor, prefix))
        feats = retriever.extract(batch_imgs, [question] * (end - start))
        handle.remove()

        all_feats.append(feats[last_layer])

    return torch.cat(all_feats, dim=0)


def plot_manipulation_tsne(steervit, retriever, dataset, device, query, scenes,
                           category, test_layers, num_db, batch_size,
                           output_dir, seed=42, perplexity=30):
    """t-SNE of LAST-layer features, showing downstream effect of hooking at each layer.

    For each manipulation layer L:
      - Hook at block[L]: add δ to grounding group's patch tokens
      - Propagate through all subsequent layers
      - Show t-SNE of the resulting last-layer features vs clean
    This matches what the decoder sees, unifying t-SNE with val acc measurement.
    """
    apply_style()

    question = query["question"]
    answer = query["answer"]
    program = query["program"]
    dataset_idx = query["dataset_idx"]
    img_filename = dataset.questions[dataset_idx]["image_filename"]
    query_scene = scenes[img_filename]
    query_info = extract_query_info(query_scene["objects"], program, answer)

    db_fnames, db_indices = build_db_pool(
        dataset, scenes, img_filename, 3, 5, num_db)

    n_conds = len(ALL_COND_NAMES)
    cond_labels = np.zeros((len(db_fnames), n_conds), dtype=bool)
    for di, fname in enumerate(db_fnames):
        db_scene = scenes.get(fname)
        if db_scene is None:
            continue
        conds, _ = label_db_image(db_scene["objects"], query_info, category)
        for ci in range(min(len(conds), n_conds)):
            cond_labels[di, ci] = conds[ci]

    # Clean pass: extract features at ALL layers (need layer L for δ, last layer for t-SNE)
    db_feats = retriever.build_steered_database(
        dataset, question, device, db_indices, batch_size=batch_size)
    last_layer = retriever.num_layers - 1
    clean_final = db_feats[last_layer].numpy()

    g_mask = cond_labels[:, COND_ANSWER] & cond_labels[:, COND_GROUNDING]
    g_db_indices = [db_indices[i] for i in range(len(db_indices)) if g_mask[i]]
    print(f"  Grounding group: {len(g_db_indices)}/{len(db_indices)} images")

    n_layers = len(test_layers)
    fig, axes = plt.subplots(2, n_layers, figsize=(3.2 * n_layers, 6.4))
    if n_layers == 1:
        axes = axes.reshape(2, 1)

    for pi, layer in enumerate(test_layers):
        feats_at_L = db_feats[layer].numpy()
        delta = compute_delta(feats_at_L, cond_labels, "grounding")

        if delta is None or len(g_db_indices) == 0:
            tsne = TSNE(n_components=2, perplexity=perplexity,
                        random_state=seed, metric="cosine")
            emb = tsne.fit_transform(clean_final)
            _plot_one_tsne_panel(axes[0, pi], emb, cond_labels, f"Manip@L{layer}")
            _plot_one_tsne_panel(axes[1, pi], emb, cond_labels, f"Manip@L{layer}")
            print(f"  L{layer} skipped (no grounding group)")
            continue

        delta_t = torch.from_numpy(delta).to(device=device, dtype=torch.bfloat16)

        # Re-run grounding group with hook → extract final-layer features
        hooked_g = _extract_final_with_hook(
            retriever, dataset, question, device, g_db_indices,
            layer, delta_t, batch_size=batch_size)

        # Build manipulated final-layer: clean non-G + hooked G
        manip_final = clean_final.copy()
        manip_final[g_mask] = hooked_g.numpy()

        # Joint t-SNE so before/after share the same embedding space
        combined = np.vstack([clean_final, manip_final])
        tsne = TSNE(n_components=2, perplexity=perplexity,
                    random_state=seed, metric="cosine")
        emb_all = tsne.fit_transform(combined)
        N = len(clean_final)
        emb_before = emb_all[:N]
        emb_after = emb_all[N:]

        _plot_one_tsne_panel(axes[0, pi], emb_before, cond_labels, f"Manip@L{layer}")
        _plot_one_tsne_panel(axes[1, pi], emb_after, cond_labels, f"Manip@L{layer}")

        print(f"  L{layer} t-SNE done")

    axes[0, 0].set_ylabel("Clean", fontsize=S_TSNE["label_fontsize"], fontweight="bold")
    axes[1, 0].set_ylabel("Manipulated", fontsize=S_TSNE["label_fontsize"], fontweight="bold")

    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=(0.75, 0.75, 0.75),
               markersize=7, label="None"),
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=tuple(FILL_COLORS[0]),
               markersize=7, label="Feature binding"),
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=tuple(FILL_COLORS[1]),
               markersize=7, label="Object match"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=(0.75, 0.75, 0.75),
               markeredgecolor=tuple(ANSWER_MATCH_COLOR),
               markeredgewidth=1.5, markersize=7, label="Retrieval match"),
    ]

    q_short = question[:60] + "..." if len(question) > 60 else question
    fig.suptitle(f"Last-layer t-SNE after object match manipulation: {q_short} → {answer}",
                 fontsize=S_TSNE["suptitle_fontsize"])
    fig.subplots_adjust(hspace=0.12, wspace=0.08, top=0.90, bottom=0.00)
    fig.legend(handles=handles, loc="upper center",
               bbox_to_anchor=(0.5, 0.016), ncol=4,
               fontsize=S_TSNE["legend_fontsize"], frameon=False)

    out_path = output_dir / "tsne_grounding_manipulation.png"
    fig.savefig(str(out_path), dpi=S_TSNE["dpi"], bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── Random direction null distribution ─────────────────────────

def run_random_control(model, steervit, retriever, dataset, scenes,
                       queries, category, test_layers, num_db,
                       batch_size, device, n_random, seed, output_dir):
    """Run N random directions as control, compare acc against grounding."""
    blocks = steervit.vision_model.trunk.blocks
    prefix = steervit.vision_model.trunk.num_prefix_tokens
    n_conds = len(ALL_COND_NAMES)

    grounding_correct = {l: [] for l in test_layers}
    random_correct = {l: [[] for _ in range(n_random)] for l in test_layers}

    for qi, query in enumerate(queries):
        question = query["question"]
        answer = str(query["answer"]).strip().lower()
        program = query["program"]
        dataset_idx = query["dataset_idx"]
        img_filename = dataset.questions[dataset_idx]["image_filename"]
        query_scene = scenes[img_filename]
        query_info = extract_query_info(query_scene["objects"], program, query["answer"])

        if qi % 10 == 0:
            print(f"  Query {qi}/{len(queries)}", flush=True)

        db_fnames, db_indices = build_db_pool(
            dataset, scenes, img_filename, 3, 5, num_db)
        cond_labels = np.zeros((len(db_fnames), n_conds), dtype=bool)
        for di, fname in enumerate(db_fnames):
            db_scene = scenes.get(fname)
            if db_scene is None:
                continue
            conds, _ = label_db_image(db_scene["objects"], query_info, category)
            for ci in range(min(len(conds), n_conds)):
                cond_labels[di, ci] = conds[ci]

        db_feats = retriever.build_steered_database(
            dataset, question, device, db_indices, batch_size=batch_size)

        query_img = dataset[dataset_idx]["image"].unsqueeze(0).to(device)

        for layer in test_layers:
            feats_np = db_feats[layer].numpy()
            g_delta = compute_delta(feats_np, cond_labels, "grounding")
            if g_delta is None:
                continue

            target_norm = float(np.linalg.norm(g_delta))

            # Grounding direction
            dt = torch.from_numpy(g_delta).to(device=device, dtype=torch.bfloat16)
            h = blocks[layer].register_forward_hook(make_manip_hook(dt, prefix))
            with torch.no_grad(), autocast(device_type="cuda", dtype=torch.bfloat16):
                g_pred = model.generate(query_img, [question])[0]
            h.remove()
            grounding_correct[layer].append(g_pred.strip().lower() == answer)

            # N random directions
            for ri in range(n_random):
                rng_i = np.random.RandomState(seed * 10000 + qi * 100 + layer * n_random + ri)
                rand_dir = rng_i.randn(feats_np.shape[1]).astype(np.float32)
                rand_dir = rand_dir / np.linalg.norm(rand_dir) * target_norm

                dt = torch.from_numpy(rand_dir).to(device=device, dtype=torch.bfloat16)
                h = blocks[layer].register_forward_hook(make_manip_hook(dt, prefix))
                with torch.no_grad(), autocast(device_type="cuda", dtype=torch.bfloat16):
                    r_pred = model.generate(query_img, [question])[0]
                h.remove()
                random_correct[layer][ri].append(r_pred.strip().lower() == answer)

    # Aggregate & report
    print(f"\n{'='*60}")
    print(f"RANDOM CONTROL (N={n_random})")
    print(f"{'='*60}")

    results = {}
    for layer in test_layers:
        g_acc = float(np.mean(grounding_correct[layer])) if grounding_correct[layer] else float("nan")
        r_accs = [float(np.mean(random_correct[layer][ri]))
                  for ri in range(n_random) if random_correct[layer][ri]]
        r_mean = float(np.mean(r_accs)) if r_accs else float("nan")
        r_std = float(np.std(r_accs)) if r_accs else float("nan")
        r_min = float(np.min(r_accs)) if r_accs else float("nan")
        r_max = float(np.max(r_accs)) if r_accs else float("nan")
        pct_worse = sum(1 for ra in r_accs if ra <= g_acc) / len(r_accs) if r_accs else float("nan")

        results[str(layer)] = {
            "grounding_acc": round(g_acc, 4),
            "random_mean": round(r_mean, 4),
            "random_std": round(r_std, 4),
            "random_min": round(r_min, 4),
            "random_max": round(r_max, 4),
            "p_value": round(pct_worse, 4),
            "n_queries": len(grounding_correct[layer]),
            "random_accs": [round(ra, 4) for ra in r_accs],
        }

        print(f"  Layer {layer:2d}: Grounding {g_acc:.3f}  |  "
              f"Random {r_mean:.3f}±{r_std:.3f} [{r_min:.3f}, {r_max:.3f}]  |  "
              f"p={pct_worse:.3f}")

    out_data = {
        "n_random": n_random, "n_queries": len(queries), "seed": seed,
        "layers_tested": test_layers, "per_layer": results,
    }
    out_path = output_dir / "random_control.json"
    with open(out_path, "w") as f:
        json.dump(out_data, f, indent=2)
    print(f"Saved: {out_path}")

    # Plot
    apply_style()
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    layers_x = list(range(len(test_layers)))
    g_accs = [results[str(l)]["grounding_acc"] for l in test_layers]
    r_means = [results[str(l)]["random_mean"] for l in test_layers]
    r_stds = [results[str(l)]["random_std"] for l in test_layers]

    ax.errorbar(layers_x, r_means, yerr=r_stds, fmt="o-", color="gray",
                label=f"Random (N={n_random})", capsize=4, markersize=6)
    ax.plot(layers_x, g_accs, "s-", color="red", label="Object match", markersize=8)
    ax.axhline(y=1.0, color="black", linestyle="--", alpha=0.3, label="Clean")
    ax.set_xticks(layers_x)
    ax.set_xticklabels([str(l) for l in test_layers])
    ax.set_xlabel("Manipulation Layer")
    ax.set_ylabel("Val Accuracy")
    ax.set_title("Object match vs Random Direction Control")
    ax.legend()
    ax.set_ylim(0, 1.05)

    out_plot = output_dir / "random_control.png"
    fig.savefig(str(out_plot), dpi=PLOT_STYLE["dpi"], bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_plot}")


# ── Main ─────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--data-root", type=str,
                   default="/home/jungchun/data/clevr/CLEVR_v1.0")
    p.add_argument("--n-queries", type=int, default=72)
    p.add_argument("--num-db", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--layers", type=str, default=None,
                   help="Comma-separated layers to test manipulation at. "
                        "Default: all GCA layers + last layer.")
    p.add_argument("--measure-acc", action="store_true",
                   help="Also measure VQA accuracy with/without manipulation hooks.")
    p.add_argument("--n-random-control", type=int, default=0,
                   help="Run N random direction controls (acc only). "
                        "0 = skip. Outputs null distribution for comparison.")
    p.add_argument("--tsne", action="store_true",
                   help="Generate before/after t-SNE plots for grounding manipulation.")
    p.add_argument("--tsne-only", action="store_true",
                   help="Only generate t-SNE plots, skip RSA/acc analysis.")
    p.add_argument("--perplexity", type=float, default=30)
    args = p.parse_args()

    apply_style()

    ckpt_dir = Path(args.checkpoint).parent
    model_name = ckpt_dir.name.replace("_s42", "")
    output_dir = Path(args.output_dir) if args.output_dir else \
        Path("outputs/analysis/grounding_manipulation") / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model, steervit, vocab = load_model(args.checkpoint, device)
    transform = steervit.get_transforms()
    retriever = AllLayerRetriever(steervit)
    num_layers = retriever.num_layers

    gca_layers = [i for i, blk in enumerate(steervit.vision_model.trunk.blocks)
                  if getattr(blk, "gated_cross_attn", None) is not None]

    if args.layers:
        test_layers = [int(x) for x in args.layers.split(",")]
    else:
        test_layers = sorted(set(gca_layers + [num_layers - 1]))

    print(f"GCA layers: {gca_layers}")
    print(f"Testing manipulation at layers: {test_layers}")

    dataset = CLEVRVQADataset(args.data_root, "val", transform)
    with open(Path(args.data_root) / "scenes" / "CLEVR_val_scenes.json") as f:
        scenes = {s["image_filename"]: s for s in json.load(f)["scenes"]}

    rng = random.Random(args.seed)
    family_index = build_family_index(dataset)

    category = "attr_query_direct"
    queries = sample_queries(dataset, family_index, category,
                             n_total=args.n_queries, rng=rng, scenes=scenes)
    print(f"Queries: {len(queries)}")

    n_conds = len(ALL_COND_NAMES)

    # ── t-SNE visualization ──
    if args.tsne or args.tsne_only:
        print("\nGenerating t-SNE plots...")
        plot_manipulation_tsne(
            steervit, retriever, dataset, device, queries[0], scenes, category,
            test_layers, args.num_db, args.batch_size, output_dir,
            seed=args.seed, perplexity=args.perplexity)
        if args.tsne_only:
            print("Done (t-SNE only).")
            return

    # ── Random control null distribution ──
    if args.n_random_control > 0:
        print(f"\nRunning random direction control (N={args.n_random_control})...")
        run_random_control(
            model, steervit, retriever, dataset, scenes, queries, category,
            test_layers, args.num_db, args.batch_size, device,
            n_random=args.n_random_control, seed=args.seed,
            output_dir=output_dir)
        print("Done (random control).")
        return

    # ── Per-layer accumulators ──
    # For each manipulation type, at each layer, collect per-query RSA values
    np_rng = np.random.RandomState(args.seed)
    _random_fn = lambda feats, conds: apply_random_manipulation(feats, conds, rng=np_rng)

    for manip_type, apply_fn in [("grounding", apply_grounding_manipulation),
                                  ("random", _random_fn),
                                  ("answer", apply_answer_manipulation)]:
        print(f"\n{'='*60}")
        print(f"MANIPULATION: {manip_type}")
        print(f"{'='*60}")

        per_layer_rsa = {l: {"grounding_before": [], "grounding_after": [],
                              "answer_before": [], "answer_after": [],
                              "retrieval_before": [], "retrieval_after": []}
                          for l in test_layers}
        per_layer_acc = {l: {"clean_correct": 0, "manip_correct": 0, "total": 0,
                              "flipped_wrong": [], "flipped_right": []}
                          for l in test_layers} if args.measure_acc else None
        group_infos = []
        skipped = 0
        blocks = steervit.vision_model.trunk.blocks
        prefix_tokens = steervit.vision_model.trunk.num_prefix_tokens

        for qi, query in enumerate(queries):
            question = query["question"]
            answer = query["answer"]
            program = query["program"]
            dataset_idx = query["dataset_idx"]
            img_filename = dataset.questions[dataset_idx]["image_filename"]
            query_scene = scenes[img_filename]

            if qi % 10 == 0:
                print(f"  Query {qi}/{len(queries)}: {question[:50]}...", flush=True)

            query_info = extract_query_info(query_scene["objects"], program, answer)

            db_fnames, db_indices = build_db_pool(
                dataset, scenes, img_filename, 3, 5, args.num_db)

            cond_labels = np.zeros((len(db_fnames), n_conds), dtype=bool)
            for di, fname in enumerate(db_fnames):
                db_scene = scenes.get(fname)
                if db_scene is None:
                    continue
                conds, _ = label_db_image(db_scene["objects"], query_info, category)
                for ci in range(min(len(conds), n_conds)):
                    cond_labels[di, ci] = conds[ci]

            db_feats = retriever.build_steered_database(
                dataset, question, device, db_indices, batch_size=args.batch_size)

            query_img = dataset[dataset_idx]["image"].unsqueeze(0).to(device)
            query_feats = retriever.extract(query_img, [question])

            binding_mask = cond_labels[:, COND_BINDING]
            answer_labels = cond_labels[:, COND_ANSWER]

            # Val acc: clean prediction (once per query)
            clean_pred = None
            clean_correct = None
            if args.measure_acc:
                with torch.no_grad(), autocast(device_type="cuda", dtype=torch.bfloat16):
                    clean_pred = model.generate(query_img, [question])[0]
                clean_correct = clean_pred.strip().lower() == str(answer).strip().lower()

            for layer in test_layers:
                feats_np = db_feats[layer].numpy()
                q_feat = query_feats[layer].numpy()[0]

                # Before manipulation
                rho_g_b, rho_a_b = compute_conditional_rsa(feats_np, cond_labels, binding_mask)
                ret_b = compute_retrieval_accuracy(q_feat, feats_np, answer_labels)

                # Apply manipulation
                if manip_type == "answer" and layer != num_layers - 1:
                    # Answer manipulation only at last layer
                    manip_feats = feats_np
                    info = {"skipped": True, "reason": "answer manipulation only at last layer"}
                else:
                    manip_feats, info = apply_fn(feats_np, cond_labels)

                if info.get("skipped"):
                    continue

                # After manipulation
                rho_g_a, rho_a_a = compute_conditional_rsa(manip_feats, cond_labels, binding_mask)
                ret_a = compute_retrieval_accuracy(q_feat, manip_feats, answer_labels)

                if rho_g_b is not None:
                    per_layer_rsa[layer]["grounding_before"].append(rho_g_b)
                if rho_g_a is not None:
                    per_layer_rsa[layer]["grounding_after"].append(rho_g_a)
                if rho_a_b is not None:
                    per_layer_rsa[layer]["answer_before"].append(rho_a_b)
                if rho_a_a is not None:
                    per_layer_rsa[layer]["answer_after"].append(rho_a_a)
                per_layer_rsa[layer]["retrieval_before"].append(float(ret_b))
                per_layer_rsa[layer]["retrieval_after"].append(float(ret_a))

                # Val acc: hooked prediction
                if args.measure_acc and per_layer_acc is not None:
                    delta_vec = compute_delta(feats_np, cond_labels, manip_type,
                                              rng=np_rng)
                    if delta_vec is not None:
                        delta_t = torch.from_numpy(delta_vec).to(
                            device=device, dtype=torch.bfloat16)
                        handle = blocks[layer].register_forward_hook(
                            make_manip_hook(delta_t, prefix_tokens))
                        with torch.no_grad(), autocast(device_type="cuda", dtype=torch.bfloat16):
                            manip_pred = model.generate(query_img, [question])[0]
                        handle.remove()
                        manip_correct = manip_pred.strip().lower() == str(answer).strip().lower()

                        per_layer_acc[layer]["clean_correct"] += int(clean_correct)
                        per_layer_acc[layer]["manip_correct"] += int(manip_correct)
                        per_layer_acc[layer]["total"] += 1
                        if clean_correct and not manip_correct:
                            per_layer_acc[layer]["flipped_wrong"].append(
                                {"qi": qi, "q": question[:60], "ans": str(answer),
                                 "pred": manip_pred})
                        elif not clean_correct and manip_correct:
                            per_layer_acc[layer]["flipped_right"].append(
                                {"qi": qi, "q": question[:60], "ans": str(answer),
                                 "clean": clean_pred, "pred": manip_pred})

            if qi == 0:
                group_infos.append(info)

        # Aggregate
        agg = {}
        for layer in test_layers:
            d = per_layer_rsa[layer]
            layer_agg = {}
            for key in ["grounding_before", "grounding_after",
                        "answer_before", "answer_after"]:
                vals = d[key]
                if vals:
                    arr = np.array(vals)
                    layer_agg[f"{key.replace('_before', '_rsa_before').replace('_after', '_rsa_after')}"] = round(float(arr.mean()), 6)
                    layer_agg[f"{key}_std"] = round(float(arr.std()), 6)
                    layer_agg[f"{key}_n"] = len(vals)

            for key in ["retrieval_before", "retrieval_after"]:
                vals = d[key]
                if vals:
                    arr = np.array(vals)
                    layer_agg[f"retrieval_acc_{key.split('_')[1]}"] = round(float(arr.mean()), 6)

            # Wilcoxon test for answer RSA change
            ab = d["answer_before"]
            aa = d["answer_after"]
            if len(ab) >= 10 and len(aa) >= 10 and len(ab) == len(aa):
                try:
                    stat, pval = wilcoxon(ab, aa)
                    layer_agg["answer_rsa_wilcoxon_p"] = round(float(pval), 6)
                except ValueError:
                    pass

            agg[str(layer)] = layer_agg

        # Add acc results to aggregation
        if per_layer_acc is not None:
            for layer in test_layers:
                d = per_layer_acc[layer]
                if d["total"] > 0:
                    agg[str(layer)]["val_acc_clean"] = round(d["clean_correct"] / d["total"], 6)
                    agg[str(layer)]["val_acc_manip"] = round(d["manip_correct"] / d["total"], 6)
                    agg[str(layer)]["val_acc_n"] = d["total"]
                    agg[str(layer)]["val_acc_flipped_wrong"] = len(d["flipped_wrong"])
                    agg[str(layer)]["val_acc_flipped_right"] = len(d["flipped_right"])

        results = {
            "manipulation": manip_type,
            "category": category,
            "n_queries": len(queries),
            "num_db": args.num_db,
            "layers_tested": test_layers,
            "gca_layers": gca_layers,
            "checkpoint": str(args.checkpoint),
            "per_layer": agg,
        }
        if per_layer_acc is not None:
            results["val_acc_details"] = {
                str(l): {"flipped_wrong": per_layer_acc[l]["flipped_wrong"],
                          "flipped_right": per_layer_acc[l]["flipped_right"]}
                for l in test_layers if per_layer_acc[l]["total"] > 0
            }

        # Print summary
        print(f"\n--- {manip_type} manipulation summary ---")
        for layer in test_layers:
            la = agg.get(str(layer), {})
            g_b = la.get("grounding_rsa_before", "n/a")
            g_a = la.get("grounding_rsa_after", "n/a")
            a_b = la.get("answer_rsa_before", "n/a")
            a_a = la.get("answer_rsa_after", "n/a")
            r_b = la.get("retrieval_acc_before", "n/a")
            r_a = la.get("retrieval_acc_after", "n/a")
            p_val = la.get("answer_rsa_wilcoxon_p", "n/a")
            acc_str = ""
            if "val_acc_clean" in la:
                acc_str = (f"  |  Acc {la['val_acc_clean']:.3f} → {la['val_acc_manip']:.3f} "
                           f"(flip -{la['val_acc_flipped_wrong']}/+{la['val_acc_flipped_right']})")
            print(f"  Layer {layer:2d}: "
                  f"Grounding ρ {g_b} → {g_a}  |  "
                  f"Answer ρ {a_b} → {a_a} (p={p_val})  |  "
                  f"Retrieval {r_b} → {r_a}{acc_str}")

        out_path = output_dir / f"manipulation_{manip_type}.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved: {out_path}")

        plot_results(results, output_dir)
        plot_retrieval(results, output_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
