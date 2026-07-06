"""Unified t-SNE visualization for all model types (GCA + MoT).

Three modes:
  qtype       — Sample balanced questions, color by question type.
  steered     — Pick one query, steer N DB images, color by conditions
                (feature binding / object match / retrieval match). v2 naming
                (docs/legacy-reference.md §1.1): "Grounding" now names the
                whole language-conditioning mechanism, never a stage; legend
                text uses "Object match" / "Retrieval match" instead of the
                old "object grounding" / "answer match" stage labels.
  cross_model — Load all available checkpoints, plot side-by-side t-SNE.

Usage:
    # qtype (GCA or MoT)
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python scripts/analysis/tsne_viz.py \
        --checkpoint outputs/clevr_siglip_decoder1l_scratch_s42/best.pt \
        --mode qtype --n-samples 500

    # steered with curated metadata
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python scripts/analysis/tsne_viz.py \
        --checkpoint outputs/clevr_siglip_decoder1l_scratch_s42/best.pt \
        --mode steered --metadata outputs/analysis/metadata/attr_direct_queries.json \
        --query-idx 0 --num-db 500

    # steered with random question
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python scripts/analysis/tsne_viz.py \
        --checkpoint outputs/clevr_mot_scratch_s42/best.pt \
        --mode steered --qtype query_attribute --num-db 500

    # cross-model comparison
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python scripts/analysis/tsne_viz.py \
        --mode cross_model --qtype equal_attribute --n-questions 2 --n-images 300
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.amp import autocast
from sklearn.manifold import TSNE

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from data.clevr import CLEVRVQADataset, ANSWER_TO_IDX
from data.clevr_programs import (
    coarse_question_type,
    extract_described_attrs,
    find_anchor,
    find_target,
    evaluate_answer,
)
from omegaconf import OmegaConf

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ── Style & constants ────────────────────────────────────────────────

_tab10 = plt.cm.tab10.colors

QTYPE_COLORS = {
    "compare_integer":  _tab10[0],
    "count":            _tab10[1],
    "exist":            _tab10[2],
    "query_attribute":  _tab10[3],
    "equal_attribute":  _tab10[4],
}

# Steered mode colors
FILL_COLORS = [
    np.array(_tab10[0][:3]),   # blue  — feature binding
    np.array(_tab10[1][:3]),   # orange — object grounding
]
ANSWER_MATCH_COLOR = np.array(_tab10[3][:3])  # red
ANCHOR_FILL_COLORS = [
    np.array(_tab10[6][:3]),   # pink   — anchor binding
    np.array(_tab10[4][:3]),   # purple — anchor grounding
]
TARGET_FILL_COLORS = [
    np.array(_tab10[0][:3]),   # blue   — target binding
    np.array(_tab10[1][:3]),   # orange — target grounding
]

# Cross-model colors
CORRECT_COLOR = _tab10[0]   # blue
WRONG_COLOR = (0.75, 0.75, 0.75)

ATTR_KEYS = ("color", "shape", "material", "size")
SHAPE_MARKERS = {"sphere": "o", "cylinder": "^", "cube": "s"}

from analysis.plot_style import PLOT_STYLE, apply_style

# Intentional overrides vs PLOT_STYLE: smaller legend (14 vs 16) for dense
# scatter legends, smaller suptitle (18 vs 20) for multi-panel t-SNE grids.
S = dict(PLOT_STYLE, legend_fontsize=14, suptitle_fontsize=18)


# ── Feature extraction ───────────────────────────────────────────────

class GCALayerRetriever:
    """Extract mean-pooled patch features at every ViT block (GCA models)."""

    def __init__(self, steervit):
        self.steervit = steervit
        self.blocks = steervit.vision_model.trunk.blocks
        self.norm = steervit.vision_model.trunk.norm
        self.prefix = steervit.vision_model.trunk.num_prefix_tokens
        self.num_layers = len(self.blocks)

    @torch.no_grad()
    def extract(self, images, questions, **kwargs):
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

    def extract_batched(self, dataset, indices, questions, device,
                        batch_size=32, **kwargs):
        all_feats = {l: [] for l in range(self.num_layers)}
        N = len(indices)
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            batch_imgs = torch.stack(
                [dataset[indices[i]]["image"] for i in range(start, end)]
            ).to(device)
            q = questions[start:end] if questions is not None else None
            feats = self.extract(batch_imgs, q)
            for l in range(self.num_layers):
                all_feats[l].append(feats[l])
            if end % 200 == 0 or end == N:
                print(f"  Extracted: {end}/{N}", flush=True)
        for l in range(self.num_layers):
            all_feats[l] = torch.cat(all_feats[l], dim=0)
        return all_feats


class MoTLayerRetriever:
    """Extract mean-pooled vision token features at every MoT layer."""

    def __init__(self, model):
        self.model = model
        self.num_layers = len(model.layers)
        self.num_patches = model.num_patches

    @torch.no_grad()
    def extract(self, images, questions, answer_ids=None, **kwargs):
        layer_out = {}
        hooks = []
        n_vision = self.num_patches
        for idx, layer in enumerate(self.model.layers):
            def make_hook(li):
                def fn(module, inp, output):
                    layer_out[li] = output[:, :n_vision, :].detach()
                return fn
            hooks.append(layer.register_forward_hook(make_hook(idx)))

        with autocast(device_type="cuda", dtype=torch.bfloat16):
            self.model(images, questions, answer_ids)

        for h in hooks:
            h.remove()

        feats = {}
        for l in range(self.num_layers):
            feats[l] = layer_out[l].float().mean(dim=1).cpu()
        return feats

    def extract_batched(self, dataset, indices, questions, device,
                        batch_size=32, answer_indices=None, vocab=None,
                        **kwargs):
        from trainer import _answers_to_decoder_ids
        all_feats = {l: [] for l in range(self.num_layers)}
        N = len(indices)
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            batch_imgs = torch.stack(
                [dataset[indices[i]]["image"] for i in range(start, end)]
            ).to(device)
            batch_ans = torch.tensor(
                answer_indices[start:end], dtype=torch.long, device=device
            )
            batch_ans_ids = _answers_to_decoder_ids(batch_ans, vocab)
            feats = self.extract(
                batch_imgs, questions[start:end], answer_ids=batch_ans_ids)
            for l in range(self.num_layers):
                all_feats[l].append(feats[l])
            if end % 200 == 0 or end == N:
                print(f"  Extracted: {end}/{N}", flush=True)
        for l in range(self.num_layers):
            all_feats[l] = torch.cat(all_feats[l], dim=0)
        return all_feats


# ── Model loading ────────────────────────────────────────────────────

def detect_model_type(cfg):
    task_type = cfg.task.get("type", cfg.task.get("name", ""))
    if task_type == "mot":
        return "mot"
    return "gca"


def load_gca_model(ckpt, cfg, device):
    from model import CrossAttnViT
    from tasks.decoder import build_decoder_model, build_clevr_decoder_vocab

    cross_attn_layers = list(cfg.model.cross_attn_layers)
    pretrained = cfg.model.get("pretrained", True)
    steervit = CrossAttnViT.from_config(
        cfg.model.backbone_name, device=device,
        cross_attn_layers=cross_attn_layers,
        resolution=cfg.model.resolution,
        pretrained=pretrained,
    )
    vocab = build_clevr_decoder_vocab()
    model_cfg = OmegaConf.create({
        "model": cfg.model, "task": cfg.task, "data": cfg.data,
    })
    model = build_decoder_model(steervit, model_cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()

    transform = steervit.get_transforms()
    retriever = GCALayerRetriever(steervit)
    gca_layers = [i for i, blk in enumerate(steervit.vision_model.trunk.blocks)
                  if getattr(blk, "gated_cross_attn", None) is not None]

    return model, retriever, transform, vocab, gca_layers


def load_mot_model(ckpt, cfg, device):
    from tasks.mot_vqa import MoTVQAModel, build_clevr_mot_vocab

    mot_cfg = cfg.task.get("mot", {})
    vocab = build_clevr_mot_vocab(cfg.data.root)
    model = MoTVQAModel(
        vocab=vocab,
        resolution=cfg.model.get("resolution", 224),
        patch_size=mot_cfg.get("patch_size", 16),
        dim=mot_cfg.get("dim", 512),
        depth=mot_cfg.get("depth", 12),
        heads=mot_cfg.get("heads", 8),
        ff_mult=mot_cfg.get("ff_mult", 4),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()

    transform = model.get_transforms()
    retriever = MoTLayerRetriever(model)
    # Odd layers to match GCA convention [1,3,5,7,9,11]
    num_layers = len(model.layers)
    show_layers = list(range(1, num_layers, 2))

    return model, retriever, transform, vocab, show_layers


def load_model(ckpt_path, device):
    """Load checkpoint, auto-detect type, return unified interface."""
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    cfg = OmegaConf.create(ckpt["config"])
    model_type = detect_model_type(cfg)

    if model_type == "mot":
        model, retriever, transform, vocab, show_layers = \
            load_mot_model(ckpt, cfg, device)
    else:
        model, retriever, transform, vocab, show_layers = \
            load_gca_model(ckpt, cfg, device)

    epoch = ckpt.get("epoch", "?")
    print(f"  Loaded {model_type} (epoch {epoch}), show layers: {show_layers}")
    return model, retriever, transform, vocab, show_layers, model_type


# ── Sampling ─────────────────────────────────────────────────────────

def _get_qtype(q):
    return coarse_question_type(q.get("program", []))


def sample_by_qtype(dataset, n_per_type=100, seed=42):
    """Sample balanced questions across question types."""
    rng = random.Random(seed)
    by_type = {}
    for i, q in enumerate(dataset.questions):
        qt = _get_qtype(q)
        if qt == "unknown":
            continue
        by_type.setdefault(qt, []).append(i)
    print(f"  Types: { {k: len(v) for k, v in by_type.items()} }")

    indices, qtypes, questions, answer_indices = [], [], [], []
    for qt in sorted(by_type.keys()):
        pool = by_type[qt]
        rng.shuffle(pool)
        for idx in pool[:n_per_type]:
            q = dataset.questions[idx]
            indices.append(idx)
            qtypes.append(qt)
            questions.append(q["question"])
            answer_indices.append(ANSWER_TO_IDX.get(q.get("answer", ""), 0))
    return indices, qtypes, questions, answer_indices


def build_db_pool(dataset, scenes, exclude_fname, min_obj, max_obj, num_db):
    """Sample unique images filtered by object count."""
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
        if len(db_fnames) >= num_db:
            break
    return db_fnames, db_indices


# ── Condition checking (steered mode) ────────────────────────────────

def check_conditions(db_objs, described_attrs, anchor_4attrs, program, gt_answer):
    """Check [binding, grounding, answer_match] for one DB scene."""
    binding = (
        any(all(o.get(k) == v for k, v in described_attrs.items()) for o in db_objs)
        if described_attrs else False
    )
    grounding = (
        any(all(o.get(k) == anchor_4attrs.get(k) for k in ATTR_KEYS) for o in db_objs)
        if anchor_4attrs else False
    )
    ans = evaluate_answer(db_objs, program)
    answer_match = str(ans).lower() == str(gt_answer).lower() if ans is not None else False
    return [binding, grounding, answer_match]


# ── Plotting: shared layout ─────────────────────────────────────────

def _make_layer_grid(n_panels, ncols=3, cell=2.8):
    """Create a grid of subplots for per-layer visualization."""
    apply_style()
    nrows = (n_panels + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(cell * ncols + 1, cell * nrows + 1))
    if nrows == 1 and ncols == 1:
        axes = np.array([axes])
    if nrows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    # Hide unused panels
    for pi in range(n_panels, len(axes)):
        axes[pi].set_visible(False)
    return fig, axes


def _finish_plot(fig, title, handles, output_path, ncol_legend=3):
    """Add suptitle, legend, save and close."""
    fig.suptitle(title, fontsize=S["suptitle_fontsize"])
    fig.subplots_adjust(hspace=0.08, wspace=0.08, top=0.90, bottom=0.00)
    fig.legend(handles=handles, loc="upper center",
               bbox_to_anchor=(0.5, 0.016), ncol=ncol_legend,
               fontsize=S["legend_fontsize"], frameon=False)
    fig.savefig(str(output_path), dpi=S["dpi"], bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def _style_ax(ax, layer_idx):
    """Apply common axis styling."""
    ax.set_title(f"L{layer_idx}", fontsize=S["subplot_title_fontsize"])
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_box_aspect(1)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)


# ── Plotting: qtype ─────────────────────────────────────────────────

def plot_qtype_tsne(embeddings, qtypes, show_layers, title, output_path):
    fig, axes = _make_layer_grid(len(show_layers))

    qtypes_arr = np.array(qtypes)
    unique_qtypes = sorted(set(qtypes))

    for pi, l in enumerate(show_layers):
        ax = axes[pi]
        emb = embeddings[l]
        for qt in unique_qtypes:
            mask = qtypes_arr == qt
            ax.scatter(emb[mask, 0], emb[mask, 1],
                       color=QTYPE_COLORS.get(qt, (0.5, 0.5, 0.5)),
                       s=6, alpha=0.6, label=qt if pi == 0 else "",
                       edgecolors="none", rasterized=True)
        _style_ax(ax, l)

    handles = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=QTYPE_COLORS.get(qt, (0.5, 0.5, 0.5)),
               markersize=7, label=qt)
        for qt in unique_qtypes
    ]
    _finish_plot(fig, title, handles, output_path, ncol_legend=3)


# ── Plotting: steered ────────────────────────────────────────────────

def _scatter_shaped(ax, emb, mask, colors, sizes, db_shapes,
                    edge_color=None, edge_width=0):
    for shape_name, marker in SHAPE_MARKERS.items():
        sm = mask & (db_shapes == shape_name)
        if not sm.any():
            continue
        kw = dict(c=colors[sm], s=sizes, marker=marker, rasterized=True)
        if edge_color is not None:
            kw.update(edgecolors=edge_color, linewidths=edge_width)
        else:
            kw["edgecolors"] = "none"
        ax.scatter(emb[sm, 0], emb[sm, 1], **kw)
    unk = mask & ~np.isin(db_shapes, list(SHAPE_MARKERS.keys()))
    if unk.any():
        kw = dict(c=colors[unk], s=sizes, marker="o", rasterized=True)
        if edge_color is not None:
            kw.update(edgecolors=edge_color, linewidths=edge_width)
        else:
            kw["edgecolors"] = "none"
        ax.scatter(emb[unk, 0], emb[unk, 1], **kw)


def _plot_steered_axes(ax, emb, labels, db_shapes, query_pt, fill_colors):
    """Draw one steered subplot with condition coloring."""
    N = labels.shape[0]
    gray = np.array([0.75, 0.75, 0.75])
    point_colors = np.tile(gray, (N, 1))
    for c in range(2):  # binding, grounding
        point_colors[labels[:, c]] = fill_colors[c]

    has_binding = labels[:, 0]
    answer_mask = labels[:, 2]

    # Gray (no binding)
    no_binding = ~has_binding
    ax.scatter(emb[no_binding, 0], emb[no_binding, 1],
               c=point_colors[no_binding], s=8, marker="o",
               edgecolors="none", rasterized=True)
    # Binding without answer match
    bind_no_ans = has_binding & ~answer_mask
    _scatter_shaped(ax, emb, bind_no_ans, point_colors, 18, db_shapes)
    # Binding with answer match — red edge
    bind_ans = has_binding & answer_mask
    _scatter_shaped(ax, emb, bind_ans, point_colors, 22, db_shapes,
                    edge_color=ANSWER_MATCH_COLOR, edge_width=1.2)
    # Query star (disabled — query point not needed)
    # if query_pt is not None:
    #     ax.scatter(query_pt[0], query_pt[1], c="gold", s=120, marker="*",
    #                edgecolors="black", linewidths=0.8, zorder=10)


def plot_steered_tsne(embeddings, labels, db_shapes, show_layers, query_emb,
                      question, answer, output_path, role=None,
                      fill_colors=None):
    """Plot steered t-SNE with condition coloring."""
    if fill_colors is None:
        fill_colors = FILL_COLORS

    fig, axes = _make_layer_grid(len(show_layers))

    for pi, l in enumerate(show_layers):
        ax = axes[pi]
        _plot_steered_axes(ax, embeddings[l], labels, db_shapes,
                           query_emb.get(l), fill_colors)
        _style_ax(ax, l)

    gray_rgba = (0.75, 0.75, 0.75)
    role_prefix = f"{role} " if role else ""
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=gray_rgba,
               markersize=7, label="None"),
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=tuple(fill_colors[0]),
               markersize=7, label=f"{role_prefix}Feature binding"),
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=tuple(fill_colors[1]),
               markersize=7, label=f"{role_prefix}Object match"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=gray_rgba,
               markeredgecolor=tuple(ANSWER_MATCH_COLOR),
               markeredgewidth=1.5, markersize=7, label="Retrieval match"),
    ]

    q_short = question[:65] + "..." if len(question) > 65 else question
    title = f"[{role}] {q_short}  →  {answer}" if role else \
            f"{q_short}  →  {answer}"
    _finish_plot(fig, title, handles, output_path, ncol_legend=2)


# ── Plotting: cross_model ───────────────────────────────────────────

def plot_cross_model(all_results, queries, perplexity, seed, output_dir, qtype):
    apply_style()
    for qi, query in enumerate(queries):
        n_models = len(all_results)
        fig, axes = plt.subplots(1, n_models, figsize=(3.5 * n_models, 3.5))
        if n_models == 1:
            axes = [axes]

        for mi, (model_name, results) in enumerate(all_results.items()):
            ax = axes[mi]
            data = results[qi]
            X, correct, acc = data["feats"], data["correct"], data["acc"]

            tsne = TSNE(n_components=2, perplexity=perplexity,
                        random_state=seed, metric="cosine")
            emb = tsne.fit_transform(X)

            wrong = ~correct
            ax.scatter(emb[wrong, 0], emb[wrong, 1],
                       color=WRONG_COLOR, s=6, alpha=0.4,
                       edgecolors="none", rasterized=True)
            ax.scatter(emb[correct, 0], emb[correct, 1],
                       color=CORRECT_COLOR, s=8, alpha=0.6,
                       edgecolors="none", rasterized=True)
            ax.set_title(f"{model_name}\n({acc:.0%})",
                         fontsize=S["subplot_title_fontsize"])
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_box_aspect(1)

        q_short = query["question"][:60]
        fig.suptitle(f"'{q_short}' → {query['answer']}",
                     fontsize=S["suptitle_fontsize"], y=1.02)
        handles = [
            Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=CORRECT_COLOR, markersize=7, label="Correct"),
            Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=WRONG_COLOR, markersize=7, label="Wrong"),
        ]
        fig.legend(handles=handles, loc="lower center", ncol=2,
                   fontsize=S["legend_fontsize"], frameon=False,
                   bbox_to_anchor=(0.5, -0.05))
        out_path = output_dir / f"cross_model_{qtype}_q{qi}.png"
        fig.savefig(str(out_path), dpi=S["dpi"], bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_path}")


# ── Mode: qtype ─────────────────────────────────────────────────────

def run_qtype(args, device):
    model, retriever, transform, vocab, show_layers, model_type = \
        load_model(args.checkpoint, device)

    dataset = CLEVRVQADataset(args.data_root, "val", transform)
    show_layers = show_layers[::args.every_n]

    ckpt_dir = Path(args.checkpoint).parent
    run_name = ckpt_dir.name.replace("_s42", "")
    output_dir = Path(args.output_dir) if args.output_dir else \
        Path("outputs/analysis/tsne") / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = output_dir / "cache_qtype.npz"

    if not args.replot:
        n_per_type = args.n_samples // 5
        indices, qtypes, questions, answer_indices = sample_by_qtype(
            dataset, n_per_type=n_per_type, seed=args.seed)
        print(f"Sampled {len(indices)} questions ({n_per_type} per type)")

        print("Extracting features...", flush=True)
        all_feats = retriever.extract_batched(
            dataset, indices, questions, device,
            batch_size=args.batch_size,
            answer_indices=answer_indices, vocab=vocab)

        save_dict = {
            "qtypes": np.array(qtypes),
            "show_layers": np.array(show_layers),
            "num_layers": retriever.num_layers,
            "indices": np.array(indices),
        }
        for l in range(retriever.num_layers):
            save_dict[f"feat_{l}"] = all_feats[l].numpy()
        np.savez(str(cache_path), **save_dict)
        print(f"Saved cache: {cache_path}")
    else:
        print(f"Loading cache: {cache_path}")
        cached = np.load(str(cache_path), allow_pickle=True)
        qtypes = cached["qtypes"].tolist()
        if "show_layers" in cached:
            show_layers = cached["show_layers"].tolist()
        elif "gca_layers" in cached:
            show_layers = cached["gca_layers"].tolist()
        else:
            num_layers = int(cached["num_layers"])
            show_layers = list(range(1, num_layers, 2))
        all_feats = {}
        for l in show_layers:
            if f"feat_{l}" in cached:
                all_feats[l] = torch.from_numpy(cached[f"feat_{l}"])

    print("Running t-SNE...", flush=True)
    embeddings = {}
    for l in show_layers:
        X = all_feats[l].numpy()
        tsne = TSNE(n_components=2, perplexity=args.perplexity,
                    random_state=args.seed, metric="cosine")
        embeddings[l] = tsne.fit_transform(X)
        print(f"  L{l} done")

    model_name = Path(args.checkpoint).parent.name.replace("_s42", "")
    plot_qtype_tsne(
        embeddings, qtypes, show_layers,
        title=f"t-SNE by Question Type — {model_name}",
        output_path=output_dir / "tsne_qtype.png",
    )


# ── Mode: steered ───────────────────────────────────────────────────

def run_steered(args, device):
    model, retriever, transform, vocab, show_layers, model_type = \
        load_model(args.checkpoint, device)

    dataset = CLEVRVQADataset(args.data_root, "val", transform)
    show_layers = show_layers[::args.every_n]

    # Load scenes
    scenes_path = Path(args.data_root) / "scenes" / "CLEVR_val_scenes.json"
    with open(scenes_path) as f:
        scene_list = json.load(f)["scenes"]
    scenes = {s["image_filename"]: s for s in scene_list}

    # Resolve query
    if args.metadata:
        with open(args.metadata) as f:
            meta_list = json.load(f)
        qi = args.query_idx
        meta = meta_list[qi]
        q_data = dataset.questions[meta["idx"]]
        query_dataset_idx = meta["idx"]
    else:
        # Pick random question of given type
        rng = random.Random(args.seed)
        candidates = [i for i, q in enumerate(dataset.questions)
                      if _get_qtype(q) == args.qtype]
        rng.shuffle(candidates)
        query_dataset_idx = candidates[0]
        q_data = dataset.questions[query_dataset_idx]
        qi = 0

    fname = q_data["image_filename"]
    program = q_data.get("program", [])
    question = q_data["question"]
    gt_answer = q_data["answer"]
    query_scene = scenes[fname]
    print(f"Query [{qi}]: {question}  →  {gt_answer}")

    # Extract program info for condition checking
    described_attrs = extract_described_attrs(program)
    anchor_obj, _ = find_anchor(query_scene["objects"], program)
    anchor_4attrs = {k: anchor_obj.get(k) for k in ATTR_KEYS} if anchor_obj else None

    # Check for same/spatial (has target)
    meta_stem = Path(args.metadata).stem if args.metadata else ""
    has_target = "same" in meta_stem or "spatial" in meta_stem
    target_described, target_4attrs = {}, None
    if has_target:
        target_obj, target_described = find_target(query_scene["objects"], program)
        if target_obj:
            target_4attrs = {k: target_obj.get(k) for k in ATTR_KEYS}
        print(f"Anchor attrs: {described_attrs}, Target attrs: {target_described}")

    # Output dir
    ckpt_dir = Path(args.checkpoint).parent
    run_name = ckpt_dir.name.replace("_s42", "")
    if args.metadata:
        sub_dir = Path(args.metadata).stem.replace("_queries", "")
    else:
        sub_dir = args.qtype
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path("outputs/analysis/tsne") / run_name / sub_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    no_ca = getattr(args, "no_ca", False)
    cache_suffix = "_noca" if no_ca else ""
    cache_path = output_dir / f"cache_q{qi}{cache_suffix}.npz"

    if not args.replot:
        # Sample DB
        db_fnames, db_indices = build_db_pool(
            dataset, scenes, fname, 3, 5, args.num_db)
        print(f"DB: {len(db_fnames)} images")

        # Label DB images
        N = len(db_fnames)
        labels = np.zeros((N, 3), dtype=bool)
        target_labels = np.zeros((N, 3), dtype=bool) if has_target else None
        db_shapes = [""] * N
        for i, db_fname in enumerate(db_fnames):
            scene = scenes.get(db_fname)
            if scene is None:
                continue
            labels[i] = check_conditions(
                scene["objects"], described_attrs, anchor_4attrs,
                program, gt_answer)
            if has_target:
                target_labels[i] = check_conditions(
                    scene["objects"], target_described, target_4attrs,
                    program, gt_answer)
            best_score, best_shape = -1, ""
            for obj in scene["objects"]:
                score = sum(1 for k, v in described_attrs.items()
                            if obj.get(k) == v)
                if score > best_score:
                    best_score = score
                    best_shape = obj.get("shape", "")
            db_shapes[i] = best_shape
        db_shapes = np.array(db_shapes)

        cond_names = ["Binding", "Grounding", "Answer Match"]
        print(f"Anchor: {dict(zip(cond_names, labels.sum(axis=0).tolist()))}")

        # Extract features
        if no_ca:
            print("Extracting DB features (no-CA)...", flush=True)
            db_feats = retriever.extract_batched(
                dataset, db_indices, None, device,
                batch_size=args.batch_size)
        else:
            print("Extracting DB features...", flush=True)
            db_questions = [question] * N
            db_answer_indices = [ANSWER_TO_IDX.get(gt_answer, 0)] * N
            db_feats = retriever.extract_batched(
                dataset, db_indices, db_questions, device,
                batch_size=args.batch_size,
                answer_indices=db_answer_indices, vocab=vocab)

        print("Extracting query features...", flush=True)
        query_img = dataset[query_dataset_idx]["image"].unsqueeze(0).to(device)
        q_text = None if no_ca else [question]
        if model_type == "mot" and not no_ca:
            from trainer import _answers_to_decoder_ids
            q_ans = torch.tensor(
                [ANSWER_TO_IDX.get(gt_answer, 0)], dtype=torch.long, device=device)
            q_ans_ids = _answers_to_decoder_ids(q_ans, vocab)
            query_feats = retriever.extract(query_img, [question], answer_ids=q_ans_ids)
        else:
            query_feats = retriever.extract(query_img, q_text)
        query_feats = {l: query_feats[l][0].cpu() for l in range(retriever.num_layers)}

        # Save cache (backward compat: write both show_layers and gca_layers)
        save_dict = {
            "labels": labels, "db_shapes": db_shapes,
            "show_layers": np.array(show_layers),
            "gca_layers": np.array(show_layers),
            "num_layers": retriever.num_layers,
            "question": question, "answer": gt_answer,
            "db_indices": np.array(db_indices),
            "has_target": has_target,
        }
        if has_target:
            save_dict["target_labels"] = target_labels
        for l in show_layers:
            save_dict[f"feat_{l}"] = db_feats[l].numpy()
            save_dict[f"query_feat_{l}"] = query_feats[l].numpy()
        np.savez(str(cache_path), **save_dict)
        print(f"Saved cache: {cache_path}")

    else:
        print(f"Loading cache: {cache_path}")
        cached = np.load(str(cache_path), allow_pickle=True)
        labels = cached["labels"]
        db_shapes = cached["db_shapes"] if "db_shapes" in cached else np.array([""] * len(labels))
        if "show_layers" in cached:
            show_layers = cached["show_layers"].tolist()
        elif "gca_layers" in cached:
            show_layers = cached["gca_layers"].tolist()
        db_feats = {l: torch.from_numpy(cached[f"feat_{l}"]) for l in show_layers}
        query_feats = {l: torch.from_numpy(cached[f"query_feat_{l}"])
                       for l in show_layers if f"query_feat_{l}" in cached}
        question = str(cached["question"])
        gt_answer = str(cached["answer"])
        has_target = bool(cached.get("has_target", False))
        target_labels = cached["target_labels"] if has_target else None

    if args.compute_only:
        return

    # t-SNE
    print("Running t-SNE...", flush=True)
    embeddings, query_emb = {}, {}
    for l in show_layers:
        X_db = db_feats[l].numpy()
        X_q = query_feats[l].numpy().reshape(1, -1) if l in query_feats else None
        X_all = np.vstack([X_db, X_q]) if X_q is not None else X_db
        tsne = TSNE(n_components=2, perplexity=args.perplexity,
                    random_state=args.seed, metric="cosine")
        emb_all = tsne.fit_transform(X_all)
        embeddings[l] = emb_all[:len(X_db)]
        if X_q is not None:
            query_emb[l] = emb_all[-1]
        print(f"  L{l} done")

    # Plot
    tag = "unsteered" if no_ca else "steered"
    if not has_target:
        plot_steered_tsne(
            embeddings, labels, db_shapes, show_layers, query_emb,
            question, gt_answer,
            output_dir / f"tsne_{tag}_q{qi}.png",
        )
    else:
        plot_steered_tsne(
            embeddings, labels, db_shapes, show_layers, query_emb,
            question, gt_answer,
            output_dir / f"tsne_{tag}_q{qi}_anchor.png",
            role="Anchor", fill_colors=ANCHOR_FILL_COLORS,
        )
        plot_steered_tsne(
            embeddings, target_labels, db_shapes, show_layers, query_emb,
            question, gt_answer,
            output_dir / f"tsne_{tag}_q{qi}_target.png",
            role="Target", fill_colors=TARGET_FILL_COLORS,
        )


# ── Mode: cross_model ──────────────────────────────────────────────

def run_cross_model(args, device):
    from torchvision import transforms as T

    rng = random.Random(args.seed)

    base = Path("outputs/model")
    model_configs = [
        ("SigLIP+GCA", base / "clevr_siglip_decoder1l_scratch_s42/best.pt"),
        ("DINOv2+GCA", base / "clevr_dinov2_decoder1l_scratch_s42/best.pt"),
        ("MAE+GCA", base / "clevr_mae_decoder1l_scratch_s42/best.pt"),
        ("MoT", base / "clevr_mot_scratch_s42/best.pt"),
        ("GCA scratch", base / "clevr_dinov2_gca_scratch_s42/best.pt"),
    ]
    model_configs = [(n, p) for n, p in model_configs if p.exists()]
    print(f"Models: {[n for n, _ in model_configs]}")

    # Sample questions & DB images
    simple_transform = T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.ToTensor(),
    ])
    dataset_raw = CLEVRVQADataset(args.data_root, "val", simple_transform)

    candidates = [i for i, q in enumerate(dataset_raw.questions)
                  if _get_qtype(q) == args.qtype]
    rng.shuffle(candidates)
    query_indices = candidates[:args.n_questions]

    queries = []
    for qi_idx in query_indices:
        q = dataset_raw.questions[qi_idx]
        queries.append({
            "idx": qi_idx, "question": q["question"],
            "answer": q.get("answer", ""), "image_filename": q["image_filename"],
        })
        print(f"Query {len(queries)}: '{q['question']}' → {q.get('answer', '')}")

    query_fnames = {q["image_filename"] for q in queries}
    all_indices = [i for i in range(len(dataset_raw))
                   if dataset_raw.questions[i]["image_filename"] not in query_fnames]
    rng.shuffle(all_indices)
    db_indices = all_indices[:args.n_images]
    print(f"DB images: {len(db_indices)}")

    output_dir = Path("outputs/analysis/tsne/cross_model")
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}
    for model_name, ckpt_path in model_configs:
        print(f"\n{'='*60}\nProcessing: {model_name}\n{'='*60}")
        model, retriever, transform, vocab, show_layers, model_type = \
            load_model(ckpt_path, device)
        target_layer = show_layers[-1]

        ds = CLEVRVQADataset(args.data_root, "val", transform)
        raw_model = model.module if hasattr(model, "module") else model
        all_results[model_name] = {}

        for qi, query in enumerate(queries):
            question = query["question"]
            gt_answer = query["answer"]
            feats_list, correct_list = [], []

            for start in range(0, len(db_indices), args.batch_size):
                end = min(start + args.batch_size, len(db_indices))
                batch_imgs = torch.stack(
                    [ds[db_indices[i]]["image"] for i in range(start, end)]
                ).to(device)
                batch_qs = [question] * (end - start)

                if model_type == "mot":
                    from trainer import _answers_to_decoder_ids
                    dummy_ans = torch.zeros(end - start, dtype=torch.long, device=device)
                    dummy_ids = _answers_to_decoder_ids(dummy_ans, vocab)
                    feats = retriever.extract(batch_imgs, batch_qs, answer_ids=dummy_ids)
                else:
                    feats = retriever.extract(batch_imgs, batch_qs)
                feats_list.append(feats[target_layer])

                with torch.no_grad(), autocast(device_type="cuda", dtype=torch.bfloat16):
                    preds = raw_model.generate(batch_imgs, batch_qs)
                for p in preds:
                    correct_list.append(p.strip().lower() == gt_answer.strip().lower())

            all_feats = torch.cat(feats_list, dim=0).numpy()
            correct = np.array(correct_list)
            acc = correct.mean()
            print(f"  Q{qi}: '{question[:50]}...' → acc={acc:.1%}")
            all_results[model_name][qi] = {
                "feats": all_feats, "correct": correct, "acc": acc,
            }

        del model, retriever
        torch.cuda.empty_cache()

    plot_cross_model(all_results, queries, args.perplexity, args.seed,
                     output_dir, args.qtype)


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="qtype",
                        choices=["qtype", "steered", "cross_model"])
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--data-root", type=str,
                        default="/home/jungchun/data/clevr/CLEVR_v1.0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--perplexity", type=float, default=30)
    parser.add_argument("--every-n", type=int, default=1,
                        help="Subsample show_layers by every N-th")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--replot", action="store_true")
    parser.add_argument("--compute-only", action="store_true")
    parser.add_argument("--no-ca", action="store_true",
                        help="Extract features without text (no cross-attention)")
    # qtype args
    parser.add_argument("--n-samples", type=int, default=500)
    # steered args
    parser.add_argument("--metadata", type=str, default=None)
    parser.add_argument("--query-idx", type=int, default=0)
    parser.add_argument("--qtype", type=str, default="query_attribute")
    parser.add_argument("--num-db", type=int, default=500)
    # cross_model args
    parser.add_argument("--n-questions", type=int, default=2)
    parser.add_argument("--n-images", type=int, default=300)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if args.mode == "qtype":
        run_qtype(args, device)
    elif args.mode == "steered":
        run_steered(args, device)
    elif args.mode == "cross_model":
        run_cross_model(args, device)


if __name__ == "__main__":
    main()
