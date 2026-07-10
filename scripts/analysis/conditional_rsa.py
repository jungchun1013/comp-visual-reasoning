"""Conditional RSA — hierarchical conditioning to handle base-rate imbalance.

Instead of binary RSA on full DB (rare conditions diluted by "both false" pairs),
each level is conditioned on the previous:

Direct:
  1. Binding | All                    — C1 on full DB
  2. Retrieval | Binding              — C2 on subset where C1=True
  3. Answer classification | Binding  — C4 on subset where C1=True

Spatial (same_attr / spatial):
  1. Anchor binding | All
  2. Target binding | All
  3. Anchor retrieval | Anc bind
  4. Target retrieval | Tgt bind
  5. Answer classification | Tgt bind

v2 naming (docs/legacy-reference.md §1.1): the 3-stage "binding → object
grounding → answer matching" pipeline is now 2-stage "Binding → Retrieval";
"Grounding" names the whole language-conditioning mechanism, never a chain
stage. The old "object grounding" condition (C2: full 4-attr object identity
match) IS the Retrieval stage (user decision 2026-07-07: no object/answer
split within Retrieval). The answer-level condition (C4) is the pre-existing
classification readout, labeled "Answer classification"; it is not a
grounding stage. Condition indices / JSON keys are unchanged.

Usage:
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python scripts/analysis/conditional_rsa.py \
        --checkpoint outputs/model/clevr_siglip_decoder1l_scratch_s42/best.pt \
        --categories attr_query_direct,attr_query_same,attr_query_spatial
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
from torch.amp import autocast

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from data.clevr import CLEVRVQADataset
from data.clevr_sampling import RETRIEVAL_CATEGORIES, build_family_index, sample_queries
from data.clevr_conditions import (
    COND_COLORS, SPATIAL_COND_COLORS,
    FAMILY_SUBCATEGORY, SPATIAL_SUBCATEGORIES,
    extract_query_info, label_db_image,
)
from omegaconf import OmegaConf

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analysis.plot_style import PLOT_STYLE as S, apply_style
from analysis.run_log import tee_stdout


# ── Chain definitions ────────────────────────────────────────────
# (name, condition_index, subset_condition_index or None)

CHAIN_DIRECT = [
    ("Binding | All", 1, None),
    ("Retrieval | Binding", 2, 1),
    ("Answer classification | Binding", 4, 1),
]
CHAIN_DIRECT_COLORS = [COND_COLORS[1], COND_COLORS[2], COND_COLORS[4]]

CHAIN_SPATIAL = [
    ("Anchor binding | All", 0, None),
    ("Target binding | All", 1, None),
    ("Anchor retrieval | Anc bind", 2, 0),
    ("Target retrieval | Tgt bind", 3, 1),
    ("Answer classification | Tgt bind", 6, 1),
]
CHAIN_SPATIAL_COLORS = [
    SPATIAL_COND_COLORS[0], SPATIAL_COND_COLORS[1],
    SPATIAL_COND_COLORS[2], SPATIAL_COND_COLORS[3],
    SPATIAL_COND_COLORS[6],
]


# ── Model loading (same as linear_probe.py) ──────────────────────

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
        """Extract features for all DB images at all layers."""
        num_layers = self.num_layers
        all_feats = {l: [] for l in range(num_layers)}
        questions = [question] * batch_size if question else None

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


# ── RSA helpers ──────────────────────────────────────────────────

def build_binary_rdm(labels):
    """Binary RDM: 0 if same label, 1 if different. Condensed form."""
    n = len(labels)
    rdm = np.empty(n * (n - 1) // 2)
    k = 0
    for i in range(n):
        for j in range(i + 1, n):
            rdm[k] = 0.0 if (labels[i] == labels[j]) else 1.0
            k += 1
    return rdm


def build_db_pool(dataset, scenes, exclude_fname, min_obj, max_obj, num_db):
    """Sample unique DB images filtered by object count."""
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


# ── Plot ─────────────────────────────────────────────────────────

def plot(stats, output_dir):
    apply_style()

    num_layers = stats["num_layers"]
    layers = list(range(num_layers))
    is_spatial = stats.get("is_spatial", False)
    colors = CHAIN_SPATIAL_COLORS if is_spatial else CHAIN_DIRECT_COLORS
    # Stored stats carry the condition names current at RUN time; rename by
    # condition index so replots always use the current naming standard.
    chain = CHAIN_SPATIAL if is_spatial else CHAIN_DIRECT
    name_by_idx = {(ci, si): name for name, ci, si in chain}

    def display_name(entry):
        key = (entry.get("condition_index"), entry.get("subset_condition_index"))
        return name_by_idx.get(key, entry["name"])
    gca_layers = stats.get("gca_layers", [1, 3, 5, 7, 9, 11])

    fig, ax = plt.subplots(1, 1, figsize=S["subplot_size"])

    # Unsteered (dashed)
    unsteered = stats.get("conditional_rsa_unsteered", [])
    for i, entry in enumerate(unsteered):
        if "per_layer" not in entry:
            continue
        color = colors[i] if i < len(colors) else "gray"
        u_means = [entry["per_layer"].get(str(l), {}).get("mean", float("nan"))
                    for l in layers]
        ax.plot(layers, u_means, color=color, linewidth=1,
                linestyle="--", alpha=0.5,
                marker=S["marker"], markersize=S["markersize"] - 2)

    # Steered (solid)
    for i, entry in enumerate(stats["conditional_rsa"]):
        color = colors[i] if i < len(colors) else "gray"
        means, sems = [], []
        for l in layers:
            v = entry["per_layer"].get(str(l))
            if v:
                means.append(v["mean"])
                sems.append(v["std"] / max(1, np.sqrt(v["n"])))
            else:
                means.append(float("nan"))
                sems.append(0.0)
        ax.plot(layers, means, color=color, linewidth=S["linewidth"],
                marker=S["marker"], markersize=S["markersize"],
                label=display_name(entry))
        ax.fill_between(layers,
                        [m - s for m, s in zip(means, sems)],
                        [m + s for m, s in zip(means, sems)],
                        alpha=0.15, color=color)

    for l in gca_layers:
        ax.axvline(x=l, color="gray", linestyle="--", linewidth=0.8, alpha=0.15)

    ax.set_xticks(layers)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Spearman ρ")
    for spine in ax.spines.values():
        spine.set_linewidth(2)

    handles, labels = ax.get_legend_handles_labels()
    fig.tight_layout()
    fig.legend(handles, labels, loc="upper center",
               bbox_to_anchor=(0.5, -0.05),
               ncol=min(len(handles), 5),
               fontsize=S["legend_fontsize"], frameon=False)

    out = output_dir / "rsa_conditional.png"
    fig.savefig(str(out), dpi=S["dpi"], bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ── Main ─────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--data-root", type=str,
                   default="/home/jungchun/data/clevr/CLEVR_v1.0")
    p.add_argument("--categories", type=str,
                   default="attr_query_direct,attr_query_same,attr_query_spatial")
    p.add_argument("--queries-per-category", type=int, default=72)
    p.add_argument("--num-db", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--replot", action="store_true")
    args = p.parse_args()

    ckpt_dir = Path(args.checkpoint).parent
    model_name = ckpt_dir.name.replace("_s42", "")
    output_dir = Path(args.output_dir) if args.output_dir else \
        Path("outputs/analysis/conditional_rsa") / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    tee_stdout(output_dir)

    if args.replot:
        for cat_dir in sorted(output_dir.iterdir()):
            stats_path = cat_dir / "rsa_conditional_stats.json"
            if stats_path.exists():
                with open(stats_path) as f:
                    stats = json.load(f)
                plot(stats, cat_dir)
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model, steervit, vocab = load_model(args.checkpoint, device)
    transform = steervit.get_transforms()
    retriever = AllLayerRetriever(steervit)
    num_layers = retriever.num_layers

    gca_layers = [i for i, blk in enumerate(steervit.vision_model.trunk.blocks)
                  if getattr(blk, "gated_cross_attn", None) is not None]

    dataset = CLEVRVQADataset(args.data_root, "val", transform)
    with open(Path(args.data_root) / "scenes" / "CLEVR_val_scenes.json") as f:
        scenes = {s["image_filename"]: s for s in json.load(f)["scenes"]}

    rng = random.Random(args.seed)
    family_index = build_family_index(dataset)

    categories = args.categories.split(",")

    for category in categories:
        print(f"\n{'='*60}")
        print(f"Category: {category}")
        print(f"{'='*60}", flush=True)

        cat_dir = output_dir / category
        cat_dir.mkdir(parents=True, exist_ok=True)

        queries = sample_queries(dataset, family_index, category,
                                 n_total=args.queries_per_category, rng=rng,
                                 scenes=scenes)
        print(f"  Queries: {len(queries)}", flush=True)
        if not queries:
            continue

        families_used = sorted(set(q["family"] for q in queries))
        subcategory = FAMILY_SUBCATEGORY.get(families_used[0], category)
        is_spatial = subcategory in SPATIAL_SUBCATEGORIES

        chain = CHAIN_SPATIAL if is_spatial else CHAIN_DIRECT
        from data.clevr_conditions import SPATIAL_COND_NAMES, ALL_COND_NAMES
        n_conds = len(SPATIAL_COND_NAMES) if is_spatial else len(ALL_COND_NAMES)
        print(f"  {'Spatial' if is_spatial else 'Direct'}, chain: {[c[0] for c in chain]}",
              flush=True)

        # Accumulators
        rsa_acc = {ci: {l: [] for l in range(num_layers)} for ci in range(len(chain))}
        rsa_acc_u = {ci: {l: [] for l in range(num_layers)} for ci in range(len(chain))}
        subset_sizes = {ci: [] for ci in range(len(chain))}

        for qi, query in enumerate(queries):
            question = query["question"]
            answer = query["answer"]
            program = query["program"]
            dataset_idx = query["dataset_idx"]

            img_filename = dataset.questions[dataset_idx]["image_filename"]
            query_scene = scenes[img_filename]

            if qi % 10 == 0:
                print(f"  Query {qi}/{len(queries)}: {question[:60]}", flush=True)

            query_info = extract_query_info(query_scene["objects"], program, answer)

            # Build DB
            db_fnames, db_indices = build_db_pool(
                dataset, scenes, img_filename, 3, 5, args.num_db)

            # Label all DB images
            cond_labels = np.zeros((len(db_fnames), n_conds), dtype=bool)
            for di, fname in enumerate(db_fnames):
                db_scene = scenes.get(fname)
                if db_scene is None:
                    continue
                conds, _ = label_db_image(db_scene["objects"], query_info, subcategory)
                for ci_idx in range(min(len(conds), n_conds)):
                    cond_labels[di, ci_idx] = conds[ci_idx]

            # Extract steered + unsteered features
            db_feats = retriever.build_steered_database(
                dataset, question, device, db_indices, batch_size=args.batch_size)
            db_feats_u = retriever.build_steered_database(
                dataset, None, device, db_indices, batch_size=args.batch_size)

            # Conditional RSA per chain level
            for ci, (chain_name, cond_idx, subset_cond_idx) in enumerate(chain):
                if subset_cond_idx is None:
                    subset_mask = np.ones(len(db_fnames), dtype=bool)
                else:
                    subset_mask = cond_labels[:, subset_cond_idx]

                subset_indices = np.where(subset_mask)[0]
                n_subset = len(subset_indices)
                subset_sizes[ci].append(n_subset)

                subset_labels = cond_labels[subset_indices, cond_idx]
                n_true = int(subset_labels.sum())
                n_false = n_subset - n_true

                for layer in range(num_layers):
                    if n_true == 0 or n_false == 0 or n_subset < 3:
                        continue

                    feats_np = db_feats[layer].numpy()[subset_indices]
                    neural_rdm = pdist(feats_np, metric="correlation")
                    hyp_rdm = build_binary_rdm(subset_labels)
                    r, _ = spearmanr(neural_rdm, hyp_rdm)
                    rsa_acc[ci][layer].append(float(r))

                    feats_u = db_feats_u[layer].numpy()[subset_indices]
                    neural_rdm_u = pdist(feats_u, metric="correlation")
                    r_u, _ = spearmanr(neural_rdm_u, hyp_rdm)
                    rsa_acc_u[ci][layer].append(float(r_u))

        # Aggregate
        def _agg(vals):
            arr = np.array(vals)
            return {"mean": round(float(arr.mean()), 6),
                    "std": round(float(arr.std()), 6), "n": len(vals)}

        results, results_u = [], []
        for ci, (chain_name, cond_idx, subset_cond_idx) in enumerate(chain):
            avg_subset = int(np.mean(subset_sizes[ci])) if subset_sizes[ci] else 0
            per_layer, per_layer_u = {}, {}
            for l in range(num_layers):
                if rsa_acc[ci][l]:
                    per_layer[str(l)] = _agg(rsa_acc[ci][l])
                if rsa_acc_u[ci][l]:
                    per_layer_u[str(l)] = _agg(rsa_acc_u[ci][l])
            results.append({"name": chain_name, "condition_index": cond_idx,
                            "subset_condition_index": subset_cond_idx,
                            "avg_subset_size": avg_subset, "per_layer": per_layer})
            results_u.append({"name": chain_name, "avg_subset_size": avg_subset,
                              "per_layer": per_layer_u})

        stats = {
            "category": category, "n_queries": len(queries),
            "num_db": args.num_db, "num_layers": num_layers,
            "is_spatial": is_spatial, "gca_layers": gca_layers,
            "checkpoint": str(args.checkpoint),
            "conditional_rsa": results,
            "conditional_rsa_unsteered": results_u,
        }

        stats_path = cat_dir / "rsa_conditional_stats.json"
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"\n  Saved: {stats_path}", flush=True)

        plot(stats, cat_dir)


if __name__ == "__main__":
    main()
