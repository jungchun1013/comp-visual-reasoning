"""Linear probing: per-layer predicate readout.

For each ViT layer, train logistic regression to predict binding,
grounding, and answer_match labels from mean-pooled patch features.
One plot per question subcategory (direct / same / spatial).

v2 naming (docs/legacy-reference.md §1.1): "answer matching" -> "Retrieval"
in figure-visible text (legend labels); the answer_match variable/JSON key
name is unchanged for schema compatibility.

Usage:
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python scripts/analysis/linear_probe.py \
        --checkpoint outputs/model/clevr_siglip_decoder1l_scratch_s42/best.pt
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from data.clevr import CLEVRVQADataset, ANSWER_TO_IDX
from data.clevr_programs import evaluate_answer
from data.clevr_sampling import RETRIEVAL_CATEGORIES, build_family_index, sample_queries
from omegaconf import OmegaConf
from torch.amp import autocast

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analysis.plot_style import apply_style, line_kwargs

ATTR_KEYS = ("color", "shape", "material", "size")
LABEL_NAMES = ["answer_match", "answer_decode"]
LABEL_COLORS = {"answer_match": "#d62728", "answer_decode": "#1f77b4"}


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
    def __init__(self, steervit, decoder_model=None):
        self.steervit = steervit
        self.decoder_model = decoder_model
        self.blocks = steervit.vision_model.trunk.blocks
        self.norm = steervit.vision_model.trunk.norm
        self.prefix = steervit.vision_model.trunk.num_prefix_tokens
        self.num_vit_layers = len(self.blocks)
        has_decoder_probe = (decoder_model is not None and
                             hasattr(decoder_model.decoder, 'pos_embedding'))
        self.num_layers = self.num_vit_layers + (1 if has_decoder_probe else 0)

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
            vit_out = self.steervit.forward(images, questions)

        for h in hooks:
            h.remove()

        feats = {}
        for l in range(self.num_vit_layers):
            normed = self.norm(layer_out[l].float())
            patches = normed[:, self.prefix:, :]
            feats[l] = patches.mean(dim=1).cpu()

        # Decoder L1: BOS hidden state (cross-attn decoder only)
        if self.decoder_model is not None:
            decoder = self.decoder_model.decoder
            if hasattr(decoder, 'pos_embedding'):
                patches = vit_out[:, self.prefix:, :].float()
                memory = decoder.visual_proj(patches)
                B = memory.size(0)
                bos_id = 0
                bos = torch.full((B, 1), bos_id, dtype=torch.long, device=memory.device)
                positions = torch.zeros(1, dtype=torch.long, device=memory.device)
                tgt = decoder.token_embedding(bos) + decoder.pos_embedding(positions)
                out = decoder._run_layers(tgt, memory)  # (B, 1, d_model)
                feats[self.num_vit_layers] = out[:, 0, :].cpu()

        return feats


def extract_and_label(query_indices, dataset, scenes, retriever, device,
                      num_db, batch_size):
    """Extract features + answer labels for a set of queries.

    Returns:
        all_feats: {layer: np.array (N, d)}
        answer_match: np.array (N,) bool — does DB scene give same answer as GT?
        answer_class: np.array (N,) int — answer index (28-class), -1 if program fails
    """
    num_layers = retriever.num_layers
    all_feats = {l: [] for l in range(num_layers)}
    all_match = []
    all_class = []

    for qi, q_idx in enumerate(query_indices):
        q_data = dataset.questions[q_idx]
        fname = q_data["image_filename"]
        question = q_data["question"]
        gt_answer = q_data["answer"]
        program = q_data.get("program", [])

        # Sample DB images
        seen = set()
        db_indices = []
        for idx in range(len(dataset)):
            db_fname = dataset.questions[idx]["image_filename"]
            if db_fname in seen or db_fname == fname:
                continue
            scene = scenes.get(db_fname)
            if scene is None:
                continue
            n_obj = len(scene["objects"])
            if n_obj < 3 or n_obj > 5:
                continue
            seen.add(db_fname)
            db_indices.append(idx)
            if len(db_indices) >= num_db:
                break

        # Labels
        match_labels = np.zeros(len(db_indices), dtype=bool)
        class_labels = np.full(len(db_indices), -1, dtype=int)
        for i, db_idx in enumerate(db_indices):
            db_fname = dataset.questions[db_idx]["image_filename"]
            scene = scenes.get(db_fname)
            if scene is None:
                continue
            ans = evaluate_answer(scene["objects"], program)
            if ans is not None:
                ans_str = str(ans).lower()
                match_labels[i] = ans_str == str(gt_answer).lower()
                class_labels[i] = ANSWER_TO_IDX.get(ans_str, -1)
        all_match.append(match_labels)
        all_class.append(class_labels)

        if (qi + 1) % 10 == 0 or qi == 0:
            n_match = match_labels.sum()
            n_valid = (class_labels >= 0).sum()
            print(f"  Q{qi+1}/{len(query_indices)}: "
                  f"match={n_match} valid_class={n_valid}/{len(db_indices)}",
                  flush=True)

        # Extract features
        db_questions = [question] * len(db_indices)
        for start in range(0, len(db_indices), batch_size):
            end = min(start + batch_size, len(db_indices))
            batch_imgs = torch.stack(
                [dataset[db_indices[i]]["image"] for i in range(start, end)]
            ).to(device)
            feats = retriever.extract(batch_imgs, db_questions[start:end])
            for l in range(num_layers):
                all_feats[l].append(feats[l])

    for l in range(num_layers):
        all_feats[l] = torch.cat(all_feats[l], dim=0).numpy()
    all_match = np.concatenate(all_match, axis=0)
    all_class = np.concatenate(all_class, axis=0)
    return all_feats, all_match, all_class


def probe_per_layer(all_feats, answer_match, answer_class, num_layers, seed):
    """Train logistic regression per layer. Returns results dict.

    Two probes:
      answer_match: binary F1 (class_weight=balanced)
      answer_decode: 28-class accuracy (multiclass)
    """
    from sklearn.metrics import accuracy_score

    results = {l: {} for l in range(num_layers)}
    # Mask for valid 28-class samples
    valid_mask = answer_class >= 0

    for l in range(num_layers):
        X = all_feats[l]

        # --- answer_match (binary F1) ---
        y_bin = answer_match.astype(int)
        if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
            results[l]["answer_match"] = {"f1": 0.0}
        else:
            sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
            tr, te = next(sss.split(X, y_bin))
            clf = make_pipeline(
                StandardScaler(),
                LogisticRegressionCV(max_iter=2000, cv=3, scoring="f1",
                                     class_weight="balanced",
                                     random_state=seed, n_jobs=-1))
            clf.fit(X[tr], y_bin[tr])
            y_pred = clf.predict(X[te])
            results[l]["answer_match"] = {
                "f1": float(f1_score(y_bin[te], y_pred, zero_division=0))}

        # --- answer_decode (28-class accuracy) ---
        X_mc = X[valid_mask]
        y_mc = answer_class[valid_mask]
        n_classes = len(np.unique(y_mc))
        if n_classes < 2:
            results[l]["answer_decode"] = {"f1": 0.0}
        else:
            sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
            tr, te = next(sss.split(X_mc, y_mc))
            clf = make_pipeline(
                StandardScaler(),
                LogisticRegressionCV(max_iter=2000, cv=3, scoring="accuracy",
                                     random_state=seed, n_jobs=-1))
            clf.fit(X_mc[tr], y_mc[tr])
            y_pred = clf.predict(X_mc[te])
            acc = accuracy_score(y_mc[te], y_pred)
            results[l]["answer_decode"] = {"f1": float(acc)}  # stored as "f1" for plot compat

        lbl = f"D1" if l == num_layers - 1 and num_layers > 12 else f"L{l:>2}"
        print(f"  {lbl}: "
              f"match_F1={results[l]['answer_match']['f1']:.3f} "
              f"decode_acc={results[l]['answer_decode']['f1']:.3f}", flush=True)
    return results


def plot_probe(results, num_layers, gca_layers, num_vit_layers, title, output_path):
    # Intentional overrides vs PLOT_STYLE: wide figsize, smaller fonts
    # (labels 14, title 16, ticks 10, legend 12) so 13 layer ticks fit.
    fig, ax = plt.subplots(figsize=(8, 4.5))
    layers = list(range(num_layers))

    # v2 naming (docs/legacy-reference.md §1.1): "answer matching" -> "Retrieval".
    # variable/key names ("answer_match") stay for schema compat; only the
    # figure-visible legend label text changes.
    plot_labels = {"answer_match": "Retrieval (F1)",
                   "answer_decode": "Answer Decode (Acc)"}
    for label_name in LABEL_NAMES:
        f1s = [results[l][label_name]["f1"] for l in layers]
        ax.plot(layers, f1s,
                **line_kwargs(label=plot_labels[label_name],
                              color=LABEL_COLORS[label_name], markersize=5))

    for gl in gca_layers:
        ax.axvline(gl, color="gray", linestyle="--", alpha=0.2, linewidth=0.8)

    # X-axis: L0..L11, D1
    tick_labels = [f"L{l}" for l in range(num_vit_layers)]
    if num_layers > num_vit_layers:
        tick_labels.append("D1")
    ax.set_xlabel("Layer", fontsize=14)
    ax.set_ylabel("Probe Score", fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.set_xticks(layers)
    ax.set_xticklabels(tick_labels, fontsize=10)
    ax.legend(fontsize=12, loc="lower right")
    ax.set_ylim(0.0, 1.02)
    ax.grid(axis="y", alpha=0.2)

    fig.savefig(str(output_path))
    plt.close(fig)
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-root", type=str,
                        default="/home/jungchun/data/clevr/CLEVR_v1.0")
    parser.add_argument("--num-db", type=int, default=500)
    parser.add_argument("--queries-per-subcat", type=int, default=72)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    apply_style()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model, steervit, vocab = load_model(args.checkpoint, device)
    transform = steervit.get_transforms()
    retriever = AllLayerRetriever(steervit, decoder_model=model)
    num_layers = retriever.num_layers

    ckpt_dir = Path(args.checkpoint).parent
    model_name = ckpt_dir.name.replace("_s42", "")
    output_dir = Path(args.output_dir) if args.output_dir else \
        Path("outputs/analysis/linear_probe") / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = CLEVRVQADataset(args.data_root, "val", transform)
    scenes_path = Path(args.data_root) / "scenes" / "CLEVR_val_scenes.json"
    with open(scenes_path) as f:
        scene_list = json.load(f)["scenes"]
    scenes = {s["image_filename"]: s for s in scene_list}

    rng = random.Random(args.seed)
    family_index = build_family_index(dataset)

    gca_layers = [i for i, blk in enumerate(steervit.vision_model.trunk.blocks)
                  if getattr(blk, "gated_cross_attn", None) is not None]

    categories = ["attr_query_direct", "attr_query_same", "attr_query_spatial"]
    all_results = {}

    for category in categories:
        print(f"\n{'='*60}")
        print(f"Category: {category}")
        print(f"{'='*60}")

        queries = sample_queries(dataset, family_index, category,
                                  n_total=args.queries_per_subcat, rng=rng,
                                  scenes=scenes)
        query_indices = [q["dataset_idx"] for q in queries]
        print(f"Queries: {len(query_indices)}, DB per query: {args.num_db}")

        feats, answer_match, answer_class = extract_and_label(
            query_indices, dataset, scenes, retriever, device,
            args.num_db, args.batch_size)

        N = len(answer_match)
        n_valid = (answer_class >= 0).sum()
        print(f"Samples: {N}")
        print(f"  Answer match: {answer_match.sum()} ({answer_match.mean()*100:.1f}%)")
        print(f"  Valid decode:  {n_valid} ({n_valid/N*100:.1f}%)")
        print(f"  Unique classes: {len(np.unique(answer_class[answer_class >= 0]))}")

        print("Probing...")
        results = probe_per_layer(feats, answer_match, answer_class,
                                  num_layers, args.seed)
        all_results[category] = {"results": results, "num_samples": N,
                                   "num_queries": len(query_indices)}

        # Short label for plot title / filename (strip "attr_query_" prefix)
        short = category.replace("attr_query_", "")
        plot_probe(results, num_layers, gca_layers, retriever.num_vit_layers,
                   title=f"Linear Probe — {model_name} ({short})",
                   output_path=output_dir / f"probe_{short}.png")

    # Save all results
    save_data = {"model": model_name, "categories": {}}
    for cat, data in all_results.items():
        save_data["categories"][cat] = data
    with open(output_dir / "probe_results.json", "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nSaved: {output_dir / 'probe_results.json'}")


if __name__ == "__main__":
    main()
