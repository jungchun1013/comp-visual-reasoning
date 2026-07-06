"""Patching t-SNE: visualize clean / corrupt / patched representations.

For each target head, run denoising (restore head in corrupt forward) and
extract final-layer features. Plot t-SNE showing how patching moves corrupt
representations back toward clean.

Usage:
    PYTHONPATH=src python scripts/analysis/patching_tsne.py \
        --checkpoint outputs/model/clevr_dinov2_decoder1l_scratch_s42/best.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.amp import autocast

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from model import CrossAttnViT
from data.clevr import CLEVRVQADataset, ANSWER_TO_IDX
from tasks.decoder import build_clevr_decoder_vocab, VQADecoder, DecoderModel
from analysis.patching_sampling import (
    build_corruption_index, collect_corruption_samples,
    collect_visual_corruption_samples,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.manifold import TSNE


# ── Style ────────────────────────────────────────────────────────

from analysis.plot_style import PLOT_STYLE, apply_style

# Intentional overrides vs PLOT_STYLE: smaller legend (12) and titles (16)
# for the dense 3-panel clean/corrupt/patched layout.
S = dict(PLOT_STYLE, legend_fontsize=12,
         subplot_title_fontsize=16, suptitle_fontsize=16)

# Answer colors (tab10-based for consistency)
_tab10 = plt.cm.tab10.colors
_CLEVR_COLORS = {
    "red": "#e74c3c", "blue": "#3498db", "green": "#2ecc71",
    "brown": "#8b4513", "purple": "#9b59b6", "cyan": "#1abc9c",
    "yellow": "#f39c12", "gray": "#7f8c8d",
    "cube": _tab10[0], "sphere": _tab10[1], "cylinder": _tab10[2],
    "metal": _tab10[3], "rubber": _tab10[4],
    "large": _tab10[0], "small": _tab10[1],
    "yes": _tab10[2], "no": _tab10[3],
}

IDX_TO_ANSWER = {v: k for k, v in ANSWER_TO_IDX.items()}
CONDITION_MARKERS = {"clean": "o", "corrupt": "X", "patched": "^"}
CONDITION_LABELS = {"clean": "Clean", "corrupt": "Corrupt", "patched": "Patched"}


# ── Model loading ────────────────────────────────────────────────

def load_model(ckpt_path, device):
    from omegaconf import OmegaConf
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    vocab = build_clevr_decoder_vocab()

    if "config" not in ckpt:
        steervit = CrossAttnViT.from_config(
            "vit_base_patch14_dinov2.lvd142m", device=device,
            cross_attn_layers=[1, 3, 5, 7, 9, 11], resolution=336)
        if "steervit_trainable_state" in ckpt:
            steervit.load_state_dict(ckpt["steervit_trainable_state"], strict=False)
        dec_sd = ckpt.get("decoder_state_dict", {})
        layer_indices = {int(k.split(".")[1]) for k in dec_sd if k.startswith("layers.")}
        decoder = VQADecoder(
            vocab_size=len(vocab), visual_dim=steervit.visual_dim,
            d_model=512, nhead=8, num_layers=len(layer_indices) or 2, max_len=8)
        if dec_sd:
            decoder.load_state_dict(dec_sd, strict=False)
        model = DecoderModel(steervit, decoder, vocab, use_steering=True)
        model.load_state_dict(ckpt.get("model_state_dict", {}), strict=False)
    else:
        cfg = OmegaConf.create(ckpt["config"])
        steervit = CrossAttnViT.from_config(
            cfg.model.backbone_name, device=device,
            cross_attn_layers=list(cfg.model.cross_attn_layers),
            resolution=cfg.model.resolution,
            pretrained=cfg.model.get("pretrained", True),
            use_gate=cfg.model.get("use_gate", True))
        dec_cfg = cfg.task.get("decoder", {})
        decoder = VQADecoder(
            vocab_size=len(vocab), visual_dim=steervit.visual_dim,
            d_model=dec_cfg.get("d_model", 512), nhead=dec_cfg.get("nhead", 8),
            num_layers=dec_cfg.get("num_layers", 2), max_len=dec_cfg.get("max_len", 8))
        model = DecoderModel(steervit, decoder, vocab,
                             use_steering=cfg.model.get("use_steering", True))
        model.load_state_dict(ckpt["model_state_dict"], strict=False)

    model = model.to(device).eval()
    transform = steervit.get_transforms()
    print(f"Loaded: {Path(ckpt_path).parent.name}")
    return steervit, decoder, vocab, transform


# ── Feature extraction with patching ─────────────────────────────

class PatchingFeatureExtractor:
    """Extract layer features under clean / corrupt / patched conditions."""

    def __init__(self, steervit, extract_layer=11):
        self.steervit = steervit
        self.blocks = steervit.vision_model.trunk.blocks
        self.norm = steervit.vision_model.trunk.norm
        self.prefix = steervit.vision_model.trunk.num_prefix_tokens
        self.extract_layer = extract_layer
        self.sa_head_dim = self.blocks[0].attn.head_dim
        self.gca_layers = []
        for idx, blk in enumerate(self.blocks):
            if getattr(blk, "gated_cross_attn", None) is not None:
                self.gca_layers.append(idx)
        if self.gca_layers:
            self.gca_head_dim = (
                self.blocks[self.gca_layers[0]]
                .gated_cross_attn.cross_attn.head_dim
            )

    def _pool(self, raw):
        """LayerNorm + mean-pool patches."""
        normed = self.norm(raw.float())
        return normed[:, self.prefix:, :].mean(dim=1).cpu()

    def _extract(self, images, questions):
        cache = {}
        def hook_fn(module, inp, output):
            out = output[0] if isinstance(output, tuple) else output
            cache["feat"] = out.detach()
        hook = self.blocks[self.extract_layer].register_forward_hook(hook_fn)
        with torch.no_grad(), autocast(device_type="cuda", dtype=torch.bfloat16):
            self.steervit.forward(images, questions)
        hook.remove()
        return self._pool(cache["feat"])

    def _record_sa_pre(self, images, questions):
        pre = {}
        hooks = []
        for idx, blk in enumerate(self.blocks):
            def make_hook(li):
                def fn(module, inp, output):
                    pre[li] = inp[0].detach().clone()
                return fn
            hooks.append(blk.attn.proj.register_forward_hook(make_hook(idx)))
        with torch.no_grad(), autocast(device_type="cuda", dtype=torch.bfloat16):
            self.steervit.forward(images, questions)
        for h in hooks:
            h.remove()
        return pre

    def _record_gca_pre(self, images, questions):
        pre = {}
        hooks = []
        for li in self.gca_layers:
            gca = self.blocks[li].gated_cross_attn
            def make_hook(layer_idx):
                def fn(module, inp, output):
                    pre[layer_idx] = inp[0].detach().clone()
                return fn
            hooks.append(gca.cross_attn.to_out.register_forward_hook(make_hook(li)))
        with torch.no_grad(), autocast(device_type="cuda", dtype=torch.bfloat16):
            self.steervit.forward(images, questions)
        for h in hooks:
            h.remove()
        return pre

    def extract_patched_sa(self, images, questions, target_layer, target_head,
                           clean_pre):
        hd = self.sa_head_dim
        h_start, h_end = target_head * hd, (target_head + 1) * hd
        cache = {}
        hooks = []
        def feat_hook(module, inp, output):
            out = output[0] if isinstance(output, tuple) else output
            cache["feat"] = out.detach()
        hooks.append(self.blocks[self.extract_layer].register_forward_hook(feat_hook))
        def restore_hook(module, inp):
            x = inp[0].clone()
            x[:, :, h_start:h_end] = clean_pre[:, :, h_start:h_end]
            return (x,)
        hooks.append(
            self.blocks[target_layer].attn.proj.register_forward_pre_hook(restore_hook))
        with torch.no_grad(), autocast(device_type="cuda", dtype=torch.bfloat16):
            self.steervit.forward(images, questions)
        for h in hooks:
            h.remove()
        return self._pool(cache["feat"])

    def extract_patched_gca(self, images, questions, target_layer, target_head,
                            clean_pre):
        hd = self.gca_head_dim
        h_start, h_end = target_head * hd, (target_head + 1) * hd
        cache = {}
        hooks = []
        def feat_hook(module, inp, output):
            out = output[0] if isinstance(output, tuple) else output
            cache["feat"] = out.detach()
        hooks.append(self.blocks[self.extract_layer].register_forward_hook(feat_hook))
        gca = self.blocks[target_layer].gated_cross_attn
        def restore_hook(module, inp):
            x = inp[0].clone()
            x[:, :, h_start:h_end] = clean_pre[:, :, h_start:h_end]
            return (x,)
        hooks.append(gca.cross_attn.to_out.register_forward_pre_hook(restore_hook))
        with torch.no_grad(), autocast(device_type="cuda", dtype=torch.bfloat16):
            self.steervit.forward(images, questions)
        for h in hooks:
            h.remove()
        return self._pool(cache["feat"])


# ── Plotting (3-panel: clean / corrupt / patched) ────────────────

def plot_patching_tsne(coords, answers, conditions, title, out_path):
    """3-panel t-SNE: clean | corrupt | patched. Color by answer."""
    apply_style()

    unique_answers = sorted(set(answers))
    # Build color map
    cmap = {}
    for i, ans in enumerate(unique_answers):
        if ans in _CLEVR_COLORS:
            cmap[ans] = _CLEVR_COLORS[ans]
        else:
            cmap[ans] = _tab10[i % 10]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))

    for pi, cond in enumerate(["clean", "corrupt", "patched"]):
        ax = axes[pi]
        mask = np.array(conditions) == cond

        for ans in unique_answers:
            ans_mask = np.array(answers) == ans
            both = mask & ans_mask
            if not both.any():
                continue
            ax.scatter(
                coords[both, 0], coords[both, 1],
                c=[cmap[ans]], s=10, alpha=0.7,
                marker=CONDITION_MARKERS[cond],
                edgecolors="none", rasterized=True,
            )

        ax.set_title(CONDITION_LABELS[cond], fontsize=S["subplot_title_fontsize"])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_box_aspect(1)

    # Shared legend
    handles = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=cmap[ans] if isinstance(cmap[ans], str)
               else cmap[ans], markersize=7, label=ans)
        for ans in unique_answers
    ]
    fig.suptitle(title, fontsize=S["suptitle_fontsize"])
    fig.subplots_adjust(wspace=0.08, top=0.85, bottom=0.00)
    fig.legend(handles=handles, loc="upper center",
               bbox_to_anchor=(0.5, 0.02), ncol=min(len(unique_answers), 8),
               fontsize=10, frameon=False)
    fig.savefig(str(out_path), dpi=S["dpi"], bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── Experiment definitions ───────────────────────────────────────

EXPERIMENTS = [
    # (head_type, layer, head, corruption_source, attr, label)
    ("gca", 9, 9,  "described", "color",    "GCA_L9_H9 (binding, color)"),
    ("gca", 9, 10, "described", "material", "GCA_L9_H10 (binding, material)"),
    ("gca", 9, 8,  "visual",   "color",    "GCA_L9_H8 (grounding, color)"),
    ("sa",  8, 3,  "visual",   "color",    "SA_L8_H3 (visual feature, color)"),
    ("sa",  3, 11, "described", "color",    "SA_L3_H11 (suppression, color)"),
    ("gca", 9, 14, "visual",   "size",     "GCA_L9_H14 (answer, size)"),
    ("sa", 10, 9,  "visual",   "material", "SA_L10_H9 (answer, material)"),
]


# ── Main ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-root", type=str,
                        default="/home/jungchun/data/clevr/CLEVR_v1.0")
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--extract-layer", type=int, default=11)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--visual-pairs-dir", type=str,
                        default="outputs/analysis/visual_corruptions")
    args = parser.parse_args()

    apply_style()

    import random
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    steervit, decoder, vocab, transform = load_model(args.checkpoint, device)
    dataset = CLEVRVQADataset(args.data_root, "val", transform)

    output_dir = Path(args.output_dir) if args.output_dir else \
        Path("outputs/analysis/patching_tsne")
    output_dir.mkdir(parents=True, exist_ok=True)

    extractor = PatchingFeatureExtractor(steervit, args.extract_layer)
    corruption_index = build_corruption_index(dataset)

    for head_type, layer, head, corr_source, attr, label in EXPERIMENTS:
        head_name = f"{'GCA' if head_type == 'gca' else 'SA'}_L{layer}_H{head}"
        exp_name = f"{head_name}_{attr}_{corr_source}"
        print(f"\n{'='*60}")
        print(f"{label}")
        print(f"{'='*60}")

        if corr_source == "visual":
            pairs_path = Path(args.visual_pairs_dir) / attr / "pairs.json"
            if not pairs_path.exists():
                print(f"  SKIP: {pairs_path} not found")
                continue
            samples = collect_visual_corruption_samples(
                dataset, str(pairs_path), args.n_samples, transform)
            visual_mode = True
        else:
            samples = collect_corruption_samples(
                dataset, corruption_index, attr, args.n_samples)
            visual_mode = False

        print(f"  Samples: {len(samples)}")
        if not samples:
            continue

        clean_feats, corrupt_feats, patched_feats = [], [], []
        answer_labels = []

        for i, sample in enumerate(samples):
            if visual_mode:
                clean_img, corrupt_img, question, answer_idx = sample
                clean_batch = clean_img.unsqueeze(0).to(device)
                corrupt_batch = corrupt_img.unsqueeze(0).to(device)
                clean_qs, corrupt_qs = [question], [question]
            else:
                image, question, corrupt_q, answer_idx = sample
                img = image.unsqueeze(0).to(device)
                clean_batch, corrupt_batch = img, img
                clean_qs, corrupt_qs = [question], [corrupt_q]

            clean_feats.append(extractor._extract(clean_batch, clean_qs))
            corrupt_feats.append(extractor._extract(corrupt_batch, corrupt_qs))

            if head_type == "sa":
                clean_pre = extractor._record_sa_pre(clean_batch, clean_qs)
                pf = extractor.extract_patched_sa(
                    corrupt_batch, corrupt_qs, layer, head, clean_pre[layer])
            else:
                clean_pre = extractor._record_gca_pre(clean_batch, clean_qs)
                pf = extractor.extract_patched_gca(
                    corrupt_batch, corrupt_qs, layer, head, clean_pre[layer])
            patched_feats.append(pf)

            answer_labels.append(IDX_TO_ANSWER.get(answer_idx, str(answer_idx)))

            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(samples)}", flush=True)

        clean_all = torch.cat(clean_feats, dim=0).numpy()
        corrupt_all = torch.cat(corrupt_feats, dim=0).numpy()
        patched_all = torch.cat(patched_feats, dim=0).numpy()

        n = len(answer_labels)
        all_feats = np.concatenate([clean_all, corrupt_all, patched_all], axis=0)
        conditions = ["clean"] * n + ["corrupt"] * n + ["patched"] * n
        answers_3x = answer_labels * 3

        np.savez(output_dir / f"cache_{exp_name}.npz",
                 feats=all_feats, conditions=conditions, answers=answers_3x)

        print(f"  Running t-SNE ({all_feats.shape})...")
        tsne = TSNE(n_components=2, perplexity=30, metric="cosine",
                    random_state=args.seed, init="pca", max_iter=1000)
        coords = tsne.fit_transform(all_feats)

        plot_patching_tsne(
            coords, answers_3x, conditions,
            title=f"Denoising: restore {head_name} ({corr_source} {attr})",
            out_path=output_dir / f"{exp_name}.png",
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
