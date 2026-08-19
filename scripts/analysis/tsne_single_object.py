"""t-SNE on single-object rendered images: no-CA vs CA, colored by attribute."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import numpy as np
import torch
from torch.amp import autocast
from sklearn.manifold import TSNE
from PIL import Image as PILImage
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl

from analysis.plot_style import (
    apply_style, ATTR_VALUE_COLORS, style_tsne_ax, make_tsne_grid,
    finish_tsne_grid, TSNE_STYLE, S,
)
from analysis.run_log import tee_stdout

SHAPE_MARKERS = {"sphere": "o", "cube": "s", "cylinder": "^"}
# Marker size encodes object size: hi/bg sizes from the shared t-SNE style
SIZE_MARKERSIZE = {"large": TSNE_STYLE["hi_size"], "small": TSNE_STYLE["bg_size"]}


def load_model(ckpt_path, device):
    from omegaconf import OmegaConf
    from model import CrossAttnViT
    from tasks.decoder import build_decoder_model, build_clevr_decoder_vocab

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = OmegaConf.create(ckpt["config"])

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
    return steervit, transform, cfg


class SingleObjRetriever:
    def __init__(self, steervit):
        self.steervit = steervit
        self.blocks = steervit.vision_model.trunk.blocks
        self.norm = steervit.vision_model.trunk.norm
        self.prefix = steervit.vision_model.trunk.num_prefix_tokens
        self.num_layers = len(self.blocks)

    @torch.no_grad()
    def extract_all(self, images, question_text=None):
        """question_text: None (no CA), a single string, or a per-image list."""
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
        with autocast(device_type="cuda", dtype=torch.bfloat16):
            self.steervit.forward(images, texts)

        for h in hooks:
            h.remove()

        feats = {}
        for l in range(self.num_layers):
            normed = self.norm(layer_out[l].float())
            patches = normed[:, self.prefix:, :]
            feats[l] = patches.mean(dim=1).cpu().numpy()
        return feats


def load_single_obj_dataset(data_dir, transform):
    """Load a controlled rendered dataset.

    Two formats: scenes.json (CLEVR-gen style; target = objects[0]) or
    metadata.json (render_single_objects.py style; target attrs at top level,
    optional 'distractors' list for multi-object scenes).
    """
    data_dir = Path(data_dir)
    if (data_dir / "metadata.json").exists():
        with open(data_dir / "metadata.json") as f:
            entries = json.load(f)
        fname_key = "filename"
    else:
        with open(data_dir / "scenes.json") as f:
            scenes = json.load(f)["scenes"]
        entries = [dict(s["objects"][0], image_filename=s["image_filename"])
                   for s in scenes]
        fname_key = "image_filename"
    images, attrs_list = [], []
    for e in entries:
        img = PILImage.open(data_dir / "images" / e[fname_key]).convert("RGB")
        images.append(transform(img))
        attrs_list.append(e)
    return torch.stack(images), attrs_list


DESC_ORDER = ["size", "color", "material", "shape"]


def described_question(attrs, query_attr):
    """CLEVR-style query describing the target by its other 3 attributes,
    e.g. query_attr='color' -> 'What is the color of the large rubber cube?'"""
    parts = [attrs[k] for k in DESC_ORDER if k != query_attr]
    desc = " ".join(parts) + (" object" if query_attr == "shape" else "")
    return f"What is the {query_attr} of the {desc}?"


REFER_PRIORITY = ["shape", "size", "material", "color"]


def minimal_referring_question(attrs, query_attr):
    """Referring question with ONE distinguishing attribute — template
    'What {query} is the {referent} object?' ('... is the {shape}?' when the
    referent is the shape). The referent is the first attribute in
    REFER_PRIORITY, excluding the queried one, whose value differs between
    target and distractor, so it uniquely picks the target. The >=2-attribute
    distractor rule guarantees such an attribute exists."""
    d = attrs["distractors"][0]
    for a in REFER_PRIORITY:
        if a != query_attr and attrs[a] != d[a]:
            if a == "shape":
                return f"What {query_attr} is the {attrs['shape']}?"
            return f"What {query_attr} is the {attrs[a]} object?"
    raise ValueError(f"no distinguishing non-query attribute: {attrs}")


def run(args):
    apply_style()
    gca_layers = [1, 3, 5, 7, 9, 11]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tee_stdout(out_dir)

    steervit = None
    attrs_list = None
    for condition, q_text in [("noca", None), ("ca", args.question)]:
        # --tag distinguishes CA runs with different questions; noca is
        # question-independent and keeps untagged names
        if condition == "ca" and args.tag:
            condition = f"ca_{args.tag}"
        cache_file = (Path(args.features_dir) / f"feats_{condition}.npz"
                      if args.features_dir else None)
        if cache_file is not None and cache_file.exists():
            data = np.load(cache_file)
            all_feats = {int(k): data[k] for k in data.files}
            with open(Path(args.features_dir) / "attrs.json") as f:
                attrs_list = json.load(f)
            print(f"Loaded cached features: {cache_file}")
        else:
            if steervit is None:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                steervit, tf, cfg = load_model(args.checkpoint, device)
                retriever = SingleObjRetriever(steervit)
                images, attrs_list = load_single_obj_dataset(args.data_dir, tf)
                with open(out_dir / "attrs.json", "w") as f:
                    json.dump(attrs_list, f)
            N = len(attrs_list)
            # Per-scene question modes (multi-object scenes need selection):
            # --describe-target names the target by its other 3 attributes;
            # --refer-minimal uses one distinguishing attribute
            per_scene_q = None
            if condition.startswith("ca") and args.describe_target:
                per_scene_q = [described_question(a, args.query_attr)
                               for a in attrs_list]
                print(f"Described-target questions, e.g.: {per_scene_q[0]}")
            elif condition.startswith("ca") and args.refer_minimal:
                per_scene_q = [minimal_referring_question(a, args.query_attr)
                               for a in attrs_list]
                print(f"Minimal referring questions, e.g.: {per_scene_q[0]}")
            print(f"\nExtracting features: {condition} ({N} images) ...")
            all_feats = {l: [] for l in range(retriever.num_layers)}
            bs = 32
            for start in range(0, N, bs):
                end = min(start + bs, N)
                batch = images[start:end].to(device)
                feats = retriever.extract_all(
                    batch, per_scene_q[start:end] if per_scene_q else q_text)
                for l in range(retriever.num_layers):
                    all_feats[l].append(feats[l])
                if end % 200 == 0 or end == N:
                    print(f"  {end}/{N}", flush=True)
            for l in range(retriever.num_layers):
                all_feats[l] = np.concatenate(all_feats[l], axis=0)
            np.savez(out_dir / f"feats_{condition}.npz",
                     **{str(l): all_feats[l] for l in all_feats})

        # All-attribute plot — shared encoding with the pooled backbone t-SNE
        # figures: hue = color, shade = material (dark = metal, light =
        # rubber), marker = shape, point size = size
        from raw_backbone_probe import _pooled_scatter
        from dino_attribute_tsne import attribute_legend_handles
        fig, axes = make_tsne_grid(len(gca_layers), ncols=3)
        if condition == "noca":
            tag = "No cross-attention"
        elif args.describe_target:
            tag = f"CA: query {args.query_attr}, target described"
        elif args.refer_minimal:
            tag = f"CA: query {args.query_attr}, minimal referring"
        else:
            tag = f"CA: \"{q_text}\""
        n_obj = 1 + len(attrs_list[0].get("distractors", []))
        scene_label = f"{n_obj}-object"

        for ax_i, layer in enumerate(gca_layers):
            ax = axes[ax_i]
            X = all_feats[layer]
            tsne = TSNE(n_components=2, perplexity=30, random_state=42)
            emb = tsne.fit_transform(X)
            _pooled_scatter(ax, emb, attrs_list)
            ax.set_title(f"L{layer}", fontsize=S["subplot_title_fontsize"])
            style_tsne_ax(ax)

        finish_tsne_grid(fig, attribute_legend_handles(),
                         suptitle=f"{scene_label} t-SNE — {tag}", ncol=5)

        fname = f"tsne_single_{condition}_allattr.png"
        out_path = out_dir / fname
        fig.savefig(out_path, dpi=S["dpi"], bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="outputs/clevr_dinov2_decoder1l_scratch/best.pt")
    ap.add_argument("--data-dir", default="data/clevr_single_object")
    ap.add_argument("--question", default="What color is the object?")
    ap.add_argument("--out-dir", default="outputs/analysis/tsne/single_object")
    ap.add_argument("--features-dir", default=None,
                    help="Dir with feats_{noca,ca}.npz + attrs.json — skip extraction")
    ap.add_argument("--tag", default=None,
                    help="Suffix for the CA condition's cache/figure names "
                         "(e.g. 'shape' for a what-shape question)")
    ap.add_argument("--describe-target", action="store_true",
                    help="Per-scene question naming the target by its other 3 "
                         "attributes (for multi-object controlled scenes)")
    ap.add_argument("--refer-minimal", action="store_true",
                    help="Per-scene referring question with ONE distinguishing "
                         "attribute ('What {query} is the {referent} object?'); "
                         "requires scenes with distractors")
    ap.add_argument("--query-attr", default="color",
                    choices=["color", "shape", "material", "size"],
                    help="Queried attribute in --describe-target mode")
    args = ap.parse_args()
    run(args)
