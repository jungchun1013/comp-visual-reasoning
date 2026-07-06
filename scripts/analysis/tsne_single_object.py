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

import matplotlib.colors as mcolors

# CLEVR base colors (from properties.json), used as-is for metal (dark/saturated)
CLEVR_COLORS_METAL = {
    "gray":   np.array([87, 87, 87]) / 255,
    "red":    np.array([173, 35, 35]) / 255,
    "blue":   np.array([42, 75, 215]) / 255,
    "green":  np.array([29, 105, 20]) / 255,
    "brown":  np.array([129, 74, 25]) / 255,
    "purple": np.array([129, 38, 192]) / 255,
    "cyan":   np.array([41, 208, 208]) / 255,
    "yellow": np.array([255, 238, 51]) / 255,
}
# Rubber = lighter version (blend toward white)
CLEVR_COLORS_RUBBER = {k: v * 0.55 + 0.45 for k, v in CLEVR_COLORS_METAL.items()}

SHAPE_MARKERS = {"sphere": "o", "cube": "s", "cylinder": "^"}
SIZE_MARKERSIZE = {"large": 55, "small": 20}


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
        layer_out = {}
        hooks = []
        for idx, blk in enumerate(self.blocks):
            def make_hook(li):
                def fn(mod, inp, out):
                    o = out[0] if isinstance(out, tuple) else out
                    layer_out[li] = o.detach()
                return fn
            hooks.append(blk.register_forward_hook(make_hook(idx)))

        texts = [question_text] * images.shape[0] if question_text else None
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
    data_dir = Path(data_dir)
    with open(data_dir / "scenes.json") as f:
        scenes = json.load(f)["scenes"]
    images, attrs_list = [], []
    for s in scenes:
        img = PILImage.open(data_dir / "images" / s["image_filename"]).convert("RGB")
        images.append(transform(img))
        attrs_list.append(s["objects"][0])
    return torch.stack(images), attrs_list


def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    steervit, tf, cfg = load_model(args.checkpoint, device)
    retriever = SingleObjRetriever(steervit)

    images, attrs_list = load_single_obj_dataset(args.data_dir, tf)
    N = len(attrs_list)
    print(f"Loaded {N} single-object images")

    gca_layers = [1, 3, 5, 7, 9, 11]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for condition, q_text in [("noca", None), ("ca", args.question)]:
        print(f"\nExtracting features: {condition} ...")
        all_feats = {l: [] for l in range(retriever.num_layers)}
        bs = 32
        for start in range(0, N, bs):
            end = min(start + bs, N)
            batch = images[start:end].to(device)
            feats = retriever.extract_all(batch, q_text)
            for l in range(retriever.num_layers):
                all_feats[l].append(feats[l])
            if end % 200 == 0 or end == N:
                print(f"  {end}/{N}", flush=True)
        for l in range(retriever.num_layers):
            all_feats[l] = np.concatenate(all_feats[l], axis=0)

        # All-attribute plot: color=CLEVR color, marker=shape, dark/light=material, size=size
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        tag = "No cross-attention" if condition == "noca" else f"CA: \"{q_text}\""
        fig.suptitle(f"Single-object t-SNE — {tag}\n"
                     "marker=shape, dark/light=material, size=object size, color=CLEVR color",
                     fontsize=14)

        for ax_i, layer in enumerate(gca_layers):
            ax = axes[ax_i // 3, ax_i % 3]
            X = all_feats[layer]
            tsne = TSNE(n_components=2, perplexity=30, random_state=42)
            emb = tsne.fit_transform(X)

            for i, a in enumerate(attrs_list):
                cmap = CLEVR_COLORS_METAL if a["material"] == "metal" else CLEVR_COLORS_RUBBER
                c = cmap[a["color"]]
                marker = SHAPE_MARKERS[a["shape"]]
                s = SIZE_MARKERSIZE[a["size"]]
                edge = "black" if a["material"] == "metal" else "none"
                ax.scatter(emb[i, 0], emb[i, 1], c=[c], marker=marker, s=s,
                           alpha=0.8, edgecolors=edge, linewidths=0.4)

            ax.set_title(f"L{layer}", fontsize=14)
            ax.set_xticks([])
            ax.set_yticks([])

        # Build legend
        from matplotlib.lines import Line2D
        legend_handles = []
        for shape, marker in SHAPE_MARKERS.items():
            legend_handles.append(Line2D([0], [0], marker=marker, color="w",
                                  markerfacecolor="gray", markersize=8, label=shape))
        legend_handles.append(Line2D([0], [0], marker="o", color="w",
                              markerfacecolor="gray", markersize=10, markeredgecolor="black",
                              markeredgewidth=1.0, label="metal (dark+edge)"))
        legend_handles.append(Line2D([0], [0], marker="o", color="w",
                              markerfacecolor="lightgray", markersize=10, label="rubber (light)"))
        legend_handles.append(Line2D([0], [0], marker="o", color="w",
                              markerfacecolor="gray", markersize=10, label="large"))
        legend_handles.append(Line2D([0], [0], marker="o", color="w",
                              markerfacecolor="gray", markersize=5, label="small"))
        fig.legend(handles=legend_handles, loc="lower center", ncol=7, fontsize=11,
                   markerscale=1.0)
        plt.tight_layout(rect=[0, 0.06, 1, 0.91])

        fname = f"tsne_single_{condition}_allattr.png"
        fig.savefig(out_dir / fname, dpi=150)
        plt.close(fig)
        print(f"  Saved {fname}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="outputs/clevr_dinov2_decoder1l_scratch/best.pt")
    ap.add_argument("--data-dir", default="data/clevr_single_object")
    ap.add_argument("--question", default="What color is the object?")
    ap.add_argument("--out-dir", default="outputs/analysis/tsne/single_object")
    args = ap.parse_args()
    run(args)
