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
    apply_style()
    gca_layers = [1, 3, 5, 7, 9, 11]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    steervit = None
    attrs_list = None
    for condition, q_text in [("noca", None), ("ca", args.question)]:
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
            print(f"\nExtracting features: {condition} ({N} images) ...")
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
            np.savez(out_dir / f"feats_{condition}.npz",
                     **{str(l): all_feats[l] for l in all_feats})

        # All-attribute plot: color=tab20c color palette, marker=shape,
        # black edge=metal, marker size=object size
        color_map = ATTR_VALUE_COLORS["color"]
        fig, axes = make_tsne_grid(len(gca_layers), ncols=3)
        tag = "No cross-attention" if condition == "noca" else f"CA: \"{q_text}\""

        # Vectorized scatter: group by (shape, size, material) for speed
        colors_arr = np.array([color_map[a["color"]][:3] for a in attrs_list])
        shapes_arr = np.array([a["shape"] for a in attrs_list])
        sizes_arr = np.array([SIZE_MARKERSIZE[a["size"]] for a in attrs_list])
        metal_arr = np.array([a["material"] == "metal" for a in attrs_list])

        for ax_i, layer in enumerate(gca_layers):
            ax = axes[ax_i]
            X = all_feats[layer]
            tsne = TSNE(n_components=2, perplexity=30, random_state=42)
            emb = tsne.fit_transform(X)

            for shape, marker in SHAPE_MARKERS.items():
                for is_metal in (False, True):
                    m = (shapes_arr == shape) & (metal_arr == is_metal)
                    if not m.any():
                        continue
                    kw = dict(c=colors_arr[m], s=sizes_arr[m], marker=marker,
                              rasterized=True)
                    if is_metal:
                        kw.update(edgecolors="black",
                                  linewidths=TSNE_STYLE["edge_width"] / 2)
                    else:
                        kw["edgecolors"] = "none"
                    ax.scatter(emb[m, 0], emb[m, 1], **kw)

            ax.set_title(f"L{layer}", fontsize=S["subplot_title_fontsize"])
            style_tsne_ax(ax)

        # Legend: colors + shapes + material/size codes, below the grid
        from matplotlib.lines import Line2D
        legend_handles = []
        for color, c in color_map.items():
            legend_handles.append(Line2D([0], [0], marker="o", color="w",
                                  markerfacecolor=c, markersize=7, label=color))
        for shape, marker in SHAPE_MARKERS.items():
            legend_handles.append(Line2D([0], [0], marker=marker, color="w",
                                  markerfacecolor=TSNE_STYLE["gray"],
                                  markersize=7, label=shape))
        legend_handles.append(Line2D([0], [0], marker="o", color="w",
                              markerfacecolor=TSNE_STYLE["gray"], markersize=7,
                              markeredgecolor="black", markeredgewidth=1.0,
                              label="metal (edge)"))
        legend_handles.append(Line2D([0], [0], marker="o", color="w",
                              markerfacecolor=TSNE_STYLE["gray"], markersize=8,
                              label="large"))
        legend_handles.append(Line2D([0], [0], marker="o", color="w",
                              markerfacecolor=TSNE_STYLE["gray"], markersize=4,
                              label="small"))
        finish_tsne_grid(fig, legend_handles, suptitle=f"Single-object t-SNE — {tag}",
                         ncol=5)

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
    args = ap.parse_args()
    run(args)
