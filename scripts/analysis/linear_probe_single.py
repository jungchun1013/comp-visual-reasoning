"""Linear probing on single-object images: attribute decodability per layer, no-CA vs CA.

For each GCA layer × attribute × condition (no-CA / CA with 3 questions),
train a logistic regression on 80% of data, report accuracy on 20%.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import numpy as np
import torch
from torch.amp import autocast
from PIL import Image as PILImage
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ATTRS = ["color", "shape", "material", "size"]
GCA_LAYERS = [1, 3, 5, 7, 9, 11]


def load_model(ckpt_path, device):
    from omegaconf import OmegaConf
    from model import CrossAttnViT
    from tasks.decoder import build_decoder_model, build_clevr_decoder_vocab

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = OmegaConf.create(ckpt["config"])
    steervit = CrossAttnViT.from_config(
        cfg.model.backbone_name, device=device,
        cross_attn_layers=list(cfg.model.cross_attn_layers),
        resolution=cfg.model.resolution,
        pretrained=cfg.model.get("pretrained", True),
    )
    vocab = build_clevr_decoder_vocab()
    model_cfg = OmegaConf.create({"model": cfg.model, "task": cfg.task, "data": cfg.data})
    model = build_decoder_model(steervit, model_cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()
    return steervit, steervit.get_transforms()


class Retriever:
    def __init__(self, steervit):
        self.steervit = steervit
        self.blocks = steervit.vision_model.trunk.blocks
        self.norm = steervit.vision_model.trunk.norm
        self.prefix = steervit.vision_model.trunk.num_prefix_tokens
        self.num_layers = len(self.blocks)

    @torch.no_grad()
    def extract_batch(self, images, question_text=None):
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


def extract_all(retriever, images, question_text, device, bs=32):
    N = images.shape[0]
    all_feats = {l: [] for l in range(retriever.num_layers)}
    for start in range(0, N, bs):
        end = min(start + bs, N)
        batch = images[start:end].to(device)
        feats = retriever.extract_batch(batch, question_text)
        for l in range(retriever.num_layers):
            all_feats[l].append(feats[l])
    for l in range(retriever.num_layers):
        all_feats[l] = np.concatenate(all_feats[l], axis=0)
    return all_feats


def probe_accuracy(X, y, n_splits=5):
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    if len(le.classes_) < 2:
        return 1.0
    pca = PCA(n_components=min(50, X.shape[1], X.shape[0]))
    X_r = pca.fit_transform(X)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    accs = []
    for train_idx, test_idx in skf.split(X_r, y_enc):
        clf = LogisticRegression(max_iter=500, C=1.0, solver="lbfgs")
        clf.fit(X_r[train_idx], y_enc[train_idx])
        accs.append(clf.score(X_r[test_idx], y_enc[test_idx]))
    return np.mean(accs)


COND_DISPLAY = {
    "noca": "no-CA",
    "ca_object": 'CA: "What color is the object?"',
    "ca_cube": 'CA: "What color is the cube?"',
}


def run(args):
    from analysis.plot_style import apply_style, S, line_kwargs, save_with_legend
    apply_style()

    out_dir = Path(args.out_dir) if args.out_dir else \
        (Path(args.features_dir) if args.features_dir
         else Path("outputs/analysis/linear_probe/single_object"))
    out_dir.mkdir(parents=True, exist_ok=True)

    # Conditions: (name, {layer: (N, D)})
    conditions = []
    if args.features_dir:
        # Reuse the t-SNE feature caches — identical setting by construction
        fd = Path(args.features_dir)
        with open(fd / "attrs.json") as f:
            attrs_list = json.load(f)
        for fpath in sorted(fd.glob("feats_*.npz")):
            cond = fpath.stem[len("feats_"):]
            data = np.load(fpath)
            conditions.append((cond, {int(k): data[k] for k in data.files}))
        print(f"Loaded {len(conditions)} cached conditions from {fd}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        steervit, tf = load_model(args.checkpoint, device)
        retriever = Retriever(steervit)

        data_dir = Path(args.data_dir)
        with open(data_dir / "scenes.json") as f:
            scenes = json.load(f)["scenes"]
        images, attrs_list = [], []
        for s in scenes:
            img = PILImage.open(data_dir / "images" / s["image_filename"]).convert("RGB")
            images.append(tf(img))
            attrs_list.append(s["objects"][0])
        images = torch.stack(images)
        print(f"Loaded {len(attrs_list)} images")

        print("\nExtracting: no-CA ...")
        conditions.append(("noca", extract_all(retriever, images, None, device)))
        for cond, q_text in [("ca_object", "What color is the object?"),
                             ("ca_cube", "What color is the cube?")]:
            print(f"\nExtracting: {cond} ...")
            conditions.append((cond, extract_all(retriever, images, q_text, device)))

    results = {}
    for cond, feats in conditions:
        results[cond] = {}
        print(f"\nProbing: {cond} ...")
        for attr in ATTRS:
            labels = [a[attr] for a in attrs_list]
            accs = []
            for layer in GCA_LAYERS:
                acc = probe_accuracy(feats[layer], labels)
                accs.append(acc)
                print(f"  L{layer:2d} {attr:10s}: {acc:.3f}")
            results[cond][attr] = accs

    with open(out_dir / "linear_probe_results.json", "w") as f:
        json.dump({"layers": GCA_LAYERS, "results": results}, f, indent=2)

    # Plot: one panel per attribute, lines = conditions (project plot style)
    _tab10 = plt.cm.tab10.colors
    cond_color = {"noca": (0.4, 0.4, 0.4), "ca_object": _tab10[0], "ca_cube": _tab10[1]}

    fig, axes = plt.subplots(1, 4, figsize=(8 * 4, 6), sharey=True)
    for ax, attr in zip(axes, ATTRS):
        for cond, _ in conditions:
            kw = line_kwargs(COND_DISPLAY.get(cond, cond),
                             color=cond_color.get(cond))
            if cond == "noca":
                kw.update(linestyle="--")
            ax.plot(GCA_LAYERS, results[cond][attr], **kw)
        ax.set_title(attr.capitalize(), fontsize=S["subplot_title_fontsize"])
        ax.set_xlabel("Layer")
        ax.set_xticks(GCA_LAYERS)
        ax.set_ylim(0, 1.05)
    axes[0].set_ylabel("5-fold CV accuracy")

    save_with_legend(fig, str(out_dir / "linear_probe.png"))
    print(f"\nSaved: {out_dir / 'linear_probe.png'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="outputs/model/clevr_dinov2_decoder1l_scratch_s42/best.pt")
    ap.add_argument("--data-dir", default="data/clevr_single_object_v3")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--features-dir", default=None,
                    help="Reuse feats_*.npz + attrs.json from a t-SNE run "
                         "(e.g. outputs/analysis/tsne/object_count/n1) — no GPU")
    args = ap.parse_args()
    run(args)
