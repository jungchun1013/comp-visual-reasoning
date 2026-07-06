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


def run(args):
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
    N = len(attrs_list)
    print(f"Loaded {N} images")

    ca_questions = [
        ("CA: color", "What color is the object?"),
        ("CA: shape", "What shape is the object?"),
        ("CA: color of cube", "What color is the cube?"),
    ]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Extract no-CA once
    print("\nExtracting: no-CA ...")
    noca_feats = extract_all(retriever, images, None, device)
    results = {}
    results["no-CA"] = {}
    for attr in ATTRS:
        labels = [a[attr] for a in attrs_list]
        accs = []
        for layer in GCA_LAYERS:
            acc = probe_accuracy(noca_feats[layer], labels)
            accs.append(acc)
            print(f"  L{layer:2d} {attr:10s}: {acc:.3f}")
        results["no-CA"][attr] = accs

    # Extract CA for each question
    for cond_name, q_text in ca_questions:
        print(f"\nExtracting: {cond_name} ...")
        feats = extract_all(retriever, images, q_text, device)
        results[cond_name] = {}
        for attr in ATTRS:
            labels = [a[attr] for a in attrs_list]
            accs = []
            for layer in GCA_LAYERS:
                acc = probe_accuracy(feats[layer], labels)
                accs.append(acc)
                print(f"  L{layer:2d} {attr:10s}: {acc:.3f}")
            results[cond_name][attr] = accs

    with open(out_dir / "linear_probe_results.json", "w") as f:
        json.dump({"layers": GCA_LAYERS, "results": results}, f, indent=2)

    # Plot: one subplot per attribute, lines = conditions
    COND_STYLES = {
        "no-CA":             {"color": "black", "ls": "--", "lw": 2.5},
        "CA: color":         {"color": "tab:red", "ls": "-", "lw": 2},
        "CA: shape":         {"color": "tab:blue", "ls": "-", "lw": 2},
        "CA: color of cube": {"color": "tab:green", "ls": "-", "lw": 2},
    }

    fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharey=True)
    fig.suptitle("Linear probe accuracy per layer — single-object images", fontsize=16)

    for ax, attr in zip(axes, ATTRS):
        for cond_name in ["no-CA"] + [c[0] for c in ca_questions]:
            style = COND_STYLES[cond_name]
            ax.plot(GCA_LAYERS, results[cond_name][attr],
                    label=cond_name, **style, marker="o", markersize=5)
        ax.set_title(attr, fontsize=14)
        ax.set_xlabel("Layer")
        ax.set_xticks(GCA_LAYERS)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("5-fold CV accuracy")
    axes[-1].legend(loc="lower right", fontsize=9)
    plt.tight_layout(rect=[0, 0, 1, 0.93])

    fig.savefig(out_dir / "linear_probe.png", dpi=150)
    plt.close(fig)
    print(f"\nSaved: {out_dir / 'linear_probe.png'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="outputs/model/clevr_dinov2_decoder1l_scratch_s42/best.pt")
    ap.add_argument("--data-dir", default="data/clevr_single_object_v3")
    ap.add_argument("--out-dir", default="outputs/analysis/linear_probe/single_object")
    args = ap.parse_args()
    run(args)
