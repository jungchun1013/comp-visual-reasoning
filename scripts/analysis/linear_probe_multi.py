"""Linear probing on multi-object CLEVR images: attribute decodability per layer.

For attribute-query questions (query_color/shape/material/size), train a linear
probe to predict the ANSWER from mean-pooled patch features.  Under no-CA the
same image yields the same features regardless of question, so the probe must
deal with genuine multi-object ambiguity.  Under CA, the question conditions
the features and should enable object selection.
"""
from __future__ import annotations
import argparse, json, random, sys
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
from analysis.run_log import tee_stdout

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

QUERY_TYPES = ["query_color", "query_shape", "query_material", "query_size"]
ATTR_LABELS = {"query_color": "color", "query_shape": "shape",
               "query_material": "material", "query_size": "size"}
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
    def extract_batch(self, images, question_texts=None):
        layer_out = {}
        hooks = []
        for idx, blk in enumerate(self.blocks):
            def make_hook(li):
                def fn(mod, inp, out):
                    o = out[0] if isinstance(out, tuple) else out
                    layer_out[li] = o.detach()
                return fn
            hooks.append(blk.register_forward_hook(make_hook(idx)))

        with autocast(device_type="cuda", dtype=torch.bfloat16):
            self.steervit.forward(images, question_texts)
        for h in hooks:
            h.remove()

        feats = {}
        for l in range(self.num_layers):
            normed = self.norm(layer_out[l].float())
            patches = normed[:, self.prefix:, :]
            feats[l] = patches.mean(dim=1).cpu().numpy()
        return feats


def sample_questions(questions_path, n_per_type=500, seed=42):
    """Sample n_per_type attribute-query questions per query type."""
    with open(questions_path) as f:
        all_qs = json.load(f)["questions"]

    by_type = {qt: [] for qt in QUERY_TYPES}
    for q in all_qs:
        prog = q.get("program", [])
        if not prog:
            continue
        last_fn = prog[-1]["function"]
        if last_fn in by_type:
            by_type[last_fn].append(q)

    rng = random.Random(seed)
    sampled = {}
    for qt, pool in by_type.items():
        rng.shuffle(pool)
        sampled[qt] = pool[:n_per_type]
        print(f"  {qt}: sampled {len(sampled[qt])} from {len(pool)}")
    return sampled


def extract_features(retriever, image_tensors, questions, device, bs=32):
    """Extract features for a list of (image_tensor, question) pairs.

    questions: list of str (CA) or None (no-CA).
    """
    N = len(image_tensors)
    all_feats = {l: [] for l in range(retriever.num_layers)}
    for start in range(0, N, bs):
        end = min(start + bs, N)
        batch_imgs = torch.stack(image_tensors[start:end]).to(device)
        if questions is not None:
            batch_qs = questions[start:end]
        else:
            batch_qs = None
        feats = retriever.extract_batch(batch_imgs, batch_qs)
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

    clevr_root = Path(args.clevr_root)
    image_dir = clevr_root / "images" / "val"
    questions_path = clevr_root / "questions" / "CLEVR_val_questions.json"

    print("Sampling questions...")
    sampled = sample_questions(questions_path, n_per_type=args.n_per_type)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tee_stdout(out_dir)

    results = {}

    for qt in QUERY_TYPES:
        attr = ATTR_LABELS[qt]
        qs = sampled[qt]
        print(f"\n=== {qt} ({len(qs)} samples) ===")

        image_tensors = []
        questions_text = []
        answers = []
        for q in qs:
            img = PILImage.open(image_dir / q["image_filename"]).convert("RGB")
            image_tensors.append(tf(img))
            questions_text.append(q["question"])
            answers.append(q["answer"])

        # No-CA
        print(f"  Extracting no-CA features...")
        noca_feats = extract_features(retriever, image_tensors, None, device)

        # CA with actual question
        print(f"  Extracting CA features...")
        ca_feats = extract_features(retriever, image_tensors, questions_text, device)

        results[qt] = {"no-CA": [], "CA": []}
        for layer in GCA_LAYERS:
            acc_noca = probe_accuracy(noca_feats[layer], answers)
            acc_ca = probe_accuracy(ca_feats[layer], answers)
            results[qt]["no-CA"].append(acc_noca)
            results[qt]["CA"].append(acc_ca)
            print(f"  L{layer:2d}  no-CA={acc_noca:.3f}  CA={acc_ca:.3f}")

    # Save results
    with open(out_dir / "linear_probe_multi_results.json", "w") as f:
        json.dump({"layers": GCA_LAYERS, "results": results}, f, indent=2)

    # Plot: one subplot per query type
    fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharey=True)
    fig.suptitle("Linear probe accuracy — multi-object CLEVR images\n"
                 "Target: answer to attribute query (the queried object's attribute)",
                 fontsize=14)

    for ax, qt in zip(axes, QUERY_TYPES):
        attr = ATTR_LABELS[qt]
        ax.plot(GCA_LAYERS, results[qt]["no-CA"],
                color="black", ls="--", lw=2.5, marker="o", markersize=5,
                label="no-CA")
        ax.plot(GCA_LAYERS, results[qt]["CA"],
                color="tab:red", ls="-", lw=2, marker="o", markersize=5,
                label="CA (actual question)")
        ax.set_title(f"{attr}", fontsize=14)
        ax.set_xlabel("Layer")
        ax.set_xticks(GCA_LAYERS)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("5-fold CV accuracy")
    axes[-1].legend(loc="lower right", fontsize=10)
    plt.tight_layout(rect=[0, 0, 1, 0.88])

    fig.savefig(out_dir / "linear_probe_multi.png", dpi=150)
    plt.close(fig)
    print(f"\nSaved: {out_dir / 'linear_probe_multi.png'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint",
                    default="outputs/model/clevr_dinov2_decoder1l_scratch_s42/best.pt")
    ap.add_argument("--clevr-root",
                    default="/nfs/turbo/coe-chaijy/jungchun/data/clevr/CLEVR_v1.0")
    ap.add_argument("--n-per-type", type=int, default=500)
    ap.add_argument("--out-dir",
                    default="outputs/analysis/linear_probe/multi_object")
    args = ap.parse_args()
    run(args)
