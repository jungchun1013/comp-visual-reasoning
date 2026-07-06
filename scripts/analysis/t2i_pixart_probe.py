"""T2I substrate analysis on PixArt-Sigma (pre-registered: docs/paper_v2_outline.md §T2I).

DIFT-style feature extraction: VAE-encode a CLEVR image, add noise at a small
timestep, run ONE denoising forward through the DiT with the *question text* as
the conditioning prompt, and capture per-block hidden states + cross-attention
maps. Three analyses on the 3 attr_query categories (RETRIEVAL_CATEGORIES):

  (a) per-block linear probing of the answer attribute — global mean-pooled AND
      referent-local (3x3 patch neighborhood at the referent's pixel_coords,
      comparable to raw_backbone_probe.py);
  (b) cross-attn localization: per block, attention mass on the referent's
      3x3 neighborhood (chance = 9/1024) — zero-shot Binding evidence;
  (c) [separate step] frozen 1-layer decoder readout — NOT in this script.

Caveat (pre-registered): questions != captions — domain mismatch for a T2I
text encoder; state in writeup.

Requires diffusers from the vault-root overlay dir:
  PYTHONPATH=<vault>/deps-t2i:src CUDA_VISIBLE_DEVICES=0 <interpreter> \
      scripts/analysis/t2i_pixart_probe.py --stage all --n-per-cat 300
Output: outputs/analysis/t2i_pixart/{features.npz,probe_results.json,*.png}
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from data.clevr_programs import _execute_program  # referent localization
from data.clevr_sampling import RETRIEVAL_CATEGORIES

CATEGORIES = ("attr_query_direct", "attr_query_same", "attr_query_spatial")
CLEVR_W, CLEVR_H = 480, 320  # native render size (pixel_coords space)
IMG_SIZE = 512               # PixArt-Sigma-512 input
LATENT_GRID = IMG_SIZE // 8 // 2  # vae /8, DiT patch 2 -> 32x32 tokens


def find_referent(objects: list[dict], program: list[dict]) -> dict | None:
    """Referent = object set at the last `unique` step before the terminal op.

    Covers direct (filter->unique->query) AND same/relate families (the final
    unique after the pivot selects the queried object). Returns None if the
    program has no unique step or execution is empty there.
    """
    results = _execute_program(objects, program)
    uniq_idx = [i for i, s in enumerate(program)
                if s.get("function", s.get("type", "")) == "unique"]
    if not uniq_idx:
        return None
    objs = results[uniq_idx[-1]]
    return objs[0] if objs else None


def referent_patch_window(obj: dict) -> list[int] | None:
    """3x3 patch-token indices around the referent center in the 32x32 grid."""
    px, py = obj["pixel_coords"][0], obj["pixel_coords"][1]
    col = int(px * IMG_SIZE / CLEVR_W) // (IMG_SIZE // LATENT_GRID)
    row = int(py * IMG_SIZE / CLEVR_H) // (IMG_SIZE // LATENT_GRID)
    idxs = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            r, c = row + dr, col + dc
            if 0 <= r < LATENT_GRID and 0 <= c < LATENT_GRID:
                idxs.append(r * LATENT_GRID + c)
    return idxs or None


class StoreCAProcessor:
    """Attention processor for DiT attn2 that also stores head-mean CA probs."""

    def __init__(self, store: dict, key: int):
        self.store = store
        self.key = key

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, **kwargs):
        ehs = hidden_states if encoder_hidden_states is None else encoder_hidden_states
        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(
                attention_mask, ehs.shape[1], hidden_states.shape[0])
        q = attn.head_to_batch_dim(attn.to_q(hidden_states))
        k = attn.head_to_batch_dim(attn.to_k(ehs))
        v = attn.head_to_batch_dim(attn.to_v(ehs))
        probs = attn.get_attention_scores(q, k, attention_mask)
        h = attn.heads
        bh, n_img, n_txt = probs.shape
        self.store[self.key] = (
            probs.view(bh // h, h, n_img, n_txt).mean(dim=1).detach()  # (B, img, txt)
        )
        out = attn.batch_to_head_dim(torch.bmm(probs, v))
        return attn.to_out[1](attn.to_out[0](out))


def load_pipeline(device: str, dtype: torch.dtype):
    from diffusers import PixArtSigmaPipeline, Transformer2DModel

    tf = Transformer2DModel.from_pretrained(
        "PixArt-alpha/PixArt-Sigma-XL-2-512-MS", subfolder="transformer",
        torch_dtype=dtype)
    pipe = PixArtSigmaPipeline.from_pretrained(
        "PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers", transformer=tf,
        torch_dtype=dtype)
    pipe.to(device)
    pipe.transformer.eval()
    pipe.text_encoder.eval()
    pipe.vae.eval()
    return pipe


def sample_queries_from_json(questions: list[dict], scenes: list[dict],
                             n_per_cat: int, seed: int):
    """Sample n questions per category whose referent is resolvable."""
    rng = np.random.default_rng(seed)
    by_family = defaultdict(list)
    for q in questions:
        by_family[q["question_family_index"]].append(q)
    samples = []
    for cat in CATEGORIES:
        pool = [q for fam in RETRIEVAL_CATEGORIES[cat] for q in by_family[fam]]
        rng.shuffle(pool)
        kept = 0
        for q in pool:
            if kept >= n_per_cat:
                break
            scene = scenes[q["image_index"]]
            ref = find_referent(scene["objects"], q["program"])
            if ref is None:
                continue
            win = referent_patch_window(ref)
            if win is None:
                continue
            samples.append({
                "category": cat, "question": q["question"],
                "answer": q["answer"], "image_filename": q["image_filename"],
                "family": q["question_family_index"], "window": win,
            })
            kept += 1
        print(f"[sample] {cat}: {kept}/{n_per_cat}")
    return samples


@torch.no_grad()
def extract(args, out_dir: Path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    pipe = load_pipeline(device, dtype)
    n_blocks = len(pipe.transformer.transformer_blocks)

    ca_store: dict[int, torch.Tensor] = {}
    for bi, blk in enumerate(pipe.transformer.transformer_blocks):
        blk.attn2.processor = StoreCAProcessor(ca_store, bi)

    hidden_store: dict[int, torch.Tensor] = {}
    hooks = []
    for bi, blk in enumerate(pipe.transformer.transformer_blocks):
        def fn(mod, inp, output, _bi=bi):
            o = output[0] if isinstance(output, tuple) else output
            hidden_store[_bi] = o.detach()
        hooks.append(blk.register_forward_hook(fn))

    data_root = Path(args.data_root)
    questions = json.load(open(data_root / "questions/CLEVR_val_questions.json"))["questions"]
    scenes = json.load(open(data_root / "scenes/CLEVR_val_scenes.json"))["scenes"]
    samples = sample_queries_from_json(questions, scenes, args.n_per_cat, args.seed)

    gen = torch.Generator(device=device).manual_seed(args.seed)
    t = torch.tensor([args.timestep], device=device)
    feats_global = np.zeros((len(samples), n_blocks, 1152), dtype=np.float16)
    feats_local = np.zeros_like(feats_global)
    ca_mass = np.zeros((len(samples), n_blocks), dtype=np.float32)
    saved_maps = {}

    for start in range(0, len(samples), args.batch_size):
        batch = samples[start:start + args.batch_size]
        imgs = []
        for s in batch:
            im = Image.open(data_root / "images/val" / s["image_filename"]).convert("RGB")
            im = im.resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC)
            imgs.append(torch.from_numpy(np.array(im)).permute(2, 0, 1))
        pix = (torch.stack(imgs).to(device, dtype) / 127.5) - 1.0

        lat = pipe.vae.encode(pix).latent_dist.sample(gen) * pipe.vae.config.scaling_factor
        noise = torch.randn(lat.shape, generator=gen, device=device, dtype=dtype)
        noisy = pipe.scheduler.add_noise(lat, noise, t)

        emb, mask, _, _ = pipe.encode_prompt(
            [s["question"] for s in batch], do_classifier_free_guidance=False,
            device=device)

        hidden_store.clear(); ca_store.clear()
        pipe.transformer(
            noisy, encoder_hidden_states=emb, encoder_attention_mask=mask,
            timestep=t.expand(len(batch)),
            added_cond_kwargs={"resolution": None, "aspect_ratio": None})

        for j, s in enumerate(batch):
            win = s["window"]
            n_txt = int(mask[j].sum().item())
            for bi in range(n_blocks):
                h = hidden_store[bi][j].float()          # (1024, 1152)
                feats_global[start + j, bi] = h.mean(0).cpu().numpy()
                feats_local[start + j, bi] = h[win].mean(0).cpu().numpy()
                # Rows (per image token) sum to 1 by construction, so localization
                # must be read column-wise: normalize each text token's column over
                # image patches, then average columns -> "which patches attend to
                # the text more than other patches do". Chance = |win|/1024.
                ca = ca_store[bi][j, :, :n_txt].float()      # (1024, n_txt)
                ca_img = (ca / ca.sum(dim=0, keepdim=True)).mean(dim=1)
                ca_mass[start + j, bi] = ca_img[win].sum().item()
            if start + j < args.save_maps:
                saved_maps[f"map_{start + j}"] = torch.stack(
                    [(lambda c: (c / c.sum(dim=0, keepdim=True)).mean(dim=1))(
                        ca_store[bi][j, :, :n_txt].float())
                     for bi in range(n_blocks)]).cpu().numpy()
        if (start // args.batch_size) % 5 == 0:
            print(f"[extract] {start + len(batch)}/{len(samples)}", flush=True)

    for h in hooks:
        h.remove()
    np.savez_compressed(
        out_dir / "features.npz", feats_global=feats_global,
        feats_local=feats_local, ca_mass=ca_mass,
        answers=np.array([s["answer"] for s in samples]),
        categories=np.array([s["category"] for s in samples]),
        families=np.array([s["family"] for s in samples]),
        windows_len=np.array([len(s["window"]) for s in samples]),
        timestep=args.timestep, **saved_maps)
    (out_dir / "samples.json").write_text(json.dumps(samples, indent=1))
    print(f"[extract] wrote {out_dir}/features.npz  n={len(samples)}")


def probe(args, out_dir: Path):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    d = np.load(out_dir / "features.npz", allow_pickle=False)
    answers, cats = d["answers"], d["categories"]
    n_blocks = d["feats_global"].shape[1]
    results = {"timestep": int(d["timestep"]), "per_category": {}}

    for cat in CATEGORIES:
        m = cats == cat
        y = answers[m]
        classes, counts = np.unique(y, return_counts=True)
        keep = np.isin(y, classes[counts >= 5])
        chance = counts[counts >= 5].max() / counts[counts >= 5].sum()
        entry = {"n": int(keep.sum()), "n_classes": int((counts >= 5).sum()),
                 "majority_baseline": float(chance)}
        for feat_key in ("feats_global", "feats_local"):
            accs = []
            for bi in range(n_blocks):
                X = d[feat_key][m][keep][:, bi].astype(np.float32)
                clf = make_pipeline(StandardScaler(),
                                    LogisticRegression(max_iter=2000, C=1.0))
                cv = StratifiedKFold(5, shuffle=True, random_state=0)
                accs.append(float(np.mean(
                    cross_val_score(clf, X, y[keep], cv=cv, scoring="accuracy"))))
            entry[feat_key] = accs
        entry["ca_mass_mean"] = d["ca_mass"][m].mean(axis=0).tolist()
        entry["ca_mass_chance"] = float(
            np.mean(d["windows_len"][m]) / (LATENT_GRID * LATENT_GRID))
        results["per_category"][cat] = entry
        print(f"[probe] {cat}: local best "
              f"{max(entry['feats_local']):.3f} @ block "
              f"{int(np.argmax(entry['feats_local']))} "
              f"(majority {chance:.3f}) | CA mass peak "
              f"{max(entry['ca_mass_mean']):.4f} vs chance "
              f"{entry['ca_mass_chance']:.4f}")

    (out_dir / "probe_results.json").write_text(json.dumps(results, indent=2))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from analysis.plot_style import apply_style

    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for cat in CATEGORIES:
        e = results["per_category"][cat]
        axes[0].plot(e["feats_local"], label=f"{cat} (local)")
        axes[0].plot(e["feats_global"], ls="--", alpha=0.6, label=f"{cat} (global)")
        axes[1].plot(e["ca_mass_mean"], label=cat)
    axes[0].axhline(results["per_category"][CATEGORIES[0]]["majority_baseline"],
                    color="gray", ls=":", label="majority")
    axes[0].set(xlabel="DiT block", ylabel="probe accuracy",
                title="Answer decodability (PixArt-Sigma, 1 denoise step)")
    axes[1].axhline(results["per_category"][CATEGORIES[0]]["ca_mass_chance"],
                    color="gray", ls=":", label="chance")
    axes[1].set(xlabel="DiT block", ylabel="CA mass on referent",
                title="Cross-attn localization on referent")
    for ax in axes:
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "t2i_probe.png", dpi=200)
    print(f"[probe] wrote {out_dir}/probe_results.json + t2i_probe.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["extract", "probe", "all"], default="all")
    ap.add_argument("--data-root", default="/home/jungchun/data/clevr/CLEVR_v1.0")
    ap.add_argument("--output-dir", default="outputs/analysis/t2i_pixart")
    ap.add_argument("--n-per-cat", type=int, default=300)
    ap.add_argument("--timestep", type=int, default=100,
                    help="DIFT-style noise timestep (of 1000)")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--save-maps", type=int, default=8,
                    help="save full per-block CA maps for the first N samples")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.stage in ("extract", "all"):
        extract(args, out_dir)
    if args.stage in ("probe", "all"):
        probe(args, out_dir)


if __name__ == "__main__":
    main()
