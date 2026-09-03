"""X21 — what a referring question does to the additive object vector of a
patch token, and whether that vector is causally additive.

Background (X19): in the frozen ViT-B with gated cross-attention (GCA) and NO
question, a patch containing an object = background token at that position +
an additive, position-invariant, object-specific vector. X20: without
cross-attention the ViT stream carries no question-related signal. This script
adds the language condition and the causal test, following
  * Song, Lepori & Pavlick 2025 (concept-vector projections, Δ_ref / Δ_nonref),
  * Feng & Steinhardt 2024 / Saravanan et al. 2025 (difference-in-means
    vector, additive swap with a norm-matched random control),
  * Darcet et al. 2024 (high-norm background tokens; norm-standardised variant),
  * Assouel et al. 2025 (identity-RSM vs position-RSM dissociation).

Conditions on the 2-object renders (n2): c0 no question; c1 question that
uniquely refers to the TARGET; c2 uniquely refers to the DISTRACTOR;
c3 non-referring "What color is the object?". n1 (target alone): c0, c1.

Three phases (X19 pattern):
    --masks-only   selection + segmentation + masks_debug (CPU)
    (default)      sparse feature extraction per condition (GPU)
    --intervene    Part B residual interventions with decoder readout (GPU)
    --replot       Part A metrics, Part C probes, all figures (CPU, from cache)

Usage (from main/ or the worktree root):
    PYTHONPATH=src CUDA_VISIBLE_DEVICES=  <py> scripts/analysis/patch_language_condition.py --masks-only
    PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 <py> scripts/analysis/patch_language_condition.py
    PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 <py> scripts/analysis/patch_language_condition.py --intervene
    PYTHONPATH=src CUDA_VISIBLE_DEVICES=  <py> scripts/analysis/patch_language_condition.py --replot
    PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 <py> scripts/analysis/patch_language_condition.py --mirror \
        --checkpoint outputs/model/<mirror run>/best.pt --out-dir outputs/analysis/patch_language_condition/mirror
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.amp import autocast
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from analysis.plot_style import apply_style, S, GCA_LAYERS, mark_gca_layers, line_kwargs
from analysis.run_log import tee_stdout
from tsne_patch_level import load_entries, build_masks, save_masks_debug
from tsne_single_object import minimal_referring_question
from patch_pca_cluster import (combo_key, offset_statistics_from_offsets,
                               CLUSTER_RGB, BACKBONE_LABELS)

ATTRS = ("color", "shape", "material", "size")
COLORS = ["red", "blue", "green", "brown", "purple", "cyan", "yellow"]  # gray excluded (segmentation)
QUERIED = "color"          # queried attribute of every question; set from --queried in main()
CONDITIONS_N2 = ["c0", "c1", "c2", "c3"]
CONDITIONS_N1 = ["c0", "c1"]
COND_LABEL = {"c0": "no question", "c1": "refer target", "c2": "refer distractor",
              "c3": "non-referring", "c4": "absent referent"}
COND_LS = {"c1": "-", "c2": "--", "c3": ":", "c4": "-."}
OWNER_RGB = {"target": CLUSTER_RGB["target"], "distractor": CLUSTER_RGB["distractor"],
             "bg": (0.45, 0.45, 0.45)}
NUM_LAYERS = 12


# ---------------------------------------------------------------------------
# Stimuli, questions, masks
# ---------------------------------------------------------------------------

def seg_ok(e):
    return e["color"] != "gray" and all(
        d["color"] not in (e["color"], "gray") for d in e["distractors"])


def swap_roles(e):
    """The distractor becomes the main object, the target its distractor."""
    d = e["distractors"][0]
    return dict({k: d[k] for k in ATTRS},
                distractors=[{k: e[k] for k in ATTRS}])


def referent_word(question):
    """'What color is the sphere?' -> 'sphere'; '... the large object?' -> 'large';
    '... the object?' -> 'object'."""
    words = question.rstrip("?").split()
    return words[-2] if words[-1] == "object" and len(words) > 4 else words[-1]


def build_questions(e, with_absent):
    q = {"c1": minimal_referring_question(e, QUERIED),
         "c2": minimal_referring_question(swap_roles(e), QUERIED),
         "c3": f"What {QUERIED} is the object?"}
    if with_absent:
        d = e["distractors"][0]
        shared = [a for a in ("shape", "size", "material") if e[a] == d[a]]
        vals = {"shape": ["cube", "sphere", "cylinder"], "size": ["large", "small"],
                "material": ["metal", "rubber"]}
        for a in shared:
            unused = [v for v in vals[a] if v != e[a]]
            if unused:
                v = unused[0]
                q["c4"] = (f"What color is the {v}?" if a == "shape"
                           else f"What color is the {v} object?")
                break
    return q


def spatial_cell(owner, oid, grid, n_cells=3):
    pos = np.nonzero(owner == oid)[0]
    if len(pos) == 0:
        return -1
    r, c = pos // grid, pos % grid
    return int(r.mean() * n_cells // grid) * n_cells + int(c.mean() * n_cells // grid)


def prepare_subsets(n1_entries, n2_entries, args, out_dir, x19_pca_pairs):
    """Segment every eligible pair for BOTH subsets, drop pairs whose
    segmentation fails on either, write labels.json + masks_debug.png."""
    rng = np.random.RandomState(args.seed)
    cand = [i for i, e in enumerate(n2_entries) if seg_ok(e)]
    if args.n_pairs and args.n_pairs < len(cand):
        cand = sorted(int(i) for i in rng.choice(cand, args.n_pairs, replace=False))
    print(f"Eligible pairs under the X19 segmentation filter: {len(cand)}")
    keep, owners = [], {"n1": [], "n2": []}
    images = {"n1": [], "n2": []}
    dirs = {"n1": args.n1_dir, "n2": args.n2_dir}
    ents = {"n1": n1_entries, "n2": n2_entries}
    for i in cand:
        try:
            got = {}
            for name in ("n1", "n2"):
                im, ow, _ = build_masks(ents[name], [i], dirs[name], args)
                got[name] = (im[0], ow[0])
        except AssertionError as ex:
            print(f"  skip pair {i}: {ex}")
            continue
        keep.append(i)
        for name in ("n1", "n2"):
            images[name].append(got[name][0])
            owners[name].append(got[name][1])
    print(f"Pairs kept after segmentation: {len(keep)}")
    labels = {"n1": [], "n2": []}
    for k, i in enumerate(keep):
        e = n2_entries[i]
        ow1, ow2 = owners["n1"][k], owners["n2"][k]
        bg_ok = np.nonzero((ow1 == 0) & (ow2 == 0))[0]
        bg_sample = sorted(int(p) for p in np.random.RandomState(args.seed + i)
                           .choice(bg_ok, min(args.bg_per_image, len(bg_ok)), replace=False))
        qs = build_questions(e, args.with_absent)
        base = {"pair_index": i, "filename": e["filename"],
                "target": {a: e[a] for a in ATTRS},
                "position": {"x": e["x"], "y": e["y"]},
                "distractors": [{a: d[a] for a in ATTRS} | {"x": d["x"], "y": d["y"]}
                                for d in e["distractors"]],
                "slot": i % 5, "in_x19_pca_set": i in x19_pca_pairs,
                "questions": qs,
                "referent_words": {c: referent_word(q) for c, q in qs.items()},
                "bg_sample": bg_sample}
        labels["n2"].append(dict(base, spatial_cell=spatial_cell(ow2, 1, args.grid),
                                 spatial_cell_distractor=spatial_cell(ow2, 2, args.grid),
                                 n_target_patches=int((ow2 == 1).sum()),
                                 n_distractor_patches=int((ow2 == 2).sum())))
        labels["n1"].append(dict(base, distractors=[], questions={"c1": qs["c1"]},
                                 referent_words={"c1": referent_word(qs["c1"])},
                                 spatial_cell=spatial_cell(ow1, 1, args.grid),
                                 n_target_patches=int((ow1 == 1).sum())))
    for name in ("n1", "n2"):
        sub = out_dir / name
        sub.mkdir(parents=True, exist_ok=True)
        with open(sub / "labels.json", "w") as f:
            json.dump(labels[name], f, indent=1)
        np.save(sub / "owner.npy", np.stack(owners[name]))
        n_dbg = min(40, len(keep))
        save_masks_debug(images[name][:n_dbg], np.stack(owners[name][:n_dbg]),
                         ents[name], keep[:n_dbg], args.grid, sub / "masks_debug.png")
    return keep, images, owners, labels


def load_or_prepare_subsets(out_dir, n1_entries, n2_entries, args, x19_pairs):
    """Reuse the pair selection recorded in <out_dir>/n2/labels.json, else select anew."""
    dirs = {"n1": args.n1_dir, "n2": args.n2_dir}
    if (out_dir / "n2" / "labels.json").exists():
        labels = {n: load_labels(out_dir / n) for n in ("n1", "n2")}
        keep = [r["pair_index"] for r in labels["n2"]]
        owners = {n: list(np.load(out_dir / n / "owner.npy")) for n in ("n1", "n2")}
        images = {n: [build_masks({"n1": n1_entries, "n2": n2_entries}[n], [i], dirs[n], args)[0][0]
                      for i in keep] for n in ("n1", "n2")}
        print(f"Reusing existing selection: {len(keep)} pairs")
        return keep, images, owners, labels
    return prepare_subsets(n1_entries, n2_entries, args, out_dir, x19_pairs)


# ---------------------------------------------------------------------------
# Sparse extraction (GPU)
# ---------------------------------------------------------------------------

class SparseExtractor:
    """Per batch: all 12 block outputs (raw + trunk.norm), GCA writes and
    per-patch attention onto the referent token; keeps object patches + the
    fixed background sample only."""

    def __init__(self, steervit):
        self.steervit = steervit
        self.trunk = steervit.vision_model.trunk
        self.blocks = self.trunk.blocks
        self.norm = self.trunk.norm
        self.prefix = self.trunk.num_prefix_tokens
        self.gca_layers = [i for i, b in enumerate(self.blocks)
                           if getattr(b, "gated_cross_attn", None) is not None]
        self.tokenizer = steervit.tokenizer

    def referent_token_index(self, questions, words):
        enc = self.tokenizer(list(questions), padding=True, return_tensors="pt")
        ids = enc["input_ids"]
        toks = [self.tokenizer.convert_ids_to_tokens(row) for row in ids]
        idx, last = [], []
        for row, w in zip(toks, words):
            cand = [j for j, t in enumerate(row) if t == "Ġ" + w]
            assert cand, f"referent token '{w}' not found in {row}"
            idx.append(cand[0])
            last.append(max(j for j, t in enumerate(row) if t == "</s>"))
        return idx, last

    @torch.no_grad()
    def run(self, images, questions, words):
        layer_out, writes = {}, {}
        hooks = []
        for li, blk in enumerate(self.blocks):
            def mk(li):
                def fn(mod, inp, out):
                    layer_out[li] = (out[0] if isinstance(out, tuple) else out).detach()
                return fn
            hooks.append(blk.register_forward_hook(mk(li)))
        if questions is not None:
            for li in self.gca_layers:
                def mkw(li):
                    def fn(mod, inp, out):
                        writes[li] = (out - inp[0]).detach()
                    return fn
                hooks.append(self.blocks[li].gated_cross_attn.register_forward_hook(mkw(li)))
                self.blocks[li].gated_cross_attn.cross_attn.save_attn = True
        try:
            if images.is_cuda:
                with autocast(device_type="cuda", dtype=torch.bfloat16):
                    self.steervit.forward(images, None if questions is None else list(questions))
            else:
                self.steervit.forward(images, None if questions is None else list(questions))
        finally:
            for h in hooks:
                h.remove()
            for li in self.gca_layers:
                self.blocks[li].gated_cross_attn.cross_attn.save_attn = False
        p = self.prefix
        raw = torch.stack([layer_out[l].float()[:, p:, :] for l in range(NUM_LAYERS)], 1)  # (B,12,P,D)
        normed = torch.stack([self.norm(layer_out[l].float())[:, p:, :]
                              for l in range(NUM_LAYERS)], 1)
        out = {"raw": raw, "normed": normed}
        if questions is not None:
            out["write"] = torch.stack([writes[l].float()[:, p:, :] for l in self.gca_layers], 1)  # (B,6,P,D)
            ridx, lidx = self.referent_token_index(questions, words)
            attn_ref, attn_sp = [], []
            for l in self.gca_layers:
                am = self.blocks[l].gated_cross_attn.cross_attn.attn_map.float().mean(1)  # (B,Tq,Tk)
                am = am[:, p:, :]
                b = torch.arange(am.shape[0], device=am.device)
                attn_ref.append(am[b, :, torch.tensor(ridx, device=am.device)])
                attn_sp.append(am[:, :, 0] + am[b, :, torch.tensor(lidx, device=am.device)])
                self.blocks[l].gated_cross_attn.cross_attn.attn_map = None
            out["attn_ref"] = torch.stack(attn_ref, 1)   # (B,6,P)
            out["attn_special"] = torch.stack(attn_sp, 1)
        return out


def ensure_model(state, args):
    if state.get("extractor") is None:
        from model.checkpoint_io import load_any_checkpoint
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading checkpoint on {device}: {args.checkpoint}")
        model, steervit, transform, vocab, task_type, meta = load_any_checkpoint(args.checkpoint, device)
        model.eval()
        state.update(extractor=SparseExtractor(steervit), transform=transform,
                     device=device, model=model, steervit=steervit, vocab=vocab)
    return state


def extract_condition_sparse(sub_dir, cond, images, owners, labels, args, state, n_obj=2):
    npz = sub_dir / f"feats_{cond}.npz"
    if npz.exists():
        print(f"Cached features exist, not re-extracting: {npz}")
        return
    ensure_model(state, args)
    ext, tf, device = state["extractor"], state["transform"], state["device"]
    questions = None if cond == "c0" else [rec["questions"][cond] for rec in labels]
    words = None if cond == "c0" else [rec["referent_words"][cond] for rec in labels]
    print(f"Extracting {cond}: {len(images)} images ...")
    acc = {k: [] for k in ("tok", "tok_img", "tok_pos", "tok_owner", "obj_mean", "bg_mean",
                           "raw_obj_mean", "raw_bg_mean", "raw_norm", "gca_write",
                           "gca_write_img", "gca_write_pos", "gca_write_owner",
                           "gca_write_norm", "gca_attn_ref", "gca_attn_special")}
    bs = args.batch_size
    for s in range(0, len(images), bs):
        e = min(s + bs, len(images))
        batch = torch.stack([tf(im) for im in images[s:e]]).to(device)
        out = ext.run(batch, None if questions is None else questions[s:e],
                      None if words is None else words[s:e])
        raw, normed = out["raw"].cpu().numpy(), out["normed"].cpu().numpy()
        for bi in range(e - s):
            i = s + bi
            ow = owners[i]
            bg = np.array(labels[i]["bg_sample"], dtype=int)
            keep = np.concatenate([np.nonzero(ow > 0)[0], bg])
            acc["tok"].append(normed[bi][:, keep, :].transpose(1, 0, 2).astype(np.float16))
            acc["tok_img"].append(np.full(len(keep), i, dtype=np.int16))
            acc["tok_pos"].append(keep.astype(np.int16))
            acc["tok_owner"].append(ow[keep].astype(np.int8))
            om = np.zeros((n_obj, NUM_LAYERS, normed.shape[-1]), np.float32)
            rom = np.zeros_like(om)
            for oid in range(1, n_obj + 1):
                sel = ow == oid
                if sel.any():
                    om[oid - 1] = normed[bi][:, sel, :].mean(1)
                    rom[oid - 1] = raw[bi][:, sel, :].mean(1)
            acc["obj_mean"].append(om.astype(np.float16))
            acc["raw_obj_mean"].append(rom.astype(np.float16))
            acc["bg_mean"].append(normed[bi][:, ow == 0, :].mean(1).astype(np.float16))
            acc["raw_bg_mean"].append(raw[bi][:, ow == 0, :].mean(1).astype(np.float16))
            acc["raw_norm"].append(np.linalg.norm(raw[bi], axis=-1).astype(np.float16))
            if "write" in out:
                w = out["write"][bi].cpu().numpy()          # (6,P,D)
                wk = np.concatenate([np.nonzero(ow > 0)[0], bg[:16]])
                acc["gca_write"].append(w[:, wk, :].transpose(1, 0, 2).astype(np.float16))
                acc["gca_write_img"].append(np.full(len(wk), i, dtype=np.int16))
                acc["gca_write_pos"].append(wk.astype(np.int16))
                acc["gca_write_owner"].append(ow[wk].astype(np.int8))
                acc["gca_write_norm"].append(np.linalg.norm(w, axis=-1).astype(np.float16))
                acc["gca_attn_ref"].append(out["attn_ref"][bi].cpu().numpy().astype(np.float16))
                acc["gca_attn_special"].append(out["attn_special"][bi].cpu().numpy().astype(np.float16))
        print(f"  {e}/{len(images)}", flush=True)
    arrays = {k: np.concatenate(v) if k.startswith(("tok", "gca_write")) and not k.endswith("norm")
              else np.stack(v) for k, v in acc.items() if v}
    arrays["owner"] = np.stack(owners)
    arrays["gca_layers"] = np.array(ext.gca_layers)
    np.savez(npz, **arrays)
    print(f"Saved: {npz} ({sum(a.nbytes for a in arrays.values()) / 1e9:.2f} GB)")


def load_sparse(sub_dir, cond):
    d = np.load(sub_dir / f"feats_{cond}.npz")
    return {k: d[k] for k in d.files}


def load_labels(sub_dir):
    with open(sub_dir / "labels.json") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Part A — projections, per-patch change, GCA write, offsets, RSA, norms
# ---------------------------------------------------------------------------

def _unit(v):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-8)


def _cos(a, b):
    return (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-8)


def _boot(x, n=1000, seed=0):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    rng = np.random.RandomState(seed)
    m = np.array([x[rng.randint(0, len(x), len(x))].mean() for _ in range(n)])
    return {"mean": float(x.mean()), "lo": float(np.percentile(m, 2.5)),
            "hi": float(np.percentile(m, 97.5)), "n": int(len(x))}


def token_table(c, layer, norm_std=False):
    """(T, D) normed tokens at `layer` (+ per-token L2 standardisation)."""
    t = c["tok"][:, layer, :].astype(np.float32)
    return _unit(t) if norm_std else t


def offsets_from_cache(c, norm_std=False):
    """Per image: target / distractor offset = obj_mean − bg_mean, (N, 2, 12, D)."""
    om, bm = c["obj_mean"].astype(np.float32), c["bg_mean"].astype(np.float32)
    if norm_std:
        om, bm = _unit(om), _unit(bm)
    return om - bm[:, None]


def part_a(caches, labels, gca_layers, norm_std=False):
    conds = [k for k in caches if k != "c0"]
    c0 = caches["c0"]
    N = len(labels)
    has_d = np.array([rec["n_distractor_patches"] > 0 for rec in labels])
    off = {k: offsets_from_cache(caches[k], norm_std) for k in caches}
    # reference directions from c0
    V = _unit(off["c0"][:, 0].mean(0))                       # (12, D) target-based
    V_dist = _unit(off["c0"][has_d, 1].mean(0))
    v_img = _unit(off["c0"])                                 # (N, 2, 12, D)
    metrics = {"cos_V_target_vs_distractor": [float(_cos(V[l], V_dist[l])) for l in range(NUM_LAYERS)],
               "proj": {}, "delta": {}, "patch_change": {}, "gca": {}, "offset_norm": {}, "rsa": {}}

    def proj(cond, oid, l, direction):
        om = caches[cond]["obj_mean"][:, oid, l, :].astype(np.float32)
        if norm_std:
            om = _unit(om)
        return (om * direction).sum(-1)                      # (N,)

    for cond in caches:
        for oid, name in ((0, "target"), (1, "distractor")):
            valid = np.ones(N, bool) if oid == 0 else has_d
            metrics["proj"][f"{cond}_{name}"] = [
                _boot(proj(cond, oid, l, V[l])[valid]) for l in range(NUM_LAYERS)]
    if "c1" in caches and "c2" in caches:
        d_ref = [proj("c1", 0, l, V[l]) - proj("c2", 0, l, V[l]) for l in range(NUM_LAYERS)]
        d_non = [(proj("c1", 1, l, V[l]) - proj("c2", 1, l, V[l]))[has_d] for l in range(NUM_LAYERS)]
        metrics["delta"]["ref"] = [_boot(x) for x in d_ref]
        metrics["delta"]["nonref"] = [_boot(x) for x in d_non]
        # same deltas projected onto each image's OWN no-question object direction
        def diff(oid, l):
            a = caches["c1"]["obj_mean"][:, oid, l].astype(np.float32)
            b = caches["c2"]["obj_mean"][:, oid, l].astype(np.float32)
            return a - b
        metrics["delta"]["ref_imgdir"] = [
            _boot((diff(0, l) * v_img[:, 0, l]).sum(-1)) for l in range(NUM_LAYERS)]
        metrics["delta"]["nonref_imgdir"] = [
            _boot((diff(1, l) * v_img[:, 1, l]).sum(-1)[has_d]) for l in range(NUM_LAYERS)]
    for cond in conds:
        metrics["delta"][f"base_target_{cond}"] = [
            _boot(proj(cond, 0, l, V[l]) - proj("c0", 0, l, V[l])) for l in range(NUM_LAYERS)]
        metrics["delta"][f"base_distractor_{cond}"] = [
            _boot((proj(cond, 1, l, V[l]) - proj("c0", 1, l, V[l]))[has_d]) for l in range(NUM_LAYERS)]
    metrics["offset_norm_ref"] = [float(np.linalg.norm(off["c0"][:, 0, l], axis=-1).mean())
                                  for l in range(NUM_LAYERS)]

    # per-patch change d = h(c) − h(c0), split by owner
    own = c0["tok_owner"]
    same = all(np.array_equal(caches[k]["tok_pos"], c0["tok_pos"]) and
               np.array_equal(caches[k]["tok_img"], c0["tok_img"]) for k in conds)
    assert same, "token tables differ across conditions"
    for cond in conds:
        pc = {}
        for l in range(NUM_LAYERS):
            h0 = token_table(c0, l, norm_std)
            d = token_table(caches[cond], l, norm_std) - h0
            for oid, name in ((1, "target"), (2, "distractor"), (0, "bg")):
                sel = own == oid
                if not sel.any():
                    continue
                pc.setdefault(name, {"norm": [], "rel_norm": [], "cos_V": [], "cos_vimg": []})
                pc[name]["norm"].append(float(np.linalg.norm(d[sel], axis=-1).mean()))
                pc[name]["rel_norm"].append(float((np.linalg.norm(d[sel], axis=-1)
                                                   / (np.linalg.norm(h0[sel], axis=-1) + 1e-8)).mean()))
                pc[name]["cos_V"].append(float(_cos(d[sel], V[l][None]).mean()))
                img = c0["tok_img"][sel]
                vi = v_img[img, 0 if oid != 2 else 1, l]
                pc[name]["cos_vimg"].append(float(_cos(d[sel], vi).mean()) if oid else float("nan"))
        metrics["patch_change"][cond] = pc
        # GCA write, attention onto the referent token
        cc = caches[cond]
        g = {}
        wn = cc["gca_write_norm"].astype(np.float32)          # (N,6,P)
        ar = cc["gca_attn_ref"].astype(np.float32)
        owner_full = cc["owner"]
        for oid, name in ((1, "target"), (2, "distractor"), (0, "bg")):
            m = owner_full == oid                              # (N,P)
            g[name] = {"write_norm": [float(wn[:, k][m].mean()) for k in range(len(gca_layers))],
                       "attn_ref": [float(ar[:, k][m].mean()) for k in range(len(gca_layers))]}
        wo = cc["gca_write_owner"]
        for oid, name in ((1, "target"), (2, "distractor"), (0, "bg")):
            sel = wo == oid
            g[name]["write_cos_V"] = [
                float(_cos(cc["gca_write"][sel, k, :].astype(np.float32), V[gl][None]).mean())
                for k, gl in enumerate(gca_layers)]
        metrics["gca"][cond] = g

    # offset norms / target-vs-distractor cosine per condition
    for cond in caches:
        o = off[cond]
        metrics["offset_norm"][cond] = {
            "target": [float(np.linalg.norm(o[:, 0, l], axis=-1).mean()) for l in range(NUM_LAYERS)],
            "distractor": [float(np.linalg.norm(o[has_d, 1, l], axis=-1).mean()) for l in range(NUM_LAYERS)],
            "target_vs_distractor_cos": [float(_cos(o[has_d, 0, l], o[has_d, 1, l]).mean())
                                         for l in range(NUM_LAYERS)]}

    # RSA: target object-mean features vs identity RDM vs position RDM (Assouel et al.)
    ident = np.array(["-".join(combo_key(rec["target"])) for rec in labels])
    pos = np.array([[rec["position"]["x"], rec["position"]["y"]] for rec in labels])
    iu = np.triu_indices(N, 1)
    rdm_id = (ident[:, None] != ident[None, :]).astype(float)[iu]
    rdm_pos = np.linalg.norm(pos[:, None] - pos[None], axis=-1)[iu]
    for cond in caches:
        rs = {"identity": [], "position": [], "identity_offset": [], "position_offset": []}
        for l in range(NUM_LAYERS):
            f = caches[cond]["obj_mean"][:, 0, l].astype(np.float32)
            if norm_std:
                f = _unit(f)
            rdm = 1 - _cos(f[:, None], f[None])[iu]
            rs["identity"].append(float(spearmanr(rdm, rdm_id).correlation))
            rs["position"].append(float(spearmanr(rdm, rdm_pos).correlation))
            fo = off[cond][:, 0, l]                              # offset = obj − bg
            rdm_o = 1 - _cos(fo[:, None], fo[None])[iu]
            rs["identity_offset"].append(float(spearmanr(rdm_o, rdm_id).correlation))
            rs["position_offset"].append(float(spearmanr(rdm_o, rdm_pos).correlation))
        metrics["rsa"][cond] = rs
    return metrics


def position_templates(cache, grid):
    """Per-position background template (P, 12, D) from the sparse cache: the
    mean over all sampled background tokens at each patch position across
    images (X19's `background_templates`, computed from the sparse table).
    Returns (template, count per position)."""
    P = grid * grid
    bg = cache["tok_owner"] == 0
    pos = cache["tok_pos"][bg].astype(int)
    tok = cache["tok"][bg].astype(np.float32)                 # (T, 12, D)
    tpl = np.zeros((P, tok.shape[1], tok.shape[2]), np.float32)
    cnt = np.bincount(pos, minlength=P).astype(np.float32)
    np.add.at(tpl, pos, tok)
    tpl /= np.maximum(cnt, 1.0)[:, None, None]
    return tpl, cnt


def offsets_template(cache, tpl, n_images):
    """Per image: object patch mean − mean of the per-position template at the
    object's own positions, (N, 2, 12, D). Zero rows where the object has no
    patches."""
    img, pos, own = cache["tok_img"].astype(int), cache["tok_pos"].astype(int), cache["tok_owner"]
    om = cache["obj_mean"].astype(np.float32)
    out = np.zeros_like(om)
    for i in range(n_images):
        sel_i = img == i
        for oid in (1, 2):
            sel = sel_i & (own == oid)
            if sel.any():
                out[i, oid - 1] = om[i, oid - 1] - tpl[pos[sel]].mean(0)
    return out


def rsa_template(caches, labels, grid):
    """RSA of the target's offset against identity / colour / position RDMs,
    with the per-position background template subtracted (the X19 template),
    alongside the image-mean-subtracted offset used in part_a."""
    N = len(labels)
    ident = np.array(["-".join(combo_key(rec["target"])) for rec in labels])
    colour = np.array([rec["target"]["color"] for rec in labels])
    pos = np.array([[rec["position"]["x"], rec["position"]["y"]] for rec in labels])
    iu = np.triu_indices(N, 1)
    rdms = {"identity": (ident[:, None] != ident[None, :]).astype(float)[iu],
            "position": np.linalg.norm(pos[:, None] - pos[None], axis=-1)[iu]}
    for attr in ATTRS:
        vals = np.array([rec["target"][attr] for rec in labels])
        key = "colour" if attr == "color" else attr
        rdms[key] = (vals[:, None] != vals[None, :]).astype(float)[iu]
    res = {"grid": grid, "n_images": N, "conditions": {}}
    for cond, c in caches.items():
        tpl, cnt = position_templates(c, grid)
        o_tpl = offsets_template(c, tpl, N)[:, 0]              # target, (N, 12, D)
        o_img = offsets_from_cache(c)[:, 0]
        r = {"template_count_min": float(cnt.min()), "template_count_mean": float(cnt.mean()),
             "template_positions_empty": int((cnt == 0).sum())}
        for tag, o in (("template", o_tpl), ("imgmean", o_img)):
            for name, rdm_m in rdms.items():
                vals = []
                for l in range(NUM_LAYERS):
                    f = o[:, l]
                    rdm = 1 - _cos(f[:, None], f[None])[iu]
                    vals.append(float(spearmanr(rdm, rdm_m).correlation))
                r[f"{name}_{tag}"] = vals
        res["conditions"][cond] = r
        print(f"RSA template {cond}: template count min/mean {cnt.min():.0f}/{cnt.mean():.1f}; "
              f"L11 identity {r['identity_template'][-1]:+.3f} colour {r['colour_template'][-1]:+.3f} "
              f"position {r['position_template'][-1]:+.3f} "
              f"(image-mean offset: position {r['position_imgmean'][-1]:+.3f})", flush=True)
    return res


def plot_rsa_template(res, label, out_path, gca_layers):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    colours = {"identity": "#2ca02c", "colour": "#d62728", "shape": "#1f77b4",
               "material": "#8c564b", "size": "#e377c2", "position": "#9467bd"}
    markers = {"identity": "o", "colour": "^", "shape": "v", "material": "D", "size": "P", "position": "s"}
    for ax, tag, title in ((axes[0], "imgmean", "offset = patch mean − image background mean"),
                           (axes[1], "template", "offset = patch mean − per-position background template")):
        for cond, r in res["conditions"].items():
            ls = COND_LS.get(cond, "-")
            for name in ("identity", "colour", "shape", "material", "size", "position"):
                if f"{name}_{tag}" not in r:
                    continue
                ax.plot(range(NUM_LAYERS), r[f"{name}_{tag}"], ls, color=colours[name],
                        marker=markers[name], markersize=3,
                        label=f"{name} RDM, {COND_LABEL[cond]}")
        ax.set_ylabel("Spearman(offset RDM, model RDM)", fontsize=10)
        ax.set_title(title, fontsize=10)
        _layers_axis(ax, gca_layers)
        ax.axhline(0, color="k", linewidth=0.6)
    h, l = axes[1].get_legend_handles_labels()
    fig.legend(h, l, fontsize=7, ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.12))
    fig.suptitle(f"{label} — RSA of the target's offset: object identity / colour / shape / position (questions ask about {QUERIED})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=S["dpi"], bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Attribute-specific directions (Song, Lepori & Pavlick 2025 concept vectors):
# does a question amplify the asked attribute of the referent, or suppress the
# non-referent's whole object vector?  Directions from the 1-object images
# (independent set), projections on the 2-object images per condition.
# ---------------------------------------------------------------------------

ATTR_VALUES = {"color": COLORS, "shape": ["cube", "sphere", "cylinder"],
               "size": ["small", "large"], "material": ["rubber", "metal"]}


def attribute_directions(cache_n1, labels_n1):
    """V[attr][value] (12, D) = unit(mean obj_mean of 1-object targets with that
    value − mean over all 1-object targets), in trunk.norm feature space."""
    om = cache_n1["obj_mean"][:, 0].astype(np.float32)               # (N, 12, D)
    mu = om.mean(0)
    V = {}
    for attr, values in ATTR_VALUES.items():
        vals = np.array([rec["target"][attr] for rec in labels_n1])
        V[attr] = {v: _unit(om[vals == v].mean(0) - mu) for v in values if (vals == v).sum() >= 5}
    return V


def attr_direction_analysis(caches_n2, labels_n2, V):
    N = len(labels_n2)
    has_d = np.array([rec["n_distractor_patches"] > 0 for rec in labels_n2])
    res = {"n_images": N, "n_with_distractor": int(has_d.sum()), "proj": {}, "delta": {}}

    def proj(cond, oid, attr, value_of):
        """Per image projection onto V[attr][own value] and mean over the other values."""
        om = caches_n2[cond]["obj_mean"][:, oid].astype(np.float32)   # (N, 12, D)
        own = np.full((N, NUM_LAYERS), np.nan, np.float32)
        other = np.full((N, NUM_LAYERS), np.nan, np.float32)
        for i in range(N):
            v = value_of(i)
            if v not in V[attr]:
                continue
            own[i] = (om[i] * V[attr][v]).sum(-1)
            others = [V[attr][u] for u in V[attr] if u != v]
            other[i] = np.mean([(om[i] * u).sum(-1) for u in others], 0)
        return own, other

    P = {}
    for cond in caches_n2:
        for oid, name, key in ((0, "target", "target"), (1, "distractor", "distractors")):
            for attr in dict.fromkeys(["color", "shape", QUERIED]):
                value_of = (lambda i, a=attr: labels_n2[i]["target"][a]) if oid == 0 else \
                           (lambda i, a=attr: labels_n2[i]["distractors"][0][a])
                own, other = proj(cond, oid, attr, value_of)
                valid = np.ones(N, bool) if oid == 0 else has_d
                P[(cond, name, attr)] = (own, other, valid)
                res["proj"][f"{cond}_{name}_{attr}"] = {
                    "own": [_boot(own[valid, l]) for l in range(NUM_LAYERS)],
                    "other": [_boot(other[valid, l]) for l in range(NUM_LAYERS)]}
    # referent − non-referent contrasts, per attribute, own vs other value
    for attr in dict.fromkeys(["color", "shape", QUERIED]):
        for what, k in (("own", 0), ("other", 1)):
            t1, t2 = P[("c1", "target", attr)], P[("c2", "target", attr)]
            d1, d2 = P[("c1", "distractor", attr)], P[("c2", "distractor", attr)]
            v = t1[2] & d1[2]
            res["delta"][f"ref_target_{attr}_{what}"] = [_boot((t1[k] - t2[k])[v, l]) for l in range(NUM_LAYERS)]
            res["delta"][f"nonref_distractor_{attr}_{what}"] = [_boot((d1[k] - d2[k])[v, l]) for l in range(NUM_LAYERS)]
            if "c0" in caches_n2:
                t0, d0 = P[("c0", "target", attr)], P[("c0", "distractor", attr)]
                res["delta"][f"refvs0_target_{attr}_{what}"] = [_boot((t1[k] - t0[k])[v, l]) for l in range(NUM_LAYERS)]
                res["delta"][f"nonrefvs0_target_{attr}_{what}"] = [_boot((t2[k] - t0[k])[v, l]) for l in range(NUM_LAYERS)]
                if "c3" in caches_n2:
                    t3 = P[("c3", "target", attr)]
                    res["delta"][f"c3vs0_target_{attr}_{what}"] = [_boot((t3[k] - t0[k])[v, l]) for l in range(NUM_LAYERS)]
    return res


def plot_attr_directions(res, label, out_path, gca_layers):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    x = range(NUM_LAYERS)

    def line(ax, key, color, ls, lab, marker="o"):
        d = res["delta"][key]
        m = np.array([q["mean"] for q in d]); lo = np.array([q["lo"] for q in d]); hi = np.array([q["hi"] for q in d])
        ax.plot(x, m, ls, color=color, marker=marker, markersize=3, label=lab)
        ax.fill_between(x, lo, hi, color=color, alpha=0.12, linewidth=0)

    contrast = "shape" if QUERIED != "shape" else "color"
    qL = "colour" if QUERIED == "color" else QUERIED
    cL = "colour" if contrast == "color" else contrast
    ax = axes[0]
    line(ax, f"ref_target_{QUERIED}_own", "#d62728", "-", f"target: own {qL} direction")
    line(ax, f"ref_target_{QUERIED}_other", "#d62728", ":", f"target: other {qL} directions (mean)")
    line(ax, f"ref_target_{contrast}_own", "#1f77b4", "-", f"target: own {cL} direction", "^")
    line(ax, f"ref_target_{contrast}_other", "#1f77b4", ":", f"target: other {cL} directions (mean)", "^")
    ax.set_title("target: refer target − refer distractor", fontsize=10)
    ax = axes[1]
    line(ax, f"nonref_distractor_{QUERIED}_own", "#d62728", "-", f"distractor: own {qL} direction")
    line(ax, f"nonref_distractor_{QUERIED}_other", "#d62728", ":", f"distractor: other {qL} directions (mean)")
    line(ax, f"nonref_distractor_{contrast}_own", "#1f77b4", "-", f"distractor: own {cL} direction", "^")
    line(ax, f"nonref_distractor_{contrast}_other", "#1f77b4", ":", f"distractor: other {cL} directions (mean)", "^")
    ax.set_title("distractor: refer target − refer distractor", fontsize=10)
    ax = axes[2]
    line(ax, f"refvs0_target_{QUERIED}_own", "#d62728", "-", f"refer target − no question, own {qL}")
    line(ax, f"nonrefvs0_target_{QUERIED}_own", "#d62728", "--", f"refer distractor − no question, own {qL}", "s")
    line(ax, f"refvs0_target_{contrast}_own", "#1f77b4", "-", f"refer target − no question, own {cL}", "^")
    line(ax, f"nonrefvs0_target_{contrast}_own", "#1f77b4", "--", f"refer distractor − no question, own {cL}", "v")
    ax.set_title("target: question − no question", fontsize=10)
    for ax in axes:
        ax.axhline(0, color="k", linewidth=0.6)
        ax.set_ylabel("Δ projection", fontsize=10)
        _layers_axis(ax, gca_layers)
        ax.legend(fontsize=6)
    fig.suptitle(f"{label} — attribute-specific directions (from 1-object images); questions ask about {QUERIED}: "
                 f"queried and contrast directions, own value vs other values")
    fig.tight_layout()
    fig.savefig(out_path, dpi=S["dpi"], bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def token_norm_stats(c0):
    rn = c0["raw_norm"].astype(np.float32)                    # (N,12,P)
    own = c0["owner"]
    out = {"median": [], "outlier_frac": [], "outlier_bg_share": [], "bg_share": float((own == 0).mean())}
    for l in range(NUM_LAYERS):
        x = rn[:, l]
        med = float(np.median(x))
        outl = x > 5 * med
        out["median"].append(med)
        out["outlier_frac"].append(float(outl.mean()))
        out["outlier_bg_share"].append(float((own[outl] == 0).mean()) if outl.any() else float("nan"))
    out["example_map"] = rn[0].tolist()
    return out


def offset_stats_by_condition(cache_n1, caches_n2, labels_n2, norm_std=False):
    """X19 statistics on the sparse cache, one block per n2 condition."""
    o1 = offsets_from_cache(cache_n1, norm_std)[:, 0]        # (N,12,D)
    combos = ["-".join(combo_key(rec["target"])) for rec in labels_n2]
    has_d = np.array([rec["n_distractor_patches"] > 0 for rec in labels_n2])
    out = {}
    for cond, c in caches_n2.items():
        o2 = offsets_from_cache(c, norm_std)
        out[cond] = {}
        for l in GCA_LAYERS:
            o_n2d = [o2[b, 1, l] if has_d[b] else None for b in range(len(labels_n2))]
            out[cond][f"L{l}"] = offset_statistics_from_offsets(o1[:, l], o2[:, 0, l], o_n2d, combos)
    return out


# ---------------------------------------------------------------------------
# Part B — residual interventions
# ---------------------------------------------------------------------------

class ResidualAdder:
    """Forward hook on trunk.blocks[layer]: x[:, prefix:, :] += alpha * delta * mask."""

    def __init__(self, trunk, layer, delta, mask, alpha):
        self.blk = trunk.blocks[layer]
        self.prefix = trunk.num_prefix_tokens
        self.delta, self.mask, self.alpha = delta, mask, alpha
        self.h = None

    def __enter__(self):
        def fn(mod, inp, out):
            x = out[0].clone()
            x[:, self.prefix:, :] = x[:, self.prefix:, :] + \
                self.alpha * self.delta[:, None, :].to(x.dtype) * self.mask[:, :, None].to(x.dtype)
            return (x, out[1], out[2])
        self.h = self.blk.register_forward_hook(fn)
        return self

    def __exit__(self, *a):
        self.h.remove()


@torch.no_grad()
def first_token_logits(model, steervit, images, questions):
    prefix = steervit.vision_model.trunk.num_prefix_tokens
    feats = steervit.forward(images, list(questions))
    patches = feats[:, prefix:, :]
    bos = torch.full((images.shape[0], 1), model.vocab["<bos>"], dtype=torch.long, device=images.device)
    return model.decoder(bos, patches)[:, 0, :]


# ---------------------------------------------------------------------------
# Readout check — where does the decoder read the answer from at the last block?
# (i) decoder cross-attention mass by patch owner; (ii) activation patching of
# background / object tokens between conditions at every block output.
# ---------------------------------------------------------------------------

class BlockCapture:
    """Forward hooks on all trunk blocks; stores each block's patch output."""

    def __init__(self, trunk):
        self.trunk, self.prefix, self.out, self.hs = trunk, trunk.num_prefix_tokens, {}, []

    def __enter__(self):
        for li, blk in enumerate(self.trunk.blocks):
            def mk(li):
                def fn(mod, inp, out):
                    self.out[li] = out[0][:, self.prefix:, :].detach().clone()
                return fn
            self.hs.append(blk.register_forward_hook(mk(li)))
        return self

    def __exit__(self, *a):
        for h in self.hs:
            h.remove()


class TokenSwapper:
    """Forward hook on trunk.blocks[layer]: masked patch tokens of the current
    (receiver) run are replaced by the donor run's output at the same block."""

    def __init__(self, trunk, layer, donor, mask):
        self.blk, self.prefix, self.donor, self.mask, self.h = trunk.blocks[layer], trunk.num_prefix_tokens, donor, mask, None

    def __enter__(self):
        def fn(mod, inp, out):
            x = out[0].clone()
            x[:, self.prefix:, :] = torch.where(self.mask[:, :, None], self.donor.to(x.dtype), x[:, self.prefix:, :])
            return (x, out[1], out[2])
        self.h = self.blk.register_forward_hook(fn)
        return self

    def __exit__(self, *a):
        self.h.remove()


class DecoderAttention:
    """Captures the decoder's cross-attention weights (B, H, Tq, P) by forcing
    need_weights=True on the MultiheadAttention call via a kwargs pre-hook."""

    def __init__(self, decoder):
        self.mha = [l.base_layer.multihead_attn for l in decoder.layers]
        self.weights, self.hs = [], []

    def __enter__(self):
        def pre(mod, args, kwargs):
            kwargs = dict(kwargs)
            kwargs["need_weights"] = True
            kwargs["average_attn_weights"] = False
            return args, kwargs

        def post(mod, args, out):
            self.weights.append(out[1].detach().float())
        for m in self.mha:
            self.hs.append(m.register_forward_pre_hook(pre, with_kwargs=True))
            self.hs.append(m.register_forward_hook(post))
        return self

    def __exit__(self, *a):
        for h in self.hs:
            h.remove()


@torch.no_grad()
def run_readout(out_dir, args, state, images_n2, owners_n2, labels_n2):
    model, steervit, device, tf = state["model"], state["steervit"], state["device"], state["transform"]
    trunk = steervit.vision_model.trunk
    prefix = trunk.num_prefix_tokens
    inv = {v: k for k, v in model.vocab.items()}
    N, bs = len(labels_n2), args.batch_size
    imgs_t = torch.stack([tf(im) for im in images_n2])
    owner_t = torch.from_numpy(np.stack(owners_n2))
    A_id = np.array([model.vocab[r["target"][QUERIED]] for r in labels_n2])
    Ad_id = np.array([model.vocab[r["distractors"][0][QUERIED]] for r in labels_n2])
    conds = ["c0", "c1", "c2"]
    bos = lambda b: torch.full((b, 1), model.vocab["<bos>"], dtype=torch.long, device=device)

    # ---- (i) baselines + decoder attention by owner ----
    base_pred = {c: np.zeros(N, int) for c in conds}
    attn_rows = {c: [] for c in conds}          # per image: (H, 3) mass on bg / target / distractor
    attn_top = {c: [] for c in conds}           # owner of the top-attended patch (head-mean)
    attn_topk = {c: [] for c in conds}          # object share among the top-8 patches (head-mean)
    for s in range(0, N, bs):
        e = min(s + bs, N)
        ow = owner_t[s:e].to(device)
        for c in conds:
            qs = None if c == "c0" else [labels_n2[i]["questions"][c] for i in range(s, e)]
            feats = steervit.forward(imgs_t[s:e].to(device), qs)
            patches = feats[:, prefix:, :]
            with DecoderAttention(model.decoder) as da:
                lg = model.decoder(bos(e - s), patches)[:, 0, :]
            base_pred[c][s:e] = lg.argmax(-1).cpu().numpy()
            w = da.weights[0][:, :, 0, :]                            # (B, H, P)
            assert torch.allclose(w.sum(-1), torch.ones_like(w.sum(-1)), atol=1e-4), "attention rows must sum to 1"
            mass = torch.stack([(w * (ow == k)[:, None, :]).sum(-1) for k in range(3)], -1)  # (B,H,3)
            attn_rows[c].append(mass.cpu().numpy())
            wm = w.mean(1)                                            # (B, P)
            top = wm.argmax(-1)
            attn_top[c].append(ow[torch.arange(e - s, device=device), top].cpu().numpy())
            topk = wm.topk(8, dim=-1).indices
            attn_topk[c].append((torch.gather(ow, 1, topk) > 0).float().mean(-1).cpu().numpy())
        if s == 0:
            gen = model.generate(imgs_t[:min(8, N)].to(device),
                                 [labels_n2[i]["questions"]["c1"] for i in range(min(8, N))])
            print(f"generate() vs first-token argmax on 8 images: {gen} | "
                  f"{[inv.get(int(t), '?') for t in base_pred['c1'][:min(8, N)]]}")
    acc = {"c1": float((base_pred["c1"] == A_id).mean()), "c2": float((base_pred["c2"] == Ad_id).mean()),
           "c0_says_target": float((base_pred["c0"] == A_id).mean()),
           "c0_says_distractor": float((base_pred["c0"] == Ad_id).mean())}
    print(f"baseline accuracy: c1 {acc['c1']:.3f}  c2 {acc['c2']:.3f}; no question → target colour "
          f"{acc['c0_says_target']:.3f}, distractor colour {acc['c0_says_distractor']:.3f}")
    n_tok = {k: float((owner_t == k).float().sum(1).mean()) for k in range(3)}
    attention = {"n_images": N, "tokens_per_owner_mean": {"bg": n_tok[0], "target": n_tok[1], "distractor": n_tok[2]},
                 "baseline_accuracy": acc, "conditions": {}}
    for c in conds:
        m = np.concatenate(attn_rows[c])                              # (N, H, 3)
        top = np.concatenate(attn_top[c])
        share = np.concatenate(attn_topk[c])
        mean_owner = m.mean(1)                                        # (N, 3) head-mean
        attention["conditions"][c] = {
            "mass_mean": {"bg": float(mean_owner[:, 0].mean()), "target": float(mean_owner[:, 1].mean()),
                          "distractor": float(mean_owner[:, 2].mean())},
            "mass_per_token": {"bg": float((mean_owner[:, 0] / np.maximum((owner_t == 0).sum(1).numpy(), 1)).mean()),
                               "target": float((mean_owner[:, 1] / np.maximum((owner_t == 1).sum(1).numpy(), 1)).mean()),
                               "distractor": float((mean_owner[:, 2] / np.maximum((owner_t == 2).sum(1).numpy(), 1)).mean())},
            "mass_per_head": {"bg": m[:, :, 0].mean(0).tolist(), "target": m[:, :, 1].mean(0).tolist(),
                              "distractor": m[:, :, 2].mean(0).tolist()},
            "top_patch_owner_frac": {"bg": float((top == 0).mean()), "target": float((top == 1).mean()),
                                     "distractor": float((top == 2).mean())},
            "object_share_in_top8": float(share.mean())}
        a = attention["conditions"][c]
        print(f"decoder attention {c}: mass bg {a['mass_mean']['bg']:.3f} target {a['mass_mean']['target']:.3f} "
              f"distractor {a['mass_mean']['distractor']:.3f} | per token ×1e3: bg {a['mass_per_token']['bg']*1e3:.2f} "
              f"target {a['mass_per_token']['target']*1e3:.2f} distractor {a['mass_per_token']['distractor']*1e3:.2f} | "
              f"top patch: bg {a['top_patch_owner_frac']['bg']:.2f} target {a['top_patch_owner_frac']['target']:.2f} "
              f"distractor {a['top_patch_owner_frac']['distractor']:.2f}", flush=True)
    with open(out_dir / "readout_attention.json", "w") as f:
        json.dump(attention, f, indent=1)

    # ---- (ii) activation patching between conditions at each block output ----
    # receiver run c1 (asks about the target); masked tokens replaced by the donor run's block output
    variants = [("c2", "bg"), ("c2", "objects"), ("c2", "target"), ("c2", "distractor"),
                ("c0", "bg"), ("c0", "objects"), ("c1", "bg")]       # last = identity control
    ok = (base_pred["c1"] == A_id) & (base_pred["c2"] == Ad_id) & (A_id != Ad_id)   # both correct, answers distinct
    print(f"token swaps on {int(ok.sum())} images (both questions correct, distinct answers)")
    counts = {(v, l): np.zeros(3, int) for v in variants for l in range(NUM_LAYERS)}   # [A, Ad, other]
    fout = open(out_dir / "readout_swap_trials.jsonl", "w")
    for s in range(0, N, bs):
        e = min(s + bs, N)
        idx = list(range(s, e))
        ims = imgs_t[s:e].to(device)
        ow = owner_t[s:e].to(device)
        donors = {}
        for c in ("c0", "c1", "c2"):
            qs = None if c == "c0" else [labels_n2[i]["questions"][c] for i in idx]
            with BlockCapture(trunk) as cap:
                feats = steervit.forward(ims, qs)
            donors[c] = cap.out
            if c == "c1":   # the block-11 capture followed by trunk.norm must equal the decoder input
                assert torch.allclose(trunk.norm(cap.out[NUM_LAYERS - 1]), feats[:, prefix:, :], atol=1e-4)
        masks = {"bg": ow == 0, "objects": ow > 0, "target": ow == 1, "distractor": ow == 2}
        qs1 = [labels_n2[i]["questions"]["c1"] for i in idx]
        for (dc, mk) in variants:
            for l in range(NUM_LAYERS):
                with TokenSwapper(trunk, l, donors[dc][l], masks[mk]):
                    lg = first_token_logits(model, steervit, ims, qs1)
                pred = lg.argmax(-1).cpu().numpy()
                for j, i in enumerate(idx):
                    if not ok[i]:
                        continue
                    k = 0 if pred[j] == A_id[i] else (1 if pred[j] == Ad_id[i] else 2)
                    counts[((dc, mk), l)][k] += 1
                    fout.write(json.dumps({"img": i, "donor": dc, "mask": mk, "layer": l,
                                           "pred": inv.get(int(pred[j]), "?"), "class": ["A", "Ad", "other"][k]}) + "\n")
        print(f"  swaps {e}/{N}", flush=True)
    fout.close()
    rows = []
    for (dc, mk) in variants:
        for l in range(NUM_LAYERS):
            cnt = counts[((dc, mk), l)]
            n = int(cnt.sum())
            rows.append({"donor": dc, "mask": mk, "layer": l, "n": n, "p_target": cnt[0] / max(n, 1),
                         "p_distractor": cnt[1] / max(n, 1), "p_other": cnt[2] / max(n, 1)})
    ident = [r for r in rows if r["donor"] == "c1"]
    assert all(r["p_target"] == 1.0 for r in ident), "identity swap must reproduce the c1 baseline"
    for (dc, mk) in variants:
        rr = [r for r in rows if r["donor"] == dc and r["mask"] == mk]
        print(f"swap donor {dc} mask {mk:<10} P(target colour) by block: " +
              " ".join(f"{r['p_target']:.2f}" for r in rr) + " | P(distractor colour): " +
              " ".join(f"{r['p_distractor']:.2f}" for r in rr), flush=True)
    swap = {"n_images_ok": int(ok.sum()), "receiver": "c1", "variants": [list(v) for v in variants], "rows": rows}
    with open(out_dir / "readout_swap.json", "w") as f:
        json.dump(swap, f, indent=1)
    return attention, swap


def plot_readout(attention, swap, label, out_path, gca_layers):
    """Two panels: at one block, a set of patch tokens of the run that asks
    about the target is replaced by the same tokens from another run (the run
    that asks about the distractor / the run without a question); the answer
    is then read by the decoder. Decoder-attention numbers go into the title."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    styles = {"bg": ("0.3", "-", "s", "background patches"),
              "objects": ("#ff7f0e", "-", "o", "both objects' patches"),
              "target": ("#1f77b4", "--", "^", "target's patches only"),
              "distractor": ("#d62728", "--", "v", "distractor's patches only")}
    panels = ((axes[0], "c2", "p_distractor",
               "tokens replaced from the forward pass with the question about the distractor",
               "P(answer = distractor's colour)"),
              (axes[1], "c0", "p_target",
               "tokens replaced from the forward pass without a question",
               "P(answer = target's colour)"))
    for ax, dc, key, title, ylab in panels:
        for mk, (col, ls, mkr, lab) in styles.items():
            rr = [r for r in swap["rows"] if r["donor"] == dc and r["mask"] == mk]
            if not rr:
                continue
            ax.plot([r["layer"] for r in rr], [r[key] for r in rr], ls, color=col, marker=mkr,
                    markersize=3, label=f"replaced: {lab}")
        ax.set_ylim(-0.02, 1.02)
        ax.set_ylabel(ylab, fontsize=10)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("ViT block at which the tokens are replaced")
        ax.set_xticks(range(NUM_LAYERS))
        mark_gca_layers(ax)
        ax.legend(fontsize=7)
    a1 = attention["conditions"]["c1"]["mass_per_token"]
    fig.suptitle(f"{label} — forward pass with the question about the target; one block's patch tokens replaced from another forward pass "
                 f"(n={swap['n_images_ok']})\n"
                 f"decoder attention per patch when asking about the target: target {a1['target']*1e3:.1f}, "
                 f"distractor {a1['distractor']*1e3:.1f}, background {a1['bg']*1e3:.1f} (×1e-3)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=S["dpi"], bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Marker causal test — is the referent marker a portable tag?  The marker
# direction m_l = mean over images of [target's raw patch mean under c1 (it is
# the referent) - the same under c2 (it is not)].  Transplanting m_l onto the
# DISTRACTOR's patches (a different image position) under the c1 question tests
# whether decoder attention and the answer follow the tag: if they do, the
# marker is a content-addressed identity code (Saravanan-style); if not, it is
# bound to the referent's position (Assouel-style position ID).
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_marker_test(out_dir, args, state, images_n2, owners_n2, labels_n2):
    model, steervit, device, tf = state["model"], state["steervit"], state["device"], state["transform"]
    trunk = steervit.vision_model.trunk
    c1 = load_sparse(out_dir / "n2", "c1")
    c2 = load_sparse(out_dir / "n2", "c2")
    marker = (c1["raw_obj_mean"][:, 0].astype(np.float32)
              - c2["raw_obj_mean"][:, 0].astype(np.float32)).mean(0)          # (12, 768)
    m_norm = np.linalg.norm(marker, axis=-1)
    print("marker norm by block: " + " ".join(f"{v:.1f}" for v in m_norm))

    N, bs = len(labels_n2), args.batch_size
    imgs_t = torch.stack([tf(im) for im in images_n2])
    owner_t = torch.from_numpy(np.stack(owners_n2))
    A_id = np.array([model.vocab[r["target"][QUERIED]] for r in labels_n2])
    Ad_id = np.array([model.vocab[r["distractors"][0][QUERIED]] for r in labels_n2])
    qs1_all = [r["questions"]["c1"] for r in labels_n2]
    bos = lambda b: torch.full((b, 1), model.vocab["<bos>"], dtype=torch.long, device=device)
    prefix = trunk.num_prefix_tokens

    def forward_with_attn(ims, qs):
        feats = steervit.forward(ims, qs)
        patches = feats[:, prefix:, :]
        with DecoderAttention(model.decoder) as da:
            lg = model.decoder(bos(ims.shape[0]), patches)[:, 0, :]
        w = da.weights[0][:, :, 0, :].mean(1)                                  # (B, P) head-mean
        return lg.argmax(-1).cpu().numpy(), w

    # baseline under c1
    base_pred = np.zeros(N, int)
    base_mass = []
    for s in range(0, N, bs):
        e = min(s + bs, N)
        pred, w = forward_with_attn(imgs_t[s:e].to(device), qs1_all[s:e])
        base_pred[s:e] = pred
        ow = owner_t[s:e].to(device)
        base_mass.append(torch.stack([(w * (ow == k)).sum(-1) for k in range(3)], -1).cpu().numpy())
    ok = (base_pred == A_id) & (A_id != Ad_id)
    bm = np.concatenate(base_mass)[ok].mean(0)
    print(f"baseline c1: accuracy {ok.mean():.3f} on {N} images; attention mass "
          f"bg {bm[0]:.3f} target {bm[1]:.3f} distractor {bm[2]:.3f}")

    rng = np.random.default_rng(args.seed)
    rand = rng.standard_normal(marker.shape).astype(np.float32)
    rand = rand / np.linalg.norm(rand, axis=-1, keepdims=True) * m_norm[:, None]
    layers = [int(l) for l in args.intervene_layers.split(",")]
    alphas = [float(a) for a in args.marker_alphas.split(",")]
    variants = ["marker_to_distractor", "marker_minus_target", "swap", "random_to_distractor"]
    rows = []
    for layer in layers:
        d_m = torch.from_numpy(marker[layer]).to(device)
        d_r = torch.from_numpy(rand[layer]).to(device)
        for alpha in alphas:
            for var in variants:
                if var == "random_to_distractor" and alpha != alphas[0]:
                    continue
                cnt = np.zeros(3, int)
                mass = []
                for s in range(0, N, bs):
                    e = min(s + bs, N)
                    ims = imgs_t[s:e].to(device)
                    ow = owner_t[s:e].to(device)
                    delta = (d_r if var == "random_to_distractor" else d_m).expand(e - s, -1)
                    adders = []
                    if var in ("marker_to_distractor", "swap", "random_to_distractor"):
                        adders.append(ResidualAdder(trunk, layer, delta, ow == 2, alpha))
                    if var in ("marker_minus_target", "swap"):
                        adders.append(ResidualAdder(trunk, layer, delta, ow == 1, -alpha))
                    for a in adders:
                        a.__enter__()
                    try:
                        pred, w = forward_with_attn(ims, qs1_all[s:e])
                    finally:
                        for a in adders:
                            a.__exit__()
                    m3 = torch.stack([(w * (ow == k)).sum(-1) for k in range(3)], -1).cpu().numpy()
                    for j, i in enumerate(range(s, e)):
                        if not ok[i]:
                            continue
                        cnt[0 if pred[j] == A_id[i] else (1 if pred[j] == Ad_id[i] else 2)] += 1
                        mass.append(m3[j])
                n = int(cnt.sum())
                mass = np.stack(mass).mean(0)
                rows.append({"layer": layer, "alpha": alpha, "variant": var, "n": n,
                             "p_target": cnt[0] / max(n, 1), "p_distractor": cnt[1] / max(n, 1),
                             "p_other": cnt[2] / max(n, 1),
                             "attn_bg": float(mass[0]), "attn_target": float(mass[1]),
                             "attn_distractor": float(mass[2])})
                r = rows[-1]
                print(f"L{layer:2d} a={alpha:g} {var:<22} P(target) {r['p_target']:.2f} "
                      f"P(distractor) {r['p_distractor']:.2f} | attn target {r['attn_target']:.3f} "
                      f"distractor {r['attn_distractor']:.3f}", flush=True)
    res = {"n_images_ok": int(ok.sum()), "queried": QUERIED,
           "marker_norm_by_block": m_norm.tolist(),
           "baseline": {"accuracy_c1": float(ok.mean()),
                        "attn_mass": {"bg": float(bm[0]), "target": float(bm[1]),
                                      "distractor": float(bm[2])}},
           "rows": rows}
    with open(out_dir / "marker_test.json", "w") as f:
        json.dump(res, f, indent=1)
    print(f"Saved: {out_dir / 'marker_test.json'}")
    return res


def plot_marker_test(res, label, out_path):
    """Left: answer flip; right: decoder-attention mass on the distractor."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    styles = {"marker_to_distractor": ("#d62728", "-", "o", "marker added to distractor patches"),
              "marker_minus_target": ("#1f77b4", "--", "^", "marker subtracted from target patches"),
              "swap": ("#9467bd", "-", "s", "both (moved from target to distractor)"),
              "random_to_distractor": ("0.4", ":", "x", "norm-matched random vector on distractor")}
    alphas = sorted({r["alpha"] for r in res["rows"]})
    a0 = alphas[0]
    for ax, key, ylab in ((axes[0], "p_distractor", "P(answer = distractor's colour)"),
                          (axes[1], "attn_distractor", "decoder attention mass on distractor patches")):
        for var, (col, ls, mk, lab) in styles.items():
            rr = [r for r in res["rows"] if r["variant"] == var and r["alpha"] == a0]
            if rr:
                ax.plot([r["layer"] for r in rr], [r[key] for r in rr], ls, color=col,
                        marker=mk, markersize=3, label=lab)
        ax.set_xlabel("ViT block at which the vector is added")
        ax.set_ylabel(ylab, fontsize=10)
        ax.set_xticks(range(NUM_LAYERS))
        ax.set_ylim(-0.02, 1.02)
        mark_gca_layers(ax)
    axes[1].axhline(res["baseline"]["attn_mass"]["distractor"], color="0.6", lw=0.8,
                    label="baseline (no intervention)")
    axes[0].legend(fontsize=7)
    axes[1].legend(fontsize=7)
    fig.suptitle(f"{label} — transplanting the referent-marker direction (difference of the target's patch mean, "
                 f"refer-target vs refer-distractor)\nquestion asks the target; n={res['n_images_ok']}, "
                 f"scale {a0:g}x", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=S["dpi"], bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Attention-vs-norm control (Darcet / attention-sink guard): is the decoder's
# attention concentration on the referent explained by token norm?
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_attn_norm_control(out_dir, args, state, images_n2, owners_n2, labels_n2):
    model, steervit, device, tf = state["model"], state["steervit"], state["device"], state["transform"]
    prefix = steervit.vision_model.trunk.num_prefix_tokens
    norm11 = load_sparse(out_dir / "n2", "c1")["raw_norm"][:, NUM_LAYERS - 1].astype(np.float32)  # (N, P)
    N, bs = len(labels_n2), args.batch_size
    imgs_t = torch.stack([tf(im) for im in images_n2])
    owner = np.stack(owners_n2)
    bos = lambda b: torch.full((b, 1), model.vocab["<bos>"], dtype=torch.long, device=device)
    res = {}
    for cond in ("c0", "c1"):
        W = []
        for s in range(0, N, bs):
            e = min(s + bs, N)
            qs = None if cond == "c0" else [labels_n2[i]["questions"]["c1"] for i in range(s, e)]
            feats = steervit.forward(imgs_t[s:e].to(device), qs)
            with DecoderAttention(model.decoder) as da:
                model.decoder(bos(e - s), feats[:, prefix:, :])
            W.append(da.weights[0][:, :, 0, :].mean(1).cpu().numpy())
        W = np.concatenate(W)                                                  # (N, P)
        rho_all, rho_bg, hi_mass, hi_bgfrac, ratio_all, ratio_excl = [], [], [], [], [], []
        for i in range(N):
            w, nm, ow = W[i], norm11[i], owner[i]
            rho_all.append(spearmanr(w, nm)[0])
            rho_bg.append(spearmanr(w[ow == 0], nm[ow == 0])[0])
            hi = nm > 5 * np.median(nm)
            hi_mass.append(w[hi].sum() if hi.any() else 0.0)
            hi_bgfrac.append((ow[hi] == 0).mean() if hi.any() else np.nan)
            per_tok = lambda m: w[m].mean() if m.any() else np.nan
            ratio_all.append(per_tok(ow == 1) / per_tok(ow == 0))
            keep = ~hi
            ratio_excl.append(per_tok((ow == 1) & keep) / per_tok((ow == 0) & keep))
        res[cond] = {
            "spearman_attn_norm_mean": float(np.nanmean(rho_all)),
            "spearman_attn_norm_bg_only_mean": float(np.nanmean(rho_bg)),
            "high_norm_tokens_per_image": float((norm11 > 5 * np.median(norm11, 1, keepdims=True)).sum(1).mean()),
            "attn_mass_on_high_norm_mean": float(np.nanmean(hi_mass)),
            "high_norm_bg_fraction_mean": float(np.nanmean(hi_bgfrac)),
            "per_token_ratio_target_over_bg": float(np.nanmean(ratio_all)),
            "per_token_ratio_excluding_high_norm": float(np.nanmean(ratio_excl))}
        r = res[cond]
        print(f"{cond}: rho(attn,norm) {r['spearman_attn_norm_mean']:.3f} (bg only "
              f"{r['spearman_attn_norm_bg_only_mean']:.3f}); high-norm tokens/image "
              f"{r['high_norm_tokens_per_image']:.1f}, their attention mass {r['attn_mass_on_high_norm_mean']:.3f} "
              f"(bg fraction {r['high_norm_bg_fraction_mean']:.2f}); target/bg per-token ratio "
              f"{r['per_token_ratio_target_over_bg']:.1f} -> {r['per_token_ratio_excluding_high_norm']:.1f} "
              f"excluding high-norm", flush=True)
    with open(out_dir / "attn_norm_control.json", "w") as f:
        json.dump(res, f, indent=1)
    print(f"Saved: {out_dir / 'attn_norm_control.json'}")
    return res


# ---------------------------------------------------------------------------
# Head scan — which attention heads carry the selection effect?  Zero-ablate
# one head (or one whole layer) with the shared HeadAblator, run the two
# referring questions, and measure per block the target's projection onto its
# own queried-attribute direction, refer target − refer distractor.
# ---------------------------------------------------------------------------

def make_selection_measurer(args, state, cache_n1, labels_n1, images_n2, owners_n2, labels_n2):
    """Returns measure(ablator) -> selection effect per block + accuracy, the
    quantity the head scan and the head-combination runs share."""
    from contextlib import nullcontext
    model, steervit, device, tf = state["model"], state["steervit"], state["device"], state["transform"]
    trunk = steervit.vision_model.trunk
    prefix, norm = trunk.num_prefix_tokens, trunk.norm
    N, bs = len(labels_n2), args.batch_size
    imgs_t = torch.stack([tf(im) for im in images_n2])
    owner_t = torch.from_numpy(np.stack(owners_n2)).to(device)
    V = attribute_directions(cache_n1, labels_n1)[QUERIED]
    Vt = torch.from_numpy(np.stack([V[r["target"][QUERIED]] for r in labels_n2])).to(device)          # (N,12,D)
    Vd = torch.from_numpy(np.stack([V[r["distractors"][0][QUERIED]] for r in labels_n2])).to(device)
    A_id = np.array([model.vocab[r["target"][QUERIED]] for r in labels_n2])
    Ad_id = np.array([model.vocab[r["distractors"][0][QUERIED]] for r in labels_n2])
    has_d = torch.from_numpy(np.array([r["n_distractor_patches"] > 0 for r in labels_n2])).to(device)
    bos = lambda b: torch.full((b, 1), model.vocab["<bos>"], dtype=torch.long, device=device)

    def measure(ablator):
        P = {}
        preds = {}
        for cond in ("c1", "c2"):
            pt = torch.zeros(N, NUM_LAYERS, device=device)
            pd = torch.zeros(N, NUM_LAYERS, device=device)
            pr = np.zeros(N, int)
            for s in range(0, N, bs):
                e = min(s + bs, N)
                qs = [labels_n2[i]["questions"][cond] for i in range(s, e)]
                ctx = ablator if ablator is not None else nullcontext()
                with ctx, BlockCapture(trunk) as cap:
                    feats = steervit.forward(imgs_t[s:e].to(device), qs)
                lg = model.decoder(bos(e - s), feats[:, prefix:, :])[:, 0, :]
                pr[s:e] = lg.argmax(-1).cpu().numpy()
                ow = owner_t[s:e]
                mt = (ow == 1).float()[:, :, None]
                md = (ow == 2).float()[:, :, None]
                for l in range(NUM_LAYERS):
                    x = norm(cap.out[l]).float()                          # (B,P,D)
                    mean_t = (x * mt).sum(1) / mt.sum(1).clamp(min=1)
                    mean_d = (x * md).sum(1) / md.sum(1).clamp(min=1)
                    pt[s:e, l] = (mean_t * Vt[s:e, l]).sum(-1)
                    pd[s:e, l] = (mean_d * Vd[s:e, l]).sum(-1)
            P[cond] = (pt, pd)
            preds[cond] = pr
        S_t = (P["c1"][0] - P["c2"][0]).mean(0).cpu().numpy()                        # (12,)
        S_d = (P["c1"][1] - P["c2"][1])[has_d].mean(0).cpu().numpy()
        return {"S_target": S_t.tolist(), "S_distractor": S_d.tolist(),
                "acc_c1": float((preds["c1"] == A_id).mean()), "acc_c2": float((preds["c2"] == Ad_id).mean())}

    return measure


@torch.no_grad()
def run_head_scan(out_dir, args, state, cache_n1, labels_n1, images_n2, owners_n2, labels_n2):
    from analysis.patching_utils import HeadAblator
    steervit = state["steervit"]
    trunk = steervit.vision_model.trunk
    N = len(labels_n2)
    measure = make_selection_measurer(args, state, cache_n1, labels_n1, images_n2, owners_n2, labels_n2)
    base = measure(None)
    print("baseline selection effect (target, by block): " + " ".join(f"{v:+.1f}" for v in base["S_target"])
          + f"  acc {base['acc_c1']:.3f}/{base['acc_c2']:.3f}", flush=True)
    gca_layers = [i for i, b in enumerate(trunk.blocks) if getattr(b, "gated_cross_attn", None) is not None]
    n_sa = trunk.blocks[0].attn.num_heads
    n_gca = trunk.blocks[gca_layers[0]].gated_cross_attn.cross_attn.num_heads
    runs = [("sa", l, h) for l in range(NUM_LAYERS) for h in range(n_sa)] + \
           [("gca", l, h) for l in gca_layers for h in range(n_gca)]
    layer_runs = [("sa", l, None) for l in range(NUM_LAYERS)] + [("gca", l, None) for l in gca_layers]
    rows = []
    fout = open(out_dir / "head_scan_rows.jsonl", "w")
    for k, (kind, l, h) in enumerate(runs + layer_runs):
        heads = [(kind, l, h)] if h is not None else [(kind, l, hh) for hh in range(n_sa if kind == "sa" else n_gca)]
        r = measure(HeadAblator(steervit, heads, mode="zero"))
        r.update(kind=kind, layer=l, head=h)
        rows.append(r)
        fout.write(json.dumps(r) + "\n")
        fout.flush()
        if h is None or (k % 24 == 0):
            print(f"  {k + 1}/{len(runs) + len(layer_runs)} {kind} L{l} H{h}: S11 {r['S_target'][-1]:+.1f} "
                  f"(base {base['S_target'][-1]:+.1f}) acc {r['acc_c1']:.2f}/{r['acc_c2']:.2f}", flush=True)
    fout.close()
    # compare with the headwise activation-patching recovery on the same checkpoint
    comp = {}
    stats_path = Path(args.patching_stats)
    if stats_path.exists():
        from scipy.stats import spearmanr
        st = json.load(open(stats_path))
        for group in ("fine_attribute_denoising", "fine_attribute_query_denoising"):
            for cat, v in st[group].items():
                for kind, key in (("sa", "sa_mean"), ("gca", "gca_mean")):
                    rec = np.array(v[key])
                    xs, ys = [], []
                    for r in rows:
                        if r["kind"] != kind or r["head"] is None:
                            continue
                        li = r["layer"] if kind == "sa" else gca_layers.index(r["layer"])
                        xs.append(rec[li, r["head"]])
                        ys.append(base["S_target"][-1] - r["S_target"][-1])
                    comp[f"{group}/{cat}/{kind}"] = {"spearman_recovery_vs_dS11": float(spearmanr(xs, ys).correlation),
                                                     "n": len(xs)}
    res = {"queried": QUERIED, "mode": "zero", "n_images": N, "baseline": base, "rows": rows,
           "gca_layers": gca_layers, "n_sa_heads": n_sa, "n_gca_heads": n_gca, "comparison_with_patching": comp}
    with open(out_dir / "head_scan.json", "w") as f:
        json.dump(res, f, indent=1)
    for k, v in comp.items():
        print(f"patching recovery vs selection drop, {k}: Spearman {v['spearman_recovery_vs_dS11']:+.2f}")
    return res


@torch.no_grad()
def run_head_combos(out_dir, args, state, cache_n1, labels_n1, images_n2, owners_n2, labels_n2):
    """Sufficiency test on GCA layers 7 and 9: keep only the top-4 heads (by the
    single-head scan), zero only the top-4, keep / zero random subsets."""
    from analysis.patching_utils import HeadAblator
    steervit = state["steervit"]
    measure = make_selection_measurer(args, state, cache_n1, labels_n1, images_n2, owners_n2, labels_n2)
    hs = json.load(open(out_dir / "head_scan.json"))
    base11 = hs["baseline"]["S_target"][-1]
    n_gca = hs["n_gca_heads"]
    top = {}
    for L in (7, 9):
        rows = [(r["head"], base11 - r["S_target"][-1]) for r in hs["rows"]
                if r["kind"] == "gca" and r["layer"] == L and r["head"] is not None]
        rows.sort(key=lambda x: -x[1])
        top[L] = [h for h, _ in rows[:4]]
    print(f"top-4 heads by single-head drop: L7 {top[7]}  L9 {top[9]}")
    rng = np.random.RandomState(args.seed)
    conds = []
    for L in (7, 9):
        allh = list(range(n_gca))
        conds.append((f"L{L}_keep_top4", [("gca", L, h) for h in allh if h not in top[L]]))
        conds.append((f"L{L}_zero_top4", [("gca", L, h) for h in top[L]]))
        for si in range(3):
            keep = list(rng.choice(allh, 4, replace=False))
            conds.append((f"L{L}_keep_rand4_s{si}", [("gca", L, h) for h in allh if h not in keep]))
            drop = list(rng.choice(allh, 8, replace=False))
            conds.append((f"L{L}_zero_rand8_s{si}", [("gca", L, int(h)) for h in drop]))
        conds.append((f"L{L}_zero_all", [("gca", L, h) for h in allh]))
    conds.append(("L7L9_keep_top4", [("gca", L, h) for L in (7, 9) for h in range(n_gca) if h not in top[L]]))
    base = measure(None)
    print("baseline S by block: " + " ".join(f"{v:+.1f}" for v in base["S_target"]))
    rows = [dict(name="baseline", n_zeroed=0, **base)]
    for name, heads in conds:
        r = measure(HeadAblator(steervit, heads, mode="zero"))
        r = dict(name=name, n_zeroed=len(heads), **r)
        rows.append(r)
        print(f"  {name:<22} zeroed {len(heads):2d}  S9 {r['S_target'][9]:+5.1f} S10 {r['S_target'][10]:+5.1f} "
              f"S11 {r['S_target'][11]:+5.1f}  acc {r['acc_c1']:.2f}/{r['acc_c2']:.2f}", flush=True)
    res = {"queried": QUERIED, "top4": {str(k): v for k, v in top.items()}, "rows": rows}
    with open(out_dir / "head_combos.json", "w") as f:
        json.dump(res, f, indent=1)
    return res


def plot_head_scan(res, label, out_path):
    """Heatmaps: change of the selection effect (target's own queried-attribute
    projection, refer target − refer distractor) at block 10 and at block 11
    when one head is zeroed; bars: all heads of one layer zeroed."""
    base = np.array(res["baseline"]["S_target"])
    gl, n_sa, n_gca = res["gca_layers"], res["n_sa_heads"], res["n_gca_heads"]
    blocks = (10, 11)
    dS = {(k, b): np.full(shape, np.nan) for k, shape in (("sa", (NUM_LAYERS, n_sa)), ("gca", (len(gl), n_gca))) for b in blocks}
    layer = {(k, b): np.full(n, np.nan) for k, n in (("sa", NUM_LAYERS), ("gca", len(gl))) for b in blocks}
    worst_acc = 1.0
    for r in res["rows"]:
        li = r["layer"] if r["kind"] == "sa" else gl.index(r["layer"])
        for b in blocks:
            if r["head"] is None:
                layer[(r["kind"], b)][li] = r["S_target"][b] - base[b]
            else:
                dS[(r["kind"], b)][li, r["head"]] = r["S_target"][b] - base[b]
        if r["head"] is not None:
            worst_acc = min(worst_acc, r["acc_c1"], r["acc_c2"])
    vmax = max(np.nanmax(np.abs(v)) for v in dS.values())
    fig, axes = plt.subplots(2, 4, figsize=(17, 7), gridspec_kw={"width_ratios": [1, 1, 0.45, 0.45]})
    for row, (kind, title) in enumerate((("sa", "self-attention heads"), ("gca", "gated cross-attention heads"))):
        ylab = "block" if kind == "sa" else "GCA layer"
        yt = list(range(NUM_LAYERS)) if kind == "sa" else gl
        for col, b in enumerate(blocks):
            ax = axes[row, col]
            im = ax.imshow(dS[(kind, b)], cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
            ax.set_title(f"{title}: one head zeroed → change of the selection effect at block {b}", fontsize=9)
            ax.set_xlabel("head"); ax.set_ylabel(ylab)
            ax.set_yticks(range(len(yt))); ax.set_yticklabels([str(l) for l in yt], fontsize=8)
            if col == 1:
                fig.colorbar(im, ax=ax, fraction=0.04)
        for col, b in enumerate(blocks):
            ax = axes[row, 2 + col]
            ys = layer[(kind, b)]
            ax.barh(range(len(ys)), ys, color="0.4")
            ax.set_yticks(range(len(ys))); ax.set_yticklabels([str(l) for l in yt], fontsize=8)
            ax.invert_yaxis()
            ax.axvline(0, color="k", linewidth=0.6)
            ax.set_xlim(min(-1.0, np.nanmin(ys) * 1.1), max(1.0, np.nanmax(ys) * 1.1))
            ax.set_title(f"all heads of one layer zeroed,\nchange at block {b}", fontsize=9)
            ax.set_xlabel("change of selection effect")
    fig.suptitle(f"{label} — head ablation scan (questions ask about {res['queried']}); baseline selection effect "
                 f"+{base[10]:.1f} at block 10, +{base[11]:.1f} at block 11; no single-head ablation lowers accuracy "
                 f"below {worst_acc:.2f} (baseline {res['baseline']['acc_c1']:.3f})", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=S["dpi"], bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def color_vectors(cache_n1, labels_n1):
    """Per value of the queried attribute: mean raw target-patch token of the
    1-object images with that value, (12, D) each; Δ_ℓ(A→B) = means[B] − means[A]."""
    rom = cache_n1["raw_obj_mean"][:, 0].astype(np.float32)   # (N,12,D)
    cols = np.array([rec["target"][QUERIED] for rec in labels_n1])
    means = {c: rom[cols == c].mean(0) for c in ATTR_VALUES[QUERIED] if (cols == c).any()}
    return means


def run_interventions(out_dir, args, state, cache_n1, labels_n1, images_n2, owners_n2, labels_n2):
    model, steervit, device, tf = state["model"], state["steervit"], state["device"], state["transform"]
    trunk = steervit.vision_model.trunk
    inv = {v: k for k, v in model.vocab.items()}
    means = color_vectors(cache_n1, labels_n1)
    np.savez(out_dir / "color_vectors.npz", **{k: v for k, v in means.items()})
    rng = np.random.RandomState(args.seed)
    N = len(labels_n2)
    # trial design: B != A round-robin
    trials = []
    for i, rec in enumerate(labels_n2):
        A = rec["target"][QUERIED]
        Ad = rec["distractors"][0][QUERIED]
        # B differs from BOTH objects' values so a flip cannot be confused with
        # answering about the other object
        others = [c for c in ATTR_VALUES[QUERIED] if c not in (A, Ad) and c in means]
        if not others:            # only possible with few values (e.g. size) — skip image
            others = [A]
        B = others[i % len(others)]
        Bd = others[(i + 1) % len(others)]
        trials.append({"img": i, "A": A, "B": B, "Ad": Ad, "Bd": Bd})
    layers = [int(l) for l in args.intervene_layers.split(",")]
    alphas = [float(a) for a in args.alphas.split(",")]
    variants = ["target_delta_c1", "target_random_c1", "bg_subset_delta_c1", "bg_all_delta_c1",
                "distractor_delta_c1", "distractor_deltaD_c2", "target_delta_c2"]
    bs = args.batch_size
    imgs_t = torch.stack([tf(im) for im in images_n2])
    owner_t = torch.from_numpy(np.stack(owners_n2))
    rows = []
    fout = open(out_dir / "intervention_trials.jsonl", "w")

    def answers(images, questions, adder=None):
        if adder is None:
            lg = first_token_logits(model, steervit, images, questions)
        else:
            with adder:
                lg = first_token_logits(model, steervit, images, questions)
        return lg.float().cpu()

    # baselines
    base = {}
    for cond in ("c1", "c2"):
        preds = []
        for s in range(0, N, bs):
            e = min(s + bs, N)
            lg = answers(imgs_t[s:e].to(device), [labels_n2[i]["questions"][cond] for i in range(s, e)])
            preds.append(lg.argmax(-1))
        base[cond] = torch.cat(preds).numpy()
    # cross-check argmax vs generate on the first batch
    gen = model.generate(imgs_t[:min(8, N)].to(device), [labels_n2[i]["questions"]["c1"] for i in range(min(8, N))])
    gen_first = [inv.get(int(t), "?") for t in base["c1"][:min(8, N)]]
    print(f"generate() vs first-token argmax on 8 images: {gen} | {gen_first}")
    truth = {"c1": np.array([model.vocab[t["A"]] for t in trials]),
             "c2": np.array([model.vocab[t["Ad"]] for t in trials])}
    base_acc = {c: float((base[c] == truth[c]).mean()) for c in ("c1", "c2")}
    print(f"baseline accuracy: c1 {base_acc['c1']:.3f}  c2 {base_acc['c2']:.3f}")

    for layer in layers:
        for alpha in alphas:
            for var in variants:
                cond = "c2" if var.endswith("c2") else "c1"
                flips, changed, logit_gap, n_ok = 0, 0, [], 0
                for s in range(0, N, bs):
                    e = min(s + bs, N)
                    idx = list(range(s, e))
                    ow = owner_t[s:e]
                    deltas, masks, targetsB, valid = [], [], [], []
                    for i in idx:
                        t = trials[i]
                        if var == "distractor_deltaD_c2":
                            A, B = t["Ad"], t["Bd"]
                        else:
                            A, B = t["A"], t["B"]
                        if A not in means or B not in means:   # colour absent from n1 (small runs)
                            valid.append(False)
                            deltas.append(np.zeros(next(iter(means.values())).shape[-1], np.float32))
                            targetsB.append(-1)
                            masks.append(torch.zeros(ow.shape[1], dtype=torch.bool))
                            continue
                        valid.append(True)
                        d = means[B][layer] - means[A][layer]
                        if var == "target_random_c1":
                            r = rng.randn(*d.shape).astype(np.float32)
                            d = r / np.linalg.norm(r) * np.linalg.norm(d)
                        deltas.append(d)
                        targetsB.append(model.vocab[B])
                        o = ow[i - s]
                        if var.startswith("target"):
                            m = o == 1
                        elif var.startswith("distractor"):
                            m = o == 2
                        elif var == "bg_all_delta_c1":
                            m = o == 0
                        else:  # bg subset of size |target|
                            m = torch.zeros_like(o, dtype=torch.bool)
                            bgpos = torch.nonzero(o == 0).flatten().numpy()
                            k = int((o == 1).sum())
                            pick = np.random.RandomState(args.seed + i).choice(bgpos, min(k, len(bgpos)), replace=False)
                            m[torch.from_numpy(pick)] = True
                        masks.append(m)
                    delta_t = torch.from_numpy(np.stack(deltas)).to(device)
                    mask_t = torch.stack(masks).to(device)
                    adder = ResidualAdder(trunk, layer, delta_t, mask_t, alpha)
                    lg = answers(imgs_t[s:e].to(device),
                                 [labels_n2[i]["questions"][cond] for i in idx], adder)
                    pred = lg.argmax(-1).numpy()
                    for j, i in enumerate(idx):
                        ok = base[cond][i] == truth[cond][i]
                        if not ok or not valid[j]:
                            continue
                        n_ok += 1
                        flip = pred[j] == targetsB[j]
                        chg = pred[j] != base[cond][i]
                        flips += int(flip)
                        changed += int(chg)
                        gap = float(lg[j, targetsB[j]] - lg[j, truth[cond][i]])
                        logit_gap.append(gap)
                        row = {"layer": layer, "alpha": alpha, "variant": var, "img": i,
                               "pred": inv.get(int(pred[j]), "?"), "flip": bool(flip),
                               "changed": bool(chg), "logit_gap": gap}
                        fout.write(json.dumps(row) + "\n")
                rows.append({"layer": layer, "alpha": alpha, "variant": var, "n": n_ok,
                             "flip_rate": flips / max(n_ok, 1), "change_rate": changed / max(n_ok, 1),
                             "logit_gap_mean": float(np.mean(logit_gap)) if logit_gap else float("nan")})
                print(f"L{layer:2d} a={alpha:<4} {var:<22} n={n_ok:3d} flip={flips / max(n_ok, 1):.3f} "
                      f"changed={changed / max(n_ok, 1):.3f} gap={rows[-1]['logit_gap_mean']:.2f}", flush=True)
    fout.close()
    summary = {"baseline_accuracy": base_acc, "layers": layers, "alphas": alphas,
               "variants": variants, "rows": rows, "n_images": N}
    with open(out_dir / "intervention_results.json", "w") as f:
        json.dump(summary, f, indent=1)
    print(f"Saved: {out_dir / 'intervention_results.json'}")
    return summary


# ---------------------------------------------------------------------------
# Part C — held-out-position probes on single patch tokens
# ---------------------------------------------------------------------------

def _subsample_tokens(c, labels, max_obj, max_bg, seed):
    rng = np.random.RandomState(seed)
    keep = []
    img, own = c["tok_img"], c["tok_owner"]
    for i in range(len(labels)):
        for oid, cap in ((1, max_obj), (2, max_obj), (0, max_bg)):
            idx = np.nonzero((img == i) & (own == oid))[0]
            if len(idx) > cap:
                idx = rng.choice(idx, cap, replace=False)
            keep.extend(idx.tolist())
    return np.array(sorted(keep))


def _fit_eval(X, y, groups, splitter):
    accs = []
    for tr, te in splitter.split(X, y, groups):
        if len(np.unique(y[tr])) < 2:
            continue
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        clf.fit(X[tr], y[tr])
        accs.append(float((clf.predict(X[te]) == y[te]).mean()))
    return float(np.mean(accs)) if accs else float("nan")


def part_c(caches, labels, args):
    layers = [int(l) for l in args.probe_layers.split(",")]
    c0 = caches["c0"]
    keep = _subsample_tokens(c0, labels, args.probe_max_tokens_per_object, args.probe_max_bg, args.seed)
    img, own, pos = c0["tok_img"][keep], c0["tok_owner"][keep], c0["tok_pos"][keep]
    slot = np.array([labels[i]["slot"] for i in img])
    cell = np.array([labels[i]["spatial_cell"] if o != 2 else labels[i]["spatial_cell_distractor"]
                     for i, o in zip(img, own)])
    is_obj = own > 0
    attr_lab = {a: np.array([(labels[i]["target"][a] if o == 1 else
                              labels[i]["distractors"][0][a] if o == 2 else "bg")
                             for i, o in zip(img, own)]) for a in ATTRS}
    splits = {"random_group_image": (GroupKFold(5), img),
              "slot_loo": (LeaveOneGroupOut(), slot),
              "spatial_loo": (LeaveOneGroupOut(), cell)}
    results = {}
    for l in layers:
        X = token_table(c0, l)[keep]
        res = {}
        for sname, (sp, g) in splits.items():
            res[f"bg_vs_object/{sname}"] = _fit_eval(X, is_obj.astype(int), g, sp)
            for a in ATTRS:
                res[f"{a}/{sname}"] = _fit_eval(X[is_obj], attr_lab[a][is_obj], g[is_obj], sp)
        # referent vs non-referent on object tokens under c1 ∪ c2 (and c0 control)
        if "c1" in caches and "c2" in caches:
            Xs, ys, gs, cs = [], [], [], []
            for cond, ref_oid in (("c1", 1), ("c2", 2)):
                Xc = token_table(caches[cond], l)[keep][is_obj]
                Xs.append(Xc)
                ys.append((own[is_obj] == ref_oid).astype(int))
                gs.append(img[is_obj])
                cs.append(cell[is_obj])
            Xr, yr, gr, cr = np.concatenate(Xs), np.concatenate(ys), np.concatenate(gs), np.concatenate(cs)
            res["referent/random_group_image"] = _fit_eval(Xr, yr, gr, GroupKFold(5))
            res["referent/spatial_loo"] = _fit_eval(Xr, yr, cr, LeaveOneGroupOut())
            X0 = np.concatenate([X[is_obj], X[is_obj]])
            res["referent_c0_control/random_group_image"] = _fit_eval(X0, yr, gr, GroupKFold(5))
        results[f"L{l}"] = res
        print(f"L{l}: " + " ".join(f"{k}={v:.3f}" for k, v in res.items() if k.endswith("random_group_image")))
    return results



# ---------------------------------------------------------------------------
# Part D — readout probes on the cached no-question tokens: which aggregation
# can read the target's attribute when no question is given?
#   mean_pool                     mean over all cached tokens -> logistic regression
#   attention_all_tokens          one learned query, content attention over all tokens
#   attention_target_tokens       same, attention restricted to the target's tokens (oracle position)
#   attention_distractor_tokens   same, restricted to the distractor's tokens (control)
# "All tokens" = the cached set (every object patch + 64 background patches).
# Feature standardisation uses token statistics of the whole layer (unsupervised).
# ---------------------------------------------------------------------------

class _AttnReadout(torch.nn.Module):
    def __init__(self, d, n_cls):
        super().__init__()
        self.q = torch.nn.Parameter(torch.zeros(d))
        self.head = torch.nn.Linear(d, n_cls)

    def forward(self, x, mask):                      # x (B,T,D), mask (B,T) bool
        s = (x @ self.q) / x.shape[-1] ** 0.5
        w = torch.softmax(s.masked_fill(~mask, -1e9), -1)
        return self.head((w[..., None] * x).sum(1))


def _pad_images(X, img, n_img):
    """(T,D) tokens with image index -> (N,Tmax,D) array, valid mask, index lists."""
    idx = [np.nonzero(img == i)[0] for i in range(n_img)]
    tmax = max(len(t) for t in idx)
    out = np.zeros((n_img, tmax, X.shape[1]), np.float32)
    valid = np.zeros((n_img, tmax), bool)
    for i, t in enumerate(idx):
        out[i, :len(t)] = X[t]
        valid[i, :len(t)] = True
    return out, valid, idx


def _train_attn(Xtr, Mtr, ytr, Xte, Mte, n_cls, seed, epochs=300):
    torch.manual_seed(seed)
    m = _AttnReadout(Xtr.shape[-1], n_cls)
    opt = torch.optim.Adam(m.parameters(), lr=1e-2, weight_decay=1e-3)
    Xtr_t, Mtr_t, ytr_t = torch.as_tensor(Xtr), torch.as_tensor(Mtr), torch.as_tensor(ytr)
    for _ in range(epochs):
        opt.zero_grad()
        torch.nn.functional.cross_entropy(m(Xtr_t, Mtr_t), ytr_t).backward()
        opt.step()
    with torch.no_grad():
        return m(torch.as_tensor(Xte), torch.as_tensor(Mte)).argmax(-1).numpy()


READOUT_KINDS = ("mean_pool", "attention_all_tokens", "attention_target_tokens", "attention_distractor_tokens")


def part_d(c0, labels, args):
    from sklearn.model_selection import KFold
    layers = [int(l) for l in args.probe_layers.split(",")]
    N = len(labels)
    img, own = c0["tok_img"], c0["tok_owner"]
    y_all = {a: np.array([labels[i]["target"][a] for i in range(N)]) for a in ATTRS}
    results = {}
    for l in layers:
        X = token_table(c0, l)
        X = (X - X.mean(0)) / (X.std(0) + 1e-6)
        Xp, valid, idx = _pad_images(X, img, N)
        own_p = np.zeros(valid.shape, int)
        for i, t in enumerate(idx):
            own_p[i, :len(t)] = own[t]
        masks = {"attention_all_tokens": valid,
                 "attention_target_tokens": valid & (own_p == 1),
                 "attention_distractor_tokens": valid & (own_p == 2)}
        pooled = np.stack([Xp[i][valid[i]].mean(0) for i in range(N)])
        res = {}
        for a in ATTRS:
            classes, y = np.unique(y_all[a], return_inverse=True)
            res[f"{a}/majority"] = float(np.bincount(y).max() / N)
            accs = {k: [] for k in READOUT_KINDS}
            for tr, te in KFold(5, shuffle=True, random_state=args.seed).split(pooled):
                clf = LogisticRegression(max_iter=2000).fit(pooled[tr], y[tr])
                accs["mean_pool"].append(float((clf.predict(pooled[te]) == y[te]).mean()))
                for k, M in masks.items():
                    pred = _train_attn(Xp[tr], M[tr], y[tr], Xp[te], M[te], len(classes), args.seed)
                    accs[k].append(float((pred == y[te]).mean()))
            for k, v in accs.items():
                res[f"{a}/{k}"] = float(np.mean(v))
        results[f"L{l}"] = res
        print(f"L{l}: " + " ".join(f"{k}={v:.3f}" for k, v in res.items()))
    return results


def plot_readout_probe(res, label, out_path):
    layers = sorted(int(k[1:]) for k in res)
    styles = {"mean_pool": ("0.4", "-"), "attention_all_tokens": ("#1f77b4", "-"),
              "attention_target_tokens": ("#d62728", "-"), "attention_distractor_tokens": ("#1f77b4", "--")}
    names = {"mean_pool": "mean pooling", "attention_all_tokens": "attention, all tokens",
             "attention_target_tokens": "attention, target tokens only (oracle position)",
             "attention_distractor_tokens": "attention, distractor tokens only (control)"}
    fig, axes = plt.subplots(1, len(ATTRS), figsize=(4.6 * len(ATTRS), 4.2), sharey=True)
    for ax, a in zip(axes, ATTRS):
        for k in READOUT_KINDS:
            c, ls = styles[k]
            ax.plot(layers, [res[f"L{l}"][f"{a}/{k}"] for l in layers], color=c, ls=ls, marker="o", lw=2, label=names[k])
        ax.plot(layers, [res[f"L{l}"][f"{a}/majority"] for l in layers], color="k", ls=":", lw=1.5, label="majority baseline")
        ax.set_title(f"target {a}", fontsize=S["subplot_title_fontsize"])
        ax.set_xlabel("ViT block", fontsize=S["label_fontsize"])
        ax.set_xticks(layers)
        ax.set_ylim(0, 1.02)
        ax.tick_params(labelsize=S["tick_labelsize"])
    axes[0].set_ylabel("5-fold accuracy", fontsize=S["label_fontsize"])
    axes[0].legend(fontsize=10, loc="lower left")
    fig.suptitle(f"{label}: readouts on no-question tokens, 2-object images (n={res[f'L{layers[0]}'].get('n', 324)})",
                 fontsize=S["suptitle_fontsize"])
    fig.tight_layout()
    fig.savefig(out_path, dpi=S["dpi"], bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")

# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _layers_axis(ax, gca_layers):
    ax.set_xticks(range(NUM_LAYERS))
    ax.set_xlabel("ViT block")
    mark_gca_layers(ax)


def plot_projection_deltas(m, label, out_path, gca_layers):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]
    series = [("ref", "Δ_ref: target, refer target − refer distractor", CLUSTER_RGB["target"], "-"),
              ("nonref", "Δ_nonref: distractor, refer target − refer distractor", CLUSTER_RGB["distractor"], "-"),
              ("base_target_c1", "target: refer target − no question", CLUSTER_RGB["target"], "--"),
              ("base_distractor_c2", "distractor: refer distractor − no question", CLUSTER_RGB["distractor"], "--"),
              ("base_target_c3", "target: non-referring − no question", CLUSTER_RGB["target"], ":"),
              ("base_distractor_c3", "distractor: non-referring − no question", CLUSTER_RGB["distractor"], ":")]
    for key, lab, col, ls in series:
        if key not in m["delta"]:
            continue
        mu = [d["mean"] for d in m["delta"][key]]
        lo = [d["lo"] for d in m["delta"][key]]
        hi = [d["hi"] for d in m["delta"][key]]
        ax.plot(range(NUM_LAYERS), mu, ls, color=col, marker="o", markersize=3, label=lab)
        ax.fill_between(range(NUM_LAYERS), lo, hi, color=col, alpha=0.12, linewidth=0)
    ax.axhline(0, color="k", linewidth=0.6)
    ax.set_ylabel("Δ projection onto V", fontsize=10)
    _layers_axis(ax, gca_layers)
    ax.legend(fontsize=7, loc="best")
    ax = axes[1]
    ref = np.array(m["offset_norm_ref"])
    for key, lab, col, ls in series[:2]:
        if key in m["delta"]:
            mu = np.array([d["mean"] for d in m["delta"][key]])
            ax.plot(range(NUM_LAYERS), mu / ref, ls, color=col, marker="o", markersize=3, label=lab)
    ax.axhline(0, color="k", linewidth=0.6)
    ax.set_ylabel("Δ / mean offset norm", fontsize=10)
    _layers_axis(ax, gca_layers)
    ax.legend(fontsize=7)
    fig.suptitle(f"{label} — question effect on the object direction V "
                 "(V = no-question offset; bands = bootstrap 95% CI over images)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=S["dpi"], bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_patch_change(m, label, out_path, gca_layers):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, key, ylab in zip(axes, ("rel_norm", "cos_V", "cos_vimg"),
                             ("relative change ‖Δh‖ / ‖h‖",
                              "cos(Δh, V)", "cos(Δh, image's own object direction)")):
        for cond, pc in m["patch_change"].items():
            for name, vals in pc.items():
                if key == "cos_vimg" and name == "bg":
                    continue
                ax.plot(range(NUM_LAYERS), vals[key], COND_LS.get(cond, "-"), color=OWNER_RGB[name],
                        marker="o", markersize=3, label=f"{name}, {COND_LABEL[cond]}")
        ax.set_ylabel(ylab)
        _layers_axis(ax, gca_layers)
        if key != "rel_norm":
            ax.axhline(0, color="k", linewidth=0.6)
    axes[0].legend(fontsize=6, ncol=1)
    fig.suptitle(f"{label} — per-patch change induced by the question, grouped by background / target / distractor")
    fig.tight_layout()
    fig.savefig(out_path, dpi=S["dpi"], bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_gca(m, label, out_path, gca_layers):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, key, ylab in zip(axes, ("write_norm", "write_cos_V", "attn_ref"),
                             ("‖GCA write‖ per patch", "cos(GCA write, V)",
                              "patch → referent word attention")):
        for cond, g in m["gca"].items():
            for name, vals in g.items():
                ax.plot(gca_layers, vals[key], COND_LS.get(cond, "-"), color=OWNER_RGB[name],
                        marker="o", markersize=3, label=f"{name}, {COND_LABEL[cond]}")
        ax.set_ylabel(ylab)
        ax.set_xticks(gca_layers)
        ax.set_xlabel("GCA layer")
    axes[0].legend(fontsize=6)
    fig.suptitle(f"{label} — what the gated cross-attention writes, grouped by background / target / distractor")
    fig.tight_layout()
    fig.savefig(out_path, dpi=S["dpi"], bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_offsets_by_condition(m, label, out_path, gca_layers):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for cond, o in m["offset_norm"].items():
        ls = COND_LS.get(cond, "-") if cond != "c0" else "-"
        axes[0].plot(range(NUM_LAYERS), o["target"], ls, color=CLUSTER_RGB["target"],
                     marker="o", markersize=3, alpha=1 if cond != "c0" else 0.4, label=COND_LABEL[cond])
        axes[1].plot(range(NUM_LAYERS), o["distractor"], ls, color=CLUSTER_RGB["distractor"],
                     marker="o", markersize=3, alpha=1 if cond != "c0" else 0.4, label=COND_LABEL[cond])
        axes[2].plot(range(NUM_LAYERS), o["target_vs_distractor_cos"], ls, color="0.2",
                     marker="o", markersize=3, alpha=1 if cond != "c0" else 0.4, label=COND_LABEL[cond])
    for ax, t in zip(axes, ("target offset norm", "distractor offset norm", "cos(target offset, distractor offset)")):
        ax.set_ylabel(t)
        _layers_axis(ax, gca_layers)
        ax.legend(fontsize=7)
    fig.suptitle(f"{label} — offset (object − background) by condition")
    fig.tight_layout()
    fig.savefig(out_path, dpi=S["dpi"], bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_rsa(m, label, out_path, gca_layers):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, suffix, title in ((axes[0], "", "target patch mean (raw token)"),
                              (axes[1], "_offset", "target offset (patch mean − background mean)")):
        for cond, rs in m["rsa"].items():
            ls = COND_LS.get(cond, "-")
            ax.plot(range(NUM_LAYERS), rs["identity" + suffix], ls, color="#2ca02c", marker="o",
                    markersize=3, label=f"object identity, {COND_LABEL[cond]}")
            ax.plot(range(NUM_LAYERS), rs["position" + suffix], ls, color="#9467bd", marker="s",
                    markersize=3, label=f"position, {COND_LABEL[cond]}")
        ax.set_ylabel("Spearman(feature RDM, model RDM)", fontsize=10)
        ax.set_title(title, fontsize=10)
        _layers_axis(ax, gca_layers)
        ax.axhline(0, color="k", linewidth=0.6)
    axes[0].legend(fontsize=6)
    fig.suptitle(f"{label} — RSA of the target's features against an identity RDM and a position RDM")
    fig.tight_layout()
    fig.savefig(out_path, dpi=S["dpi"], bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_interventions(summary, label, out_path):
    rows = summary["rows"]
    alphas, layers = summary["alphas"], summary["layers"]
    fig, axes = plt.subplots(len(alphas), 2, figsize=(11, 3.2 * len(alphas)), squeeze=False)
    cmap = plt.get_cmap("tab10")
    for r, a in enumerate(alphas):
        for c, key in enumerate(("flip_rate", "logit_gap_mean")):
            ax = axes[r, c]
            for k, var in enumerate(summary["variants"]):
                ys = [next(x[key] for x in rows if x["layer"] == l and x["alpha"] == a and x["variant"] == var)
                      for l in layers]
                ax.plot(layers, ys, "-", color=cmap(k), marker="o", markersize=3, label=var)
            ax.set_xticks(layers)
            ax.set_xlabel("edited block")
            ax.set_ylabel(f"{'flip rate to B' if key == 'flip_rate' else 'logit(B) − logit(A)'}  (α={a})")
            mark_gca_layers(ax)
            if key == "flip_rate":
                ax.set_ylim(0, 1.02)
            else:
                ax.axhline(0, color="k", linewidth=0.6)
    axes[0, 0].legend(fontsize=6)
    fig.suptitle(f"{label} — additive colour-vector interventions (baseline acc c1 "
                 f"{summary['baseline_accuracy']['c1']:.2f}, c2 {summary['baseline_accuracy']['c2']:.2f})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=S["dpi"], bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_probes(res, label, out_path):
    layers = sorted(int(k[1:]) for k in res)
    tasks = ["bg_vs_object", "color", "shape", "material", "size", "referent"]
    fig, axes = plt.subplots(1, len(tasks), figsize=(3.1 * len(tasks), 3.6), sharey=True)
    ls = {"random_group_image": "-", "slot_loo": "--", "spatial_loo": ":"}
    for ax, task in zip(axes, tasks):
        for sname, l_ in ls.items():
            ys = [res[f"L{l}"].get(f"{task}/{sname}", np.nan) for l in layers]
            if np.all(np.isnan(ys)):
                continue
            ax.plot(layers, ys, l_, color="#1f77b4", marker="o", markersize=3, label=sname)
        if task == "referent":
            ys = [res[f"L{l}"].get("referent_c0_control/random_group_image", np.nan) for l in layers]
            ax.plot(layers, ys, "-", color="0.5", marker="x", markersize=3, label="no-question control")
        ax.set_title(task, fontsize=9)
        ax.set_xticks(layers)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("block")
    axes[0].set_ylabel("accuracy")
    axes[0].legend(fontsize=6)
    fig.suptitle(f"{label} — single-patch linear probes, three split schemes")
    fig.tight_layout()
    fig.savefig(out_path, dpi=S["dpi"], bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_token_norms(tn, label, out_path, grid):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    axes[0].plot(range(NUM_LAYERS), tn["median"], marker="o", markersize=3)
    axes[0].set_ylabel("median token norm (pre-norm residual)")
    axes[1].plot(range(NUM_LAYERS), tn["outlier_frac"], marker="o", markersize=3, label="fraction > 5× median")
    axes[1].plot(range(NUM_LAYERS), tn["outlier_bg_share"], marker="s", markersize=3, label="share of outliers that are background")
    axes[1].axhline(tn["bg_share"], color="0.5", linewidth=0.7, label="background share of all patches")
    axes[1].legend(fontsize=7)
    for ax in axes[:2]:
        ax.set_xticks(range(NUM_LAYERS))
        ax.set_xlabel("block")
    im = axes[2].imshow(np.array(tn["example_map"])[NUM_LAYERS - 1].reshape(grid, grid), cmap="magma")
    axes[2].set_title("token norm map, first image, last block", fontsize=9)
    axes[2].axis("off")
    fig.colorbar(im, ax=axes[2], shrink=0.8)
    fig.suptitle(f"{label} — token norms (Darcet et al. control)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=S["dpi"], bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Relational questions on 3-object scenes (--relational {same,spatial}).
# Roles per scene: A = the object named in the question (anchor), T = the
# answer of the clean run, D = the third object (answer of the corrupted run).
# Conditions: c0 no question; c1 clean run; c2 corrupted run (same template,
# other anchor for `same`, opposite relation for `spatial`).
# ---------------------------------------------------------------------------

ROLES = ("A", "T", "D")
ROLE_RGB = {"A": (0.58, 0.40, 0.74), "T": CLUSTER_RGB["target"], "D": CLUSTER_RGB["distractor"],
            "bg": OWNER_RGB["bg"]}
ROLE_LABEL = {"A": "anchor A (named in the question)", "T": "answer object T (clean run)",
              "D": "other object D (corrupted-run answer)", "bg": "background"}
SPATIAL_RELATIONS = ["left of", "right of", "in front of", "behind"]
SPATIAL_OPPOSITE = {"left of": "right of", "right of": "left of", "in front of": "behind", "behind": "in front of"}
SPATIAL_MARGIN = 2          # patches along the relation axis


def scene_objects(e):
    return [{k: e[k] for k in ATTRS}] + [{k: d[k] for k in ATTRS} for d in e["distractors"]]


def seg_ok3(e):
    """Three objects with pairwise-distinct, non-gray colours (colour segmentation)."""
    cols = [e["color"]] + [d["color"] for d in e["distractors"]]
    return len(e["distractors"]) == 2 and "gray" not in cols and len(set(cols)) == 3


def mask_centroids(owner, n_obj, grid):
    """(n_obj, 2) centroid (row, column) of each object's patches in the patch grid."""
    c = np.full((n_obj, 2), np.nan)
    for oid in range(n_obj):
        pos = np.nonzero(owner == oid + 1)[0]
        if len(pos):
            c[oid] = [(pos // grid).mean(), (pos % grid).mean()]
    return c


def same_question(attr, anchor_color):
    return f"There is another thing that is the same {attr} as the {anchor_color} object; what is its color?"


def spatial_question(rel, anchor_color):
    return f"What color is the object {rel} the {anchor_color} object?"


def check_same(objs, anchor, attr, answer):
    """Ground-truth rule: exactly one non-anchor object shares `attr` with the anchor, and it is `answer`."""
    return [j for j in range(len(objs)) if j != anchor and objs[j][attr] == objs[anchor][attr]] == [answer]


def spatial_axis_sign(rel):
    """(axis, sign): axis 1 = column for left/right, 0 = row for front/behind;
    sign so that `sign * (coord_object - coord_anchor) > 0` means the relation holds
    (front = larger row)."""
    axis = 1 if rel in ("left of", "right of") else 0
    sign = -1 if rel in ("left of", "behind") else 1
    return axis, sign


def spatial_holds(cent, j, anchor, rel, margin=SPATIAL_MARGIN):
    axis, sign = spatial_axis_sign(rel)
    return sign * (cent[j, axis] - cent[anchor, axis]) >= margin


def check_spatial(cent, anchor, rel, answer):
    """Ground-truth rule: exactly one non-anchor object satisfies `rel` w.r.t. the anchor
    (margin SPATIAL_MARGIN patches), and it is `answer`."""
    return [j for j in range(len(cent)) if j != anchor and spatial_holds(cent, j, anchor, rel)] == [answer]


def assign_roles(mode, objs, cent, grid):
    """First (anchor, attribute | relation) in fixed order that leaves exactly one
    answer object (and, for spatial, exactly one object on the opposite side);
    returns the role record or None."""
    n = len(objs)
    for anchor in range(n):
        others = [j for j in range(n) if j != anchor]
        if mode == "same":
            for a in ("shape", "material", "size"):
                shares = [j for j in others if objs[j][a] == objs[anchor][a]]
                if len(shares) != 1:
                    continue
                T = shares[0]
                D = [j for j in others if j != T][0]
                assert check_same(objs, anchor, a, T) and check_same(objs, T, a, anchor)
                return {"A": anchor, "T": T, "D": D, "attribute": a,
                        "questions": {"c1": same_question(a, objs[anchor]["color"]),
                                      "c2": same_question(a, objs[T]["color"])},
                        "referent_words": {"c1": objs[anchor]["color"], "c2": objs[T]["color"]},
                        "answers": {"c1": objs[T]["color"], "c2": objs[anchor]["color"]}}
        else:
            for rel in SPATIAL_RELATIONS:
                fwd = [j for j in others if spatial_holds(cent, j, anchor, rel)]
                opp = [j for j in others if spatial_holds(cent, j, anchor, SPATIAL_OPPOSITE[rel])]
                if len(fwd) != 1 or len(opp) != 1:
                    continue
                T, D = fwd[0], opp[0]
                assert check_spatial(cent, anchor, rel, T) and check_spatial(cent, anchor, SPATIAL_OPPOSITE[rel], D)
                axis, _ = spatial_axis_sign(rel)
                centre = (grid - 1) / 2
                return {"A": anchor, "T": T, "D": D, "relation": rel, "opposite": SPATIAL_OPPOSITE[rel],
                        "axis": "column" if axis == 1 else "row",
                        "same_side": bool((cent[T, axis] - centre) * (cent[D, axis] - centre) > 0),
                        "questions": {"c1": spatial_question(rel, objs[anchor]["color"]),
                                      "c2": spatial_question(SPATIAL_OPPOSITE[rel], objs[anchor]["color"])},
                        "referent_words": {"c1": objs[anchor]["color"], "c2": objs[anchor]["color"]},
                        "answers": {"c1": objs[T]["color"], "c2": objs[D]["color"]}}
    return None


def prepare_relational(entries, args, out_dir):
    """Segment the 3-object scenes, assign roles, write relational_records.json,
    owner.npy and masks_debug.png. Returns (records, images, owners)."""
    mode = args.relational
    cand = [i for i, e in enumerate(entries) if seg_ok3(e)]
    print(f"Scenes with three distinct non-gray colours: {len(cand)} / {len(entries)}")
    order = cand
    if args.n_pairs and args.n_pairs < len(cand):
        order = [int(i) for i in np.random.RandomState(args.seed).permutation(cand)]
    records, images, owners = [], [], []
    n_seg_fail = n_role_fail = 0
    for i in order:
        if args.n_pairs and len(records) >= args.n_pairs:
            break
        e = entries[i]
        try:
            im, ow, _ = build_masks(entries, [i], args.three_dir, args)
        except AssertionError as ex:
            print(f"  skip scene {i}: {ex}")
            n_seg_fail += 1
            continue
        ow = ow[0]
        objs = scene_objects(e)
        cent = mask_centroids(ow, len(objs), args.grid)
        roles = assign_roles(mode, objs, cent, args.grid)
        if roles is None:
            n_role_fail += 1
            continue
        bg_ok = np.nonzero(ow == 0)[0]
        bg_sample = sorted(int(p) for p in np.random.RandomState(args.seed + i)
                           .choice(bg_ok, min(args.bg_per_image, len(bg_ok)), replace=False))
        rec = dict(scene_index=i, filename=e["filename"], mode=mode, objects=objs,
                   centroids_row_col=[[float(v) for v in c] for c in cent],
                   n_patches=[int((ow == k + 1).sum()) for k in range(len(objs))],
                   bg_sample=bg_sample, **roles)
        records.append(rec)
        images.append(im[0])
        owners.append(ow)
    idx = np.argsort([r["scene_index"] for r in records])
    records = [records[k] for k in idx]
    images = [images[k] for k in idx]
    owners = [owners[k] for k in idx]
    print(f"Accepted scenes ({mode}): {len(records)}  (segmentation failures {n_seg_fail}, "
          f"no valid role assignment {n_role_fail})")
    assert records, "no scene accepted"
    with open(out_dir / "relational_records.json", "w") as f:
        json.dump(records, f, indent=1)
    np.save(out_dir / "owner.npy", np.stack(owners))
    n_dbg = min(40, len(records))
    save_masks_debug(images[:n_dbg], np.stack(owners[:n_dbg]), entries,
                     [r["scene_index"] for r in records[:n_dbg]], args.grid, out_dir / "masks_debug.png")
    return records, images, owners


def _role_index(records, role):
    return np.array([r[role] for r in records], dtype=int)


def relational_projection(caches, records, u_color, gca_layers):
    """Per block and role: projection of the object's mean-token change (c1 − c0,
    c2 − c0) onto that object's own colour direction u, and the object-vector
    norm ratio ‖o(c)‖ / ‖o(c0)‖ with o = obj_mean − bg_mean."""
    N = len(records)
    off = {c: offsets_from_cache(caches[c]) for c in caches}                  # (N, 3, 12, D)
    om = {c: caches[c]["obj_mean"].astype(np.float32) for c in caches}
    ar = np.arange(N)
    res = {"n_scenes": N, "proj_delta": {}, "proj_abs": {}, "norm_ratio": {}, "n_with_direction": {}}
    for role in ROLES:
        idx = _role_index(records, role)
        cols = [r["objects"][j]["color"] for r, j in zip(records, idx)]
        U = np.stack([u_color[c] if c in u_color else np.full((NUM_LAYERS, om["c0"].shape[-1]), np.nan, np.float32)
                      for c in cols])                                            # (N, 12, D)
        res["n_with_direction"][role] = int(sum(c in u_color for c in cols))
        for cond in caches:
            p_abs = (om[cond][ar, idx] * U).sum(-1)                                # (N, 12)
            res["proj_abs"][f"{cond}_{role}"] = [_boot(p_abs[:, l]) for l in range(NUM_LAYERS)]
            if cond == "c0":
                continue
            d = ((om[cond][ar, idx] - om["c0"][ar, idx]) * U).sum(-1)
            res["proj_delta"][f"{cond}_{role}"] = [_boot(d[:, l]) for l in range(NUM_LAYERS)]
            ratio = (np.linalg.norm(off[cond][ar, idx], axis=-1)
                     / (np.linalg.norm(off["c0"][ar, idx], axis=-1) + 1e-8))
            res["norm_ratio"][f"{cond}_{role}"] = [_boot(ratio[:, l]) for l in range(NUM_LAYERS)]
    # decoder accuracy (first-token argmax, recorded by run_relational_transplant)
    if all("pred_c1" in r for r in records):
        ok1 = np.array([r["pred_c1"] == r["answers"]["c1"] for r in records])
        ok2 = np.array([r["pred_c2"] == r["answers"]["c2"] for r in records])
        res["accuracy"] = {"c1": float(ok1.mean()), "c2": float(ok2.mean()), "n": N}
        if records[0]["mode"] == "spatial":
            ss = np.array([r["same_side"] for r in records])
            res["accuracy"]["same_side"] = {"n": int(ss.sum()), "c1": float(ok1[ss].mean()) if ss.any() else float("nan"),
                                            "c2": float(ok2[ss].mean()) if ss.any() else float("nan")}
            res["accuracy"]["opposite_side"] = {"n": int((~ss).sum()), "c1": float(ok1[~ss].mean()) if (~ss).any() else float("nan"),
                                                "c2": float(ok2[~ss].mean()) if (~ss).any() else float("nan")}
    return res


def plot_relational_projection(res, mode, label, out_path, gca_layers):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
    x = range(NUM_LAYERS)
    for ax, key, ylab, ref in ((axes[0], "proj_delta", "Δ projection onto own colour direction u", 0.0),
                               (axes[1], "proj_abs", "projection onto own colour direction u", None),
                               (axes[2], "norm_ratio", "‖o(question)‖ / ‖o(no question)‖", 1.0)):
        for cond in ("c0", "c1", "c2"):
            for role in ROLES:
                k = f"{cond}_{role}"
                if k not in res[key]:
                    continue
                d = res[key][k]
                m = [q["mean"] for q in d]; lo = [q["lo"] for q in d]; hi = [q["hi"] for q in d]
                ls = {"c0": ":", "c1": "-", "c2": "--"}[cond]
                cl = {"c0": "no question", "c1": "clean run", "c2": "corrupted run"}[cond]
                ax.plot(x, m, ls, color=ROLE_RGB[role], marker="o", markersize=3, label=f"{ROLE_LABEL[role]}, {cl}")
                ax.fill_between(x, lo, hi, color=ROLE_RGB[role], alpha=0.10, linewidth=0)
        if ref is not None:
            ax.axhline(ref, color="k", linewidth=0.6)
        ax.set_ylabel(ylab, fontsize=10)
        _layers_axis(ax, gca_layers)
        ax.legend(fontsize=6)
    axes[0].set_title("mean token change (question − no question) per role", fontsize=10)
    axes[1].set_title("absolute projection per role and condition", fontsize=10)
    axes[2].set_title("object-vector norm ratio per role", fontsize=10)
    acc = res.get("accuracy")
    acc_txt = ""
    if acc:
        acc_txt = f"; decoder accuracy clean run {acc['c1']:.2f}, corrupted run {acc['c2']:.2f} (n={acc['n']})"
        if "same_side" in acc:
            acc_txt += (f"; T and D on the same side of the image centre: clean {acc['same_side']['c1']:.2f} "
                        f"(n={acc['same_side']['n']}), opposite sides: clean {acc['opposite_side']['c1']:.2f} "
                        f"(n={acc['opposite_side']['n']})")
    fig.suptitle(f"{label} — relational question ({mode}), 3-object scenes: h = b + o; o projected onto the "
                 f"object's own colour direction u{acc_txt}", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=S["dpi"], bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def relational_gca_write(caches, records, gca_layers):
    """Mean GCA write norm per GCA layer per role (A / T / D / background) under c1 and c2."""
    res = {"gca_layers": list(gca_layers), "write_norm": {}, "attn_anchor": {}}
    for cond in ("c1", "c2"):
        wn = caches[cond]["gca_write_norm"].astype(np.float32)                    # (N, 6, P)
        ar = caches[cond]["gca_attn_ref"].astype(np.float32)
        owner = caches[cond]["owner"]
        for role in ROLES + ("bg",):
            m = owner == 0 if role == "bg" else owner == (_role_index(records, role) + 1)[:, None]
            res["write_norm"][f"{cond}_{role}"] = [float(wn[:, k][m].mean()) for k in range(len(gca_layers))]
            res["attn_anchor"][f"{cond}_{role}"] = [float(ar[:, k][m].mean()) for k in range(len(gca_layers))]
    return res


def plot_relational_gca_write(res, mode, label, out_path):
    gl = res["gca_layers"]
    fig, ax = plt.subplots(1, 1, figsize=(6.5, 4.2))
    for cond, ls, cl in (("c1", "-", "clean run"), ("c2", "--", "corrupted run")):
        for role in ROLES + ("bg",):
            ax.plot(gl, res["write_norm"][f"{cond}_{role}"], ls, color=ROLE_RGB[role], marker="o",
                    markersize=3, label=f"{ROLE_LABEL[role]}, {cl}")
    ax.set_xticks(gl)
    ax.set_xlabel("GCA layer")
    ax.set_ylabel("‖GCA write‖ per patch")
    ax.legend(fontsize=6)
    fig.suptitle(f"{label} — relational question ({mode}): what the gated cross-attention writes, by role", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=S["dpi"], bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


@torch.no_grad()
def run_relational_transplant(out_dir, args, state, images, owners, records):
    """Baseline decoder answers under c0/c1/c2 (stored into the records), then the
    causal test: at block l, one patch group of the clean run (c1) is replaced by
    the corrupted run's (c2) tokens; control: replaced by the clean run's own tokens."""
    model, steervit, device, tf = state["model"], state["steervit"], state["device"], state["transform"]
    trunk = steervit.vision_model.trunk
    prefix = trunk.num_prefix_tokens
    inv = {v: k for k, v in model.vocab.items()}
    N, bs = len(records), args.batch_size
    imgs_t = torch.stack([tf(im) for im in images])
    owner_t = torch.from_numpy(np.stack(owners))
    role_idx = {role: _role_index(records, role) for role in ROLES}
    colour_id = {role: np.array([model.vocab[r["objects"][j]["color"]] for r, j in zip(records, role_idx[role])])
                 for role in ROLES}
    ans_id = {c: np.array([model.vocab[r["answers"][c]] for r in records]) for c in ("c1", "c2")}

    base = {}
    for cond in ("c0", "c1", "c2"):
        preds = []
        for s in range(0, N, bs):
            e = min(s + bs, N)
            qs = None if cond == "c0" else [records[i]["questions"][cond] for i in range(s, e)]
            ims = imgs_t[s:e].to(device)
            feats = steervit.forward(ims, qs)
            bos = torch.full((e - s, 1), model.vocab["<bos>"], dtype=torch.long, device=device)
            preds.append(model.decoder(bos, feats[:, prefix:, :])[:, 0, :].argmax(-1).cpu().numpy())
        base[cond] = np.concatenate(preds)
    gen = model.generate(imgs_t[:min(8, N)].to(device), [records[i]["questions"]["c1"] for i in range(min(8, N))])
    print(f"generate() vs first-token argmax on 8 scenes: {gen} | {[inv.get(int(t), '?') for t in base['c1'][:8]]}")
    for i, r in enumerate(records):
        for cond in ("c0", "c1", "c2"):
            r[f"pred_{cond}"] = inv.get(int(base[cond][i]), "?")
    with open(out_dir / "relational_records.json", "w") as f:
        json.dump(records, f, indent=1)
    acc = {c: float((base[c] == ans_id[c]).mean()) for c in ("c1", "c2")}
    print(f"baseline accuracy: clean run {acc['c1']:.3f}  corrupted run {acc['c2']:.3f}  (n={N}); "
          f"no question answers T {float((base['c0'] == colour_id['T']).mean()):.2f} "
          f"A {float((base['c0'] == colour_id['A']).mean()):.2f} D {float((base['c0'] == colour_id['D']).mean()):.2f}")

    ok = (base["c1"] == ans_id["c1"]) & (base["c2"] == ans_id["c2"])
    print(f"token replacement on {int(ok.sum())} scenes (both runs answered correctly)")
    groups = ("A", "T", "D", "bg")
    donors = ("c2", "c1")
    counts = {(d, g, l): np.zeros(3, int) for d in donors for g in groups for l in range(NUM_LAYERS)}   # [T, A, D]
    agree = {(d, g, l): 0 for d in donors for g in groups for l in range(NUM_LAYERS)}
    for s in range(0, N, bs):
        e = min(s + bs, N)
        idx = list(range(s, e))
        ims = imgs_t[s:e].to(device)
        ow = owner_t[s:e].to(device)
        cap_out = {}
        for c in donors:
            qs = [records[i]["questions"][c] for i in idx]
            with BlockCapture(trunk) as cap:
                feats = steervit.forward(ims, qs)
            cap_out[c] = cap.out
            if c == "c1":
                assert torch.allclose(trunk.norm(cap.out[NUM_LAYERS - 1]), feats[:, prefix:, :], atol=1e-4)
        masks = {"bg": ow == 0}
        for role in ROLES:
            masks[role] = ow == torch.from_numpy(role_idx[role][s:e] + 1).to(device)[:, None]
        qs1 = [records[i]["questions"]["c1"] for i in idx]
        for d in donors:
            for g in groups:
                for l in range(NUM_LAYERS):
                    with TokenSwapper(trunk, l, cap_out[d][l], masks[g]):
                        pred = first_token_logits(model, steervit, ims, qs1).argmax(-1).cpu().numpy()
                    for j, i in enumerate(idx):
                        agree[(d, g, l)] += int(pred[j] == base["c1"][i])
                        if ok[i]:
                            k = (0 if pred[j] == colour_id["T"][i] else 1 if pred[j] == colour_id["A"][i]
                                 else 2 if pred[j] == colour_id["D"][i] else 3)
                            if k < 3:
                                counts[(d, g, l)][k] += 1
        print(f"  replacements {e}/{N}", flush=True)
    rows = []
    for d in donors:
        for g in groups:
            for l in range(NUM_LAYERS):
                cnt = counts[(d, g, l)]
                n = int(ok.sum())
                rows.append({"donor": d, "group": g, "layer": l, "n": n,
                             "p_T": cnt[0] / max(n, 1), "p_A": cnt[1] / max(n, 1), "p_D": cnt[2] / max(n, 1),
                             "p_other": 1 - cnt.sum() / max(n, 1),
                             "agree_with_clean": agree[(d, g, l)] / N})
    ctrl = [r for r in rows if r["donor"] == "c1"]
    bad = [(r["group"], r["layer"], r["agree_with_clean"]) for r in ctrl if r["agree_with_clean"] != 1.0]
    print(f"clean-self control: {len(ctrl)} (group, block) cells; agreement with the clean run "
          f"{'1.00 everywhere' if not bad else f'NOT 1.0 at {bad}'}")
    assert not bad, "replacing tokens from the clean run itself must reproduce the clean run"
    for g in groups:
        rr = [r for r in rows if r["donor"] == "c2" and r["group"] == g]
        print(f"replace {g:<3} from corrupted run: P(T colour) " + " ".join(f"{r['p_T']:.2f}" for r in rr)
              + " | P(A colour) " + " ".join(f"{r['p_A']:.2f}" for r in rr)
              + " | P(D colour) " + " ".join(f"{r['p_D']:.2f}" for r in rr), flush=True)
    res = {"mode": records[0]["mode"], "n_scenes": N, "n_scenes_ok": int(ok.sum()), "baseline_accuracy": acc,
           "receiver": "c1 (clean run)", "donors": list(donors), "groups": list(groups), "rows": rows}
    with open(out_dir / "relational_transplant.json", "w") as f:
        json.dump(res, f, indent=1)
    print(f"Saved: {out_dir / 'relational_transplant.json'}")
    return res


def plot_relational_transplant(res, label, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
    styles = {"A": (ROLE_RGB["A"], "^"), "T": (ROLE_RGB["T"], "o"), "D": (ROLE_RGB["D"], "v"), "bg": (ROLE_RGB["bg"], "s")}
    mode = res["mode"]
    corrupted_answer = "A" if mode == "same" else "D"
    for ax, key, role in zip(axes, ("p_T", "p_A", "p_D"), ROLES):
        for g, (col, mk) in styles.items():
            rr = [r for r in res["rows"] if r["donor"] == "c2" and r["group"] == g]
            ax.plot([r["layer"] for r in rr], [r[key] for r in rr], "-", color=col, marker=mk, markersize=3,
                    label=f"replaced: {ROLE_LABEL[g]} patches" if g != "bg" else "replaced: background patches")
            if key == "p_T":
                rc = [r for r in res["rows"] if r["donor"] == "c1" and r["group"] == g]
                ax.plot([r["layer"] for r in rc], [r[key] for r in rc], ":", color=col, linewidth=0.8,
                        label="control: replaced from the clean run itself" if g == "bg" else None)
        ax.set_ylim(-0.02, 1.02)
        ax.set_ylabel(f"P(answer = {role}'s colour)" + (" (clean-run answer)" if role == "T" else
                      " (corrupted-run answer)" if role == corrupted_answer else ""), fontsize=10)
        ax.set_xlabel("ViT block at which the tokens are replaced")
        ax.set_xticks(range(NUM_LAYERS))
        mark_gca_layers(ax)
        ax.legend(fontsize=6)
    fig.suptitle(f"{label} — relational question ({mode}): one block's patch tokens of the clean run replaced by the "
                 f"corrupted run's tokens (n={res['n_scenes_ok']} scenes with both runs correct; baseline accuracy "
                 f"clean {res['baseline_accuracy']['c1']:.2f}, corrupted {res['baseline_accuracy']['c2']:.2f})", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=S["dpi"], bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def _r2(X, y):
    X1 = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
    ss_res = float(((y - X1 @ beta) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return 1 - ss_res / max(ss_tot, 1e-12)


def relational_write_position(caches, records, gca_layers, grid):
    """Spatial only. Background patches pooled across scenes: per GCA layer, R² of
    (c1 − c2) GCA write norm, and of the c1 write norm alone, regressed on the
    patch's absolute coordinate along the relation axis, on its coordinate
    relative to the anchor centroid, and on both. `oriented`: coordinates signed
    so that + is the direction of the clean-run relation; `raw`: unsigned grid
    coordinate (row or column)."""
    wn1 = caches["c1"]["gca_write_norm"].astype(np.float32)
    wn2 = caches["c2"]["gca_write_norm"].astype(np.float32)
    owner = caches["c1"]["owner"]
    P = grid * grid
    coord = {0: np.arange(P) // grid, 1: np.arange(P) % grid}
    xs_abs, xs_rel, sgn, y_diff, y_c1 = [], [], [], [], []
    for i, r in enumerate(records):
        axis, sign = spatial_axis_sign(r["relation"])
        bg = owner[i] == 0
        c = coord[axis][bg].astype(np.float32)
        xs_abs.append(c)
        xs_rel.append(c - r["centroids_row_col"][r["A"]][axis])
        sgn.append(np.full(bg.sum(), sign, np.float32))
        y_diff.append(wn1[i][:, bg] - wn2[i][:, bg])
        y_c1.append(wn1[i][:, bg])
    xa, xr, sg = np.concatenate(xs_abs), np.concatenate(xs_rel), np.concatenate(sgn)
    Y = {"diff_c1_c2": np.concatenate(y_diff, 1), "c1": np.concatenate(y_c1, 1)}        # (6, T)
    res = {"gca_layers": list(gca_layers), "n_scenes": len(records), "n_bg_patches": int(len(xa)), "r2": {}}
    for orient, (a, rl) in (("oriented", (sg * xa, sg * xr)), ("raw", (xa, xr))):
        for yname, ys in Y.items():
            for design, X in (("absolute", a[:, None]), ("relative_to_anchor", rl[:, None]),
                              ("both", np.column_stack([a, rl]))):
                res["r2"][f"{orient}/{yname}/{design}"] = [_r2(X, ys[k]) for k in range(len(gca_layers))]
    for k in sorted(res["r2"]):
        if k.startswith("oriented"):
            print(f"write-position R² {k:<40} " + " ".join(f"{v:.3f}" for v in res["r2"][k]))
    return res


def plot_relational_write_position(res, label, out_path):
    gl = res["gca_layers"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    styles = {"absolute": ("#1f77b4", "o", "absolute coordinate along the relation axis"),
              "relative_to_anchor": ("#d62728", "^", "coordinate relative to the anchor centroid"),
              "both": ("0.3", "s", "both")}
    for ax, yname, title in ((axes[0], "diff_c1_c2", "GCA write norm, clean run − corrupted run (opposite relation words)"),
                             (axes[1], "c1", "GCA write norm, clean run")):
        for design, (col, mk, lab) in styles.items():
            ax.plot(gl, res["r2"][f"oriented/{yname}/{design}"], "-", color=col, marker=mk, markersize=4, label=lab)
            ax.plot(gl, res["r2"][f"raw/{yname}/{design}"], ":", color=col, marker=mk, markersize=3, alpha=0.6,
                    label=f"{lab} (unsigned)")
        ax.set_xticks(gl)
        ax.set_xlabel("GCA layer")
        ax.set_title(title, fontsize=9)
        ax.set_ylim(-0.02, 1.02)
    axes[0].set_ylabel("R² over background patches")
    axes[0].legend(fontsize=6)
    fig.suptitle(f"{label} — relational question (spatial): does the GCA write to background patches depend on "
                 f"position? (n={res['n_scenes']} scenes, {res['n_bg_patches']} background patches)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=S["dpi"], bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def run_relational(args, out_dir, label):
    mode = args.relational
    entries = load_entries(args.three_dir)
    n3 = sum(len(e["distractors"]) == 2 for e in entries)
    print(f"3-object scenes in {args.three_dir}: {n3} / {len(entries)}")
    state = {}
    if args.replot:
        with open(out_dir / "relational_records.json") as f:
            records = json.load(f)
        print(f"--replot: {len(records)} recorded scenes")
    else:
        records, images, owners = prepare_relational(entries, args, out_dir)
        for cond in ("c0", "c1", "c2"):
            extract_condition_sparse(out_dir, cond, images, owners, records, args, state, n_obj=3)
        ensure_model(state, args)
        run_relational_transplant(out_dir, args, state, images, owners, records)
    caches = {c: load_sparse(out_dir, c) for c in ("c0", "c1", "c2")}
    gca_layers = [int(l) for l in caches["c0"]["gca_layers"]]
    d_dir = Path(args.directions_dir) / "n1"
    u_color = attribute_directions(load_sparse(d_dir, "c0"), load_labels(d_dir))["color"]
    print(f"colour directions u from {d_dir}: {sorted(u_color)}")
    res = relational_projection(caches, records, u_color, gca_layers)
    for key in sorted(res["proj_delta"]):
        print(f"proj Δ {key:<6} " + " ".join(f"{q['mean']:+.2f}" for q in res["proj_delta"][key]))
    for key in sorted(res["norm_ratio"]):
        print(f"norm ratio {key:<6} " + " ".join(f"{q['mean']:.2f}" for q in res["norm_ratio"][key]))
    if "accuracy" in res:
        print(f"accuracy: {res['accuracy']}")
    with open(out_dir / "relational_projection.json", "w") as f:
        json.dump(res, f, indent=1)
    plot_relational_projection(res, mode, label, out_dir / "relational_projection.png", gca_layers)
    res = relational_gca_write(caches, records, gca_layers)
    for key in sorted(res["write_norm"]):
        print(f"GCA write norm {key:<6} " + " ".join(f"{v:.2f}" for v in res["write_norm"][key]))
    with open(out_dir / "relational_gca_write.json", "w") as f:
        json.dump(res, f, indent=1)
    plot_relational_gca_write(res, mode, label, out_dir / "relational_gca_write.png")
    if (out_dir / "relational_transplant.json").exists():
        with open(out_dir / "relational_transplant.json") as f:
            plot_relational_transplant(json.load(f), label, out_dir / "relational_transplant.png")
    if mode == "spatial":
        res = relational_write_position(caches, records, gca_layers, args.grid)
        with open(out_dir / "relational_write_position.json", "w") as f:
            json.dump(res, f, indent=1)
        plot_relational_write_position(res, label, out_dir / "relational_write_position.png")


# ---------------------------------------------------------------------------
# Mirror model (--mirror).  The ViT is vanilla; the gated cross-attention sits
# inside RoBERTa-large (text tokens = query, ViT patch tokens h = key/value);
# the 1-layer decoder reads the connector-projected RoBERTa tokens.  Same n2
# pairs, masks and questions as the ViT-side pipeline: c1 = question refers to
# the target, c2 = refers to the distractor; c0 = the c1 question with the
# cross-attention disabled (text only, no image), the no-image control for the
# probes.  The vanilla patch cache n2/feats_c0.npz (shared SparseExtractor)
# gives the object-identity contrast for the fetch test.
# ---------------------------------------------------------------------------

MIRROR_TOKEN_TYPES = ("referent", "bos", "last", "mean_all")
MIRROR_TOKEN_LABEL = {"referent": "referent word token", "bos": "first token <s>",
                      "last": "last token </s>", "mean_all": "mean over all question tokens"}
MIRROR_READOUTS = ("mean", "bos", "last")
MIRROR_READOUT_LABEL = {"mean": "mean over question tokens", "bos": "first token <s>", "last": "last token </s>"}


class MirrorCapture:
    """Around one encode_text_mirror call: head-mean attention map (B, T, P) of
    every text-GCA layer, and the RoBERTa hidden state after every layer
    (index 0 = embeddings, l = output of encoder layer l-1; state l is captured
    before the text-GCA hook at layer l, if any, modifies it)."""

    def __init__(self, steervit):
        self.sv = steervit
        self.keys = list(steervit.text_gca.keys())
        self.hidden, self.attn, self.hs = {}, {}, []

    def __enter__(self):
        tm = self.sv.text_model
        self.hs.append(tm.embeddings.register_forward_hook(
            lambda m, i, o: self.hidden.__setitem__(0, o.detach())))
        for li, layer in enumerate(tm.encoder.layer):
            def mk(li):
                def fn(m, i, o):
                    self.hidden[li + 1] = (o[0] if isinstance(o, tuple) else o).detach()
                return fn
            self.hs.append(layer.register_forward_hook(mk(li)))
        for k in self.keys:
            self.sv.text_gca[k].cross_attn.save_attn = True
        return self

    def __exit__(self, *a):
        for h in self.hs:
            h.remove()
        for k in self.keys:
            ca = self.sv.text_gca[k].cross_attn
            if ca.attn_map is not None:
                self.attn[int(k)] = ca.attn_map.float().mean(1)              # (B, T, P)
            ca.attn_map = None
            ca.save_attn = False


@torch.no_grad()
def mirror_forward(state, images, questions, use_image=True, kv_delta=None, kv_mask=None, alpha=0.0):
    """Vanilla ViT patches -> (optionally edited) key/value -> RoBERTa with the
    text-GCA hooks -> decoder first-token logits and decoder attention over the
    text tokens.  kv_delta (B, D) is added to the patches selected by kv_mask
    (B, P) scaled by alpha, in the key/value tensor only."""
    model, sv, device = state["model"], state["steervit"], state["device"]
    prefix = sv.vision_model.trunk.num_prefix_tokens
    kv = sv.forward(images, None)[:, prefix:, :]
    if kv_delta is not None:
        kv = kv + alpha * kv_delta[:, None, :].to(kv.dtype) * kv_mask[:, :, None].to(kv.dtype)
    with MirrorCapture(sv) as cap:
        mem, mask = sv.encode_text_mirror(list(questions), kv if use_image else None)
    bos = torch.full((images.shape[0], 1), model.vocab["<bos>"], dtype=torch.long, device=device)
    with DecoderAttention(model.decoder) as da:
        lg = model.decoder(bos, mem, memory_key_padding_mask=~mask)[:, 0, :]
    return {"logits": lg, "kv": kv, "mask": mask, "attn": cap.attn, "hidden": cap.hidden,
            "dec_attn": da.weights[0][:, :, 0, :]}                                     # (B, H, T)


def mirror_token_index(ext, questions, words):
    """(referent, last </s>, queried-attribute word) token positions, RoBERTa BPE."""
    ridx, lidx = ext.referent_token_index(questions, words)
    qidx, _ = ext.referent_token_index(questions, [QUERIED] * len(questions))
    return np.array(ridx), np.array(lidx), np.array(qidx)


def _owner_mass(rows, ow):
    """rows (B, P) attention rows, ow (B, P) owner -> (B, 3) mass on bg / target / distractor."""
    return torch.stack([(rows * (ow == k)).sum(-1) for k in range(3)], -1)


def _referent_mass(out, ridx, ow, gca_layers):
    """(B, n_gca, 3) mass of the referent token's attention by owner, per text-GCA layer."""
    b = torch.arange(ow.shape[0], device=ow.device)
    r = torch.as_tensor(ridx, device=ow.device)
    return torch.stack([_owner_mass(out["attn"][l][b, r], ow) for l in gca_layers], 1)


@torch.no_grad()
def run_mirror_pass(out_dir, args, state, images_n2, owners_n2, labels_n2):
    """One pass per condition: caches text hidden states (text_feats_c*.npz),
    decoder answers, text-GCA attention mass by owner (mirror_attention.json,
    with the error split) and decoder attention over the text tokens
    (mirror_decoder_attention.json)."""
    model, sv, device, tf, ext = state["model"], state["steervit"], state["device"], state["transform"], state["extractor"]
    inv = {v: k for k, v in model.vocab.items()}
    gca_layers = sorted(int(k) for k in sv.text_gca.keys())
    n_hid = len(sv.text_model.encoder.layer) + 1
    N, bs = len(labels_n2), args.batch_size
    imgs_t = torch.stack([tf(im) for im in images_n2])
    owner_t = torch.from_numpy(np.stack(owners_n2))
    A_id = np.array([model.vocab[r["target"][QUERIED]] for r in labels_n2])
    Ad_id = np.array([model.vocab[r["distractors"][0][QUERIED]] for r in labels_n2])
    qs_all = {c: [r["questions"][c] for r in labels_n2] for c in ("c1", "c2")}
    words_all = {c: [r["referent_words"][c] for r in labels_n2] for c in ("c1", "c2")}
    Tmax = max(ext.tokenizer(qs_all[c], padding=True, return_tensors="pt")["input_ids"].shape[1] for c in qs_all)
    D = sv.text_dim
    H = model.decoder.layers[0].base_layer.multihead_attn.num_heads
    preds, mass, dec_mass = {}, {}, {}
    for cond in ("c0", "c1", "c2"):
        q_cond = "c1" if cond == "c0" else cond            # c0 = c1 question, cross-attention disabled
        hid = np.zeros((N, n_hid, Tmax, D), np.float16)
        tok_mask = np.zeros((N, Tmax), bool)
        ids = np.zeros((N, Tmax), np.int32)
        pred = np.zeros(N, int)
        dec = np.zeros((N, H, Tmax), np.float32)
        m_tok = np.zeros((N, len(gca_layers), len(MIRROR_TOKEN_TYPES), 3), np.float32)
        ridx_all, lidx_all, qidx_all = mirror_token_index(ext, qs_all[q_cond], words_all[q_cond])
        for s in range(0, N, bs):
            e = min(s + bs, N)
            qs = qs_all[q_cond][s:e]
            ims = imgs_t[s:e].to(device)
            ow = owner_t[s:e].to(device)
            out = mirror_forward(state, ims, qs, use_image=cond != "c0")
            if cond == "c1" and s == 0:
                # sanity: the key/value patch tokens do not depend on the question
                out2 = mirror_forward(state, ims, qs_all["c2"][s:e])
                diff = float((out["kv"] - out2["kv"]).abs().max())
                assert diff == 0.0, f"key/value patches differ between c1 and c2: max abs diff {diff}"
                print(f"sanity: key/value patch tokens identical under c1 and c2 (max abs diff {diff})")
            ridx, lidx = ext.referent_token_index(qs, words_all[q_cond][s:e])
            assert list(ridx) == list(ridx_all[s:e]) and list(lidx) == list(lidx_all[s:e])
            T = out["mask"].shape[1]
            mk = out["mask"].cpu().numpy()
            tok_mask[s:e, :T] = mk
            ids[s:e, :T] = ext.tokenizer(qs, padding=True, return_tensors="pt")["input_ids"].numpy()
            for l in range(n_hid):
                hid[s:e, l, :T] = out["hidden"][l].float().cpu().numpy().astype(np.float16)
            pred[s:e] = out["logits"].argmax(-1).cpu().numpy()
            dec[s:e, :, :T] = out["dec_attn"].cpu().numpy()
            if cond != "c0":
                b = torch.arange(e - s, device=device)
                mkt = out["mask"].float()
                for k, l in enumerate(gca_layers):
                    am = out["attn"][l]                                          # (B, T, P)
                    rows = {"referent": am[b, torch.as_tensor(ridx, device=device)],
                            "bos": am[:, 0],
                            "last": am[b, torch.as_tensor(lidx, device=device)],
                            "mean_all": (am * mkt[:, :, None]).sum(1) / mkt.sum(1)[:, None]}
                    for t, name in enumerate(MIRROR_TOKEN_TYPES):
                        m_tok[s:e, k, t] = _owner_mass(rows[name], ow).cpu().numpy()
            print(f"  {cond} {e}/{N}", flush=True)
        np.savez(out_dir / f"text_feats_{cond}.npz", hid=hid, mask=tok_mask, ids=ids, pred=pred,
                 ref_idx=ridx_all, last_idx=lidx_all, q_idx=qidx_all, gca_layers=np.array(gca_layers))
        print(f"Saved: {out_dir / f'text_feats_{cond}.npz'} ({hid.nbytes / 1e9:.2f} GB)")
        preds[cond], mass[cond], dec_mass[cond] = pred, m_tok, dec
        if cond == "c1":
            gen = model.generate(imgs_t[:min(8, N)].to(device), qs_all["c1"][:min(8, N)])
            print(f"generate() vs first-token argmax on 8 images: {gen} | "
                  f"{[inv.get(int(t), '?') for t in pred[:min(8, N)]]}")
    acc = {"c1": float((preds["c1"] == A_id).mean()), "c2": float((preds["c2"] == Ad_id).mean()),
           "c0_text_only_says_target": float((preds["c0"] == A_id).mean())}
    print(f"decoder accuracy: c1 {acc['c1']:.3f}  c2 {acc['c2']:.3f}; text only (no image) -> target colour "
          f"{acc['c0_text_only_says_target']:.3f}")

    # ---- text-GCA attention mass by owner (2) + error split (6) ----
    attention = {"n_images": N, "text_gca_layers": gca_layers, "token_types": list(MIRROR_TOKEN_TYPES),
                 "tokens_per_owner_mean": {n: float((owner_t == k).float().sum(1).mean()) for k, n in
                                           enumerate(("bg", "target", "distractor"))},
                 "decoder_accuracy": acc, "conditions": {}, "error_split": {}}
    for cond in ("c1", "c2"):
        attention["conditions"][cond] = {
            t: {n: mass[cond][:, :, ti, k].mean(0).tolist() for k, n in enumerate(("bg", "target", "distractor"))}
            for ti, t in enumerate(MIRROR_TOKEN_TYPES)}
        for t in MIRROR_TOKEN_TYPES:
            a = attention["conditions"][cond][t]
            print(f"{cond} {t:<9} mass on target " + " ".join(f"{v:.3f}" for v in a["target"]) +
                  " | distractor " + " ".join(f"{v:.3f}" for v in a["distractor"]), flush=True)
    correct = preds["c1"] == A_id
    ref_last = mass["c1"][:, -1, 0]                                              # (N, 3) referent token, last GCA layer
    for name, sel in (("correct", correct), ("incorrect", ~correct)):
        attention["error_split"][name] = {
            "n": int(sel.sum()),
            "referent_mass_last_layer": {k: float(ref_last[sel, i].mean()) if sel.any() else float("nan")
                                         for i, k in enumerate(("bg", "target", "distractor"))}}
        r = attention["error_split"][name]["referent_mass_last_layer"]
        print(f"c1 {name} (n={int(sel.sum())}): referent-token mass at text-GCA layer {gca_layers[-1]}: "
              f"target {r['target']:.3f} distractor {r['distractor']:.3f} bg {r['bg']:.3f}")
    attention["per_image"] = {"correct_c1": correct.tolist(),
                              "referent_mass_last_layer_c1": ref_last.tolist()}
    with open(out_dir / "mirror_attention.json", "w") as f:
        json.dump(attention, f, indent=1)

    # ---- decoder attention over the text tokens under c1 (3) ----
    ids1 = np.load(out_dir / "text_feats_c1.npz")
    ridx, lidx, qidx = ids1["ref_idx"], ids1["last_idx"], ids1["q_idx"]
    w = dec_mass["c1"]                                                           # (N, H, T)
    assert np.allclose(w.sum(-1), 1.0, atol=1e-4), "decoder attention rows must sum to 1"
    ar = np.arange(N)
    cats = {"referent": w[ar, :, ridx], "bos": w[:, :, 0], "queried_word": w[ar, :, qidx], "last": w[ar, :, lidx]}
    cats["rest"] = 1.0 - sum(cats.values())
    dec_res = {"n_images": N, "condition": "c1", "queried_word": QUERIED, "n_heads": int(H),
               "mass_mean": {k: float(v.mean()) for k, v in cats.items()},
               "mass_per_head": {k: v.mean(0).tolist() for k, v in cats.items()}}
    print("decoder attention over text tokens (c1): " +
          " ".join(f"{k} {v:.3f}" for k, v in dec_res["mass_mean"].items()))
    with open(out_dir / "mirror_decoder_attention.json", "w") as f:
        json.dump(dec_res, f, indent=1)
    return attention, dec_res


def mirror_readout_probe(out_dir, args, labels_n2):
    """(4) Linear probe of the target colour from the cached RoBERTa hidden
    states at every layer: mean over question tokens / <s> / </s>, under c1 and
    under the text-only control c0; decoder accuracy from mirror_attention.json."""
    N = len(labels_n2)
    y = np.array([r["target"][QUERIED] for r in labels_n2])
    groups = np.arange(N)
    res = {"n_images": N, "queried": QUERIED, "readouts": list(MIRROR_READOUTS),
           "majority": float(np.bincount(np.unique(y, return_inverse=True)[1]).max() / N), "conditions": {}}
    for cond in ("c1", "c0"):
        c = np.load(out_dir / f"text_feats_{cond}.npz")
        hid, mask, lidx = c["hid"], c["mask"], c["last_idx"]
        n_hid = hid.shape[1]
        acc = {k: [] for k in MIRROR_READOUTS}
        for l in range(n_hid):
            h = hid[:, l].astype(np.float32)                                     # (N, T, D)
            feats = {"mean": (h * mask[:, :, None]).sum(1) / mask.sum(1)[:, None],
                     "bos": h[:, 0], "last": h[np.arange(N), lidx]}
            for k, X in feats.items():
                acc[k].append(_fit_eval(X, y, groups, GroupKFold(5)))
        res["conditions"][cond] = acc
        for k in MIRROR_READOUTS:
            print(f"probe {cond} {k:<5} " + " ".join(f"{v:.2f}" for v in acc[k]), flush=True)
    with open(out_dir / "mirror_attention.json") as f:
        att = json.load(f)
    res["decoder_accuracy"] = att["decoder_accuracy"]
    res["text_gca_layers"] = att["text_gca_layers"]
    with open(out_dir / "mirror_readout.json", "w") as f:
        json.dump(res, f, indent=1)
    return res


@torch.no_grad()
def run_mirror_fetch_test(out_dir, args, state, images_n2, owners_n2, labels_n2):
    """(5) Content-addressed fetch test.  d = mean over images of (target patch
    mean − distractor patch mean) of the vanilla ViT output (n2/feats_c0.npz,
    trunk.norm space = the key/value space).  alpha·d is added to the
    DISTRACTOR's patches in the key/value tensor only, under the c1 question;
    controls: norm-matched random vector on the distractor, d on the background."""
    model, sv, device, tf, ext = state["model"], state["steervit"], state["device"], state["transform"], state["extractor"]
    gca_layers = sorted(int(k) for k in sv.text_gca.keys())
    c0 = load_sparse(out_dir / "n2", "c0")
    om = c0["obj_mean"][:, :, NUM_LAYERS - 1].astype(np.float32)                  # (N, 2, D) last block, normed
    d = (om[:, 0] - om[:, 1]).mean(0)
    d_norm = float(np.linalg.norm(d))
    kv_norm = float(np.linalg.norm(om[:, 0], axis=-1).mean())
    print(f"object-identity contrast d: norm {d_norm:.2f} (mean target patch-mean norm {kv_norm:.2f})")
    rng = np.random.default_rng(args.seed)
    rand = rng.standard_normal(d.shape).astype(np.float32)
    rand = rand / np.linalg.norm(rand) * d_norm
    N, bs = len(labels_n2), args.batch_size
    imgs_t = torch.stack([tf(im) for im in images_n2])
    owner_t = torch.from_numpy(np.stack(owners_n2))
    A_id = np.array([model.vocab[r["target"][QUERIED]] for r in labels_n2])
    Ad_id = np.array([model.vocab[r["distractors"][0][QUERIED]] for r in labels_n2])
    qs1 = [r["questions"]["c1"] for r in labels_n2]
    words1 = [r["referent_words"]["c1"] for r in labels_n2]

    def run(delta=None, mask_id=None, alpha=0.0):
        pred, mass = np.zeros(N, int), []
        for s in range(0, N, bs):
            e = min(s + bs, N)
            ims, ow = imgs_t[s:e].to(device), owner_t[s:e].to(device)
            kw = {}
            if delta is not None:
                kw = dict(kv_delta=torch.from_numpy(delta).to(device).expand(e - s, -1),
                          kv_mask=ow == mask_id, alpha=alpha)
            out = mirror_forward(state, ims, qs1[s:e], **kw)
            ridx, _ = ext.referent_token_index(qs1[s:e], words1[s:e])
            pred[s:e] = out["logits"].argmax(-1).cpu().numpy()
            mass.append(_referent_mass(out, ridx, ow, gca_layers).cpu().numpy())
        return pred, np.concatenate(mass)                                        # (N,), (N, n_gca, 3)

    base_pred, base_mass = run()
    ok = (base_pred == A_id) & (A_id != Ad_id)
    bm = base_mass[ok].mean(0)
    print(f"baseline c1: accuracy {ok.mean():.3f} on {N} images; referent-token mass on distractor by layer "
          + " ".join(f"{v:.3f}" for v in bm[:, 2]))
    alphas = [float(a) for a in args.marker_alphas.split(",")]
    variants = {"contrast_to_distractor": (d, 2), "random_to_distractor": (rand, 2), "contrast_to_bg": (d, 0)}
    rows = []
    for alpha in alphas:
        for var, (vec, mid) in variants.items():
            pred, mass = run(vec, mid, alpha)
            p, m = pred[ok], mass[ok].mean(0)
            rows.append({"variant": var, "alpha": alpha, "n": int(ok.sum()),
                         "p_target": float((p == A_id[ok]).mean()), "p_distractor": float((p == Ad_id[ok]).mean()),
                         "p_other": float(((p != A_id[ok]) & (p != Ad_id[ok])).mean()),
                         "attn_ref_target": m[:, 1].tolist(), "attn_ref_distractor": m[:, 2].tolist(),
                         "attn_ref_bg": m[:, 0].tolist()})
            r = rows[-1]
            print(f"a={alpha:g} {var:<22} P(target) {r['p_target']:.2f} P(distractor) {r['p_distractor']:.2f} | "
                  f"referent mass on distractor by layer " + " ".join(f"{v:.3f}" for v in m[:, 2]), flush=True)
    res = {"n_images_ok": int(ok.sum()), "queried": QUERIED, "text_gca_layers": gca_layers,
           "contrast_norm": d_norm, "kv_token_norm_mean": kv_norm, "alphas": alphas,
           "baseline": {"accuracy_c1": float(ok.mean()),
                        "attn_ref_target": bm[:, 1].tolist(), "attn_ref_distractor": bm[:, 2].tolist(),
                        "attn_ref_bg": bm[:, 0].tolist()},
           "rows": rows}
    with open(out_dir / "mirror_fetch_test.json", "w") as f:
        json.dump(res, f, indent=1)
    print(f"Saved: {out_dir / 'mirror_fetch_test.json'}")
    return res


def _text_gca_axis(ax, gca_layers):
    ax.set_xticks(gca_layers)
    ax.set_xlabel("text-GCA layer (RoBERTa layer index)", fontsize=10)


def plot_mirror_attention(att, label, out_path):
    gl = att["text_gca_layers"]
    fig, axes = plt.subplots(1, len(MIRROR_TOKEN_TYPES), figsize=(4.2 * len(MIRROR_TOKEN_TYPES), 4), sharey=True)
    for ax, t in zip(axes, MIRROR_TOKEN_TYPES):
        for cond, ls, cl in (("c1", "-", "clean run"), ("c2", "--", "corrupted run")):
            for name in ("target", "distractor", "bg"):
                ax.plot(gl, att["conditions"][cond][t][name], ls, color=OWNER_RGB[name], marker="o",
                        markersize=3, label=f"{name}, {cl}")
        ax.set_title(MIRROR_TOKEN_LABEL[t], fontsize=10)
        ax.set_ylim(-0.02, 1.02)
        _text_gca_axis(ax, gl)
    axes[0].set_ylabel("attention mass on the patch group", fontsize=10)
    axes[0].legend(fontsize=6)
    es = att["error_split"]
    fig.suptitle(f"{label} — text-token → patch attention inside RoBERTa, by patch group "
                 f"(clean run = question refers to the target, corrupted run = refers to the distractor; n={att['n_images']})\n"
                 f"decoder accuracy clean {att['decoder_accuracy']['c1']:.2f}, corrupted {att['decoder_accuracy']['c2']:.2f}; "
                 f"referent token at the last text-GCA layer, clean run: correct (n={es['correct']['n']}) target "
                 f"{es['correct']['referent_mass_last_layer']['target']:.2f} / distractor "
                 f"{es['correct']['referent_mass_last_layer']['distractor']:.2f}, incorrect (n={es['incorrect']['n']}) target "
                 f"{es['incorrect']['referent_mass_last_layer']['target']:.2f} / distractor "
                 f"{es['incorrect']['referent_mass_last_layer']['distractor']:.2f}", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=S["dpi"], bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_mirror_decoder_attention(res, label, out_path):
    cats = list(res["mass_mean"])
    names = {"referent": "referent word", "bos": "<s>", "queried_word": f"'{res['queried_word']}'",
             "last": "</s>", "rest": "other tokens"}
    fig, ax = plt.subplots(1, 1, figsize=(6.5, 4))
    x = np.arange(len(cats))
    ax.bar(x, [res["mass_mean"][c] for c in cats], color="0.6", label="mean over heads")
    for c, xi in zip(cats, x):
        ph = res["mass_per_head"][c]
        ax.plot(np.full(len(ph), xi) + np.linspace(-0.2, 0.2, len(ph)), ph, "o", color="#1f77b4",
                markersize=3, label="single head" if xi == 0 else None)
    ax.set_xticks(x)
    ax.set_xticklabels([names[c] for c in cats], fontsize=9)
    ax.set_ylabel("decoder attention mass")
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=7)
    fig.suptitle(f"{label} — decoder cross-attention over the RoBERTa tokens, question refers to the target "
                 f"(n={res['n_images']}, {res['n_heads']} heads)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=S["dpi"], bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_mirror_readout(res, label, out_path):
    colours = {"mean": "#d62728", "bos": "#1f77b4", "last": "#2ca02c"}
    fig, ax = plt.subplots(1, 1, figsize=(8, 4.2))
    for cond, ls, cl in (("c1", "-", "with image"), ("c0", ":", "text only (cross-attention disabled)")):
        for k in MIRROR_READOUTS:
            ys = res["conditions"][cond][k]
            ax.plot(range(len(ys)), ys, ls, color=colours[k], marker="o", markersize=3,
                    label=f"{MIRROR_READOUT_LABEL[k]}, {cl}")
    ax.axhline(res["decoder_accuracy"]["c1"], color="k", linewidth=1.0, label="decoder accuracy, question refers to the target")
    ax.axhline(res["majority"], color="0.5", linewidth=0.8, linestyle="--", label="majority class")
    for l in res["text_gca_layers"]:
        ax.axvline(l + 0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.3)
    n_hid = len(res["conditions"]["c1"]["mean"])
    ax.set_xticks(range(0, n_hid, 2))
    ax.set_xlabel("RoBERTa hidden state (0 = embeddings; grey lines = text-GCA applied before that layer)")
    ax.set_ylabel("5-fold probe accuracy")
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=6, loc="lower right")
    fig.suptitle(f"{label} — linear probe of the target's {res['queried']} from the RoBERTa tokens (n={res['n_images']})",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=S["dpi"], bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_mirror_fetch_test(res, label, out_path):
    gl = res["text_gca_layers"]
    styles = {"contrast_to_distractor": ("#d62728", "o", "target − distractor contrast added to distractor patches"),
              "random_to_distractor": ("0.4", "x", "norm-matched random vector added to distractor patches"),
              "contrast_to_bg": ("#1f77b4", "s", "same contrast added to background patches")}
    alphas = res["alphas"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    ax = axes[0]
    xs = np.arange(len(alphas))
    width = 0.8 / len(styles)
    for i, (var, (col, mk, lab)) in enumerate(styles.items()):
        rr = [next(r for r in res["rows"] if r["variant"] == var and r["alpha"] == a) for a in alphas]
        ax.bar(xs + (i - 1) * width, [r["p_distractor"] for r in rr], width, color=col, label=lab)
        ax.bar(xs + (i - 1) * width, [r["p_target"] for r in rr], width, bottom=[r["p_distractor"] for r in rr],
               color=col, alpha=0.3, hatch="//", label="P(answer = target's colour), stacked" if i == 0 else None)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"scale {a:g}" for a in alphas])
    ax.set_ylabel("P(answer = distractor's colour)")
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=6)
    ax = axes[1]
    ax.plot(gl, res["baseline"]["attn_ref_distractor"], "-", color="0.7", marker="o", markersize=3, label="baseline (no edit)")
    for var, (col, mk, lab) in styles.items():
        for a, ls in zip(alphas, ("-", "--", ":", "-.")):
            r = next(r for r in res["rows"] if r["variant"] == var and r["alpha"] == a)
            ax.plot(gl, r["attn_ref_distractor"], ls, color=col, marker=mk, markersize=3, label=f"{lab}, scale {a:g}")
    ax.set_ylabel("referent token → distractor patches\nattention mass", fontsize=10)
    ax.set_ylim(-0.02, 1.02)
    _text_gca_axis(ax, gl)
    ax.legend(fontsize=5)
    fig.suptitle(f"{label} — content-addressed fetch test: vector added to the key/value patch tokens only, "
                 f"question refers to the target (n={res['n_images_ok']}; contrast norm {res['contrast_norm']:.1f}, "
                 f"mean patch norm {res['kv_token_norm_mean']:.1f})", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=S["dpi"], bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def run_mirror(args, out_dir, label, state, n1_entries, n2_entries, x19_pairs):
    if not args.replot:
        keep, images, owners, labels = load_or_prepare_subsets(out_dir, n1_entries, n2_entries, args, x19_pairs)
        extract_condition_sparse(out_dir / "n2", "c0", images["n2"], owners["n2"], labels["n2"], args, state)
        ensure_model(state, args)
        assert getattr(state["steervit"], "text_gca", None) is not None, "--mirror needs a mirror checkpoint (text_gca)"
        run_mirror_pass(out_dir, args, state, images["n2"], owners["n2"], labels["n2"])
        mirror_readout_probe(out_dir, args, labels["n2"])
        run_mirror_fetch_test(out_dir, args, state, images["n2"], owners["n2"], labels["n2"])
    for name, plot in (("mirror_attention", plot_mirror_attention),
                       ("mirror_decoder_attention", plot_mirror_decoder_attention),
                       ("mirror_readout", plot_mirror_readout),
                       ("mirror_fetch_test", plot_mirror_fetch_test)):
        if (out_dir / f"{name}.json").exists():
            with open(out_dir / f"{name}.json") as f:
                plot(json.load(f), label, out_dir / f"{name}.png")


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="outputs/model/clevr_dinov2_decoder1l_scratch_s42/best.pt")
    ap.add_argument("--n1-dir", default="data/clevr_object_count/n1")
    ap.add_argument("--n2-dir", default="data/clevr_object_count/n2")
    ap.add_argument("--out-dir", default="outputs/analysis/patch_language_condition")
    ap.add_argument("--x19-dir", default="outputs/analysis/patch_pca_cluster")
    ap.add_argument("--masks-only", action="store_true")
    ap.add_argument("--replot", action="store_true")
    ap.add_argument("--attr-directions", action="store_true",
                    help="only: attribute-specific direction projections (new files)")
    ap.add_argument("--rsa-template", action="store_true",
                    help="only: RSA with the per-position background template (new files)")
    ap.add_argument("--intervene", action="store_true")
    ap.add_argument("--queried", default="color", choices=["color", "shape", "material", "size"],
                    help="queried attribute of every question (c1/c2 refer by another attribute)")
    ap.add_argument("--head-combos", action="store_true",
                    help="only: keep/zero head subsets of GCA layers 7 and 9 (needs head_scan.json)")
    ap.add_argument("--head-scan", action="store_true",
                    help="only: zero-ablate every SA / GCA head and measure the selection effect per block (new files)")
    ap.add_argument("--patching-stats", default="outputs/analysis/activation_patching/clevr_dinov2_decoder1l_scratch/headwise_by_type_stats.json")
    ap.add_argument("--readout", action="store_true",
                    help="only: decoder attention by owner + token swaps between conditions (new files)")
    ap.add_argument("--marker-test", action="store_true",
                    help="only: transplant the referent-marker direction onto the distractor's patches (new files)")
    ap.add_argument("--marker-alphas", default="1,2")
    ap.add_argument("--attn-norm-control", action="store_true",
                    help="only: decoder attention vs token norm (attention-sink guard, new files)")
    ap.add_argument("--with-absent", action="store_true")
    ap.add_argument("--n-pairs", type=int, default=0, help="subsample eligible pairs (0 = all)")
    ap.add_argument("--bg-per-image", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--grid", type=int, default=24)
    ap.add_argument("--resolution", type=int, default=336)
    ap.add_argument("--coverage-thresh", type=float, default=0.2)
    ap.add_argument("--sat-thresh", type=float, default=0.18)
    ap.add_argument("--hue-thresh", type=float, default=0.17)
    ap.add_argument("--alphas", default="0.5,1,2")
    ap.add_argument("--intervene-layers", default=",".join(str(l) for l in range(NUM_LAYERS)))
    ap.add_argument("--readout-probe", action="store_true",
                    help="Part D: mean-pool / attention / oracle-position readouts on no-question tokens")
    ap.add_argument("--probe-layers", default=",".join(str(l) for l in GCA_LAYERS))
    ap.add_argument("--probe-max-tokens-per-object", type=int, default=6)
    ap.add_argument("--probe-max-bg", type=int, default=12)
    ap.add_argument("--skip-probes", action="store_true")
    ap.add_argument("--model-label", default=None)
    ap.add_argument("--relational", default=None, choices=["same", "spatial"],
                    help="only: relational questions on 3-object scenes (new --out-dir); "
                         "same = 'same {attribute} as the {colour} object', spatial = 'left of / right of / "
                         "in front of / behind the {colour} object'")
    ap.add_argument("--three-dir", default="data/clevr_three_object_v2")
    ap.add_argument("--directions-dir", default="outputs/analysis/patch_language_condition",
                    help="--relational: directory whose n1/ cache gives the colour directions u")
    ap.add_argument("--mirror", action="store_true",
                    help="only: mirror checkpoint (cross-attention inside RoBERTa, decoder reads text tokens); "
                         "attention mass / decoder attention / probe ladder / fetch test (new --out-dir)")
    args = ap.parse_args()

    apply_style()
    out_dir = Path(args.out_dir)
    assert out_dir.resolve() != Path(args.x19_dir).resolve(), "refusing to write into the X19 directory"
    if (args.relational or args.mirror) and not args.replot:
        assert not (out_dir.exists() and any(out_dir.iterdir())), \
            f"--relational/--mirror need a new --out-dir (non-empty: {out_dir}); use --replot to regenerate figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    tee_stdout(out_dir)
    if args.model_label is None:
        stem = Path(args.checkpoint).parent.name
        args.model_label = next((v for k, v in BACKBONE_LABELS.items() if f"_{k}_" in f"_{stem}_"), stem)
    label = args.model_label
    global QUERIED
    QUERIED = args.queried
    print(f"args: {vars(args)}")
    if args.relational:
        run_relational(args, out_dir, label)
        return

    n1_entries, n2_entries = load_entries(args.n1_dir), load_entries(args.n2_dir)
    x19_pairs = set()
    x19_labels = Path(args.x19_dir) / "n2" / "labels.json"
    if x19_labels.exists():
        with open(x19_labels) as f:
            x19_pairs = {r["pair_index"] for r in json.load(f) if r.get("in_pca_set")}
    state = {}

    if args.mirror:
        run_mirror(args, out_dir, label, state, n1_entries, n2_entries, x19_pairs)
        return

    if not args.replot:
        keep, images, owners, labels = load_or_prepare_subsets(out_dir, n1_entries, n2_entries, args, x19_pairs)
        if args.masks_only:
            print("\n--masks-only: inspect n1/n2 masks_debug.png, then rerun.")
            return
        conds = {"n1": CONDITIONS_N1, "n2": CONDITIONS_N2 + (["c4"] if args.with_absent else [])}
        if args.with_absent:
            # c4 exists only for pairs sharing an attribute; extract on that subset is not
            # supported by the shared token table — keep the design simple: require all.
            missing = [r["pair_index"] for r in labels["n2"] if "c4" not in r["questions"]]
            if missing:
                print(f"c4 unavailable for {len(missing)} pairs; c4 extraction skipped")
                conds["n2"] = CONDITIONS_N2
        for name in ("n1", "n2"):
            for cond in conds[name]:
                extract_condition_sparse(out_dir / name, cond, images[name], owners[name],
                                         labels[name], args, state)
        if args.head_combos:
            ensure_model(state, args)
            cache_n1 = load_sparse(out_dir / "n1", "c0")
            run_head_combos(out_dir, args, state, cache_n1, labels["n1"], images["n2"], owners["n2"], labels["n2"])
            return
        if args.head_scan:
            ensure_model(state, args)
            cache_n1 = load_sparse(out_dir / "n1", "c0")
            res = run_head_scan(out_dir, args, state, cache_n1, labels["n1"], images["n2"], owners["n2"], labels["n2"])
            plot_head_scan(res, label, out_dir / "head_scan.png")
            return
        if args.readout:
            ensure_model(state, args)
            attention, swap = run_readout(out_dir, args, state, images["n2"], owners["n2"], labels["n2"])
            gca_layers = [int(l) for l in np.load(out_dir / "n2" / "feats_c0.npz")["gca_layers"]]
            plot_readout(attention, swap, label, out_dir / "readout.png", gca_layers)
            return
        if args.marker_test:
            ensure_model(state, args)
            res = run_marker_test(out_dir, args, state, images["n2"], owners["n2"], labels["n2"])
            plot_marker_test(res, label, out_dir / "marker_test.png")
            return
        if args.attn_norm_control:
            ensure_model(state, args)
            run_attn_norm_control(out_dir, args, state, images["n2"], owners["n2"], labels["n2"])
            return
        if args.intervene:
            ensure_model(state, args)
            cache_n1 = load_sparse(out_dir / "n1", "c1")
            summary = run_interventions(out_dir, args, state, cache_n1, labels["n1"],
                                        images["n2"], owners["n2"], labels["n2"])
            plot_interventions(summary, label, out_dir / "intervention_flip.png")
            return

    # ---- analyses from cache ----
    labels = {n: load_labels(out_dir / n) for n in ("n1", "n2")}
    caches_n2 = {c: load_sparse(out_dir / "n2", c) for c in CONDITIONS_N2
                 if (out_dir / "n2" / f"feats_{c}.npz").exists()}
    cache_n1 = load_sparse(out_dir / "n1", "c0")
    gca_layers = [int(l) for l in caches_n2["c0"]["gca_layers"]]
    if args.readout_probe:
        print("\nPart D readout probes ...")
        res = part_d(caches_n2["c0"], labels["n2"], args)
        with open(out_dir / "readout_probe.json", "w") as f:
            json.dump(res, f, indent=1)
        plot_readout_probe(res, label, out_dir / "readout_probe.png")
        return
    if args.attr_directions:
        V = attribute_directions(cache_n1, labels["n1"])
        print("directions per attribute: " + ", ".join(f"{a}: {sorted(V[a])}" for a in V))
        res = attr_direction_analysis(caches_n2, labels["n2"], V)
        for key in (f"ref_target_{QUERIED}_own", f"refvs0_target_{QUERIED}_own", f"nonrefvs0_target_{QUERIED}_own",
                    "ref_target_color_own", "ref_target_color_other", "ref_target_shape_own",
                    "nonref_distractor_color_own", "nonref_distractor_shape_own",
                    "refvs0_target_color_own", "refvs0_target_shape_own",
                    "nonrefvs0_target_color_own", "nonrefvs0_target_shape_own", "c3vs0_target_color_own"):
            if key in res["delta"]:
                print(f"{key:<30} " + " ".join(f"{q['mean']:+.2f}" for q in res["delta"][key]))
        with open(out_dir / "partA_attr_directions.json", "w") as f:
            json.dump(res, f, indent=1)
        plot_attr_directions(res, label, out_dir / "attr_directions.png", gca_layers)
        return
    if args.rsa_template:
        res = rsa_template(caches_n2, labels["n2"], args.grid)
        with open(out_dir / "partA_rsa_template.json", "w") as f:
            json.dump(res, f, indent=1)
        plot_rsa_template(res, label, out_dir / "rsa_template.png", gca_layers)
        return

    for tag, norm_std in (("", False), ("_normstd", True)):
        m = part_a(caches_n2, labels["n2"], gca_layers, norm_std)
        with open(out_dir / f"partA_metrics{tag}.json", "w") as f:
            json.dump(m, f, indent=1)
        print(f"\nPart A{tag}: Δ_ref by block: " +
              " ".join(f"{d['mean']:+.3f}" for d in m["delta"].get("ref", [])))
        print(f"Part A{tag}: Δ_nonref by block: " +
              " ".join(f"{d['mean']:+.3f}" for d in m["delta"].get("nonref", [])))
        plot_projection_deltas(m, label, out_dir / f"projection_deltas{tag}.png", gca_layers)
        plot_patch_change(m, label, out_dir / f"patch_change{tag}.png", gca_layers)
        plot_gca(m, label, out_dir / f"gca_write{tag}.png", gca_layers)
        plot_offsets_by_condition(m, label, out_dir / f"offset_by_condition{tag}.png", gca_layers)
        plot_rsa(m, label, out_dir / f"rsa_identity_vs_position{tag}.png", gca_layers)
        stats = offset_stats_by_condition(cache_n1, caches_n2, labels["n2"], norm_std)
        for cond, st in stats.items():
            with open(out_dir / f"offset_stats_{cond}{tag}.json", "w") as f:
                json.dump(st, f, indent=1)
            s = st["L11"]
            print(f"offset stats {cond}{tag} L11: within {s['n1_within_combo_cos']['mean']:.3f} "
                  f"between {s['n1_between_combo_cos']['mean']:.3f} n1~n2 "
                  f"{s['n1_vs_n2_target_same_pair_cos']['mean']:.3f} "
                  f"t~d {s['n2_target_vs_distractor_cos']['mean']:.3f}")
    # X19 reproduction check on the 30 X19 PCA pairs, c0 only
    idx = [b for b, rec in enumerate(labels["n2"]) if rec["in_x19_pca_set"]]
    if idx:
        o1 = offsets_from_cache(cache_n1)[idx, 0]
        o2 = offsets_from_cache(caches_n2["c0"])[idx]
        combos = ["-".join(combo_key(labels["n2"][b]["target"])) for b in idx]
        rep = {f"L{l}": offset_statistics_from_offsets(
            o1[:, l], o2[:, 0, l], [o2[k, 1, l] for k in range(len(idx))], combos) for l in GCA_LAYERS}
        with open(out_dir / "x19_reproduction_c0.json", "w") as f:
            json.dump(rep, f, indent=1)
        print(f"X19 reproduction on {len(idx)} pairs, L11: within {rep['L11']['n1_within_combo_cos']['mean']:.3f} "
              f"between {rep['L11']['n1_between_combo_cos']['mean']:.3f} "
              f"n1~n2 {rep['L11']['n1_vs_n2_target_same_pair_cos']['mean']:.3f}")
    tn = token_norm_stats(caches_n2["c0"])
    with open(out_dir / "token_norm_stats.json", "w") as f:
        json.dump(tn, f)
    plot_token_norms(tn, label, out_dir / "token_norms.png", args.grid)
    if (out_dir / "intervention_results.json").exists():
        with open(out_dir / "intervention_results.json") as f:
            plot_interventions(json.load(f), label, out_dir / "intervention_flip.png")
    if (out_dir / "readout_swap.json").exists():
        with open(out_dir / "readout_attention.json") as f:
            att = json.load(f)
        with open(out_dir / "readout_swap.json") as f:
            plot_readout(att, json.load(f), label, out_dir / "readout.png", gca_layers)
    if (out_dir / "readout_probe.json").exists():
        with open(out_dir / "readout_probe.json") as f:
            plot_readout_probe(json.load(f), label, out_dir / "readout_probe.png")
    if not args.skip_probes:
        print("\nPart C probes ...")
        res = part_c(caches_n2, labels["n2"], args)
        with open(out_dir / "probe_results.json", "w") as f:
            json.dump(res, f, indent=1)
        plot_probes(res, label, out_dir / "probe_patch.png")


if __name__ == "__main__":
    main()
