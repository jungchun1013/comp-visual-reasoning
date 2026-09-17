"""Quantify how attributes organize the pooled scene representation.

Three measures on the SAME cached block features the pooled t-SNE panels use
(no forward passes):
1. Variance partitioning — fraction of representational variance explained by
   each attribute (n1: balanced factorial, orthogonal main effects + cells
   model; n2: joint OLS over target + distractor factors, unique variance).
2. PCA spectrum ownership — per-PC one-way eta^2 for each attribute, weighted
   by the PC's explained variance ("who owns the leading axes").
3. Nesting (substructure) — global silhouette of attribute B vs. silhouette of
   B restricted to the groups of attribute A, both directions; permutation
   nulls as chance baselines.

CPU-only. Results cached to results.json; --replot redraws figures only.

Run from main/:
  PYTHONPATH=src <venv>/python scripts/analysis/variance_partitioning.py
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from analysis.run_log import tee_stdout

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ATTRS = ["color", "shape", "material", "size"]
LAYERS = list(range(12))
MAIN_LAYER = 11
GCA_LAYERS = [1, 3, 5, 7, 9, 11]
N_PERM = 100

TRAINED_CONDS = {"n1": ["noca", "ca_color_object", "ca_shape_object"],
                 "n2": ["noca", "ca_color_object", "ca_shape_object",
                        "ca_color_refer", "ca_shape_refer"]}
COND_DISPLAY = {
    "noca": "no-CA",
    "ca_color_object": 'CA "What color is the object?"',
    "ca_shape_object": 'CA "What shape is the object?"',
    "ca_color_refer": "CA referring (query color)",
    "ca_shape_refer": "CA referring (query shape)",
}
RAW_LABELS = ["DINOv2", "SigLIP", "Sup-ViT", "MAE"]


# ── label / linear-algebra helpers ───────────────────────────────

def subset_rows(subset_labels):
    """pair_index list of an object-level labels.json (X21), to restrict the 480-row
    scene caches to the same pairs; None when no subset is requested."""
    if not subset_labels:
        return None
    return [r["pair_index"] for r in json.load(open(subset_labels))]


def load_labels(data_root, rows=None):
    labels = {}
    for ds in ("n1", "n2"):
        meta = json.load(open(Path(data_root) / ds / "metadata.json"))
        if rows is not None:
            meta = [meta[i] for i in rows]
        L = {a: np.array([m[a] for m in meta]) for a in ATTRS}
        if ds == "n2":
            for a in ATTRS:
                L[f"d_{a}"] = np.array([m["distractors"][0][a] for m in meta])
        labels[ds] = L
    return labels


def one_hot(y):
    classes = sorted(set(y))
    M = np.zeros((len(y), len(classes) - 1))  # drop-one; intercept implied
    for j, c in enumerate(classes[:-1]):
        M[:, j] = (y == c)
    return M - M.mean(0)  # centered => no intercept column needed


def between_share(Xc, y, sst):
    """SS_between / SS_total for one factor (exact single-factor R^2)."""
    ss = 0.0
    for c in set(y):
        m = Xc[y == c].mean(0)
        ss += (y == c).sum() * float(m @ m)
    return ss / sst


def ols_r2(Xc, D, sst):
    beta, *_ = np.linalg.lstsq(D, Xc, rcond=None)
    pred = D @ beta
    return float((pred ** 2).sum()) / sst


def partition(Xc, L, factors, sst):
    """Marginal share per factor + joint-model unique shares + residual."""
    marg = {f: between_share(Xc, L[f], sst) for f in factors}
    designs = {f: one_hot(L[f]) for f in factors}
    D_full = np.concatenate([designs[f] for f in factors], axis=1)
    r2_full = ols_r2(Xc, D_full, sst)
    uniq = {}
    for f in factors:
        D_minus = np.concatenate([designs[g] for g in factors if g != f],
                                 axis=1)
        uniq[f] = r2_full - ols_r2(Xc, D_minus, sst)
    return {"marginal": marg, "unique": uniq, "r2_full": r2_full}


def joint_label(L, keys):
    return np.array(["|".join(t) for t in zip(*[L[k] for k in keys])])


def silhouette_pair(X50, L, attr_b, attr_a):
    """(global s(B), mean within-A-group s(B))."""
    from sklearn.metrics import silhouette_score
    y_b, y_a = L[attr_b], L[attr_a]
    s_global = float(silhouette_score(X50, y_b))
    parts, weights = [], []
    for a in set(y_a):
        m = y_a == a
        if len(set(y_b[m])) < 2:
            continue
        parts.append(float(silhouette_score(X50[m], y_b[m])))
        weights.append(m.sum())
    s_within = float(np.average(parts, weights=weights)) if parts else None
    return s_global, s_within


# ── per-condition computation ────────────────────────────────────

def analyze_condition(feats_by_layer, L, ds, rng):
    """feats_by_layer: {layer: (480, D) float64}."""
    from sklearn.decomposition import PCA

    factors = ATTRS + ([f"d_{a}" for a in ATTRS] if ds == "n2" else [])
    out = {"layers": {}}
    for layer, X in feats_by_layer.items():
        Xc = X - X.mean(0)
        sst = float((Xc ** 2).sum())
        part = partition(Xc, L, factors, sst)
        row = {"shares": part}
        if ds == "n1":  # balanced factorial: cells model + interaction
            r2_cells = between_share(Xc, joint_label(L, ATTRS), sst)
            row["r2_cells"] = r2_cells
            row["interaction"] = r2_cells - sum(part["marginal"].values())
        out["layers"][str(layer)] = row

    # Main-layer extras: PCA spectrum, nesting silhouettes, permutation nulls
    X = feats_by_layer[MAIN_LAYER]
    Xc = X - X.mean(0)
    sst = float((Xc ** 2).sum())
    pca = PCA(n_components=20, random_state=42).fit(Xc)
    scores = pca.transform(Xc)
    pc_eta = {a: [between_share(scores[:, i:i + 1] - scores[:, i:i + 1].mean(0),
                                L[a],
                                float(((scores[:, i] - scores[:, i].mean()) ** 2).sum()))
                  for i in range(20)] for a in ATTRS}
    out["pca"] = {"evr": pca.explained_variance_ratio_.tolist(),
                  "eta2": {a: [float(v) for v in pc_eta[a]] for a in ATTRS}}

    from sklearn.decomposition import PCA as _P
    X50 = _P(n_components=50, random_state=42).fit_transform(Xc)
    L2 = dict(L)
    L2["shape_material"] = joint_label(L, ["shape", "material"])
    nest = {}
    for b, a in (("color", "shape_material"), ("shape_material", "color")):
        g, w = silhouette_pair(X50, L2, b, a)
        nest[f"{b}|{a}"] = {"global": g, "within": w}
    out["nesting"] = nest

    null = {"eta2": {}, "silhouette": {}}
    for a in ATTRS + ["shape_material"]:
        y = L2[a] if a == "shape_material" else L[a]
        vals_e, vals_s = [], []
        for _ in range(N_PERM):
            yp = rng.permutation(y)
            vals_e.append(between_share(Xc, yp, sst))
        null["eta2"][a] = {"mean": float(np.mean(vals_e)),
                           "p95": float(np.quantile(vals_e, 0.95))}
        from sklearn.metrics import silhouette_score
        for _ in range(20):  # silhouette is costlier; 20 perms suffice
            yp = rng.permutation(y)
            vals_s.append(float(silhouette_score(X50, yp)))
        null["silhouette"][a] = {"mean": float(np.mean(vals_s)),
                                 "p95": float(np.quantile(vals_s, 0.95))}
    out["null"] = null
    return out


def iter_conditions(args, labels):
    rows = subset_rows(getattr(args, "subset_labels", None))
    sel = (lambda X: X) if rows is None else (lambda X: X[rows])
    raw_dir = Path(args.raw_dir)
    for lbl in ([] if getattr(args, "skip_raw", False) else RAW_LABELS):
        for ds in ("n1", "n2"):
            F = np.load(raw_dir / f"feats_{lbl}_{ds}.npz")["feats"].astype(np.float64)
            yield f"raw_{lbl}_{ds}", {l: sel(F[:, l]) for l in LAYERS}, labels[ds], ds
    for ds, conds in TRAINED_CONDS.items():
        tdir = Path(args.trained_root) / ds
        for cond in conds:
            data = np.load(tdir / f"feats_{cond}.npz")
            yield (f"trained_{cond}_{ds}",
                   {l: sel(data[str(l)].astype(np.float64)) for l in LAYERS},
                   labels[ds], ds)


def compute(args, out_dir):
    labels = load_labels(args.data_root, subset_rows(getattr(args, "subset_labels", None)))
    rng = np.random.RandomState(args.seed)
    results = {}
    for key, feats, L, ds in iter_conditions(args, labels):
        print(f"analyzing {key} ...", flush=True)
        results[key] = analyze_condition(feats, L, ds, rng)

    # Sanity: n1 balanced factorial — main effects + interaction == cells R2
    # (only holds for the full balanced 480; a pair subset is not balanced)
    for key, res in results.items():
        if key.endswith("_n1") and not getattr(args, "subset_labels", None):
            row = res["layers"][str(MAIN_LAYER)]
            total = sum(row["shares"]["marginal"].values()) + row["interaction"]
            assert abs(total - row["r2_cells"]) < 1e-6, (key, total, row["r2_cells"])
    return results


def compute_own_axis(args):
    """Per condition x attribute: silhouette inside the attribute's OWN
    top-2 eta^2 PCs (block 11). An attribute low in the variance spectrum can
    still be discretely clustered on its dedicated axes — full-space
    silhouette misses that because distances are dominated by the leading
    PCs."""
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score
    labels = load_labels(args.data_root)
    rng = np.random.RandomState(args.seed)
    out = {}
    for key, feats, L, ds in iter_conditions(args, labels):
        X = feats[MAIN_LAYER]
        Xc = X - X.mean(0)
        scores = PCA(n_components=20, random_state=42).fit_transform(Xc)
        row = {}
        for a in ATTRS:
            eta = []
            for i in range(20):
                s = scores[:, i:i + 1] - scores[:, i:i + 1].mean(0)
                eta.append(between_share(s, L[a], float((s ** 2).sum())))
            top2 = [int(i) for i in np.argsort(eta)[::-1][:2]]
            sub = scores[:, top2]
            null = [float(silhouette_score(sub, rng.permutation(L[a])))
                    for _ in range(20)]
            row[a] = {"pcs": [i + 1 for i in top2],
                      "eta2": [float(eta[i]) for i in top2],
                      "silhouette": float(silhouette_score(sub, L[a])),
                      "null_p95": float(np.quantile(null, 0.95))}
        out[key] = row
        print(key, {a: (row[a]["pcs"], round(row[a]["silhouette"], 3))
                    for a in ATTRS}, flush=True)
    return out


# ── figures ──────────────────────────────────────────────────────

def _cond_name(key):
    if key.startswith("raw_"):
        _, lbl, ds = key.split("_")
        return f"raw {lbl}"
    cond = key[len("trained_"):].rsplit("_", 1)[0]
    return COND_DISPLAY[cond]


def plot_variance_shares(results, out_dir):
    from analysis.plot_style import apply_style, S, save_with_legend
    apply_style()
    t10 = plt.cm.tab10.colors
    attr_color = dict(zip(ATTRS, [t10[0], t10[1], t10[2], t10[4]]))

    trained = [("n1", ["noca", "ca_color_object", "ca_shape_object"]),
               ("n2", ["noca", "ca_color_object", "ca_shape_object",
                       "ca_color_refer", "ca_shape_refer"])]
    fig, axes = plt.subplots(
        1, 3, figsize=(20, 6),
        gridspec_kw={"width_ratios": [3, 5, 2]}, sharey=True)

    def draw(ax, keys, ds):
        names = []
        for i, key in enumerate(keys):
            row = results[key]["layers"][str(MAIN_LAYER)]
            uniq = row["shares"]["unique"]
            bottom = 0.0
            for a in ATTRS:  # target attrs
                ax.bar(i, uniq[a], bottom=bottom, color=attr_color[a],
                       width=0.62)
                bottom += uniq[a]
            if ds == "n2":  # distractor attrs, hatched, same hues
                for a in ATTRS:
                    ax.bar(i, uniq[f"d_{a}"], bottom=bottom,
                           color=attr_color[a], width=0.62, hatch="///",
                           edgecolor="white", linewidth=0)
                    bottom += uniq[f"d_{a}"]
            ax.bar(i, row["shares"]["r2_full"] - bottom
                   if row["shares"]["r2_full"] > bottom else 0.0,
                   bottom=bottom, color="0.85", width=0.62)
            names.append(_cond_name(key))
        ax.set_xticks(range(len(keys)))
        ax.set_xticklabels(names, rotation=20, ha="right",
                           fontsize=S["tick_labelsize"])

    draw(axes[0], [f"trained_{c}_n1" for c in trained[0][1]], "n1")
    axes[0].set_title("1 object — trained model",
                      fontsize=S["subplot_title_fontsize"])
    draw(axes[1], [f"trained_{c}_n2" for c in trained[1][1]], "n2")
    axes[1].set_title("2 objects — trained model",
                      fontsize=S["subplot_title_fontsize"])
    for i, (key, ds) in enumerate((("raw_DINOv2_n1", "n1"),
                                   ("raw_DINOv2_n2", "n2"))):
        row = results[key]["layers"][str(MAIN_LAYER)]
        uniq = row["shares"]["unique"]
        bottom = 0.0
        for a in ATTRS:
            axes[2].bar(i, uniq[a], bottom=bottom, color=attr_color[a],
                        width=0.62)
            bottom += uniq[a]
        if ds == "n2":
            for a in ATTRS:
                axes[2].bar(i, uniq[f"d_{a}"], bottom=bottom,
                            color=attr_color[a], width=0.62, hatch="///",
                            edgecolor="white", linewidth=0)
                bottom += uniq[f"d_{a}"]
    axes[2].set_xticks([0, 1])
    axes[2].set_xticklabels(["raw DINOv2\n1 object", "raw DINOv2\n2 objects"],
                            fontsize=S["tick_labelsize"])
    axes[2].set_title("ViT backbone (separate baseline)",
                      fontsize=S["subplot_title_fontsize"])

    axes[0].set_ylabel("fraction of representational variance (block 11)")
    from matplotlib.patches import Patch
    handles = [Patch(color=attr_color[a], label=f"target {a}") for a in ATTRS]
    handles += [Patch(facecolor="0.4", hatch="///", edgecolor="white",
                      label="distractor attrs (hatched, by hue)")]
    fig.suptitle("Variance partitioning — pooled features",
                 fontsize=S["suptitle_fontsize"])
    save_with_legend(fig, str(out_dir / "variance_shares.png"),
                     handles=handles,
                     labels=[h.get_label() for h in handles])
    print(f"Saved: {out_dir / 'variance_shares.png'}")


def plot_queried_share_by_layer(results, out_dir):
    from analysis.plot_style import (apply_style, S, line_kwargs,
                                     save_with_legend)
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    for ax, q in zip(axes, ("color", "shape")):
        series = [(f"trained_noca_n1", "no-CA (1 obj)", {"linestyle": "--"}),
                  (f"trained_ca_{q}_object_n1", "CA (1 obj)", {}),
                  (f"trained_noca_n2", "no-CA (2 obj)",
                   {"linestyle": "--", "alpha": 0.6}),
                  (f"trained_ca_{q}_object_n2", "CA non-referring (2 obj)",
                   {"alpha": 0.6}),
                  (f"trained_ca_{q}_refer_n2", "CA referring (2 obj)", {})]
        for key, name, style in series:
            vals = [results[key]["layers"][str(l)]["shares"]["unique"][q]
                    for l in LAYERS]
            kw = line_kwargs(name)
            kw.update(style)
            ax.plot(LAYERS, vals, **kw)
        ax.set_title(f"unique variance share of target {q}",
                     fontsize=S["subplot_title_fontsize"])
        ax.set_xlabel("block")
    axes[0].set_ylabel("fraction of representational variance")
    save_with_legend(fig, str(out_dir / "queried_share_by_layer.png"))
    print(f"Saved: {out_dir / 'queried_share_by_layer.png'}")


def plot_pc_eta_heatmap(results, out_dir):
    from analysis.plot_style import apply_style, S
    apply_style()
    panels = [("raw_DINOv2_n1", "1 object — ViT backbone"),
              ("raw_DINOv2_n2", "2 objects — ViT backbone"),
              ("trained_ca_color_object_n1", "1 object — +CA, color query"),
              ("trained_ca_color_refer_n2",
               "2 objects — +CA, color query (referring)"),
              ("trained_ca_shape_object_n1", "1 object — +CA, shape query"),
              ("trained_ca_shape_refer_n2",
               "2 objects — +CA, shape query (referring)")]
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    for ax, (key, title) in zip(axes.ravel(), panels):
        pca = results[key]["pca"]
        M = np.array([[e * v for e, v in zip(pca["eta2"][a], pca["evr"])]
                      for a in ATTRS])
        im = ax.imshow(M, aspect="auto", cmap="viridis", vmin=0)
        ax.set_yticks(range(len(ATTRS)))
        ax.set_yticklabels(ATTRS, fontsize=S["tick_labelsize"])
        ax.set_xticks(range(0, 20, 2))
        ax.set_xticklabels(range(1, 21, 2), fontsize=S["tick_labelsize"])
        ax.set_xlabel("PC", fontsize=S["label_fontsize"])
        ax.set_title(title, fontsize=S["subplot_title_fontsize"])
        fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    fig.suptitle("PC spectrum ownership — eta$^2$ weighted by PC explained "
                 "variance (block 11)", fontsize=S["suptitle_fontsize"])
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(str(out_dir / "pc_eta_heatmap.png"), dpi=S["dpi"],
                bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_dir / 'pc_eta_heatmap.png'}")


def plot_nesting(results, out_dir):
    from analysis.plot_style import apply_style, S, save_with_legend
    apply_style()
    t10 = plt.cm.tab10.colors
    conds = {"n1": [("raw_DINOv2_n1", "raw DINOv2"),
                    ("trained_noca_n1", "no-CA"),
                    ("trained_ca_color_object_n1", "CA color"),
                    ("trained_ca_shape_object_n1", "CA shape")],
             "n2": [("raw_DINOv2_n2", "raw DINOv2"),
                    ("trained_noca_n2", "no-CA"),
                    ("trained_ca_color_refer_n2", "CA color refer"),
                    ("trained_ca_shape_refer_n2", "CA shape refer")]}
    pairs = [("color|shape_material", "color: global vs within shape×material"),
             ("shape_material|color", "shape×material: global vs within color")]
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharey=True)
    for r, (pair, ptitle) in enumerate(pairs):
        for c, ds in enumerate(("n1", "n2")):
            ax = axes[r][c]
            keys = conds[ds]
            xs = np.arange(len(keys))
            g = [results[k]["nesting"][pair]["global"] for k, _ in keys]
            w = [results[k]["nesting"][pair]["within"] for k, _ in keys]
            ax.bar(xs - 0.18, g, width=0.34, color=t10[0], label="global")
            ax.bar(xs + 0.18, w, width=0.34, color=t10[1],
                   label="within groups")
            attr = pair.split("|")[0]
            null = np.mean([results[k]["null"]["silhouette"][attr]["p95"]
                            for k, _ in keys])
            ax.axhline(null, color="gray", linestyle=":", linewidth=1,
                       label="permutation null (p95)")
            ax.set_xticks(xs)
            ax.set_xticklabels([n for _, n in keys], rotation=15, ha="right",
                               fontsize=S["tick_labelsize"])
            ax.set_title(f"{ptitle} — {'1 object' if ds == 'n1' else '2 objects'}",
                         fontsize=S["subplot_title_fontsize"])
    axes[0][0].set_ylabel("silhouette")
    axes[1][0].set_ylabel("silhouette")
    handles, labels = axes[0][0].get_legend_handles_labels()
    save_with_legend(fig, str(out_dir / "nesting_silhouette.png"),
                     handles=handles, labels=labels)
    print(f"Saved: {out_dir / 'nesting_silhouette.png'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir",
                    default="outputs/analysis/raw_backbone_probe/pooled_n1n2_v2")
    ap.add_argument("--trained-root", default="outputs/analysis/tsne/object_count_v2")
    ap.add_argument("--data-root", default="data/clevr_object_count")
    ap.add_argument("--out-dir", default="outputs/analysis/variance_partitioning")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--replot", action="store_true",
                    help="Figures from cached results.json only")
    ap.add_argument("--subset-labels", default=None,
                    help="object-level labels.json (X21): restrict every 480-row cache to "
                         "its pair_index rows (the 324 pairs of the object-level analysis)")
    ap.add_argument("--skip-raw", action="store_true", help="trained-model conditions only")
    ap.add_argument("--own-axis", action="store_true",
                    help="Compute only the own-axis subspace silhouettes "
                         "(results_own_axis.json); no figures")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tee_stdout(out_dir)
    if args.own_axis:
        path = out_dir / "results_own_axis.json"
        path.write_text(json.dumps(compute_own_axis(args), indent=1))
        print(f"Wrote {path}")
        return
    results_path = out_dir / "results.json"

    if args.replot or results_path.exists():
        results = json.loads(results_path.read_text())
        print(f"loaded {results_path}")
    else:
        results = compute(args, out_dir)
        results_path.write_text(json.dumps(results, indent=1))
        print(f"Wrote {results_path}")

    plot_variance_shares(results, out_dir)
    plot_queried_share_by_layer(results, out_dir)
    plot_pc_eta_heatmap(results, out_dir)
    plot_nesting(results, out_dir)


if __name__ == "__main__":
    main()
