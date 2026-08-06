#!/usr/bin/env python
"""Anchor dissipation (experiment 二c): pure post-hoc CPU analysis over
rsa_per_query.json.

New file (reason): this is a pure post-hoc CPU analysis that reads the
per-query RSA dump (rsa_per_query.json, produced by conditional_rsa.py
--dump-per-query) — there is no model forward and no existing host script to
extend.

Per query it computes a dissipation index (anchor-binding ρ at its peak layer
minus at a late layer, default L11) and the layer-wise (anchor − target)
difference curve. It correlates the dissipation index with answer correctness
(point-biserial + a logistic fit) and plots the per-layer anchor-vs-target
difference with per-query spread.

Underpowered guard: dissipation↔correct needs incorrect trials. The stats json
always reports n_incorrect; if <10 it flags underpowered=true (whether to widen
the query set is left to the user).

Usage (from main/):
  PYTHONPATH=src <interp> scripts/analysis/anchor_dissipation.py \
    --baseline outputs/analysis/conditional_rsa/..._pos_only/attr_query_same/rsa_per_query.json \
    --ablation outputs/analysis/conditional_rsa/..._head_ablation/attr_query_same/rsa_per_query.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from analysis.run_log import tee_stdout

ANCHOR_CHAIN = "Anchor binding | All"
TARGET_CHAIN = "Target binding | All"


def _chain_by_name(record, name):
    for c in record.get("chains", []):
        if c.get("name") == name:
            return c
    return None


def _curve(rho_dict, num_layers):
    """{str(layer): rho} -> float array length num_layers, nan where missing."""
    arr = np.full(num_layers, np.nan)
    for k, v in (rho_dict or {}).items():
        li = int(k)
        if 0 <= li < num_layers:
            arr[li] = v
    return arr


def _infer_num_layers(records):
    mx = -1
    for r in records:
        for c in r.get("chains", []):
            for k in (c.get("per_layer_rho") or {}):
                mx = max(mx, int(k))
    return mx + 1 if mx >= 0 else 12


def analyze_source(records, late_layer):
    from scipy.stats import pointbiserialr

    num_layers = _infer_num_layers(records)
    late = min(late_layer, num_layers - 1)

    dissipations, corrects, diff_curves = [], [], []
    for r in records:
        anc = _chain_by_name(r, ANCHOR_CHAIN)
        tgt = _chain_by_name(r, TARGET_CHAIN)
        if anc is None or tgt is None:
            continue
        a = _curve(anc.get("per_layer_rho"), num_layers)
        t = _curve(tgt.get("per_layer_rho"), num_layers)
        if np.all(np.isnan(a)):
            continue
        peak_idx = int(np.nanargmax(a))
        late_val = a[late]
        if np.isnan(late_val):
            valid = np.where(~np.isnan(a))[0]
            late_val = a[valid[-1]]
        dissipations.append(float(a[peak_idx] - late_val))
        corrects.append(bool(r.get("correct", False)))
        diff_curves.append(a - t)

    dissipations = np.array(dissipations)
    corrects = np.array(corrects, dtype=bool)
    n = len(dissipations)
    n_incorrect = int((~corrects).sum())

    # point-biserial + logistic fit (need both classes present)
    pb = {"r": None, "p": None}
    logistic = {"coef": None, "intercept": None}
    if n >= 3 and corrects.sum() >= 2 and n_incorrect >= 2:
        r_pb, p_pb = pointbiserialr(corrects.astype(int), dissipations)
        pb = {"r": float(r_pb), "p": float(p_pb)}
        from sklearn.linear_model import LogisticRegression
        lr = LogisticRegression(max_iter=500).fit(dissipations.reshape(-1, 1),
                                                  corrects.astype(int))
        logistic = {"coef": float(lr.coef_[0][0]), "intercept": float(lr.intercept_[0])}

    diff_arr = np.vstack(diff_curves) if diff_curves else np.empty((0, num_layers))
    diff_curve = {}
    for l in range(num_layers):
        col = diff_arr[:, l]
        col = col[~np.isnan(col)]
        if col.size:
            diff_curve[str(l)] = {"mean": float(col.mean()), "std": float(col.std()),
                                  "n": int(col.size)}

    return {
        "n_queries": n, "n_incorrect": n_incorrect,
        "underpowered": n_incorrect < 10,
        "late_layer": late, "num_layers": num_layers,
        "dissipation": {"mean": float(dissipations.mean()) if n else None,
                        "std": float(dissipations.std()) if n else None},
        "dissipation_vs_correct": {"pointbiserial": pb, "logistic": logistic},
        "diff_curve": diff_curve,
        "_diff_arr": diff_arr,  # for plotting; stripped before json dump
    }


def plot(sources, out_path):
    from analysis.plot_style import apply_style, PLOT_STYLE, line_kwargs, save_with_legend
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    apply_style()
    tab10 = plt.cm.tab10.colors
    fig, ax = plt.subplots(figsize=PLOT_STYLE["subplot_size"])
    for i, (label, res) in enumerate(sources):
        num_layers = res["num_layers"]
        xs = list(range(num_layers))
        means = [res["diff_curve"].get(str(l), {}).get("mean", np.nan) for l in xs]
        stds = [res["diff_curve"].get(str(l), {}).get("std", 0.0) for l in xs]
        color = tab10[i % 10]
        ax.plot(xs, means, color=color, **line_kwargs(label=label))
        ax.fill_between(xs, [m - s for m, s in zip(means, stds)],
                        [m + s for m, s in zip(means, stds)],
                        color=color, alpha=0.15)
    ax.axhline(y=0, color="black", linewidth=1, alpha=0.4)
    for l in [1, 3, 5, 7, 9, 11]:
        ax.axvline(x=l, color="gray", linestyle="--", linewidth=0.8, alpha=0.15)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Anchor − Target binding ρ")
    save_with_legend(fig, out_path)
    print(f"Saved: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-query", action="append", default=[],
                    help="rsa_per_query.json path (repeatable)")
    ap.add_argument("--baseline", type=str, default=None)
    ap.add_argument("--ablation", type=str, default=None)
    ap.add_argument("--late-layer", type=int, default=11)
    ap.add_argument("--output-dir", type=str,
                    default="outputs/analysis/anchor_dissipation/clevr_dinov2_decoder1l_scratch")
    args = ap.parse_args()

    # Build labeled source list.
    src_paths = []
    if args.baseline:
        src_paths.append(("baseline", args.baseline))
    if args.ablation:
        src_paths.append(("ablation", args.ablation))
    for p in args.per_query:
        src_paths.append((Path(p).parent.name or Path(p).stem, p))
    if not src_paths:
        ap.error("provide at least one of --baseline/--ablation/--per-query")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tee_stdout(out_dir)

    sources, stats = [], {}
    for label, path in src_paths:
        with open(path) as f:
            records = json.load(f)
        res = analyze_source(records, args.late_layer)
        sources.append((label, res))
        pb = res["dissipation_vs_correct"]["pointbiserial"]
        print(f"[{label}] n={res['n_queries']} n_incorrect={res['n_incorrect']}"
              f" underpowered={res['underpowered']} pb_r={pb['r']}", flush=True)
        res_json = {k: v for k, v in res.items() if k != "_diff_arr"}
        res_json["source_path"] = str(path)
        stats[label] = res_json

    (out_dir / "dissipation_stats.json").write_text(json.dumps(stats, indent=2))
    print(f"Saved: {out_dir/'dissipation_stats.json'}", flush=True)
    plot(sources, str(out_dir / "anchor_dissipation.png"))


if __name__ == "__main__":
    main()
