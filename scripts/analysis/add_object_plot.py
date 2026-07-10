"""E7 add-object hallucination bar (claim A1.3).

Reads every add_object_eval_*.json under outputs/analysis/add_object/<attr>/
and plots hallucination_rate as grouped bars: x = query attribute, one bar per
model. Prefers the *_fixed json when both exist for a model. No GPU — pure
aggregation, rerunnable whenever a new eval json lands.

New script because nothing under scripts/ plots add_object results
(add_object_eval*.py only evaluate, render_add_object.py only renders).

Usage (from main/):
  PYTHONPATH=src <interpreter> scripts/analysis/add_object_plot.py
Output: outputs/analysis/add_object/hallucination_bar.png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analysis.plot_style import apply_style, S
from analysis.run_log import tee_stdout

ATTRS = ["color", "material", "shape", "size"]

MODEL_DISPLAY = {
    "clevr_dinov2_concat_decoder1l_scratch_s42": "SteerViT (GCA)",
    "clevr_dinov2_concat_decoder1l_nogca_scratch_s42": "no-GCA",
    "odd_scratch_decoder_1l": "scratch-ViT",
    "clevr_flamingo_dinov2_early_s42": "Flamingo (4ep)",
    "clevr_flamingo_dinov2_frozenllm_s42": "Flamingo (8ep)",
}


def load_results(root: Path):
    """{attr: {model: summary}} — *_fixed json wins over its plain twin."""
    results = {}
    for attr in ATTRS:
        per_model = {}
        for f in sorted((root / attr).glob("add_object_eval_*.json")):
            d = json.loads(f.read_text())
            model = d["model"]
            if model in per_model and not f.stem.endswith("_fixed"):
                continue  # keep the _fixed one already loaded / to come
            per_model[model] = d
        results[attr] = per_model
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="outputs/analysis/add_object")
    ap.add_argument("--metric", default="hallucination_rate")
    args = ap.parse_args()

    root = Path(args.root)
    tee_stdout(root)
    apply_style()

    results = load_results(root)
    models = sorted({m for per in results.values() for m in per},
                    key=lambda m: list(MODEL_DISPLAY).index(m)
                    if m in MODEL_DISPLAY else 99)
    _tab10 = plt.cm.tab10.colors
    colors = {m: _tab10[i % 10] for i, m in enumerate(models)}

    fig, ax = plt.subplots(figsize=(8, 6))
    width = 0.8 / max(len(models), 1)
    x = np.arange(len(ATTRS))
    for i, m in enumerate(models):
        vals = [results[a].get(m, {}).get(args.metric, np.nan) for a in ATTRS]
        ax.bar(x + (i - (len(models) - 1) / 2) * width, vals, width,
               label=MODEL_DISPLAY.get(m, m), color=colors[m])
        print(f"{m}: " + "  ".join(f"{a}={v:.3f}" if v == v else f"{a}=--"
                                   for a, v in zip(ATTRS, vals)))
    ax.set_xticks(x)
    ax.set_xticklabels([a.capitalize() for a in ATTRS])
    ax.set_ylabel(args.metric.replace("_", " "))
    ax.set_title("Add-object perturbation (E7)",
                 fontsize=S["subplot_title_fontsize"])

    fig.tight_layout()
    fig.legend(loc="upper center", bbox_to_anchor=(0.5, -0.02),
               ncol=min(len(models), 5), fontsize=S["legend_fontsize"],
               frameon=False)
    out = root / f"{args.metric}_bar.png" if args.metric != "hallucination_rate" \
        else root / "hallucination_bar.png"
    fig.savefig(out, dpi=S["dpi"], bbox_inches="tight")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
