"""Aggregate linear-probe results across models into one readout × backbone
table and one layer-curve figure (X20 story-vs-evidence check).

Reads every `outputs/analysis/linear_probe/*/probe_results.json` produced by
`linear_probe.py` (X10 protocol) and reports, per category, the L11
answer_decode accuracy / answer_match F1, the peak layer and the half-rise
layer (first ViT layer reaching half of the peak). Cells without a run show
"—". Nothing is recomputed; the script never touches a model.

Usage (from main/):
    PYTHONPATH=src <interpreter> scripts/analysis/probe_table.py \
        [--probe-root outputs/analysis/linear_probe] [--category attr_query_direct]

`probe_section_lines()` is imported by `aggregate_results.py` so the same
table lands in docs/results_tables.md.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analysis.plot_style import apply_style, line_kwargs, mark_gca_layers, S

PROBE_ROOT = Path("outputs/analysis/linear_probe")
CATEGORIES = ["attr_query_direct", "attr_query_same", "attr_query_spatial"]
NUM_VIT_LAYERS = 12

# Site naming (docs/site): readouts are named by what they read.
READOUTS = ["CLS token", "local patches", "local patches + question"]
BACKBONES = ["DINOv2", "SigLIP", "Sup-ViT", "MAE"]
BACKBONE_KEY = {"dinov2": "DINOv2", "siglip": "SigLIP", "sup": "Sup-ViT", "mae": "MAE"}
_tab10 = plt.get_cmap("tab10").colors
BACKBONE_COLORS = {"DINOv2": _tab10[0], "SigLIP": _tab10[1], "Sup-ViT": _tab10[2], "MAE": _tab10[3]}

# Ablations are reported as separate rows, not grid cells.
ABLATIONS = {"nogca": "−CA (local patches + question)", "nogate": "ungated CA (local patches)"}

# Runs produced by the pre-2026-08-26 linear_probe.py loader, which dropped
# `use_gate` and nulled the GCA output of ungated checkpoints (see
# linear_probe.load_model). Kept on disk, never aggregated.
INVALID_DIRS = {
    "clevr_dinov2_nogate_scratch": "loader dropped use_gate (GCA nulled)",
    "clevr_dinov2_nogate_scratch_s43": "loader dropped use_gate (GCA nulled)",
    "clevr_siglip_nogate_scratch_s43": "loader dropped use_gate (GCA nulled)",
}

_DIR_RE = re.compile(
    r"^clevr_(?P<bb>dinov2|siglip|sup|mae)_"
    r"(?P<kind>cls|decoder1l|concat_decoder1l|nogate)"
    r"(?P<abl>_nogca)?_scratch(?P<seed>_s\d+)?(?P<ver>_v\d+)?$")


def classify(dir_name: str):
    """Map a probe dir name → (readout, backbone, ablation-or-None, seed) or None."""
    if dir_name == "concat_decoder_1l":  # legacy dir name of the DINOv2 main model
        return "local patches + question", "DINOv2", None, ""
    m = _DIR_RE.match(dir_name)
    if not m:
        return None
    bb = BACKBONE_KEY[m.group("bb")]
    kind, abl, seed = m.group("kind"), m.group("abl"), m.group("seed") or ""
    if kind == "nogate":
        return "local patches", bb, "nogate", seed
    readout = {"cls": "CLS token", "decoder1l": "local patches",
               "concat_decoder1l": "local patches + question"}[kind]
    return readout, bb, ("nogca" if abl else None), seed


def load_runs(probe_root: Path):
    runs = {}
    for d in sorted(probe_root.iterdir()):
        f = d / "probe_results.json"
        if not d.is_dir() or not f.exists():
            continue
        if d.name in INVALID_DIRS:
            print(f"skip {d.name}: {INVALID_DIRS[d.name]}")
            continue
        info = classify(d.name)
        if info is None:
            continue
        with open(f) as fh:
            data = json.load(fh)
        if "categories" not in data:  # legacy pilot schema (single_object/multi_object)
            continue
        runs[d.name] = {"info": info, "data": data}
    return runs


def curve(cat_block, signal):
    """Per-layer scores for ViT layers 0..11 (decoder probe D1 dropped)."""
    res = cat_block["results"]
    out = []
    for l in range(NUM_VIT_LAYERS):
        r = res.get(str(l)) or res.get(l)
        out.append(float(r[signal]["f1"]) if r else float("nan"))
    return out


def summarize(vals):
    finite = [(i, v) for i, v in enumerate(vals) if v == v]
    if not finite:
        return None
    peak_l, peak = max(finite, key=lambda t: t[1])
    # Half-rise = first layer at or above the midpoint between the L0 score
    # (chance-level, no GCA yet) and the peak. Reproduces the site's landmarks
    # for the DINOv2 local patches model (decode L1, match L5).
    base = finite[0][1]
    half = next((i for i, v in finite if v >= base + 0.5 * (peak - base)), None)
    return {"L11": vals[11], "peak": peak, "peak_layer": peak_l, "half_rise_layer": half}


def build_table(runs, category):
    """Return {(readout, backbone or ablation label): {signal: summary}}."""
    table = {}
    for name, r in runs.items():
        readout, bb, abl, seed = r["info"]
        if seed not in ("", "_s42"):
            continue  # one seed per cell; extra seeds are reported nowhere yet
        cat = r["data"]["categories"].get(category)
        if cat is None:
            continue
        key = (ABLATIONS[abl], bb) if abl else (readout, bb)
        table[key] = {sig: summarize(curve(cat, sig)) for sig in ("answer_decode", "answer_match")}
        table[key]["dir"] = name
        table[key]["curves"] = {sig: curve(cat, sig) for sig in ("answer_decode", "answer_match")}
    return table


def fmt(x):
    return "—" if x is None else f"{x:.3f}"


def probe_section_lines(probe_root: Path = PROBE_ROOT, category: str = "attr_query_direct"):
    """Markdown lines for docs/results_tables.md (imported by aggregate_results.py)."""
    runs = load_runs(Path(probe_root))
    table = build_table(runs, category)
    short = category.replace("attr_query_", "")
    lines = ["", f"## Linear probe — readout × backbone ({short}, seed 42)", "",
             "Cell = L11 answer_decode acc / answer_match F1 (X10 protocol, "
             "`linear_probe.py`; aggregated by `probe_table.py`). "
             "Peak and half-rise layers in the second table.", "",
             "| readout | " + " | ".join(BACKBONES) + " |",
             "|---|" + "---|" * len(BACKBONES)]
    rows = READOUTS + list(ABLATIONS.values())
    for ro in rows:
        cells = []
        for bb in BACKBONES:
            c = table.get((ro, bb))
            cells.append("—" if not c else f"{fmt(c['answer_decode']['L11'])} / {fmt(c['answer_match']['L11'])}")
        lines.append(f"| {ro} | " + " | ".join(cells) + " |")
    lines += ["", "| readout | backbone | decode peak (layer) | decode half-rise | "
              "match peak (layer) | match half-rise | run dir |", "|---|---|---|---|---|---|---|"]
    for ro in rows:
        for bb in BACKBONES:
            c = table.get((ro, bb))
            if not c:
                continue
            d, m = c["answer_decode"], c["answer_match"]
            lines.append(f"| {ro} | {bb} | {fmt(d['peak'])} (L{d['peak_layer']}) | L{d['half_rise_layer']} | "
                         f"{fmt(m['peak'])} (L{m['peak_layer']}) | L{m['half_rise_layer']} | `{c['dir']}` |")
    return lines, table


def plot_curves(table, category, out_path: Path):
    apply_style()
    fig, axes = plt.subplots(2, len(READOUTS), figsize=(4.6 * len(READOUTS), 7.2),
                             sharex=True, sharey=True)
    signals = [("answer_decode", "Answer Decode (Acc)"), ("answer_match", "Retrieval (F1)")]
    layers = list(range(NUM_VIT_LAYERS))
    for row, (sig, sig_label) in enumerate(signals):
        for col, ro in enumerate(READOUTS):
            ax = axes[row, col]
            for bb in BACKBONES:
                c = table.get((ro, bb))
                if c:
                    ax.plot(layers, c["curves"][sig], **line_kwargs(label=bb, color=BACKBONE_COLORS[bb]))
            for abl, abl_label in ABLATIONS.items():
                if ABLATIONS[abl].endswith(f"({ro})"):
                    for bb in BACKBONES:
                        c = table.get((abl_label, bb))
                        if c:
                            ax.plot(layers, c["curves"][sig],
                                    **line_kwargs(label=f"{abl_label.split(' (')[0]}, {bb}",
                                                  color=BACKBONE_COLORS[bb],
                                                  linestyle="--" if abl == "nogca" else ":"))
            mark_gca_layers(ax)
            if row == 0:
                ax.set_title(ro)
            if col == 0:
                ax.set_ylabel(sig_label)
            if row == 1:
                ax.set_xlabel("ViT layer")
            ax.set_xticks(layers)
            ax.set_ylim(0, 1.02)
            ax.grid(axis="y", alpha=0.2)
            ax.legend(fontsize=S.get("legend_fontsize", 9), loc="lower right")
    short = category.replace("attr_query_", "")
    fig.suptitle(f"Linear probe per layer — {short} (seed 42)")
    fig.tight_layout()
    fig.savefig(str(out_path))
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe-root", default=str(PROBE_ROOT))
    ap.add_argument("--category", default="attr_query_direct", choices=CATEGORIES)
    args = ap.parse_args()
    root = Path(args.probe_root)
    lines, table = probe_section_lines(root, args.category)
    short = args.category.replace("attr_query_", "")
    md = root / f"probe_table_{short}.md"
    md.write_text("\n".join(lines).lstrip("\n") + "\n")
    print("\n".join(lines))
    js = {f"{ro} | {bb}": {k: v for k, v in c.items() if k != "curves"} | {"curves": c["curves"]}
          for (ro, bb), c in table.items()}
    (root / f"probe_table_{short}.json").write_text(json.dumps(js, indent=2))
    plot_curves(table, args.category, root / f"probe_table_{short}.png")
    print(f"Saved: {md}")


if __name__ == "__main__":
    main()
