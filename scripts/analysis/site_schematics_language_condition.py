"""Site figures for the language-condition analyses (no existing schematic code
in the repo — every other script plots measured curves only).

1. schematic_token_swap_design.png — design of the between-condition token
   swap: two forward passes, one block's patch-token group replaced, decoder
   reads the answer.
2. schematic_mechanism_by_block.png — 12-block × 5-row grid of the measured
   quantities per backbone (numbers from the result JSONs, nothing hand-set).

Run from main/ (CPU):
  PYTHONPATH=src <py> scripts/analysis/site_schematics_language_condition.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle, Circle

from analysis.plot_style import S, apply_style

NUM_LAYERS = 12
GRID = 6            # patch grid drawn in the design schematic
BG = "#d9d9d9"
TARGET_RGB = "#d62728"
DISTRACTOR_RGB = "#1f77b4"


# ---------------------------------------------------------------------------
# 1. design schematic
# ---------------------------------------------------------------------------

def _scene(ax, x0, y0, w, h, target_on=True):
    """A small 6×6 patch grid with a target (red, cube) and a distractor
    (blue, sphere); returns the sets of patch indices for each."""
    cell = w / GRID
    t_cells = {(1, 1), (2, 1), (1, 2), (2, 2)}
    d_cells = {(4, 3), (4, 4)}
    for r in range(GRID):
        for c in range(GRID):
            col = BG
            if (c, r) in t_cells:
                col = TARGET_RGB
            elif (c, r) in d_cells:
                col = DISTRACTOR_RGB
            ax.add_patch(Rectangle((x0 + c * cell, y0 + (GRID - 1 - r) * cell), cell, cell,
                                   facecolor=col, edgecolor="white", linewidth=0.6))
    ax.add_patch(Rectangle((x0, y0), w, h, fill=False, edgecolor="0.4", linewidth=0.8))


def _block_row(ax, x0, y, label, highlight):
    """Row of 12 block boxes starting at x0; returns the x-centre of each."""
    w, gap = 1.25, 0.15
    xs = []
    for l in range(NUM_LAYERS):
        x = x0 + l * (w + gap)
        fc = "#ffe8cc" if highlight == l else "white"
        ax.add_patch(FancyBboxPatch((x, y - 0.42), w, 0.84, boxstyle="round,pad=0.02",
                                    facecolor=fc, edgecolor="0.3", linewidth=0.8))
        ax.text(x + w / 2, y, f"{l}", ha="center", va="center", fontsize=8)
        xs.append(x + w / 2)
    ax.text(x0, y + 0.7, label, ha="left", va="bottom", fontsize=9)
    return xs, x0 + NUM_LAYERS * (w + gap) - gap


def draw_design(out_path: Path, layer: int = 9):
    fig, ax = plt.subplots(figsize=(12, 5.2))
    ax.set_xlim(0, 29.5)
    ax.set_ylim(0, 10.5)
    ax.axis("off")
    yA, yB = 8.0, 3.2
    x0 = 7.6

    # scenes and questions
    for y, q, who in ((yA, "\"What color is the cube?\"", "question about the target"),
                      (yB, "\"What color is the sphere?\"", "question about the distractor\n(or no question)")):
        _scene(ax, 2.2, y - 0.2, 1.7, 1.7)
        ax.text(0.4, y - 0.65, q, fontsize=8, va="top")
        ax.text(0.4, y - 1.1, who, fontsize=7.5, va="top", color="0.35")
    ax.text(0.4, 10.1, "same image, two questions", fontsize=8.5, ha="left")

    # block rows
    xsA, xendA = _block_row(ax, x0, yA, "pass A", layer)
    xsB, xendB = _block_row(ax, x0, yB, "pass B", layer)
    ax.text(x0 + 1.6, yA + 0.7, "ViT blocks (frozen ViT with cross-attention layers; the two passes differ only in the question)",
            ha="left", va="bottom", fontsize=7.5, color="0.35")

    # replacement arrow B → A at the highlighted block
    ax.add_patch(FancyArrowPatch((xsB[layer], yB + 0.5), (xsA[layer], yA - 0.5), arrowstyle="-|>",
                                 mutation_scale=16, color="#e6550d", linewidth=2.0))
    ax.text(xsA[layer] - 0.6, 6.8, f"at the output of block {layer}, one group of patch tokens\n"
            "in pass A is replaced by pass B's values", fontsize=8, color="#e6550d", va="center", ha="right")
    # groups legend under the arrow
    groups = [("background patches", BG), ("both objects' patches", "#9467bd"),
              ("target's patches only", TARGET_RGB), ("distractor's patches only", DISTRACTOR_RGB)]
    gx = xsA[layer] - 7.2
    for k, (name, col) in enumerate(groups):
        y = 5.45 - k * 0.42
        ax.add_patch(Rectangle((gx, y - 0.14), 0.42, 0.28, facecolor=col, edgecolor="0.3", linewidth=0.6))
        ax.text(gx + 0.55, y, name, va="center", fontsize=7.5)
    ax.text(gx, 5.9, "replaced group (one per run):", fontsize=7.5, color="0.35")

    # decoder
    ax.add_patch(FancyBboxPatch((xendA + 0.6, yA - 0.6), 2.3, 1.2, boxstyle="round,pad=0.03",
                                facecolor="#e8f0fa", edgecolor="0.3"))
    ax.text(xendA + 1.75, yA, "decoder\n(local patches)", ha="center", va="center", fontsize=8)
    ax.annotate("", xy=(xendA + 0.6, yA), xytext=(xendA + 0.05, yA), arrowprops=dict(arrowstyle="->", color="0.3"))
    ax.text(xendA + 1.75, yA + 0.95, "blocks after the replacement\nrun unchanged", ha="center", fontsize=7.5, color="0.35")
    ax.annotate("", xy=(xendA + 3.4, yA), xytext=(xendA + 2.95, yA), arrowprops=dict(arrowstyle="->", color="0.3"))
    ax.text(xendA + 3.5, yA + 0.55, "answer read as", fontsize=8, va="center")
    ax.text(xendA + 3.5, yA + 0.15, "target's colour", fontsize=8, color=TARGET_RGB, va="center")
    ax.text(xendA + 3.5, yA - 0.25, "distractor's colour", fontsize=8, color=DISTRACTOR_RGB, va="center")
    ax.text(xendA + 3.5, yA - 0.65, "other", fontsize=8, color="0.4", va="center")
    ax.annotate("", xy=(xendB + 0.6, yB), xytext=(xendB + 0.05, yB), arrowprops=dict(arrowstyle="->", color="0.6"))
    ax.text(xendB + 0.7, yB, "not read", fontsize=7.5, color="0.5", va="center")

    ax.text(x0, 1.2, "controls: replacing from pass A itself reproduces the baseline at every block (1.00); "
            "the block of replacement is swept 0–11; n = images correct under both questions",
            fontsize=7.5, color="0.35")
    fig.suptitle("Between-question token replacement: which patch tokens change the decoder's answer, and at which block",
                 fontsize=10)
    fig.savefig(out_path, dpi=S["dpi"], bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# 2. block-by-block summary grid from the result JSONs
# ---------------------------------------------------------------------------

ROWS = [
    ("asked attribute (colour) on the target:\nquestion − no question", "refvs0_target_color_own", "attr", "div"),
    ("unasked attribute (shape) on the target:\nquestion − no question", "refvs0_target_shape_own", "attr", "div"),
    ("selection: target's own colour,\nrefer target − refer distractor", "ref_target_color_own", "attr", "div"),
    ("answer = distractor's colour after replacing\nboth objects' tokens from the other question", ("c2", "objects"), "swap", "seq"),
    ("answer = distractor's colour after replacing\nbackground tokens from the other question", ("c2", "bg"), "swap", "seq"),
]


def load_rows(out_dir: Path):
    attr = json.load(open(out_dir / "partA_attr_directions.json"))["delta"]
    swap = json.load(open(out_dir / "readout_swap.json"))["rows"]
    vals = []
    for _, key, src, _ in ROWS:
        if src == "attr":
            vals.append([q["mean"] for q in attr[key]])
        else:
            dc, mk = key
            rr = sorted((r for r in swap if r["donor"] == dc and r["mask"] == mk), key=lambda r: r["layer"])
            vals.append([r["p_distractor"] for r in rr])
    return np.array(vals)


def draw_summary(dirs: dict[str, Path], out_path: Path):
    fig, axes = plt.subplots(len(dirs), 1, figsize=(12, 3.0 * len(dirs)), squeeze=False)
    for ax, (label, d) in zip(axes[:, 0], dirs.items()):
        v = load_rows(d)
        img = np.zeros((len(ROWS), NUM_LAYERS, 4))
        for i, (_, _, _, kind) in enumerate(ROWS):
            row = v[i]
            if kind == "div":
                m = np.abs(row).max() or 1.0
                img[i] = plt.get_cmap("RdBu_r")(0.5 + 0.5 * row / m)
            else:
                img[i] = plt.get_cmap("Oranges")(0.1 + 0.8 * row)
        ax.imshow(img, aspect="auto", interpolation="nearest")
        for i in range(len(ROWS)):
            for l in range(NUM_LAYERS):
                x = v[i, l]
                txt = ("0" if abs(x) < 0.5 else f"{x:+.0f}") if ROWS[i][3] == "div" else f"{x:.2f}"
                lum = 0.299 * img[i, l, 0] + 0.587 * img[i, l, 1] + 0.114 * img[i, l, 2]
                ax.text(l, i, txt, ha="center", va="center", fontsize=7.5, color="white" if lum < 0.5 else "black")
        ax.set_yticks(range(len(ROWS)))
        ax.set_yticklabels([r[0] for r in ROWS], fontsize=7.5)
        ax.set_xticks(range(NUM_LAYERS))
        ax.set_xticklabels([str(l) for l in range(NUM_LAYERS)], fontsize=9)
        ax.set_xlabel("ViT block", fontsize=9)
        ax.set_title(label, fontsize=10, loc="left")
        for l in (1, 3, 5, 7, 9, 11):
            ax.axvline(l - 0.5, color="white", linewidth=0.4)
        ax.set_ylim(len(ROWS) - 0.5, -0.5)
    fig.suptitle("Block-by-block summary of the measured effects (rows 1–3: projection differences, units of the "
                 "feature space, red = positive; rows 4–5: proportion of images, n ≥ 320)", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=S["dpi"], bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="outputs/analysis/patch_language_condition")
    ap.add_argument("--swap-layer", type=int, default=9)
    args = ap.parse_args()
    apply_style()
    out = Path(args.out_dir)
    draw_design(out / "schematic_token_swap_design.png", args.swap_layer)
    draw_summary({"DINOv2": out, "SigLIP": out / "siglip"}, out / "schematic_mechanism_by_block.png")


if __name__ == "__main__":
    main()
