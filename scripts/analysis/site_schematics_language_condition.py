"""Site figures for the language-condition analyses (no existing schematic code
in the repo — every other script plots measured curves only).

1. schematic_token_swap_design.png — design of the between-condition token
   swap: two forward passes, one block's patch-token group replaced, decoder
   reads the answer.
2. schematic_mechanism_by_block.png — 12-block × 5-row grid of the measured
   quantities per backbone (numbers from the result JSONs, nothing hand-set).
3. schematic_vector_decomposition.png — the vector relations (background vector,
   object vector, question-general component q, selection component s) and
   their measured projections per run.

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

def rows_for(queried: str):
    other = "shape" if queried == "color" else "color"
    q, o = ("colour" if queried == "color" else queried), ("colour" if other == "color" else other)
    return [
        (f"queried attribute ({q}) on the target:\nquestion − no question", f"refvs0_target_{queried}_own", "attr", "div"),
        (f"unqueried attribute ({o}) on the target:\nquestion − no question", f"refvs0_target_{other}_own", "attr", "div"),
        (f"selection: target's own {q},\nrefer target − refer distractor", f"ref_target_{queried}_own", "attr", "div"),
        ("answer = distractor's value after replacing\nboth objects' tokens from the other question", ("c2", "objects"), "swap", "seq"),
        ("answer = distractor's value after replacing\nbackground tokens from the other question", ("c2", "bg"), "swap", "seq"),
    ]


def load_rows(out_dir: Path, rows):
    attr = json.load(open(out_dir / "partA_attr_directions.json"))["delta"]
    swap = json.load(open(out_dir / "readout_swap.json"))["rows"]
    vals = []
    for _, key, src, _ in rows:
        if src == "attr":
            vals.append([q["mean"] for q in attr[key]])
        else:
            dc, mk = key
            rr = sorted((r for r in swap if r["donor"] == dc and r["mask"] == mk), key=lambda r: r["layer"])
            vals.append([r["p_distractor"] for r in rr])
    return np.array(vals)


def draw_summary(panels: list, out_path: Path):
    """panels: list of (label, out_dir, queried)."""
    fig, axes = plt.subplots(len(panels), 1, figsize=(12, 3.0 * len(panels)), squeeze=False)
    for ax, (label, d, queried) in zip(axes[:, 0], panels):
        rows = rows_for(queried)
        v = load_rows(d, rows)
        img = np.zeros((len(rows), NUM_LAYERS, 4))
        for i, (_, _, _, kind) in enumerate(rows):
            row = v[i]
            if kind == "div":
                m = np.abs(row).max() or 1.0
                img[i] = plt.get_cmap("RdBu_r")(0.5 + 0.5 * row / m)
            else:
                img[i] = plt.get_cmap("Oranges")(0.1 + 0.8 * row)
        ax.imshow(img, aspect="auto", interpolation="nearest")
        for i in range(len(rows)):
            for l in range(NUM_LAYERS):
                x = v[i, l]
                txt = ("0" if abs(x) < 0.5 else f"{x:+.0f}") if rows[i][3] == "div" else f"{x:.2f}"
                lum = 0.299 * img[i, l, 0] + 0.587 * img[i, l, 1] + 0.114 * img[i, l, 2]
                ax.text(l, i, txt, ha="center", va="center", fontsize=7.5, color="white" if lum < 0.5 else "black")
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([r[0] for r in rows], fontsize=7.5)
        ax.set_xticks(range(NUM_LAYERS))
        ax.set_xticklabels([str(l) for l in range(NUM_LAYERS)], fontsize=9)
        ax.set_xlabel("ViT block", fontsize=9)
        ax.set_title(f"{label}, questions ask about {'colour' if queried == 'color' else queried}", fontsize=10, loc="left")
        for l in (1, 3, 5, 7, 9, 11):
            ax.axvline(l - 0.5, color="white", linewidth=0.4)
        ax.set_ylim(len(rows) - 0.5, -0.5)
    fig.suptitle("Block-by-block summary of the measured effects (rows 1–3: projection differences, units of the "
                 "feature space, red = positive; rows 4–5: proportion of images)", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=S["dpi"], bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# 3. vector decomposition: who minus whom is what
# ---------------------------------------------------------------------------

RUNS_DECOMP = [  # (label, subdir, queried attribute, block)
    ("DINOv2\ncolour\nblock 8", ".", "color", 8),
    ("DINOv2\ncolour\nblock 11", ".", "color", 11),
    ("DINOv2\nshape\nblock 11", "shape", "shape", 11),
    ("DINOv2\nmaterial\nblock 11", "material", "material", 11),
    ("DINOv2\nsize\nblock 11", "size", "size", 11),
    ("SigLIP\ncolour\nblock 11", "siglip", "color", 11),
    ("MAE\ncolour\nblock 11", "mae", "color", 11),
]


def _decomp_rows(root):
    """Per run: q = <q, v> (non-referring question − no question),
    s_ref = (referent − no question) − q, s_nonref = (non-referent − no question) − q.
    All three are projections of the target's mean token on its own
    queried-attribute direction, from partA_attr_directions.json."""
    rows = []
    for label, sub, attr, blk in RUNS_DECOMP:
        f = root / sub / "partA_attr_directions.json"
        if not f.exists():
            continue
        d = json.load(open(f))["delta"]
        q = d[f"c3vs0_target_{attr}_own"][blk]["mean"]
        r = d[f"refvs0_target_{attr}_own"][blk]["mean"]
        n = d[f"nonrefvs0_target_{attr}_own"][blk]["mean"]
        rows.append((label, q, r - q, n - q))
    return rows


def _arrow(ax, a, b, color, lw=2.5, ls="-", z=3):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=18,
                                 color=color, lw=lw, linestyle=ls, zorder=z))


def _vector_panel(ax, mode):
    """mode = 'removal' (DINOv2 / SigLIP) or 'marking' (MAE)."""
    ax.set_xlim(-0.7, 4.7)
    ax.set_ylim(-0.9, 3.2)
    ax.set_aspect("equal")
    ax.axis("off")
    # axes: horizontal = the object's own queried-attribute direction v
    _arrow(ax, (-0.4, 0), (4.5, 0), "k", lw=1.5, z=1)
    _arrow(ax, (0, -0.3), (0, 2.9), "k", lw=1.5, z=1)
    ax.text(2.2, -0.55, "own queried-attribute direction $u$", ha="center", va="top", fontsize=12)
    ax.text(-0.15, 2.85, "other directions", ha="right", va="top", fontsize=12, rotation=90)
    ax.plot([0], [0], "ko", ms=6, zorder=4)
    ax.text(-0.15, 0.12, "$b_\\ell(p)$\nbackground\nvector", ha="right", va="bottom", fontsize=10)
    P0 = (1.9, 1.0)      # no question
    _arrow(ax, (0, 0), P0, "0.35")
    ax.text(0.75, 0.75, "$o^{\\varnothing}$", fontsize=14, color="0.35", ha="right")
    ax.plot([P0[0]], [P0[1]], "o", color="0.35", ms=6, zorder=4)
    ax.text(P0[0] - 0.1, P0[1] + 0.12, "no question", fontsize=11, color="0.35", ha="right")
    if mode == "removal":
        P1 = (3.6, 1.4)  # after the question-general component q
        _arrow(ax, P0, P1, "#2ca02c")
        ax.text(2.7, 1.35, "$q$", fontsize=14, color="#2ca02c")
        ax.plot([P1[0]], [P1[1]], "o", color=TARGET_RGB, ms=8, zorder=5)
        ax.text(P1[0] + 0.1, P1[1] - 0.05, "referent\n$s_{\\mathrm{ref}} \\approx 0$", fontsize=11, color=TARGET_RGB, va="top")
        P2 = (0.8, 1.9)
        _arrow(ax, P1, P2, DISTRACTOR_RGB)
        ax.text(2.3, 1.9, "$s_{\\mathrm{non\\text{-}ref}}$", fontsize=14, color=DISTRACTOR_RGB)
        ax.plot([P2[0]], [P2[1]], "o", color=DISTRACTOR_RGB, ms=8, zorder=5)
        ax.text(P2[0], P2[1] + 0.18, "non-referent", fontsize=11, color=DISTRACTOR_RGB, ha="center")
        # projections on v
        for P, c in ((P0, "0.35"), (P1, TARGET_RGB), (P2, DISTRACTOR_RGB)):
            ax.plot([P[0], P[0]], [0, P[1]], ls=":", color=c, lw=1.2, zorder=2)
        ax.text(P0[0], -0.1, "$\\pi^{\\varnothing}$", ha="center", va="top", fontsize=12, color="0.35")
        ax.text(P1[0], -0.1, "$\\pi_{\\mathrm{ref}}$", ha="center", va="top", fontsize=12, color=TARGET_RGB)
        ax.text(P2[0], -0.1, "$\\pi_{\\mathrm{non\\text{-}ref}}$", ha="center", va="top", fontsize=12, color=DISTRACTOR_RGB)
        ax.set_title("DINOv2, SigLIP: selection by removal", fontsize=S["subplot_title_fontsize"])
    else:
        ax.plot([P0[0]], [P0[1]], "o", color=DISTRACTOR_RGB, ms=8, zorder=5)
        ax.text(P0[0] + 0.15, P0[1] - 0.05, "non-referent:\nunchanged", fontsize=11, color=DISTRACTOR_RGB, va="top")
        P1 = (1.9, 2.5)
        _arrow(ax, P0, P1, TARGET_RGB)
        ax.text(2.05, 1.7, "$m$: referent marker,\n$\\perp v$", fontsize=12, color=TARGET_RGB)
        ax.plot([P1[0]], [P1[1]], "o", color=TARGET_RGB, ms=8, zorder=5)
        ax.text(P1[0] + 0.1, P1[1] + 0.05, "referent", fontsize=11, color=TARGET_RGB)
        ax.plot([P0[0], P0[0]], [0, P0[1]], ls=":", color="0.35", lw=1.2, zorder=2)
        ax.text(P0[0], -0.1, "$\\pi^{\\varnothing} = \\pi_{\\mathrm{ref}} = \\pi_{\\mathrm{non\\text{-}ref}}$",
                ha="center", va="top", fontsize=12)
        ax.text(4.6, 0.55, "decoder attention selects\nthe marked object", fontsize=11, ha="right", va="top")
        ax.set_title("MAE: selection by marking", fontsize=S["subplot_title_fontsize"])


def draw_vectors(root, out_path):
    rows = _decomp_rows(root)
    fig = plt.figure(figsize=(22, 6.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.5], wspace=0.3)
    _vector_panel(fig.add_subplot(gs[0]), "removal")
    _vector_panel(fig.add_subplot(gs[1]), "marking")
    ax = fig.add_subplot(gs[2])
    x = np.arange(len(rows))
    w = 0.27
    ax.bar(x - w, [r[1] for r in rows], w, color="#2ca02c", label="$\\langle q, u\\rangle$: question, both objects")
    ax.bar(x, [r[2] for r in rows], w, color=TARGET_RGB, label="$\\langle s_{\\mathrm{ref}}, u\\rangle$: referent only")
    ax.bar(x + w, [r[3] for r in rows], w, color=DISTRACTOR_RGB, label="$\\langle s_{\\mathrm{non\\text{-}ref}}, u\\rangle$: non-referent only")
    ax.axhline(0, color="k", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels([r[0] for r in rows], fontsize=11)
    ax.set_ylabel("projection change vs. no question", fontsize=S["label_fontsize"])
    ax.tick_params(axis="y", labelsize=S["tick_labelsize"], width=S["tick_width"])
    ax.legend(fontsize=12, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, frameon=False)
    ax.set_title("Measured decomposition (target object)", fontsize=S["subplot_title_fontsize"])
    fig.suptitle("$h(p) = b(p) + o(i)$;   after a question: $o = o^{\\varnothing} + q + s$",
                 fontsize=S["suptitle_fontsize"], y=1.02)
    fig.savefig(out_path, dpi=S["dpi"], bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")
    for r in rows:
        print("  %-22s q=%+6.1f  s_ref=%+6.1f  s_nonref=%+6.1f" % (r[0].replace("\n", " "), r[1], r[2], r[3]))


# ---------------------------------------------------------------------------
# 4. background-vector definition
# ---------------------------------------------------------------------------

def _mini_grid(ax, x0, y0, n, cell, objs=(), hl=None, excl=False):
    """n x n grid at (x0,y0). objs: {(r,c): colour}. hl: (r,c) highlighted position.
    excl: cross over the highlighted cell (position not background in this image)."""
    for r in range(n):
        for c in range(n):
            col = objs.get((r, c), "#e8e6e1")
            ax.add_patch(Rectangle((x0 + c * cell, y0 + (n - 1 - r) * cell), cell, cell,
                                   facecolor=col, edgecolor="white", lw=0.6))
    if hl is None:
        return None
    r, c = hl
    x, y = x0 + c * cell, y0 + (n - 1 - r) * cell
    ax.add_patch(Rectangle((x, y), cell, cell, facecolor="none", edgecolor="k", lw=2.2, zorder=5))
    if excl:
        ax.plot([x + 0.12 * cell, x + 0.88 * cell], [y + 0.12 * cell, y + 0.88 * cell], "k-", lw=1.6, zorder=6)
        ax.plot([x + 0.12 * cell, x + 0.88 * cell], [y + 0.88 * cell, y + 0.12 * cell], "k-", lw=1.6, zorder=6)
    return x + cell / 2, y + cell / 2


def draw_template(out_path):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(21, 8.0), gridspec_kw={"width_ratios": [1.35, 1]})
    for ax, xmax in ((a1, 15.2), (a2, 11.2)):
        ax.set_xlim(0, xmax)
        ax.set_ylim(-2.6, 6.6)
        ax.set_aspect("equal")
        ax.axis("off")
    n, cell = 6, 0.55
    gw = n * cell                                   # 3.3
    HL = (3, 2)                                     # the position p, same in every image
    OBJ1 = {(1, 4): TARGET_RGB, (1, 5): TARGET_RGB, (2, 4): TARGET_RGB}
    OBJ2 = {(3, 2): DISTRACTOR_RGB, (3, 3): DISTRACTOR_RGB, (4, 2): DISTRACTOR_RGB}   # covers p
    OBJ3 = {(4, 0): TARGET_RGB, (5, 0): TARGET_RGB, (5, 1): TARGET_RGB}
    y0 = 1.6
    centers = []
    for k, (name, objs, excl) in enumerate([("image 1", OBJ1, False), ("image 2", OBJ2, True),
                                            ("image 3", OBJ3, False)]):
        xg = 0.2 + k * (gw + 0.55)
        cx, cy = _mini_grid(a1, xg, y0, n, cell, objs, HL, excl)
        a1.text(xg + gw / 2, y0 - 0.32, name, ha="center", va="top", fontsize=14, clip_on=True)
        a1.text(xg + gw / 2, y0 + gw + 0.12,
                "$p$ on background ✓" if not excl else "$p$ on an object ✗",
                ha="center", va="bottom", fontsize=13, color="k" if not excl else "0.45", clip_on=True)
        if not excl:
            centers.append((cx, cy))
    a1.text(0.2 + 3 * (gw + 0.55) + 0.15, y0 + gw / 2, "· · ·", fontsize=16, va="center")
    tx, ty = 13.5, y0 + gw / 2 - 0.4
    a1.add_patch(Rectangle((tx, ty), 0.9, 0.9, facecolor="#c8c4bc", edgecolor="k", lw=2.4))
    a1.text(tx + 0.45, ty + 1.05, "$\\hat b_\\ell(p)$", ha="center", fontsize=18)
    a1.text(tx + 0.45, ty - 0.3, "mean of the\n✓ tokens", ha="center", va="top", fontsize=13)
    for cx, cy in centers:
        a1.add_patch(FancyArrowPatch((cx, cy), (tx, ty + 0.45), arrowstyle="-|>", mutation_scale=13,
                                     color="0.45", lw=1.3, connectionstyle="arc3,rad=-0.22"))
    a1.set_title("Step 1 — estimate the background vector", fontsize=21, pad=14)
    a1.text(7.6, -1.0,
            "$\\hat b_\\ell(p) \\;=\\; \\frac{1}{|\\mathcal{I}_p|}\\sum_{I \\in \\mathcal{I}_p} h_\\ell(p;\\, I)$,"
            "$\\qquad \\mathcal{I}_p = \\{\\,\\mathrm{images}\\ I:\\ p\\ \\mathrm{is\\ background\\ in}\\ I\\,\\}$",
            ha="center", va="top", fontsize=17)
    a1.text(7.6, -2.0, "$|\\mathcal{I}_p|$ = 19–60 per position (mean 36); estimated once, from the no-question passes only",
            ha="center", va="top", fontsize=13, color="0.35")
    # ---- step 2
    OBJ = {(2, 1): TARGET_RGB, (2, 2): TARGET_RGB, (3, 1): TARGET_RGB}
    x0 = 0.3
    _mini_grid(a2, x0, y0, n, cell, OBJ)
    pcs = []
    for (r, c) in sorted(OBJ):
        x, y = x0 + c * cell, y0 + (n - 1 - r) * cell
        a2.add_patch(Rectangle((x, y), cell, cell, facecolor="none", edgecolor="k", lw=2.0, zorder=5))
        pcs.append((x + cell / 2, y + cell / 2))
    a2.text(x0 + gw / 2, y0 - 0.32, "one image; object $i$ occupies $p_1, p_2, p_3$",
            ha="center", va="top", fontsize=14)
    colx = 5.4
    rows = [4.7, 3.3, 1.9]
    for (cx, cy), ry, k in zip(pcs, rows, (1, 2, 3)):
        a2.add_patch(FancyArrowPatch((cx, cy), (colx - 0.35, ry + 0.27), arrowstyle="-|>", mutation_scale=12,
                                     color="0.45", lw=1.2, connectionstyle="arc3,rad=0.12"))
        a2.add_patch(Rectangle((colx, ry), 0.55, 0.55, facecolor=TARGET_RGB, edgecolor="k", lw=1.2))
        a2.text(colx + 0.7, ry + 0.27, "$-$", fontsize=15, va="center")
        a2.add_patch(Rectangle((colx + 0.95, ry), 0.55, 0.55, facecolor="#c8c4bc", edgecolor="k", lw=1.2))
        a2.text(colx + 1.68, ry + 0.27, f"$h(p_{k}) - \\hat b(p_{k})$", fontsize=15, va="center")
    a2.add_patch(FancyArrowPatch((colx + 2.2, 1.65), (colx + 2.2, 0.85), arrowstyle="-|>", mutation_scale=15,
                                 color="k", lw=1.8))
    a2.text(colx + 2.5, 1.25, "mean over the\nobject's patches", fontsize=13, va="center")
    a2.text(colx + 2.2, 0.35, "$\\hat o_\\ell(i)$", ha="center", fontsize=18)
    a2.set_title("Step 2 — recover the object vector", fontsize=21, pad=14)
    a2.text(5.6, -1.0,
            "$\\hat o_\\ell(i) \\;=\\; \\frac{1}{|P_i|}\\sum_{p \\in P_i}\\left[\\,h_\\ell(p) - \\hat b_\\ell(p)\\,\\right]$",
            ha="center", va="top", fontsize=17)
    a2.text(5.6, -2.0, "the same $\\hat b_\\ell$ is subtracted in all four question conditions",
            ha="center", va="top", fontsize=13, color="0.35")
    fig.suptitle("Additive model:  $h_\\ell(p) \\;=\\; b_\\ell(p) \\;+\\; o_\\ell(i)\\,\\cdot\\,"
                 "\\mathbf{1}[\\,p \\in \\mathrm{object}\\ i\\,]$"
                 "  —  every patch contains the background term; only object patches add an object term",
                 fontsize=20, y=1.02)
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
    panels = [("DINOv2", out, "color")]
    if (out / "shape" / "readout_swap.json").exists():
        panels.append(("DINOv2", out / "shape", "shape"))
    if (out / "siglip" / "readout_swap.json").exists():
        panels.append(("SigLIP", out / "siglip", "color"))
    draw_summary(panels, out / "schematic_mechanism_by_block.png")
    draw_vectors(out, out / "schematic_vector_decomposition.png")
    draw_template(out / "schematic_background_template.png")


if __name__ == "__main__":
    main()
