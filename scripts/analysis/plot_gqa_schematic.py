"""Explanatory schematic for the GQA real-image validation.

No existing script draws a schematic: every other figure in
`patch_language_condition.py` plots one measured series per panel, and the point
here is to put the four outcomes and the two transfer directions into one picture
that can be read without the registry. The depth curves are the measured
per-block values read out of the result JSONs, not drawn by hand, so the figure
stays honest; only the boxes and arrows are illustration.

Run (CPU, no model):
  OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES= PYTHONPATH=src <interp> \
      scripts/analysis/plot_gqa_schematic.py
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

sys.path.insert(0, "src")
from analysis.plot_style import S, apply_style  # noqa: E402

R = Path("outputs/analysis/patch_language_condition")
OUT = R / "x23_schematic"

_t10 = plt.cm.tab10.colors
C_A, C_T, C_D, C_BG = _t10[4], _t10[3], _t10[0], (0.55,) * 3   # anchor / answer / other / background
C_OK, C_NO, C_GREY = _t10[2], _t10[3], (0.72,) * 3


def box(ax, x, y, w, h, fc, ec=None, lw=1.8, r=0.02, alpha=1.0, z=2):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                       fc=fc, ec=ec or fc, lw=lw, alpha=alpha, zorder=z)
    ax.add_patch(p)
    return p


def arrow(ax, xy0, xy1, color="0.3", lw=2.2, style="-|>", ms=14, z=3, ls="-"):
    ax.add_patch(FancyArrowPatch(xy0, xy1, arrowstyle=style, mutation_scale=ms,
                                 lw=lw, color=color, zorder=z, linestyle=ls,
                                 shrinkA=2, shrinkB=2))


def blank(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")


def main():
    apply_style()
    direct = json.loads((R / "x23_gqa_direct" / "x23_results.json").read_text())
    spatial = json.loads((R / "x23_gqa_spatial_h2" / "x23_results.json").read_text())

    ref = [b["mean"] for b in direct["h1"]["per_block"]["ref"]]
    nonref = [b["mean"] for b in direct["h1"]["per_block"]["nonref"]]
    tmd = [b["mean"] for b in spatial["h4"]["per_block"]["T_minus_D"]]
    probe_c1 = spatial["h2_observational"]["r2"]["c1"]
    probe_sh = spatial["h2_observational"]["r2"]["c1_shuffled"]
    mass = direct["decoder_attention"]["c1"]["per_token_mass"]
    ratio = direct["decoder_attention"]["c1"]["ratio_referent_over_bg_b"]

    fig = plt.figure(figsize=(19, 13.5))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.15, 1.0, 0.95],
                          hspace=0.34, wspace=0.22,
                          left=0.05, right=0.97, top=0.90, bottom=0.05)

    fig.suptitle("Referent selection on real photographs: what reproduced, what could not be tested",
                 fontsize=S["suptitle_fontsize"], y=0.965)

    # ---------------------------------------------------------------- row 0 --
    # (a) the stimulus and the two questions that differ only in which object is asked about
    ax = fig.add_subplot(gs[0, 0])
    blank(ax)
    ax.set_title("a  One image, two questions", fontsize=S["subplot_title_fontsize"], loc="left")
    box(ax, 0.02, 0.40, 0.96, 0.52, "0.94", ec="0.6", lw=1.5, r=0.02, z=1)
    for x, y, w, h, c, lab in [(0.10, 0.52, 0.22, 0.26, C_A, "A  anchor"),
                               (0.40, 0.58, 0.24, 0.24, C_T, "T  asked about"),
                               (0.72, 0.48, 0.20, 0.22, C_D, "D  third object")]:
        ax.add_patch(Rectangle((x, y), w, h, fc=c, ec="k", lw=1.6, alpha=0.75, zorder=3))
        ax.text(x + w / 2, y - 0.055, lab, ha="center", va="top",
                fontsize=S["tick_labelsize"], color=c, weight="bold")
    ax.text(0.5, 0.885, "photograph, several plausible objects of each kind",
            ha="center", fontsize=S["tick_labelsize"] - 1, style="italic", color="0.35")
    ax.text(0.02, 0.27, "clean:  “what colour is T …?”", fontsize=S["tick_labelsize"], color=C_T)
    ax.text(0.02, 0.15, "paired: “what colour is D …?”", fontsize=S["tick_labelsize"], color=C_D)
    ax.text(0.02, 0.03, "the only difference is which object is referred to",
            fontsize=S["tick_labelsize"] - 1, style="italic", color="0.35")

    # (b) what the patches do with depth — measured
    ax = fig.add_subplot(gs[0, 1])
    ax.set_title("b  The two objects move in opposite directions",
                 fontsize=S["subplot_title_fontsize"], loc="left")
    ax.axhline(0, color="k", lw=1)
    ax.axvspan(6.5, 11.5, color=C_GREY, alpha=0.30, zorder=0)
    ax.plot(range(12), ref, color=C_T, lw=S["linewidth"], marker=S["marker"],
            ms=S["markersize"], label="referent")
    ax.plot(range(12), nonref, color=C_D, lw=S["linewidth"], marker=S["marker"],
            ms=S["markersize"], label="non-referent")
    ax.annotate("nothing happens\nfor six blocks", xy=(3, -0.15), xytext=(0.4, -2.4),
                fontsize=S["tick_labelsize"] - 1, color="0.35",
                arrowprops=dict(arrowstyle="->", color="0.45", lw=1.5))
    ax.annotate("onset at block 7", xy=(7, ref[7]), xytext=(7.4, -1.6),
                fontsize=S["tick_labelsize"], color="0.2",
                arrowprops=dict(arrowstyle="->", color="0.2", lw=1.8))
    ax.set_xlabel("block", fontsize=S["label_fontsize"])
    ax.set_ylabel("shift along the image's own\nselection direction", fontsize=S["label_fontsize"])
    ax.set_xticks(range(0, 12, 2))
    ax.legend(fontsize=S["legend_fontsize"] - 3, frameon=False, loc="upper left")

    # (c) what the decoder reads
    ax = fig.add_subplot(gs[0, 2])
    ax.set_title("c  The decoder then reads the referent", fontsize=S["subplot_title_fontsize"], loc="left")
    names = ["referent", "non-referent", "background"]
    vals = [mass["T"]["mean"] if isinstance(mass["T"], dict) else mass["T"],
            mass["D"]["mean"] if isinstance(mass["D"], dict) else mass["D"],
            mass["bg_b"]["mean"] if isinstance(mass["bg_b"], dict) else mass["bg_b"]]
    ax.bar(names, vals, color=[C_T, C_D, C_BG], width=0.6)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.0012, f"{v:.3f}", ha="center", fontsize=S["tick_labelsize"])
    ax.set_ylabel("attention mass per patch", fontsize=S["label_fontsize"])
    ax.set_ylim(0, max(vals) * 1.35)
    ax.text(0.5, 0.80, f"{ratio['mean']:.0f}× background\n[{ratio['lo']:.0f}, {ratio['hi']:.0f}]",
            transform=ax.transAxes, ha="center", fontsize=S["label_fontsize"],
            color=C_T, weight="bold")

    # ---------------------------------------------------------------- row 1 --
    # (d) the marker, measured
    ax = fig.add_subplot(gs[1, 0])
    ax.set_title("d  The selected object gets the same marker\n     as in a one-object question",
                 fontsize=S["subplot_title_fontsize"], loc="left")
    ax.axhline(0, color="k", lw=1)
    ax.axvspan(8.5, 11.5, color=C_GREY, alpha=0.30, zorder=0)
    ax.plot(range(12), tmd, color="k", lw=S["linewidth"], marker=S["marker"], ms=S["markersize"])
    ax.set_xlabel("block", fontsize=S["label_fontsize"])
    ax.set_ylabel("answer object − third object\n(projection on the marker)",
                  fontsize=S["label_fontsize"])
    ax.set_xticks(range(0, 12, 2))
    w = spatial["h4"]["window_mean"]["T_minus_D"]
    ax.text(0.04, 0.90, f"blocks 9–11: {w['mean']:+.2f}  [{w['lo']:+.2f}, {w['hi']:+.2f}]",
            transform=ax.transAxes, fontsize=S["tick_labelsize"], weight="bold")
    ax.text(0.04, 0.80, "same onset block as in b", transform=ax.transAxes,
            fontsize=S["tick_labelsize"] - 1, style="italic", color="0.35")

    # (e) the probe that could not be tested — the honest panel
    ax = fig.add_subplot(gs[1, 1])
    ax.set_title("e  The position probe never worked at all",
                 fontsize=S["subplot_title_fontsize"], loc="left")
    ax.axhline(0, color="k", lw=1.4)
    ax.plot(range(12), probe_c1, color=C_A, lw=S["linewidth"], marker=S["marker"],
            ms=S["markersize"], label="real anchor positions")
    ax.plot(range(12), probe_sh, color=C_GREY, lw=1.6, ls="--", marker="x",
            ms=S["markersize"], label="positions shuffled (control)")
    ax.set_xlabel("block", fontsize=S["label_fontsize"])
    ax.set_ylabel("cross-validated R²", fontsize=S["label_fontsize"])
    ax.set_xticks(range(0, 12, 2))
    ax.set_ylim(-0.20, 0.13)
    ax.text(0.03, 0.86, "R² < 0 at every block: worse than predicting the mean.\n"
                        "The shuffled control scores HIGHER. A probe that never\n"
                        "decodes cannot tell us whether the mechanism is there.",
            transform=ax.transAxes, fontsize=S["tick_labelsize"] - 2, color="0.25",
            va="top")
    ax.legend(fontsize=S["legend_fontsize"] - 4, frameon=False, loc="lower right")

    # (f) verdict cards
    ax = fig.add_subplot(gs[1, 2])
    blank(ax)
    ax.set_title("f  Four registered tests", fontsize=S["subplot_title_fontsize"], loc="left")
    cards = [
        ("selection of the referent", "reproduces", C_OK,
         "+1.17 [0.87, 1.49] · 184 q / 148 img"),
        ("marker on the selected object", "reproduces", C_OK,
         "+1.74 [0.90, 2.56] · 35 q / 27 img"),
        ("anchor position in background", "cannot be tested", C_GREY,
         "27 images: the probe has no power"),
        ("head-level localization", "nothing to ablate", C_GREY,
         "0 heads meet the registered rule"),
    ]
    for i, (title, verdict, col, detail) in enumerate(cards):
        y = 0.78 - i * 0.245
        box(ax, 0.0, y, 1.0, 0.205, "0.97", ec=col, lw=2.4, r=0.015)
        ax.text(0.03, y + 0.145, title, fontsize=S["tick_labelsize"], weight="bold")
        ax.text(0.03, y + 0.075, verdict, fontsize=S["tick_labelsize"], color=col, weight="bold")
        ax.text(0.03, y + 0.022, detail, fontsize=S["tick_labelsize"] - 3, color="0.35")

    # ---------------------------------------------------------------- row 2 --
    ax = fig.add_subplot(gs[2, :])
    blank(ax)
    ax.set_title("g  Neither training distribution lends the mechanism to the other — "
                 "so it is learned, not supplied by the architecture",
                 fontsize=S["subplot_title_fontsize"], loc="left")

    def model_box(x, label, sub, col):
        box(ax, x, 0.42, 0.19, 0.34, "0.97", ec=col, lw=2.6, r=0.02)
        ax.text(x + 0.095, 0.645, label, ha="center", fontsize=S["label_fontsize"], weight="bold")
        ax.text(x + 0.095, 0.545, sub, ha="center", fontsize=S["tick_labelsize"] - 2, color="0.35")
        return x

    model_box(0.03, "trained on CLEVR", "synthetic scenes", C_D)
    model_box(0.78, "trained on GQA", "real photographs", C_T)
    box(ax, 0.335, 0.40, 0.33, 0.38, "0.99", ec="0.55", lw=1.8, r=0.02)
    ax.text(0.50, 0.715, "evaluated on the OTHER dataset", ha="center",
            fontsize=S["tick_labelsize"], color="0.3", style="italic")
    ax.text(0.50, 0.615, "CLEVR model on GQA:  selection +0.29 [−0.39, +1.13]",
            ha="center", fontsize=S["tick_labelsize"], color=C_NO)
    ax.text(0.50, 0.545, "attention on the asked-about object only 4× background",
            ha="center", fontsize=S["tick_labelsize"] - 2, color="0.35")
    ax.text(0.50, 0.470, "GQA model on CLEVR:  selection ≈ 0 through block 10",
            ha="center", fontsize=S["tick_labelsize"], color=C_NO)
    arrow(ax, (0.225, 0.59), (0.330, 0.59), color=C_D)
    arrow(ax, (0.775, 0.59), (0.670, 0.59), color=C_T)
    ax.text(0.50, 0.315, "interval contains zero in both directions", ha="center",
            fontsize=S["tick_labelsize"] - 1, color=C_NO, weight="bold")

    ax.text(0.50, 0.16,
            "Same frozen backbone, same architecture, same one-layer decoder — only the training "
            "distribution differs.\nThe selection mechanism appears only in the distribution it was "
            "trained on: evidence that the training data induces it,\nrather than the backbone or "
            "the architecture supplying it.",
            ha="center", va="center", fontsize=S["label_fontsize"] - 1, color="0.15")

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "gqa_findings_schematic.png"
    fig.savefig(path, dpi=S["dpi"], bbox_inches="tight")
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
