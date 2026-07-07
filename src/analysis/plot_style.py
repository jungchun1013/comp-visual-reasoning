"""Centralized plot style for all SteerViT analysis figures."""

from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

# ---------------------------------------------------------------------------
# Core style constants
# ---------------------------------------------------------------------------
PLOT_STYLE = {
    "linewidth": 2,
    "std_alpha": 0.2,
    "tick_width": 2,
    "tick_labelsize": 14,
    "label_fontsize": 16,
    "legend_fontsize": 16,
    "legend_loc": "outside lower center",
    "subplot_title_fontsize": 18,
    "suptitle_fontsize": 20,
    "marker": "o",
    "markersize": 6,
    "dpi": 150,
    "subplot_size": (6.4, 4.8),
}

S = PLOT_STYLE  # short alias

# ---------------------------------------------------------------------------
# Colour palettes
# ---------------------------------------------------------------------------
_tab10 = plt.cm.tab10.colors    # 10 categorical colours (default for lines)
_tab20c = plt.cm.tab20c.colors  # 20 colours, 5 families × 4 shades

# Corruption-type colours: Tab20c blue gradient, dark=conceptual light=perceptual
# (only case where Tab20c gradient is used)
CORRUPTION_COLORS = {
    "attribute":  _tab10[0],   # blue
    "shape":      _tab10[1],   # orange
    "spatial":    _tab10[2],   # green
    "quantifier": _tab10[3],   # red
}
CORRUPTION_ORDER = ["attribute", "shape", "spatial", "quantifier"]

# Answer-type colours: Tab10 categorical
ANSWER_TYPE_COLORS = {
    "boolean":  _tab10[3],  # red
    "counting": _tab10[1],  # orange
    "color":    _tab10[2],  # green
    "shape":    _tab10[0],  # blue
    "material": _tab10[4],  # purple
    "size":     _tab10[5],  # brown
}
ANSWER_TYPE_ORDER = ["boolean", "counting", "color", "shape", "material", "size"]

# Steered / unsteered pair: Tab10
CONDITION_COLORS = {
    "steered":   _tab10[0],  # blue
    "unsteered": _tab10[3],  # red
}

GCA_LAYERS = [1, 3, 5, 7, 9, 11]  # odd layers

# Attribute-value colours for t-SNE scatters: Tab10 (user-specified).
# CLEVR colour values are hue-faithful via tab10's named hues (red = _tab10[3],
# the same red as the answer-match condition — no conflict, these panels carry
# no condition marks).
ATTR_VALUE_COLORS = {
    "color": {
        "blue":   _tab10[0],
        "cyan":   _tab10[9],
        "red":    _tab10[3],
        "brown":  _tab10[5],
        "yellow": _tab10[8],   # olive — nearest tab10 hue to yellow
        "green":  _tab10[2],
        "purple": _tab10[4],
        "gray":   _tab10[7],
    },
    "shape":    {"cube": _tab10[0], "sphere": _tab10[1], "cylinder": _tab10[2]},
    "material": {"metal": _tab10[0], "rubber": _tab10[1]},
    "size":     {"large": _tab10[0], "small": _tab10[1]},
}
ATTR_VALUE_ORDER = {
    "color":    ["blue", "cyan", "red", "brown", "yellow", "green", "purple", "gray"],
    "shape":    ["cube", "sphere", "cylinder"],
    "material": ["metal", "rubber"],
    "size":     ["large", "small"],
}


# t-SNE scatter style (legacy run_tsne_*.py convention):
# square cells in a tight grid, no ticks, thick spines, gray background
# points, rasterized scatters, shared fig.legend below the grid.
TSNE_STYLE = {
    "cell": 2.8,        # square panel edge, inches
    "bg_size": 8,       # unhighlighted points
    "mid_size": 18,     # highlighted points
    "hi_size": 22,      # highlighted points carrying an edge
    "edge_width": 1.2,  # highlight edge width
    "gray": (0.75, 0.75, 0.75),
}


def style_tsne_ax(ax):
    """t-SNE axis style: square box, thick spines, no ticks."""
    ax.set_box_aspect(1)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(2)


def make_tsne_grid(n_panels: int, ncols: int = 3, cell: float = None):
    """Legacy t-SNE grid: figsize=(cell*ncols+1, cell*nrows+1), flat axes."""
    if cell is None:
        cell = TSNE_STYLE["cell"]
    nrows = (n_panels + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(cell * ncols + 1, cell * nrows + 1))
    return fig, np.atleast_1d(axes).flatten()


def finish_tsne_grid(fig, handles, suptitle: str = None, ncol: int = None):
    """Legacy t-SNE finish: tight grid spacing, fig.legend below, frameless."""
    if suptitle:
        fig.suptitle(suptitle, fontsize=S["suptitle_fontsize"])
    fig.subplots_adjust(hspace=0.05, wspace=0.05, top=0.90, bottom=0.00)
    if ncol is None:
        ncol = min(len(handles), 5)
    fig.legend(handles=handles, loc="upper center",
               bbox_to_anchor=(0.5, 0.016), ncol=ncol,
               fontsize=S["legend_fontsize"], frameon=False)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def apply_style():
    """Apply global rcParams."""
    mpl.rcParams.update({
        "axes.labelsize": S["label_fontsize"],
        "axes.titlesize": S["subplot_title_fontsize"],
        "xtick.labelsize": S["tick_labelsize"],
        "ytick.labelsize": S["tick_labelsize"],
        "xtick.major.width": S["tick_width"],
        "ytick.major.width": S["tick_width"],
        "legend.fontsize": S["legend_fontsize"],
        "figure.dpi": S["dpi"],
        "savefig.dpi": S["dpi"],
        "savefig.bbox": "tight",
        "font.family": "sans-serif",
    })


def make_figure(nrows: int = 1, ncols: int = 1, **kwargs):
    """Create figure sized by subplot_size."""
    w, h = S["subplot_size"]
    figsize = kwargs.pop("figsize", (w * ncols, h * nrows))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, **kwargs)
    return fig, axes


def save_with_legend(fig, path: str, *, handles=None, labels=None, ncol=None,
                     ax_for_legend=None):
    """Save figure with legend placed below all axes.

    If handles/labels not given, collects from all axes.
    """
    if handles is None or labels is None:
        if ax_for_legend is not None:
            handles, labels = ax_for_legend.get_legend_handles_labels()
        else:
            # Collect from all axes, deduplicate
            all_h, all_l = [], []
            seen = set()
            for ax in fig.axes:
                for h, l in zip(*ax.get_legend_handles_labels()):
                    if l not in seen:
                        all_h.append(h)
                        all_l.append(l)
                        seen.add(l)
            handles, labels = all_h, all_l

    # Remove per-axis legends
    for ax in fig.axes:
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()

    fig.tight_layout()

    if handles:
        if ncol is None:
            ncol = min(len(handles), 5)
        fig.legend(handles, labels, loc="upper center",
                   bbox_to_anchor=(0.5, -0.05), ncol=ncol,
                   fontsize=S["legend_fontsize"], frameon=False)

    fig.savefig(path, dpi=S["dpi"], bbox_inches="tight")
    plt.close(fig)


def mark_gca_layers(ax, color="gray", alpha=0.15):
    """Add subtle vertical lines at GCA layer positions."""
    for l in GCA_LAYERS:
        ax.axvline(x=l, color=color, linestyle="--", linewidth=0.8, alpha=alpha)


def line_kwargs(label: str = None, color=None, **overrides):
    """Standard kwargs for ax.plot()."""
    kw = dict(
        linewidth=S["linewidth"],
        marker=S["marker"],
        markersize=S["markersize"],
    )
    if label is not None:
        kw["label"] = label
    if color is not None:
        kw["color"] = color
    kw.update(overrides)
    return kw


def rename_legacy_keys(d: dict) -> dict:
    """Rename 'object' → 'shape' in stats dicts from old runs."""
    if "object" in d:
        d["shape"] = d.pop("object")
    return d
