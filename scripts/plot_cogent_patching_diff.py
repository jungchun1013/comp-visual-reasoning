"""Plot CoGenT before/after head patching comparison.

Loads JSON stats from two runs and produces:
1. Described attr: 3 rows (Before/After/Diff) × 4 cols (color/shape/material/size)
2. Query attr: same layout

Usage:
    python scripts/plot_cogent_patching_diff.py \
        --before outputs/analysis/cogent_patching/before_ft \
        --after outputs/analysis/cogent_patching/after_ft \
        --output-dir outputs/analysis/cogent_patching/comparison
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from analysis.plot_style import apply_style, S


FINE_ATTRS = ["color", "shape", "material", "size"]


def build_column_layout(gca_layers, sa_heads, gca_heads):
    """Build SA+GCA interleaved column layout."""
    gca_set = set(gca_layers)
    col_labels, col_is_gca = [], []
    for l in range(12):
        if l in gca_set:
            col_labels.append(f"GCA{l}")
            col_is_gca.append(True)
        col_labels.append(f"SA{l}")
        col_is_gca.append(False)
    max_heads = max(sa_heads, gca_heads)
    return col_labels, col_is_gca, max_heads


def build_combined_heatmap(sa_hm, gca_hm, gca_layers, sa_heads, gca_heads, n_cols, max_heads):
    """Interleave SA and GCA into (n_cols, max_heads) array."""
    gca_set = set(gca_layers)
    combined = np.full((n_cols, max_heads), np.nan)
    gca_idx, ci = 0, 0
    for l in range(12):
        if l in gca_set:
            combined[ci, :gca_heads] = gca_hm[gca_idx]
            gca_idx += 1
            ci += 1
        combined[ci, :sa_heads] = sa_hm[l]
        ci += 1
    return combined


def plot_heatmap_grid(maps_list, row_labels, col_labels_list, col_is_gca, max_heads,
                      col_titles, suptitle, save_path, vmax=None):
    """Plot grid of heatmaps.

    Args:
        maps_list: list of dicts {attr: (n_cols, max_heads) array}
        row_labels: list of row names (e.g. ["Before FT", "After FT", "Diff"])
        col_labels_list: list of module labels for x-axis
        col_is_gca: bool list
        max_heads: int
        col_titles: list of attr names for columns
        suptitle: figure title
        save_path: output path
        vmax: shared vmax (auto if None)
    """
    n_rows = len(maps_list)
    n_cols_plot = len(col_titles)
    n_modules = len(col_labels_list)

    fig, axes = plt.subplots(n_rows, n_cols_plot,
                             figsize=(10 * n_cols_plot, 5.5 * n_rows),
                             constrained_layout=True)
    if n_rows == 1:
        axes = axes[np.newaxis, :]
    if n_cols_plot == 1:
        axes = axes[:, np.newaxis]

    # Auto vmax from first two rows (before/after), not diff
    if vmax is None:
        all_vals = []
        for row_maps in maps_list[:2]:
            for attr in col_titles:
                if attr in row_maps:
                    all_vals.append(np.nanmax(np.abs(row_maps[attr])))
        vmax = max(all_vals) if all_vals else 2.5

    for ri, (row_maps, row_label) in enumerate(zip(maps_list, row_labels)):
        # Diff row uses separate vmax
        row_vmax = vmax
        if "Diff" in row_label or "diff" in row_label.lower():
            diff_vals = [np.nanmax(np.abs(row_maps[a])) for a in col_titles if a in row_maps]
            row_vmax = max(diff_vals) if diff_vals else vmax

        for ci, attr in enumerate(col_titles):
            ax = axes[ri, ci]
            if attr not in row_maps:
                ax.axis("off")
                continue
            hm = row_maps[attr]
            masked = np.ma.array(hm, mask=np.isnan(hm))
            im = ax.imshow(masked.T, aspect="auto", cmap="RdBu_r",
                           vmin=-row_vmax, vmax=row_vmax, origin="lower")
            if ri == 0:
                ax.set_title(attr, fontsize=S["subplot_title_fontsize"])
            if ci == 0:
                ax.set_ylabel(f"{row_label}\nHead Index", fontsize=S["label_fontsize"])
            else:
                ax.set_ylabel("")
            if ri == n_rows - 1:
                ax.set_xlabel("Module", fontsize=S["label_fontsize"])
            ax.set_xticks(range(n_modules))
            ax.set_xticklabels(col_labels_list, rotation=60, ha="right", fontsize=7)
            ax.set_yticks(range(max_heads))
            for mi in range(n_modules):
                if col_is_gca[mi]:
                    rect = Rectangle((mi - 0.5, -0.5), 1, max_heads,
                                     linewidth=1.5, edgecolor="red",
                                     facecolor="none", linestyle="--")
                    ax.add_patch(rect)
            plt.colorbar(im, ax=ax, label=r"$\Delta$ logit")

    fig.suptitle(suptitle, fontsize=S["suptitle_fontsize"])
    fig.savefig(str(save_path), dpi=S["dpi"], bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close(fig)


def load_described_attr(stats_path):
    """Load described_attr_stats.json → {attr: (sa_mean, gca_mean)}."""
    with open(stats_path) as f:
        stats = json.load(f)
    data = {}
    key = "fine_attribute_denoising"
    if key in stats:
        for attr in FINE_ATTRS:
            if attr in stats[key]:
                data[attr] = (
                    np.array(stats[key][attr]["sa_mean"]),
                    np.array(stats[key][attr]["gca_mean"]),
                )
    return data, stats["gca_layers"], stats["sa_num_heads"], stats["gca_num_heads"]


def load_query_attr(stats_path):
    """Load query_attr_stats.json → {attr: (sa_mean, gca_mean)}."""
    with open(stats_path) as f:
        stats = json.load(f)
    data = {}
    for attr in FINE_ATTRS:
        sa_key, gca_key = f"{attr}_sa", f"{attr}_gca"
        if sa_key in stats and gca_key in stats:
            data[attr] = (np.array(stats[sa_key]), np.array(stats[gca_key]))
    return data, stats["gca_layers"], stats["sa_num_heads"], stats["gca_num_heads"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--before", type=str, required=True, help="Before FT results dir")
    p.add_argument("--after", type=str, required=True, help="After FT results dir")
    p.add_argument("--output-dir", type=str, required=True)
    args = p.parse_args()

    apply_style()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    before_dir = Path(args.before)
    after_dir = Path(args.after)

    # ── Plot 1: Described attr ──
    before_desc, gca_layers, sa_heads, gca_heads = load_described_attr(
        before_dir / "described_attr_stats.json")
    after_desc, _, _, _ = load_described_attr(after_dir / "described_attr_stats.json")

    col_labels, col_is_gca, max_heads = build_column_layout(gca_layers, sa_heads, gca_heads)
    n_modules = len(col_labels)

    def to_combined(data_dict):
        return {attr: build_combined_heatmap(sa, gca, gca_layers, sa_heads, gca_heads, n_modules, max_heads)
                for attr, (sa, gca) in data_dict.items()}

    before_combined = to_combined(before_desc)
    after_combined = to_combined(after_desc)
    diff_combined = {attr: after_combined[attr] - before_combined[attr]
                     for attr in FINE_ATTRS if attr in before_combined and attr in after_combined}

    plot_heatmap_grid(
        [before_combined, after_combined, diff_combined],
        ["Before FT", "After FT", "Diff (After - Before)"],
        col_labels, col_is_gca, max_heads,
        FINE_ATTRS,
        "Described Attr Head Patching — CoGenT Before/After FT",
        output_dir / "described_attr_comparison.png",
    )

    # ── Plot 2: Query attr ──
    before_query, gca_layers, sa_heads, gca_heads = load_query_attr(
        before_dir / "query_attr_stats.json")
    after_query, _, _, _ = load_query_attr(after_dir / "query_attr_stats.json")

    before_combined = to_combined(before_query)
    after_combined = to_combined(after_query)
    diff_combined = {attr: after_combined[attr] - before_combined[attr]
                     for attr in FINE_ATTRS if attr in before_combined and attr in after_combined}

    plot_heatmap_grid(
        [before_combined, after_combined, diff_combined],
        ["Before FT", "After FT", "Diff (After - Before)"],
        col_labels, col_is_gca, max_heads,
        FINE_ATTRS,
        "Query Attr Head Patching — CoGenT Before/After FT",
        output_dir / "query_attr_comparison.png",
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
