"""Paper figures composed from existing result JSONs (no model, no GPU).

Reads only files under outputs/analysis/patch_language_condition/ and writes
PDF + PNG + a provenance JSON per figure into --out-dir. Nothing under
outputs/ is written. New file because no existing script composes multi-panel
paper figures from result JSONs (each analysis script plots its own run dir).

Figures
  attribute_alignment : normalised own-attribute alignment change of the SAME
                        object under referent / non-referent / non-unique
                        questions, four backbones (colour) + DINOv2 four
                        queried attributes.
  functional_tests    : B1 projection / write-mask interventions (alignment at
                        block 11 + accuracy when asking A and asking B) and
                        X25 add-back flip rates with the referent efficacy
                        control.
  gqa_grounding       : example images with boxes, H1 / H4-crossfit / R2
                        per-block series (separate estimators, separate
                        panels), B2 answer-choice fractions with controls.

Usage (from main/ or the worktree):
  PYTHONPATH=src <interp> scripts/analysis/paper_figures.py --out-dir <dir> [--only NAME]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, Patch

from analysis.plot_style import apply_style, PLOT_STYLE as S, line_kwargs, mark_gca_layers, _tab10

ROOT = Path("outputs/analysis/patch_language_condition")
BACKBONES = [("DINOv2", ""), ("SigLIP", "siglip"), ("Sup-ViT", "sup"), ("MAE", "mae")]
ATTR_DIRS = [("colour", ""), ("shape", "shape"), ("material", "material"), ("size", "size")]
BLOCKS = np.arange(12)
C_REF, C_NONREF, C_NONUNIQUE, C_BG, C_ANCHOR = _tab10[0], _tab10[1], (0.5, 0.5, 0.5), _tab10[4], _tab10[2]
GQA_IMAGES = Path("/home/jungchun/data/gqa/images")


def load(rel):
    return json.load(open(ROOT / rel))


def arr(series, key="mean"):
    return np.array([s[key] for s in series])


def band(ax, x, series, color, label, ls="-"):
    ax.plot(x, arr(series), **line_kwargs(label=label, color=color, linestyle=ls))
    ax.fill_between(x, arr(series, "lo"), arr(series, "hi"), color=color, alpha=S["std_alpha"], linewidth=0)


def save(fig, out_dir, name, prov, ncol=None, handles=None, labels=None):
    fig.tight_layout()
    if handles is None:
        hs, ls_, seen = [], [], set()
        for ax in fig.axes:
            for h, l in zip(*ax.get_legend_handles_labels()):
                if l not in seen:
                    hs.append(h); ls_.append(l); seen.add(l)
        handles, labels = hs, ls_
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.02),
               ncol=ncol or min(len(handles), 5), fontsize=S["legend_fontsize"], frameon=False)
    for ext in ("pdf", "png"):
        p = out_dir / f"{name}.{ext}"
        fig.savefig(p, dpi=S["dpi"], bbox_inches="tight")
        print(f"Saved: {p}")
    plt.close(fig)
    (out_dir / f"{name}_provenance.json").write_text(json.dumps(prov, indent=2))
    print(f"Saved: {out_dir / (name + '_provenance.json')}")


# ---------------------------------------------------------------- figure 1
def fig_attribute_alignment(out_dir):
    fig, axes = plt.subplots(2, 4, figsize=(S["subplot_size"][0] * 4, S["subplot_size"][1] * 2))
    prov = {"panels": []}

    def panel(ax, d, title, queried):
        keys = {f"refvs0_target_{queried}_own": ("asked about the target (c1 − c0)", C_REF),
                f"nonrefvs0_target_{queried}_own": ("asked about the distractor (c2 − c0)", C_NONREF),
                f"c3vs0_target_{queried}_own": ("question without a unique referent (c3 − c0)", C_NONUNIQUE)}
        for k, (lab, col) in keys.items():
            band(ax, BLOCKS, d["delta"][k], col, lab)
        ax.axhline(0, color="k", lw=0.8)
        mark_gca_layers(ax)
        ax.set_title(title, fontsize=S["subplot_title_fontsize"])
        ax.set_xlabel("block")
        ax.set_xticks(BLOCKS[1::2])
        return {"title": title, "n_images": d["n_images"], "queried": queried,
                "keys": list(keys), "y": "cosine(unit object-mean, unit own-value direction) minus no-question value"}

    for j, (name, sub) in enumerate(BACKBONES):
        rel = f"{sub}/attr_directions_v2/partA_attr_directions_normstd.json".lstrip("/")
        d = load(rel)
        p = panel(axes[0, j], d, f"{name} — queried: colour", "color")
        p["source"] = str(ROOT / rel); prov["panels"].append(p)
    for j, (attr, sub) in enumerate(ATTR_DIRS):
        rel = f"{sub}/attr_directions_v2/partA_attr_directions_normstd.json".lstrip("/")
        d = load(rel)
        q = d["queried"]
        p = panel(axes[1, j], d, f"DINOv2 — queried: {attr}", q)
        p["source"] = str(ROOT / rel); prov["panels"].append(p)
    for ax in axes[:, 0]:
        ax.set_ylabel("Δ alignment with own value")
    prov.update({
        "model": "clevr_<backbone>_decoder1l_scratch_s42 (DINOv2 root dir = clevr_dinov2_decoder1l_scratch_s42), best.pt",
        "data": "paired two-object renders, 324 images, target object measured under c0 no question / c1 asked about the target / c2 asked about the distractor / c3 'What color is the object?'",
        "feature": "mean of the object's patch tokens after trunk.norm, unit-normalised; attribute-value directions = unit(mean of 1-object means of that value − grand mean), estimated on the 1-object set (n1)",
        "statistic": "per-image paired difference of cosine alignment, mean with 95% bootstrap CI over images (B=1000)",
        "not_excluded": "rotation/redistribution of information; direction quality (DINOv2 late blocks: held-out colour classification 0.40–0.43); a change along one direction is not information removal",
        "registry": "X24 A (2026-09-13), Sup-ViT 2026-09-14",
    })
    save(fig, out_dir, "attribute_alignment", prov, ncol=3)


# ---------------------------------------------------------------- figure 2
B1_SPECS = [("none", "no intervention"),
            ("project_7_marker", "project out\nref. direction @7"),
            ("project_8_marker", "project out @8"),
            ("project_10_marker", "project out @10"),
            ("project_7_random", "random direction @7\n(5 seeds)"),
            ("gcamask_1-3-5", "mask GCA writes\n1,3,5 (all patches)"),
            ("gcamask_1-3-5_target", "mask GCA 1,3,5\n(object A patches)"),
            ("gcamask_9-11", "mask GCA writes\n9,11")]


def b1_cells(sub):
    rows = []
    for spec, label in B1_SPECS:
        if spec == "project_7_random":
            ds = [load(f"{sub}/n2_int_project_7_random_{s}/partA_attr_directions_normstd.json".lstrip("/")) for s in range(5)]
            bs = [load(f"{sub}/n2_int_project_7_random_{s}/behaviour.json".lstrip("/")) for s in range(5)]
            g = lambda key: np.mean([d["delta"][key][11]["mean"] for d in ds])
            row = dict(label=label, nonref=g("nonrefvs0_target_color_own"), ref=g("refvs0_target_color_own"),
                       nonref_lo=min(d["delta"]["nonrefvs0_target_color_own"][11]["lo"] for d in ds),
                       nonref_hi=max(d["delta"]["nonrefvs0_target_color_own"][11]["hi"] for d in ds),
                       acc_c1=np.mean([b["c1"]["accuracy"]["mean"] for b in bs]),
                       acc_c2=np.mean([b["c2"]["accuracy"]["mean"] for b in bs]),
                       seeds=[d["delta"]["nonrefvs0_target_color_own"][11]["mean"] for d in ds])
        else:
            d = load(f"{sub}/n2_int_{spec}/partA_attr_directions_normstd.json".lstrip("/"))
            b = load(f"{sub}/n2_int_{spec}/behaviour.json".lstrip("/"))
            nr = d["delta"]["nonrefvs0_target_color_own"][11]
            row = dict(label=label, nonref=nr["mean"], nonref_lo=nr["lo"], nonref_hi=nr["hi"],
                       ref=d["delta"]["refvs0_target_color_own"][11]["mean"],
                       acc_c1=b["c1"]["accuracy"]["mean"], acc_c2=b["c2"]["accuracy"]["mean"])
        row["spec"] = spec
        rows.append(row)
    return rows


def fig_functional_tests(out_dir):
    fig = plt.figure(figsize=(S["subplot_size"][0] * 3.2, S["subplot_size"][1] * 3.1))
    gs = GridSpec(3, 6, figure=fig)
    prov = {"B1": {}, "X25": {}}
    for j, (name, sub) in enumerate([("DINOv2", ""), ("SigLIP", "siglip")]):
        rows = b1_cells(sub)
        x = np.arange(len(rows))
        ax = fig.add_subplot(gs[0, 3 * j:3 * j + 3])
        ax.bar(x - 0.2, [r["nonref"] for r in rows], 0.4, color=C_NONREF, label="object A when asked about B (c2 − c0)")
        ax.errorbar(x - 0.2, [r["nonref"] for r in rows],
                    yerr=[[r["nonref"] - r["nonref_lo"] for r in rows], [r["nonref_hi"] - r["nonref"] for r in rows]],
                    fmt="none", ecolor="k", lw=1)
        ax.bar(x + 0.2, [r["ref"] for r in rows], 0.4, color=C_REF, label="object A when asked about A (c1 − c0)")
        for r, xi in zip(rows, x):
            if "seeds" in r:
                ax.scatter([xi - 0.2] * 5, r["seeds"], s=12, color="k", zorder=3)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(x); ax.set_xticklabels([r["label"] for r in rows], rotation=35, ha="right", fontsize=S["tick_labelsize"] - 4)
        ax.set_title(f"{name}: block-11 alignment of object A", fontsize=S["subplot_title_fontsize"])
        if j == 0:
            ax.set_ylabel("Δ alignment with own colour")
        ax2 = fig.add_subplot(gs[1, 3 * j:3 * j + 3])
        ax2.plot(x, [r["acc_c1"] for r in rows], **line_kwargs(label="accuracy, question about A (c1)", color=C_REF))
        ax2.plot(x, [r["acc_c2"] for r in rows], **line_kwargs(label="accuracy, question about B (c2)", color=C_NONREF, linestyle="--"))
        ax2.set_ylim(0, 1.05)
        ax2.set_xticks(x); ax2.set_xticklabels([r["label"] for r in rows], rotation=35, ha="right", fontsize=S["tick_labelsize"] - 4)
        ax2.set_title(f"{name}: accuracy, same interventions", fontsize=S["subplot_title_fontsize"])
        if j == 0:
            ax2.set_ylabel("accuracy (324 images)")
        prov["B1"][name] = {"rows": rows, "source_dirs": f"{ROOT}/{sub}/n2_int_<spec>/{{partA_attr_directions_normstd.json,behaviour.json}}"}

    specs = ["7", "8", "9", "10", "11", "9–11"]
    for j, (name, sub) in enumerate([("DINOv2", ""), ("SigLIP", "siglip"), ("MAE", "mae")]):
        r = load(f"{sub}/n2_restore/restoration.json".lstrip("/"))
        ax = fig.add_subplot(gs[2, 2 * j:2 * j + 2])
        n = [x["n"] for x in r["rows"] if x["variant"] == "own"][0]

        def series(variant, alpha, seeds=None):
            out = []
            for spec in r["layer_specs"]:
                if seeds is None:
                    row = [x for x in r["rows"] if x["variant"] == variant and x["cond"] == "c1" and x["alpha"] == alpha and x["layers"] == spec][0]
                    out.append((row["flip_to_Ad"]["mean"], row["flip_to_Ad"]["lo"], row["flip_to_Ad"]["hi"]))
                else:
                    ms = [[x for x in r["rows"] if x["variant"] == f"random_{s}" and x["cond"] == "c1" and x["alpha"] == alpha and x["layers"] == spec][0]["flip_to_Ad"]["mean"] for s in seeds]
                    out.append((np.mean(ms), min(ms), max(ms)))
            return np.array(out)

        xs = np.arange(len(specs))
        for variant, alpha, col, ls, lab, seeds in [
                ("own", 2.0, C_NONREF, "-", "own colour of B added on B, α=2", None),
                ("own", 4.0, C_NONREF, ":", "own colour of B added on B, α=4", None),
                ("own_on_referent", 2.0, C_REF, "-", "same vector added on A (efficacy control), α=2", None),
                ("own_on_referent", 4.0, C_REF, ":", "same vector added on A, α=4", None),
                ("random", 4.0, C_NONUNIQUE, "-", "random direction on B, α=4 (5 seeds)", range(5)),
                ("own_on_bg", 4.0, C_BG, "-", "same vector on background, α=4", None)]:
            v = series(variant, alpha, seeds)
            ax.plot(xs, v[:, 0], **line_kwargs(label=lab, color=col, linestyle=ls))
            ax.fill_between(xs, v[:, 1], v[:, 2], color=col, alpha=S["std_alpha"], linewidth=0)
        ax.set_xticks(xs); ax.set_xticklabels(specs)
        ax.set_ylim(-0.02, 1.0)
        ax.set_xlabel("block(s) where the vector is added")
        dose_note = " (dose s_L ≈ 0)" if abs(r["dose_s"][11]) < 0.5 else ""
        ax.set_title(f"{name}: add-back, n = {n} (asked about A){dose_note}", fontsize=S["subplot_title_fontsize"])
        if j == 0:
            ax.set_ylabel("fraction of answers flipped to B's colour")
        prov["X25"][name] = {"n": n, "dose_s_by_block": r["dose_s"], "alphas": r["alphas"], "source": str(ROOT / sub / "n2_restore/restoration.json")}
    prov.update({
        "models": "clevr_{dinov2,siglip,mae}_decoder1l_scratch_s42 best.pt",
        "B1_design": "Object A fixed. c1 = question about A, c2 = question about B, c0 no question. 'project out' removes x − (x·v)v on ALL patch tokens at the output of block L, v = unit(mean over pairs of raw target mean c1 − c2), cross-fit by pair-index parity (NOT by image id — see delivery note). Random = isotropic Gaussian unit vector, not orthogonalised to v, not energy-matched; removed energy |x·v| not logged. Mask = zero the GCA write at the listed blocks on the listed patches.",
        "X25_design": "vector = α·|s_L|·V_raw[colour of B] added on B's patches (or on A's / background) at the listed blocks; s_L = mean raw own-colour projection change of the object under c2 − c0 at block L (dose_s_by_block); flips counted on items whose baseline answer is A's colour (n per backbone); MAE has s_L ≈ 0 so its dose is near zero.",
        "statistic": "means with 95% image-bootstrap CI; for random directions the band is min–max over 5 seeds",
        "not_excluded": "B1: perturbation magnitude of random ≠ marker; projection acts on every token, not only object roles. X25: additive restoration ≠ two-value replacement; zero flips is not exact zero probability change (ΔP(B) ≈ 1e−5).",
        "registry": "X24 B1 Results (2026-09-13), X25 Results + clarification (2026-09-13/14)",
    })
    save(fig, out_dir, "functional_tests", prov, ncol=3)


# ---------------------------------------------------------------- figure 3
def fig_gqa_grounding(out_dir, example_idx=(0, 3)):
    fig = plt.figure(figsize=(S["subplot_size"][0] * 3.2, S["subplot_size"][1] * 3.1))
    gs = GridSpec(3, 6, figure=fig)
    recs = load("x23_step0c_direct/relational_records.json")
    prov = {"examples": []}
    from PIL import Image
    for j, idx in enumerate(example_idx):
        rec = recs[idx]
        ax = fig.add_subplot(gs[0, 3 * j:3 * j + 3])
        im = Image.open(GQA_IMAGES / rec["filename"]).convert("RGB")
        ax.imshow(im); ax.set_xticks([]); ax.set_yticks([])
        for role, col in (("T", C_REF), ("D", C_NONREF)):
            o = rec["objects"][rec[role]]
            x, y, w, h = o["box_xywh"]
            ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor=col, lw=2.5))
            ax.text(x, max(y - 4, 10), o["name"], color="w", fontsize=S["tick_labelsize"],
                    bbox=dict(facecolor=col, edgecolor="none", pad=1.5))
        ax.set_title(f'c1: "{rec["questions"]["c1"]}" → {rec["answers"]["c1"]}\n'
                     f'c2: "{rec["questions"]["c2"]}" → {rec["answers"]["c2"]}', fontsize=S["tick_labelsize"])
        prov["examples"].append({"record_index": idx, "image_id": rec["image_id"], "question_id": rec["question_id"],
                                 "pair_question_id": rec["pair_question_id"], "T": rec["objects"][rec["T"]]["id"],
                                 "D": rec["objects"][rec["D"]]["id"], "boxes_xywh": {r: rec["objects"][rec[r]]["box_xywh"] for r in ("T", "D")},
                                 "role": "illustrative example from the 184-question direct cohort; not a selected success case"})

    x23 = load("x23_gqa_direct/x23_results.json")
    ax = fig.add_subplot(gs[1, 0:2])
    band(ax, BLOCKS, x23["h1"]["per_block"]["ref"], C_REF, "referent object (asked, c1 − c2)")
    band(ax, BLOCKS, x23["h1"]["per_block"]["nonref"], C_NONREF, "non-referent object (c1 − c2)")
    ax.axhline(0, color="k", lw=0.8); mark_gca_layers(ax)
    ax.set_title("Direct: per-image object − background direction\n(184 questions / 148 images, c1 acc 0.679)", fontsize=S["tick_labelsize"])
    ax.set_ylabel("projection change"); ax.set_xlabel("block")

    cf = load("x23_gqa_spatial_h2/h4_marker_crossfit.json")
    ax = fig.add_subplot(gs[1, 2:4])
    for k, col, lab in (("T", C_REF, "target (spatial)"), ("D", C_NONREF, "third object"), ("A", C_ANCHOR, "anchor"), ("bg", C_BG, "background")):
        band(ax, BLOCKS, cf["projection"][k], col, lab)
    ax.axhline(0, color="k", lw=0.8); mark_gca_layers(ax)
    wm = cf["window_mean"]["T_minus_D"]
    ax.set_title(f"Spatial: direction estimated on direct pairs, cross-fit\n(35 q / 27 images; T − D b9–11 = {wm['mean']:+.2f} [{wm['lo']:+.2f}, {wm['hi']:+.2f}])", fontsize=S["tick_labelsize"])
    ax.set_ylabel("projection (c1 − c2)"); ax.set_xlabel("block")

    r2 = load("x24_gqa_attr/attr_decomposition.json")
    ax = fig.add_subplot(gs[1, 4:6])
    pr = r2["projection"]
    band(ax, BLOCKS, pr["ref_queried"]["series"], C_REF, "referent, queried attribute (c1 − c0)")
    band(ax, BLOCKS, pr["nonref_queried"]["series"], C_NONREF, "non-referent, queried attribute (c1 − c0)")
    ax.axhline(0, color="k", lw=0.8); mark_gca_layers(ax)
    ax.set_title(f"Direct: alignment with own value of the queried attribute\n({pr['ref_queried']['n']} object-role rows / {pr['ref_queried']['n_images']} images; pool {r2['pool_n']} objects)", fontsize=S["tick_labelsize"])
    ax.set_ylabel("Δ alignment"); ax.set_xlabel("block")

    inj = load("x23_gqa_direct_inject/marker_injection.json")
    ax = fig.add_subplot(gs[2, 0:6])
    cells = [(l, a) for l in inj["layers"] for a in inj["alphas"]]
    xs = np.arange(len(cells)); w = 0.2

    def get(variant, l, a):
        return [x for x in inj["rows"] if x["variant"] == variant and x["layer"] == l and x["alpha"] == a][0]

    bars = {}
    for k, (variant, col, lab) in enumerate([("marker_to_D", C_NONREF, "direction added on the third object"),
                                             ("random", C_NONUNIQUE, "random direction on the third object (5 seeds)"),
                                             ("marker_to_bg", C_BG, "direction added on background"),
                                             ("marker_minus_T", C_REF, "direction subtracted from the referent")]):
        vals, lo, hi = [], [], []
        for l, a in cells:
            if variant == "random":
                ms = [get(f"random_to_D:{s}", l, a)["pick_D"]["mean"] for s in range(5)]
                vals.append(np.mean(ms)); lo.append(min(ms)); hi.append(max(ms))
            else:
                row = get(variant, l, a)["pick_D"]
                vals.append(row["mean"]); lo.append(row["lo"]); hi.append(row["hi"])
        ax.bar(xs + (k - 1.5) * w, vals, w, color=col, label=lab)
        ax.errorbar(xs + (k - 1.5) * w, vals, yerr=[np.array(vals) - np.array(lo), np.array(hi) - np.array(vals)], fmt="none", ecolor="k", lw=1)
        bars[variant] = dict(zip([f"b{l}_a{a:g}" for l, a in cells], vals))
    for i, (l, a) in enumerate(cells):
        acc = get("marker_minus_T", l, a)["acc"]["mean"]
        ax.text(xs[i] + 1.5 * w, bars["marker_minus_T"][f"b{l}_a{a:g}"] + 0.01, f"acc {acc:.2f}", ha="center", fontsize=S["tick_labelsize"] - 3)
    ax.set_xticks(xs); ax.set_xticklabels([f"block {l}, α = {a:g}" for l, a in cells])
    ax.set_ylabel("fraction of answers = third object's value")
    ax.set_title(f"Direct, injection: {inj['n_items']} clean-correct questions / {inj['n_images']} images (baseline fraction 0)", fontsize=S["subplot_title_fontsize"])
    prov.update({
        "model": "gqa_siglip_decoder1l_scratch_s42 best.pt (= last epoch 17, val 0.6358)",
        "panels": {
            "H1": {"source": str(ROOT / "x23_gqa_direct/x23_results.json"), "field": "h1.per_block.{ref,nonref}", "cohort": "184 questions / 148 images, step-0c eligible, no correctness filter", "estimator": "per-image direction = mean(object patches) − mean(background patches) at c0; projection of raw object mean, c1 − c2; image bootstrap"},
            "H4_crossfit": {"source": str(ROOT / "x23_gqa_spatial_h2/h4_marker_crossfit.json"), "field": "projection.{T,D,A,bg}, window_mean.T_minus_D", "cohort": "35 spatial questions / 27 images; direction from 184 direct pairs, images 2400661/2407031 use a direction fitted without their own pairs", "estimator": "raw object-mean projection on unit(mean c1 − c2 of direct targets), c1 − c2"},
            "R2": {"source": str(ROOT / "x24_gqa_attr/attr_decomposition.json"), "field": "projection.{ref_queried,nonref_queried}.series", "cohort": f"{pr['ref_queried']['n']} object-role rows / {pr['ref_queried']['n_images']} images from the 184 records; directions from an independent pool of {r2['pool_n']} scene-graph objects on other images", "estimator": "cosine of unit object mean with unit own-value direction, c1 − c0"},
            "B2": {"source": str(ROOT / "x23_gqa_direct_inject/marker_injection.json"), "field": "rows[].pick_D (fraction of items answered with the third object's value; NOT p_D softmax), rows[].acc", "cohort": "125 items / 107 images: clean-correct, both values in vocabulary and different", "controls": "5 random unit directions of the same norm; same direction on background; alpha = 0 self-control asserted in code"},
        },
        "not_excluded": "background injection has nearly the same effect as object injection (b9 α2: 0.040 vs 0.064) → not an object-specific role signal; H2 position probing not decidable; H3 registered head test not estimable (0 heads qualified); no error-cohort analysis exists",
        "registry": "X23 steps 1–4 (2026-09-12), X24 A4/B2, X25 R2",
    })
    save(fig, out_dir, "gqa_grounding", prov, ncol=2)


# ---------------------------------------------------------------- functional tests, split (2026-09-17)
# Codex figure spec F4 (writing/STORY_REVISION_AND_FIGURE_SPEC_CODEX_2026-09-17.md): one
# palette for every panel — red = the referent (the object the question is about, or the
# object the edit is applied to while it is the referent), blue = the non-referent, grey =
# random control, purple = background.  Object A is always the measured object in the
# dependence figure; A / B are the fixed object ids of labels.json.
C_A, C_B = (0.839, 0.153, 0.157), (0.121, 0.467, 0.706)
DEP_LABELS = {"none": "none",
              "project_7_marker": "remove ref.\ndirection\nblock 7",
              "project_8_marker": "remove ref.\ndirection\nblock 8",
              "project_10_marker": "remove ref.\ndirection\nblock 10",
              "project_7_random": "remove random\ndirection\nblock 7",
              "gcamask_1-3-5": "mask text\nwrites 1,3,5\nall patches",
              "gcamask_1-3-5_target": "mask text\nwrites 1,3,5\nA patches",
              "gcamask_9-11": "mask text\nwrites 9,11\nall patches"}


def fig_functional_referent_edit(out_dir, alpha=1.0):
    """X21 Part B: colour-difference vector added on patches at one block; y = fraction of
    baseline-correct items whose answer becomes the injected colour."""
    models = [("DINOv2", ""), ("SigLIP", "siglip"), ("MAE", "mae")]
    fig, axes = plt.subplots(1, 3, figsize=(S["subplot_size"][0] * 3, S["subplot_size"][1]), sharey=True)
    prov = {"alpha": alpha, "models": {}}
    lines = [("target_delta_c1", C_A, "-", "on the referent (A, question about A)"),
             ("distractor_delta_c1", C_B, "-", "on the non-referent (B, question about A)"),
             ("bg_all_delta_c1", C_BG, "-", "on all background patches (question about A)"),
             ("target_random_c1", C_NONUNIQUE, "-", "random direction of the same norm on the referent"),
             ("distractor_deltaD_c2", C_A, "--", "on the referent (B, question about B)"),
             ("target_delta_c2", C_B, "--", "on the non-referent (A, question about B)")]
    for ax, (name, sub) in zip(axes, models):
        r = load(f"{sub}/intervention_results.json".lstrip("/"))
        n = None
        for var, col, ls, lab in lines:
            rows = sorted([x for x in r["rows"] if x["variant"] == var and x["alpha"] == alpha], key=lambda x: x["layer"])
            n = rows[0]["n"]
            ax.plot([x["layer"] for x in rows], [x["flip_rate"] for x in rows],
                    **line_kwargs(label=lab, color=col, linestyle=ls))
        mark_gca_layers(ax)
        ax.set_title(f"{name} (n = {n})", fontsize=S["subplot_title_fontsize"])
        ax.set_xlabel("block where the vector is added"); ax.set_xticks(BLOCKS[1::2]); ax.set_ylim(-0.02, 1.02)
        prov["models"][name] = {"source": str(ROOT / sub / "intervention_results.json"), "n_baseline_correct": n,
                                "baseline_accuracy": r["baseline_accuracy"], "variants": r["variants"], "alphas": r["alphas"]}
    axes[0].set_ylabel("fraction of answers switched to the injected colour")
    prov.update({"design": "vector = alpha * (mean raw target mean of colour B − of colour A) from the 1-object set, added at the output of one block on the listed patches; readout = decoder first-token argmax; flip counted on items whose baseline answer is correct",
                 "palette": "red = referent, blue = non-referent, grey = random, purple = background; dashed = question about B (roles swapped)",
                 "not_excluded": "additive edit at one block is not a two-value replacement; background addition switches answers at late blocks in SigLIP/MAE (loss of specificity)",
                 "registry": "X21 Part B results (2026-08-27)"})
    save(fig, out_dir, "functional_referent_edit", prov, ncol=3)


def fig_functional_addback(out_dir, spec=(9, 10, 11)):
    """X25: the non-referent's own colour direction added back on B at blocks 9–11, dose
    alpha × |s_L|; y = fraction of items switching to B's colour."""
    models = [("DINOv2", ""), ("SigLIP", "siglip"), ("MAE", "mae")]
    fig, axes = plt.subplots(1, 3, figsize=(S["subplot_size"][0] * 3, S["subplot_size"][1]), sharey=True)
    prov = {"layer_spec": list(spec), "models": {}}
    lines = [("own", C_B, "-", "own colour of B added on B (non-referent)", None),
             ("own_on_referent", C_A, "-", "same vector added on A (referent, efficacy control)", None),
             ("random", C_NONUNIQUE, "-", "random direction on B (mean, min–max of 5 seeds)", range(5)),
             ("own_on_bg", C_BG, "-", "same vector added on the background", None)]
    for ax, (name, sub) in zip(axes, models):
        r = load(f"{sub}/n2_restore/restoration.json".lstrip("/"))
        alphas = r["alphas"]
        n = [x["n"] for x in r["rows"] if x["variant"] == "own"][0]
        for var, col, ls, lab, seeds in lines:
            m, lo, hi = [], [], []
            for a in alphas:
                if seeds is None:
                    row = [x for x in r["rows"] if x["variant"] == var and x["cond"] == "c1" and x["alpha"] == a and x["layers"] == list(spec)][0]
                    m.append(row["flip_to_Ad"]["mean"]); lo.append(row["flip_to_Ad"]["lo"]); hi.append(row["flip_to_Ad"]["hi"])
                else:
                    ms = [[x for x in r["rows"] if x["variant"] == f"random_{s_}" and x["cond"] == "c1" and x["alpha"] == a and x["layers"] == list(spec)][0]["flip_to_Ad"]["mean"] for s_ in seeds]
                    m.append(np.mean(ms)); lo.append(min(ms)); hi.append(max(ms))
            ax.plot(alphas, m, **line_kwargs(label=lab, color=col, linestyle=ls))
            ax.fill_between(alphas, lo, hi, color=col, alpha=S["std_alpha"], linewidth=0)
        dose = r["dose_s"][11]
        ax.set_title(f"{name} (n = {n}; |s| at block 11 = {abs(dose):.2f})", fontsize=S["subplot_title_fontsize"])
        ax.set_xlabel("dose α (× measured removal |s|)"); ax.set_xticks(alphas); ax.set_ylim(-0.02, 1.02)
        prov["models"][name] = {"source": str(ROOT / sub / "n2_restore/restoration.json"), "n": n, "dose_s_by_block": r["dose_s"], "alphas": alphas}
    axes[0].set_ylabel("fraction of answers switched to B's colour")
    prov.update({"design": "vector = alpha * |s_L| * V_raw[colour of B] added on the listed patches at blocks 9, 10, 11 under the question about A; s_L = mean raw own-colour projection change of the non-referent (question about the other object − no question) at block L; y = fraction of items whose answer becomes B's colour among items whose baseline answer is A's colour",
                 "palette": "blue = non-referent B, red = referent A (efficacy control), grey = random, purple = background",
                 "not_excluded": "additive restoration is not a two-value replacement; zero switches is not zero probability change (ΔP ≈ 1e−5); MAE's measured dose is ≈ 0 so its add-back is near zero by construction",
                 "registry": "X25 results and clarification (2026-09-13/14)"})
    save(fig, out_dir, "functional_addback", prov, ncol=2)


def fig_functional_dependence(out_dir):
    """X24 B1 (appendix): block-11 own-colour alignment of object A and two-object accuracy
    under the same eight interventions; original B1 estimator (in-sample directions,
    parity cross-fit of the reference direction), not the X26 cross-fit."""
    fig = plt.figure(figsize=(S["subplot_size"][0] * 3.6, S["subplot_size"][1] * 2.4))
    gs = GridSpec(2, 2, figure=fig)
    prov = {"B1": {}}
    for j, (name, sub) in enumerate([("DINOv2", ""), ("SigLIP", "siglip")]):
        rows = b1_cells(sub)
        x = np.arange(len(rows))
        labels = [DEP_LABELS[r["spec"]] for r in rows]
        ax = fig.add_subplot(gs[0, j])
        ax.bar(x - 0.2, [r["ref"] for r in rows], 0.4, color=C_A, label="object A, question about A (A is the referent)")
        ax.bar(x + 0.2, [r["nonref"] for r in rows], 0.4, color=C_B, label="object A, question about B (A is the non-referent)")
        ax.errorbar(x + 0.2, [r["nonref"] for r in rows],
                    yerr=[[r["nonref"] - r["nonref_lo"] for r in rows], [r["nonref_hi"] - r["nonref"] for r in rows]],
                    fmt="none", ecolor="k", lw=1)
        for r, xi in zip(rows, x):
            if "seeds" in r:
                ax.scatter([xi + 0.2] * 5, r["seeds"], s=12, color="k", zorder=3)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=S["tick_labelsize"] - 6)
        ax.set_title(f"{name}: alignment of object A, block 11", fontsize=S["subplot_title_fontsize"])
        if j == 0:
            ax.set_ylabel("Δ alignment with own colour")
        ax2 = fig.add_subplot(gs[1, j])
        ax2.plot(x, [r["acc_c1"] for r in rows], **line_kwargs(label="accuracy, question about A", color=C_A))
        ax2.plot(x, [r["acc_c2"] for r in rows], **line_kwargs(label="accuracy, question about B", color=C_B, linestyle="--"))
        ax2.set_ylim(0, 1.05)
        ax2.set_xticks(x); ax2.set_xticklabels(labels, fontsize=S["tick_labelsize"] - 6)
        ax2.set_title(f"{name}: accuracy, same interventions", fontsize=S["subplot_title_fontsize"])
        if j == 0:
            ax2.set_ylabel("accuracy (324 images)")
        prov["B1"][name] = {"rows": rows, "source_dirs": f"{ROOT}/{sub}/n2_int_<spec>/{{partA_attr_directions_normstd.json,behaviour.json}}"}
    prov.update({"design": "object A fixed; alignment = unit-normalised own-colour projection change vs no question at block 11 (original B1 estimator, directions in-sample, reference direction cross-fit by pair-index parity); 'project out' removes (x·v)v on all patch tokens at the block output; random = isotropic unit vector, not energy-matched; mask = zero the cross-attention write at the listed blocks on the listed patches",
                 "palette": "red = A is the referent, blue = A is the non-referent",
                 "not_excluded": "removed energy of random vs reference direction not matched; X26 cross-fitting does not retroactively validate these folds",
                 "registry": "X24 B1 results (2026-09-13)"})
    save(fig, out_dir, "functional_dependence", prov, ncol=2)


FIGS = {"attribute_alignment": fig_attribute_alignment, "functional_tests": fig_functional_tests, "gqa_grounding": fig_gqa_grounding,
        "functional_referent_edit": fig_functional_referent_edit, "functional_addback": fig_functional_addback,
        "functional_dependence": fig_functional_dependence}

# ---------------------------------------------------------------- figure 4
RSA_JSON = Path("outputs/analysis/conditional_rsa/clevr_dinov2_decoder1l_scratch/attr_query_direct/rsa_conditional_stats.json")


def fig_rsa_direct(out_dir):
    """Binding (condition 1, all scenes) and retrieval (condition 4 = answer value,
    within binding-positive scenes), with vs without question, ± SEM across queries."""
    d = json.load(open(RSA_JSON))
    specs = [(1, None, "Binding (all reference scenes)", _tab10[0]),
             (4, 1, "Retrieval (within binding-positive scenes)", _tab10[3])]
    fig, axes = plt.subplots(1, 2, figsize=(S["subplot_size"][0] * 2, S["subplot_size"][1]), sharey=True)
    prov = {"source": str(RSA_JSON), "checkpoint": d["checkpoint"], "n_queries": d["n_queries"], "num_db": d["num_db"], "panels": []}
    for ax, (cond, sub, title, col) in zip(axes, specs):
        row = next(r for r in d["conditional_rsa"] if r["condition_index"] == cond and r["subset_condition_index"] == sub)
        ctrl = [r for r in d["conditional_rsa_unsteered"] if r["name"] == row["name"]]
        assert len(ctrl) == 1
        for rec, lab, ls, alpha in ((row, "with question", "-", 1.0), (ctrl[0], "without question", "--", 0.5)):
            m = np.array([rec["per_layer"][str(k)]["mean"] for k in BLOCKS])
            sem = np.array([rec["per_layer"][str(k)]["std"] / np.sqrt(rec["per_layer"][str(k)]["n"]) for k in BLOCKS])
            ax.plot(BLOCKS, m, **line_kwargs(label=lab, color=col, linestyle=ls, alpha=alpha,
                                              linewidth=S["linewidth"] if ls == "-" else 1))
            ax.fill_between(BLOCKS, m - sem, m + sem, color=col, alpha=S["std_alpha"] * alpha, linewidth=0)
        mark_gca_layers(ax)
        ax.set_title(title, fontsize=S["subplot_title_fontsize"])
        ax.set_xlabel("block"); ax.set_xticks(BLOCKS[1::2]); ax.set_ylim(0, 0.85)
        prov["panels"].append({"title": title, "condition_index": cond, "subset_condition_index": sub, "historical_name": row["name"],
                               "block11_with": row["per_layer"]["11"]["mean"], "block11_without": ctrl[0]["per_layer"]["11"]["mean"],
                               "band": "± SEM = std / sqrt(n) over per-query Spearman rho, n = 72"})
    axes[0].set_ylabel("Spearman ρ")
    prov["not_shown"] = "condition 2 (full four-attribute profile match, block 11 = 0.247) → appendix"
    save(fig, out_dir, "rsa_direct", prov, ncol=2)


FIGS["rsa_direct"] = fig_rsa_direct


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--only", default=None, choices=list(FIGS))
    args = ap.parse_args()
    apply_style()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    for name, fn in FIGS.items():
        if args.only and name != args.only:
            continue
        fn(out)
