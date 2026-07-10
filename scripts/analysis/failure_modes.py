#!/usr/bin/env python
"""Failure-mode analysis: per-question dump + per-family / confusion aggregations (E5).

Nothing existing computes per-family accuracy or answer-confusion structure — the
evaluator only aggregates per coarse question type. This script dumps one record per
question and derives the failure tables the v2 §A5 section needs:

  - per-question-type and per-family accuracy (worst families first)
  - yes/no subset: full confusion + answer-prior bias (majority-"no" collapse?)
  - counting subset: signed error distribution (off-by-one vs random)
  - global answer marginals: gt vs pred drift

Outputs (new dir per model, never overwrites):
  outputs/analysis/failure_modes/<model_name>/records.jsonl
  outputs/analysis/failure_modes/<model_name>/failure_summary.json
  outputs/analysis/failure_modes/<model_name>/failure_summary.md

Usage (from main/):
  PYTHONPATH=src <interpreter> scripts/analysis/failure_modes.py \
      --checkpoint outputs/model/clevr_dinov2_concat_decoder1l_scratch_s42/best.pt \
      [--stride 8]   # every 8th val question (~18.7k) — plenty for family stats

GPU: ~full-val 150k questions ≈ the cost of one eval_generalization run; use
--stride to subsample when the GPU budget is tight (policy 2026-07-05: long acc
evals must not block mainline work).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from data.clevr import CLEVRVQADataset, clevr_collate_fn  # noqa: E402
from analysis.run_log import tee_stdout  # noqa: E402

YESNO = {"yes", "no"}
DIGITS = {str(i) for i in range(11)}

QTYPE_ORDER = ["query_attribute", "equal_attribute", "exist", "count",
               "compare_integer"]


def plot_summary(summary, out_dir):
    """failure_modes.png: per-qtype acc, yes/no confusion, counting errors, depth."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from analysis.plot_style import apply_style, S, line_kwargs
    apply_style()
    _tab10 = plt.cm.tab10.colors

    fig, axes = plt.subplots(1, 4, figsize=(6.4 * 4, 4.8))

    # (a) per-question-type accuracy
    ax = axes[0]
    qt = summary["per_qtype"]
    keys = [k for k in QTYPE_ORDER if k in qt] + \
        sorted(k for k in qt if k not in QTYPE_ORDER)
    ax.bar(range(len(keys)), [qt[k]["accuracy"] for k in keys], color=_tab10[0])
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([k.replace("_", "\n") for k in keys],
                       fontsize=S["tick_labelsize"] - 2)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("accuracy")
    ax.set_title("Per question type", fontsize=S["subplot_title_fontsize"])

    # (b) yes/no confusion (row-normalized) with counts
    ax = axes[1]
    conf = summary["yesno"]["confusion"]
    mat = np.array([[conf.get("yes->yes", 0), conf.get("yes->no", 0)],
                    [conf.get("no->yes", 0), conf.get("no->no", 0)]], float)
    norm = mat / mat.sum(axis=1, keepdims=True)
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{norm[i, j]:.3f}\n({int(mat[i, j])})",
                    ha="center", va="center",
                    color="white" if norm[i, j] > 0.5 else "black",
                    fontsize=S["tick_labelsize"])
    ax.set_xticks([0, 1]); ax.set_xticklabels(["pred yes", "pred no"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["gt yes", "gt no"])
    ax.set_title(f"Yes/No (pred-no {summary['yesno']['pred_no_rate']:.3f}, "
                 f"gt-no {summary['yesno']['gt_no_rate']:.3f})",
                 fontsize=S["subplot_title_fontsize"] - 2)

    # (c) counting signed-error histogram (log y)
    ax = axes[2]
    hist = summary["counting"]["signed_error_hist"]
    numeric = {int(k): v for k, v in hist.items() if k.lstrip("-").isdigit()}
    xs = sorted(numeric)
    ax.bar(xs, [numeric[x] for x in xs],
           color=[_tab10[2] if x == 0 else _tab10[3] for x in xs])
    ax.set_yscale("log")
    ax.set_xticks(xs)
    ax.set_xlabel("pred − gt")
    ax.set_ylabel("questions (log)")
    nn = hist.get("non-numeric", 0)
    ax.set_title(f"Counting errors (acc {summary['counting']['acc']:.3f}"
                 + (f", non-numeric {nn}" if nn else "") + ")",
                 fontsize=S["subplot_title_fontsize"] - 2)

    # (d) accuracy vs program depth (productivity axis)
    ax = axes[3]
    pd_ = {int(k): v for k, v in summary["per_depth"].items()}
    xs = sorted(pd_)
    ax.plot(xs, [pd_[x]["accuracy"] for x in xs], **line_kwargs(color=_tab10[4]))
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("program depth")
    ax.set_ylabel("accuracy")
    ax.set_title("Depth (productivity)", fontsize=S["subplot_title_fontsize"])

    fig.suptitle(f"{summary['model']} — failure modes (n={summary['n']}, "
                 f"overall {summary['overall_acc']:.3f})",
                 fontsize=S["suptitle_fontsize"])
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = Path(out_dir) / "failure_modes.png"
    fig.savefig(out, dpi=S["dpi"], bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ── E5 autonomous diagnosis (CPU-only: records ⋈ CLEVR questions+scenes) ──

def _exec_program(program, scene):
    """Mini CLEVR program executor — list of per-node outputs (sets/ints/strs).

    Ground-truth side only (uses scene annotations, not the model); lets the
    diagnosis recover latent quantities the answer hides, e.g. the two counts
    feeding a compare_integer node.
    """
    objs = scene["objects"]
    rel = scene["relationships"]
    out = []
    for node in program:
        f = node["function"]
        ins = [out[i] for i in node["inputs"]]
        vals = node.get("value_inputs", [])
        if f == "scene":
            r = set(range(len(objs)))
        elif f.startswith("filter_"):
            attr = f.split("_", 1)[1]
            r = {o for o in ins[0] if objs[o][attr] == vals[0]}
        elif f == "unique":
            (r,) = ins[0]
        elif f == "relate":
            r = set(rel[vals[0]][ins[0]])
        elif f.startswith("same_"):
            attr = f.split("_", 1)[1]
            r = {o for o in range(len(objs))
                 if objs[o][attr] == objs[ins[0]][attr] and o != ins[0]}
        elif f == "count":
            r = len(ins[0])
        elif f == "exist":
            r = "yes" if ins[0] else "no"
        elif f.startswith("query_"):
            r = objs[ins[0]][f.split("_", 1)[1]]
        elif f in ("equal_integer", "greater_than", "less_than"):
            r = {"equal_integer": ins[0] == ins[1],
                 "greater_than": ins[0] > ins[1],
                 "less_than": ins[0] < ins[1]}[f]
            r = "yes" if r else "no"
        elif f.startswith("equal_"):
            r = "yes" if ins[0] == ins[1] else "no"
        elif f == "union":
            r = ins[0] | ins[1]
        elif f == "intersect":
            r = ins[0] & ins[1]
        else:
            raise ValueError(f"unknown CLEVR function: {f}")
        out.append(r)
    return out


def _rate_table(pairs):
    """[(key, correct_bool)] → {key: {accuracy, count}} sorted by key."""
    stats = defaultdict(lambda: [0, 0])
    for k, c in pairs:
        stats[k][0] += c
        stats[k][1] += 1
    return {str(k): {"accuracy": c / n, "count": n}
            for k, (c, n) in sorted(stats.items())}


def diagnose(model_dir, questions, scenes_by_img):
    """D1–D4 follow-up analyses on an existing records.jsonl (no GPU)."""
    out_dir = Path(model_dir)
    records = [json.loads(l) for l in open(out_dir / "records.jsonl")]
    print(f"[{out_dir.name}] {len(records)} records")

    exec_agree = [0, 0]  # executor-vs-gt sanity over all executed programs
    cnt_by_gt = defaultdict(lambda: [0, 0, 0.0, 0, 0])  # n, correct, err_sum, under, over
    cnt_by_scene = []      # (n_objects, correct) for count questions
    all_by_scene = []      # (n_objects, correct) for every question
    cmp_by_delta = []      # (|c1-c2|, correct)
    cmp_mixed = []         # ((delta, op), correct)
    yn_by_relate = []      # (n_relate_ops, correct) on yes/no gt
    yn_err_family = Counter()
    depth_comp = defaultdict(Counter)  # depth → qtype counter (errors only)

    for r in records:
        q = questions[r["q_idx"]]
        scene = scenes_by_img[q["image_index"]]
        n_obj = len(scene["objects"])
        prog = q["program"]
        outs = _exec_program(prog, scene)
        exec_agree[0] += str(outs[-1]).lower() == r["gt"]
        exec_agree[1] += 1
        all_by_scene.append((n_obj, r["correct"]))

        if r["gt"] in DIGITS:  # D1 counting
            g = int(r["gt"])
            s = cnt_by_gt[g]
            s[0] += 1
            s[1] += r["correct"]
            if r["pred"] in DIGITS:
                e = int(r["pred"]) - g
                s[2] += e
                s[3] += e < 0
                s[4] += e > 0
            cnt_by_scene.append((n_obj, r["correct"]))

        final = prog[-1]["function"]
        if final in ("equal_integer", "greater_than", "less_than"):  # D2
            c1, c2 = (outs[i] for i in prog[-1]["inputs"])
            delta = abs(c1 - c2)
            cmp_by_delta.append((delta, r["correct"]))
            cmp_mixed.append((f"{final}|{delta}", r["correct"]))

        if r["gt"] in YESNO:  # D3
            n_rel = sum(n["function"] == "relate" for n in prog)
            yn_by_relate.append((n_rel, r["correct"]))
            if not r["correct"]:
                yn_err_family[r["family"]] += 1

        if not r["correct"]:  # D4
            depth_comp[r["depth"]][r["qtype"]] += 1

    diag = {
        "model": out_dir.name,
        "n": len(records),
        "executor_gt_agreement": exec_agree[0] / max(exec_agree[1], 1),
        "d1_counting_by_gt_count": {
            str(g): {"count": n, "accuracy": c / n,
                     "mean_signed_error": es / max(n, 1),
                     "undercount": u, "overcount": o}
            for g, (n, c, es, u, o) in sorted(cnt_by_gt.items())},
        "d1_counting_acc_by_scene_size": _rate_table(cnt_by_scene),
        "d1_overall_acc_by_scene_size": _rate_table(all_by_scene),
        "d2_compare_acc_by_abs_delta": _rate_table(cmp_by_delta),
        "d2_compare_acc_by_op_delta": _rate_table(cmp_mixed),
        "d3_yesno_acc_by_n_relate": _rate_table(yn_by_relate),
        "d3_yesno_top_error_families": dict(yn_err_family.most_common(15)),
        "d4_error_qtype_by_depth": {
            str(d): dict(c.most_common()) for d, c in sorted(depth_comp.items())},
    }
    (out_dir / "diagnosis.json").write_text(json.dumps(diag, indent=2))
    plot_diagnosis(diag, out_dir)
    print(f"[{out_dir.name}] executor/gt agreement "
          f"{diag['executor_gt_agreement']:.4f}; wrote diagnosis.json")
    return diag


def plot_diagnosis(diag, out_dir):
    """diagnosis.png: counting capacity, clutter, compare margin, relate chain."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from analysis.plot_style import apply_style, S, line_kwargs
    apply_style()
    _tab10 = plt.cm.tab10.colors

    fig, axes = plt.subplots(1, 4, figsize=(6.4 * 4, 4.8))

    # (a) D1: counting acc + mean signed error vs gt count
    ax = axes[0]
    d1 = {int(k): v for k, v in diag["d1_counting_by_gt_count"].items()}
    xs = sorted(d1)
    ax.bar(xs, [d1[x]["accuracy"] for x in xs], color=_tab10[0], alpha=0.8)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(xs)
    ax.set_xlabel("gt count")
    ax.set_ylabel("accuracy")
    ax2 = ax.twinx()
    ax2.plot(xs, [d1[x]["mean_signed_error"] for x in xs],
             **line_kwargs(color=_tab10[3]))
    ax2.axhline(0, color="gray", lw=0.8)
    ax2.set_ylabel("mean signed error", color=_tab10[3])
    ax.set_title("Counting vs gt count", fontsize=S["subplot_title_fontsize"])

    # (b) D1: counting vs overall acc as scene clutter grows
    ax = axes[1]
    for key, color, label in [("d1_counting_acc_by_scene_size", _tab10[0], "count qs"),
                              ("d1_overall_acc_by_scene_size", _tab10[7], "all qs")]:
        d = {int(k): v for k, v in diag[key].items()}
        xs = sorted(d)
        ax.plot(xs, [d[x]["accuracy"] for x in xs],
                label=label, **line_kwargs(color=color))
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("objects in scene")
    ax.set_ylabel("accuracy")
    ax.legend(fontsize=S["legend_fontsize"], frameon=False)
    ax.set_title("Scene clutter", fontsize=S["subplot_title_fontsize"])

    # (c) D2: compare_integer acc vs |count1 − count2|
    ax = axes[2]
    d2 = {int(k): v for k, v in diag["d2_compare_acc_by_abs_delta"].items()}
    xs = sorted(d2)
    ax.bar(xs, [d2[x]["accuracy"] for x in xs], color=_tab10[2])
    for x in xs:
        ax.text(x, d2[x]["accuracy"] + 0.02, str(d2[x]["count"]),
                ha="center", fontsize=S["tick_labelsize"] - 2)
    ax.set_ylim(0, 1.1)
    ax.set_xticks(xs)
    ax.set_xlabel("|count$_1$ − count$_2$|")
    ax.set_ylabel("accuracy")
    ax.set_title("Integer comparison margin", fontsize=S["subplot_title_fontsize"])

    # (d) D3: yes/no accuracy vs number of relate ops in the program
    ax = axes[3]
    d3 = {int(k): v for k, v in diag["d3_yesno_acc_by_n_relate"].items()}
    xs = sorted(d3)
    ax.bar(xs, [d3[x]["accuracy"] for x in xs], color=_tab10[4])
    for x in xs:
        ax.text(x, d3[x]["accuracy"] + 0.02, str(d3[x]["count"]),
                ha="center", fontsize=S["tick_labelsize"] - 2)
    ax.set_ylim(0, 1.1)
    ax.set_xticks(xs)
    ax.set_xlabel("relate ops in program")
    ax.set_ylabel("accuracy")
    ax.set_title("Yes/No vs spatial chain", fontsize=S["subplot_title_fontsize"])

    fig.suptitle(f"{diag['model']} — E5 diagnosis (n={diag['n']})",
                 fontsize=S["suptitle_fontsize"])
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = Path(out_dir) / "diagnosis.png"
    fig.savefig(out, dpi=S["dpi"], bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def predict_batch(model, task_type, images, questions, inv_answer, device):
    if task_type == "decoder":
        return [p.strip().lower() for p in model.generate(images, questions)]
    logits = model(images, questions)
    return [inv_answer.get(int(i), "?") for i in logits.argmax(dim=-1).cpu()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint")
    ap.add_argument("--data-root",
                    default=os.environ.get("CLEVR_ROOT",
                                           "/home/jungchun/data/clevr/CLEVR_v1.0"))
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--stride", type=int, default=1,
                    help="evaluate every k-th val question")
    ap.add_argument("--output-root", default="outputs/analysis/failure_modes")
    ap.add_argument("--replot", default=None, metavar="MODEL|all",
                    help="regenerate failure_modes.png from failure_summary.json "
                         "(no GPU): a model dir name under output-root, or 'all'")
    ap.add_argument("--diagnose", default=None, metavar="MODEL|all",
                    help="E5 follow-up diagnosis from records.jsonl joined with "
                         "CLEVR val questions/scenes (no GPU); writes "
                         "diagnosis.{json,png} into the model dir")
    args = ap.parse_args()

    if args.replot or args.diagnose:
        sel = args.replot or args.diagnose
        root = Path(args.output_root)
        dirs = sorted(d for d in root.iterdir() if d.is_dir()) \
            if sel == "all" else [root / sel]
        if args.replot:
            for d in dirs:
                summary = json.loads((d / "failure_summary.json").read_text())
                plot_summary(summary, d)
            return
        droot = Path(args.data_root)
        print("Loading CLEVR val questions/scenes ...")
        questions = json.loads(
            (droot / "questions/CLEVR_val_questions.json").read_text())["questions"]
        scenes = json.loads(
            (droot / "scenes/CLEVR_val_scenes.json").read_text())["scenes"]
        scenes_by_img = {s["image_index"]: s for s in scenes}
        for d in dirs:
            tee_stdout(d)
            diagnose(d, questions, scenes_by_img)
        return
    if not args.checkpoint:
        ap.error("--checkpoint is required unless --replot is given")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from model.checkpoint_io import load_any_checkpoint
    model, _sv, transform, _vocab, task_type, meta = \
        load_any_checkpoint(args.checkpoint, device)
    print(f"Loaded {meta['name']} task={task_type} epoch={meta['epoch']} "
          f"val_acc={meta['val_acc']} device={device}")

    from data.clevr import ANSWER_TO_IDX
    inv_answer = {v: k for k, v in ANSWER_TO_IDX.items()}

    out_dir = Path(args.output_root) / meta["name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    tee_stdout(out_dir)

    ds = CLEVRVQADataset(args.data_root, "val", transform)
    indices = list(range(0, len(ds), args.stride))
    loader = DataLoader(Subset(ds, indices), batch_size=args.batch_size,
                        shuffle=False, num_workers=8, collate_fn=clevr_collate_fn,
                        pin_memory=True)

    records = []
    cursor = 0
    with torch.no_grad(), open(out_dir / "records.jsonl", "w") as fout:
        for batch in tqdm(loader, desc="failure-modes"):
            images = batch["image"].to(device)
            questions = batch["question"]
            preds = predict_batch(model, task_type, images, questions,
                                  inv_answer, device)
            for i, pred in enumerate(preds):
                q = ds.questions[indices[cursor]]
                rec = {
                    "q_idx": indices[cursor],
                    "family": q.get("question_family_index"),
                    "qtype": batch["question_type"][i],
                    "depth": int(batch["program_depth"][i]),
                    "question": q["question"],
                    "gt": str(q.get("answer", "")).lower(),
                    "pred": pred,
                }
                rec["correct"] = rec["pred"] == rec["gt"]
                records.append(rec)
                fout.write(json.dumps(rec) + "\n")
                cursor += 1

    # ── aggregate ────────────────────────────────────────────────────
    def acc_table(key):
        stats = defaultdict(lambda: [0, 0])
        for r in records:
            stats[r[key]][0] += r["correct"]
            stats[r[key]][1] += 1
        return {k: {"accuracy": c / n, "count": n}
                for k, (c, n) in stats.items()}

    per_qtype = acc_table("qtype")
    per_family = acc_table("family")
    per_depth = acc_table("depth")  # productivity axis: acc vs program depth

    yn = [r for r in records if r["gt"] in YESNO]
    yn_conf = Counter((r["gt"], r["pred"]) for r in yn)
    yn_pred_no = sum(1 for r in yn if r["pred"] == "no") / max(len(yn), 1)
    yn_gt_no = sum(1 for r in yn if r["gt"] == "no") / max(len(yn), 1)

    cnt = [r for r in records if r["gt"] in DIGITS]
    cnt_err = Counter(
        (int(r["pred"]) - int(r["gt"])) if r["pred"] in DIGITS else "non-numeric"
        for r in cnt)

    gt_marg = Counter(r["gt"] for r in records)
    pred_marg = Counter(r["pred"] for r in records)
    drift = {a: pred_marg.get(a, 0) - gt_marg.get(a, 0)
             for a in set(gt_marg) | set(pred_marg)}

    summary = {
        "model": meta["name"], "checkpoint": str(args.checkpoint),
        "task_type": task_type, "stride": args.stride, "n": len(records),
        "overall_acc": sum(r["correct"] for r in records) / max(len(records), 1),
        "per_qtype": per_qtype,
        "per_family": per_family,
        "per_depth": {str(k): v for k, v in sorted(per_depth.items())},
        "yesno": {
            "n": len(yn),
            "acc": sum(r["correct"] for r in yn) / max(len(yn), 1),
            "confusion": {f"{g}->{p}": c for (g, p), c in yn_conf.items()},
            "pred_no_rate": yn_pred_no, "gt_no_rate": yn_gt_no,
        },
        "counting": {
            "n": len(cnt),
            "acc": sum(r["correct"] for r in cnt) / max(len(cnt), 1),
            "signed_error_hist": {str(k): v for k, v in sorted(
                cnt_err.items(), key=lambda kv: str(kv[0]))},
        },
        "answer_marginal_drift_top": dict(sorted(
            drift.items(), key=lambda kv: -abs(kv[1]))[:12]),
    }
    (out_dir / "failure_summary.json").write_text(json.dumps(summary, indent=2))

    lines = [f"# Failure modes — {meta['name']} (n={len(records)}, stride={args.stride})",
             "", f"Overall acc: {summary['overall_acc']:.4f}", "",
             "## Per question type", "", "| qtype | acc | n |", "|---|---|---|"]
    for k, v in sorted(per_qtype.items(), key=lambda kv: kv[1]["accuracy"]):
        lines.append(f"| {k} | {v['accuracy']:.4f} | {v['count']} |")
    lines += ["", "## Worst 20 families", "", "| family | acc | n |", "|---|---|---|"]
    for k, v in sorted(per_family.items(), key=lambda kv: kv[1]["accuracy"])[:20]:
        lines.append(f"| {k} | {v['accuracy']:.4f} | {v['count']} |")
    yn_s = summary["yesno"]
    lines += ["", "## Yes/No",
              f"n={yn_s['n']} acc={yn_s['acc']:.4f} "
              f"pred-no rate={yn_s['pred_no_rate']:.3f} (gt-no {yn_s['gt_no_rate']:.3f})",
              f"confusion: {yn_s['confusion']}"]
    c_s = summary["counting"]
    lines += ["", "## Counting",
              f"n={c_s['n']} acc={c_s['acc']:.4f}",
              f"signed error hist: {c_s['signed_error_hist']}",
              "", "## Answer marginal drift (pred − gt, top 12)",
              f"{summary['answer_marginal_drift_top']}"]
    (out_dir / "failure_summary.md").write_text("\n".join(lines) + "\n")
    plot_summary(summary, out_dir)
    print(f"Wrote {out_dir}/failure_summary.{{json,md}} + records.jsonl")
    print("\n".join(lines[:20]))


if __name__ == "__main__":
    main()
