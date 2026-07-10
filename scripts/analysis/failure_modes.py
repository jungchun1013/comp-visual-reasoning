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
    args = ap.parse_args()

    if args.replot:
        root = Path(args.output_root)
        dirs = sorted(d for d in root.iterdir() if d.is_dir()) \
            if args.replot == "all" else [root / args.replot]
        for d in dirs:
            summary = json.loads((d / "failure_summary.json").read_text())
            plot_summary(summary, d)
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
