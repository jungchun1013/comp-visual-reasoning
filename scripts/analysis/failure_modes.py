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


def predict_batch(model, task_type, images, questions, inv_answer, device):
    if task_type == "decoder":
        return [p.strip().lower() for p in model.generate(images, questions)]
    logits = model(images, questions)
    return [inv_answer.get(int(i), "?") for i in logits.argmax(dim=-1).cpu()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data-root",
                    default=os.environ.get("CLEVR_ROOT",
                                           "/home/jungchun/data/clevr/CLEVR_v1.0"))
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--stride", type=int, default=1,
                    help="evaluate every k-th val question")
    ap.add_argument("--output-root", default="outputs/analysis/failure_modes")
    args = ap.parse_args()

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
    print(f"Wrote {out_dir}/failure_summary.{{json,md}} + records.jsonl")
    print("\n".join(lines[:20]))


if __name__ == "__main__":
    main()
