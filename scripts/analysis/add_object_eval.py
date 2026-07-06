#!/usr/bin/env python
"""Evaluate add-object hallucination pairs (E7 companion to render_add_object.py).

For each rendered pair (base re-render vs +distractor), ask the original question
and measure:
  acc_base           accuracy on the re-rendered original scene (domain-shift control)
  acc_added          accuracy with the distractor present
  hallucination_rate fraction of pairs where pred_added == the distractor's
                     queried-attribute value (the bait)
  bait_share_of_errors  among wrong added-image answers, fraction that are the bait
  flip_rate          pred changed base→added

Usage (from main/):
  PYTHONPATH=src <interpreter> scripts/analysis/add_object_eval.py \
      --checkpoint outputs/model/clevr_dinov2_concat_decoder1l_scratch_s42/best.pt \
      --pairs outputs/analysis/add_object/color/pairs.json
Writes results next to pairs.json: add_object_eval_<model>.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--pairs", required=True,
                    help="pairs.json from render_add_object.py")
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from model.checkpoint_io import load_any_checkpoint
    model, _sv, transform, _vocab, task_type, meta = \
        load_any_checkpoint(args.checkpoint, device)
    assert task_type == "decoder", "add-object eval expects a decoder model"
    print(f"Loaded {meta['name']} on {device}")

    pairs_path = Path(args.pairs)
    pairs = json.loads(pairs_path.read_text())
    img_dir = pairs_path.parent / "images"

    def predict(names, questions):
        preds = []
        for s in range(0, len(names), args.batch_size):
            batch_n = names[s:s + args.batch_size]
            batch_q = questions[s:s + args.batch_size]
            imgs = torch.stack([
                transform(Image.open(img_dir / n).convert("RGB"))
                for n in batch_n]).to(device)
            with torch.no_grad():
                preds += [p.strip().lower() for p in model.generate(imgs, batch_q)]
        return preds

    questions = [p["question"] for p in pairs]
    pred_base = predict([p["base_image"] for p in pairs], questions)
    pred_added = predict([p["added_image"] for p in pairs], questions)

    records, n = [], len(pairs)
    for p, pb, pa in zip(pairs, pred_base, pred_added):
        records.append({**p, "pred_base": pb, "pred_added": pa,
                        "base_correct": pb == p["answer"],
                        "added_correct": pa == p["answer"],
                        "hallucinated_bait": pa == p["bait_answer"]})

    wrong_added = [r for r in records if not r["added_correct"]]
    summary = {
        "model": meta["name"], "pairs": str(pairs_path), "n": n,
        "acc_base": sum(r["base_correct"] for r in records) / n,
        "acc_added": sum(r["added_correct"] for r in records) / n,
        "hallucination_rate": sum(r["hallucinated_bait"] for r in records) / n,
        "bait_share_of_errors": (
            sum(r["hallucinated_bait"] for r in wrong_added) / len(wrong_added)
            if wrong_added else 0.0),
        "flip_rate": sum(r["pred_base"] != r["pred_added"] for r in records) / n,
        "records": records,
    }
    out = pairs_path.parent / f"add_object_eval_{meta['name']}.json"
    out.write_text(json.dumps(summary, indent=2))
    for k in ("acc_base", "acc_added", "hallucination_rate",
              "bait_share_of_errors", "flip_rate"):
        print(f"{k}: {summary[k]:.4f}")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
