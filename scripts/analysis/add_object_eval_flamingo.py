"""E7 add-object eval adapter for the Flamingo-style baseline.

`clevr_flamingo_dinov2_early_s42/last.pt` stores a custom format
(model_state_dict + gca_layers + llm_name) incompatible with
checkpoint_io.load_any_checkpoint, and generation goes through an LLM
(FlamingoModel.generate_answer) instead of model.generate(imgs, questions).
This adapter reuses FlamingoModel/DINOv2 setup from train_flamingo_clevr.py and
emits the SAME summary JSON schema as add_object_eval.py, so E7 tables stay
uniform across models.

Usage (from main/):
  PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 <interpreter> \
      scripts/analysis/add_object_eval_flamingo.py \
      --checkpoint outputs/model/clevr_flamingo_dinov2_early_s42/last.pt \
      --pairs outputs/analysis/add_object/color/pairs.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import timm
import torch
from PIL import Image
from torch.amp import autocast

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from train_flamingo_clevr import FlamingoModel  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--dinov2-name", default="vit_base_patch14_dinov2.lvd142m")
    ap.add_argument("--resolution", type=int, default=336)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    name = Path(args.checkpoint).parent.name

    dinov2 = timm.create_model(args.dinov2_name, pretrained=True)
    if args.resolution != 518:
        dinov2.set_input_size(img_size=args.resolution)
    data_cfg = timm.data.resolve_data_config(dinov2.pretrained_cfg)
    data_cfg["input_size"] = (3, args.resolution, args.resolution)
    transform = timm.data.create_transform(**data_cfg, is_training=False)
    num_prefix = dinov2.num_prefix_tokens
    dinov2 = dinov2.to(device).eval()
    for p in dinov2.parameters():
        p.requires_grad = False

    model = FlamingoModel(
        llm_name=ckpt["llm_name"], dinov2_dim=dinov2.embed_dim,
        gca_layers=ckpt["gca_layers"], lora_r=args.lora_r,
        use_lora=True, device=device)
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    trainable_missing = [k for k in missing if "lora" in k or "gca" in k.lower()
                         or "connector" in k]
    assert not trainable_missing and not unexpected, \
        f"state dict mismatch: missing={trainable_missing} unexpected={unexpected[:5]}"
    model.eval()

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(ckpt["llm_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # decoder-only generation

    pairs_path = Path(args.pairs)
    pairs = json.loads(pairs_path.read_text())
    img_dir = pairs_path.parent / "images"

    @torch.no_grad()
    def predict(names, questions):
        preds = []
        for s in range(0, len(names), args.batch_size):
            batch_n = names[s:s + args.batch_size]
            batch_q = questions[s:s + args.batch_size]
            imgs = torch.stack([
                transform(Image.open(img_dir / n).convert("RGB"))
                for n in batch_n]).to(device)
            feats = dinov2.forward_features(imgs)[:, num_prefix:, :].half()
            prompts = [f"Question: {q}\nAnswer:" for q in batch_q]
            enc = tokenizer(prompts, return_tensors="pt", padding=True,
                            truncation=True, max_length=64).to(device)
            with autocast(device_type="cuda", dtype=torch.bfloat16):
                raw = model.generate_answer(
                    feats, enc["input_ids"], enc["attention_mask"], tokenizer)
            for pred in raw:
                p = pred.split("Answer:")[-1].strip().split()[0] if pred else ""
                preds.append(p.strip(".,!?").lower())
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
        "model": name, "pairs": str(pairs_path), "n": n,
        "acc_base": sum(r["base_correct"] for r in records) / n,
        "acc_added": sum(r["added_correct"] for r in records) / n,
        "hallucination_rate": sum(r["hallucinated_bait"] for r in records) / n,
        "bait_share_of_errors": (
            sum(r["hallucinated_bait"] for r in wrong_added) / len(wrong_added)
            if wrong_added else 0.0),
        "flip_rate": sum(r["pred_base"] != r["pred_added"] for r in records) / n,
        "records": records,
    }
    out = pairs_path.parent / f"add_object_eval_{name}.json"
    out.write_text(json.dumps(summary, indent=2))
    for k in ("acc_base", "acc_added", "hallucination_rate",
              "bait_share_of_errors", "flip_rate"):
        print(f"{k}: {summary[k]:.4f}")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
