"""Unified evaluation with per-type accuracy breakdown."""

from __future__ import annotations

from collections import defaultdict

import torch
from torch.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm


def evaluate_classification(model, dataloader: DataLoader, device: torch.device) -> dict:
    """Evaluate classification model with per-type breakdown.

    Returns:
        {"accuracy": float, "total": int, "breakdown": {type: {"accuracy": float, "count": int}}}
    """
    model.eval()
    correct = 0
    total = 0
    type_stats = defaultdict(lambda: [0, 0])

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            images = batch["image"].to(device)
            questions = batch["question"]
            answers = batch["answer"].to(device)
            oracle_prompts = batch.get("oracle_prompt")

            with autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(images, questions, oracle_prompts=oracle_prompts)

            preds = logits.argmax(dim=-1)
            matches = (preds == answers)
            correct += matches.sum().item()
            total += answers.size(0)

            # Per-type breakdown (CLEVR: question_type, GQA: types dict)
            if "question_type" in batch:
                for i, qtype in enumerate(batch["question_type"]):
                    m = matches[i].item()
                    type_stats[f"qtype/{qtype}"][0] += m
                    type_stats[f"qtype/{qtype}"][1] += 1
            elif "types" in batch:
                for i, t in enumerate(batch["types"]):
                    m = matches[i].item()
                    structural = t.get("structural", "unknown")
                    semantic = t.get("semantic", "unknown")
                    type_stats[f"structural/{structural}"][0] += m
                    type_stats[f"structural/{structural}"][1] += 1
                    type_stats[f"semantic/{semantic}"][0] += m
                    type_stats[f"semantic/{semantic}"][1] += 1

    acc = correct / max(total, 1)
    breakdown = {
        k: {"accuracy": c / max(n, 1), "count": n}
        for k, (c, n) in sorted(type_stats.items())
    }
    return {"accuracy": acc, "total": total, "breakdown": breakdown}


def evaluate_decoder(model, dataloader: DataLoader, device: torch.device, vocab: dict) -> dict:
    """Evaluate decoder model by generating answers and computing exact match.

    Returns same format as evaluate_classification.
    """
    model.eval()
    raw = model.module if hasattr(model, "module") else model
    correct = 0
    total = 0
    type_stats = defaultdict(lambda: [0, 0])

    inv_vocab = {v: k for k, v in vocab.items() if v >= 3}

    for batch in tqdm(dataloader, desc="Evaluating", leave=False):
        images = batch["image"].to(device)
        questions = batch["question"]
        gt_indices = batch["answer"]

        pred_texts = raw.generate(images, questions)
        gt_texts = [inv_vocab.get(i.item() + 3, "?") for i in gt_indices]

        for i, (pred, gt) in enumerate(zip(pred_texts, gt_texts)):
            match = pred.strip().lower() == gt.strip().lower()
            correct += int(match)
            total += 1

            if "question_type" in batch:
                qtype = batch["question_type"][i]
                type_stats[f"qtype/{qtype}"][0] += int(match)
                type_stats[f"qtype/{qtype}"][1] += 1
            elif "types" in batch:
                t = batch["types"][i]
                structural = t.get("structural", "unknown")
                semantic = t.get("semantic", "unknown")
                type_stats[f"structural/{structural}"][0] += int(match)
                type_stats[f"structural/{structural}"][1] += 1
                type_stats[f"semantic/{semantic}"][0] += int(match)
                type_stats[f"semantic/{semantic}"][1] += 1

    acc = correct / max(total, 1)
    breakdown = {
        k: {"accuracy": c / max(n, 1), "count": n}
        for k, (c, n) in sorted(type_stats.items())
    }
    return {"accuracy": acc, "total": total, "breakdown": breakdown}


def format_results(results: dict) -> str:
    """Format results dict into readable table."""
    lines = [f"Overall: {results['accuracy']:.4f} ({results['total']} samples)"]
    if results.get("breakdown"):
        for k, v in sorted(results["breakdown"].items()):
            lines.append(f"  {k:<30} {v['accuracy']:.4f} ({v['count']})")
    return "\n".join(lines)
