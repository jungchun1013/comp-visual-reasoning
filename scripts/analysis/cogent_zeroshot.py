"""Zero-shot compositional generalization on CLEVR-CoGenT via activation interpolation.

Sweep α ∈ {0, 0.25, 0.5, 0.75, 1.0}:
  x_corrected = (1-α) * x_target + α * (x_q1 + x_q2 - x_q3)

α=0 → direct (baseline), α=1 → full composition.

Usage:
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python scripts/analysis/cogent_zeroshot.py \
        --checkpoint <cogent_A_checkpoint> --n-questions 50
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.amp import autocast
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from model import CrossAttnViT
from tasks.decoder import build_decoder_model, build_clevr_decoder_vocab
from data.clevr import CLEVRVQADataset
from analysis.run_log import tee_stdout

# ── CoGenT conditions ───────────────────────────────────────────────

COGENT_A = {
    "cube": {"gray", "blue", "brown", "yellow"},
    "cylinder": {"red", "green", "purple", "cyan"},
    "sphere": {"gray", "red", "blue", "green", "brown", "purple", "cyan", "yellow"},
}


def is_unseen_in_A(color, shape):
    return color not in COGENT_A.get(shape, set())


def find_unseen_combo_in_program(program):
    """Find (color, shape) pairs that are chained filter_color → filter_shape
    on the same object AND are unseen in CoGenT-A."""
    for i in range(len(program) - 1):
        if (program[i]["function"] == "filter_color"
                and program[i + 1]["function"] == "filter_shape"
                and i in program[i + 1].get("inputs", [])):
            color = program[i]["value_inputs"][0]
            shape = program[i + 1]["value_inputs"][0]
            if is_unseen_in_A(color, shape):
                return color, shape
    return None, None


def make_basis_questions(question, target_color, target_shape):
    safe_colors = COGENT_A.get(target_shape, set())
    if not safe_colors:
        return None
    safe_color = sorted(safe_colors)[0]
    q1 = question.replace(target_shape, "sphere")
    q2 = question.replace(target_color, safe_color)
    q3 = question.replace(target_color, safe_color).replace(target_shape, "sphere")
    return q1, q2, q3, safe_color


# ── Model loading ───────────────────────────────────────────────────

def load_model(ckpt_path, data_root, device):
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    steervit = CrossAttnViT.from_config(
        "vit_base_patch14_dinov2.lvd142m", device=device,
        cross_attn_layers=[1, 3, 5, 7, 9, 11], resolution=336)
    if "steervit_trainable_state" in ckpt:
        steervit.load_state_dict(ckpt["steervit_trainable_state"], strict=False)

    vocab = build_clevr_decoder_vocab()
    cfg = OmegaConf.create({
        "model": {"backbone_name": "vit_base_patch14_dinov2.lvd142m",
                  "resolution": 336, "cross_attn_layers": [1, 3, 5, 7, 9, 11]},
        "task": {"type": "decoder", "decoder": {"d_model": 512, "nhead": 8,
                 "num_layers": 1, "max_len": 8}, "use_text_gca": False},
        "data": {"root": data_root},
    })
    model = build_decoder_model(steervit, cfg).to(device)
    if "decoder_state_dict" in ckpt:
        model.decoder.load_state_dict(ckpt["decoder_state_dict"], strict=False)
    model.eval()
    transform = steervit.get_transforms()
    print(f"Loaded CoGenT-A model (epoch {ckpt.get('epoch', '?')})")
    return model, steervit, vocab, transform


# ── Activation extraction & interpolation ───────────────────────────

def get_layer_activations(steervit, image, question):
    """Run forward, hook all layer outputs. Returns dict {layer_idx: tensor}."""
    blocks = steervit.vision_model.trunk.blocks
    layer_out = {}
    hooks = []
    for idx, blk in enumerate(blocks):
        def make_hook(li):
            def fn(module, inp, output):
                out = output[0] if isinstance(output, tuple) else output
                layer_out[li] = out.detach().clone()
            return fn
        hooks.append(blk.register_forward_hook(make_hook(idx)))

    with torch.no_grad(), autocast(device_type="cuda", dtype=torch.bfloat16):
        steervit.forward(image, [question])

    for h in hooks:
        h.remove()
    return layer_out


def decode_with_activations(model, steervit, image, composed_acts, question, device):
    """Forward with hooked activations replaced, then decode."""
    blocks = steervit.vision_model.trunk.blocks

    hook_handles = []
    for idx, blk in enumerate(blocks):
        def make_replace_hook(li):
            def fn(module, inp, output):
                if isinstance(output, tuple):
                    return (composed_acts[li].to(output[0].dtype),) + output[1:]
                return composed_acts[li].to(output.dtype)
            return fn
        hook_handles.append(blk.register_forward_hook(make_replace_hook(idx)))

    with torch.no_grad(), autocast(device_type="cuda", dtype=torch.bfloat16):
        vit_out = steervit.forward(image, [question])

    for h in hook_handles:
        h.remove()

    prefix = steervit.vision_model.trunk.num_prefix_tokens
    norm = steervit.vision_model.trunk.norm
    normed = norm(vit_out.float())
    patches = normed[:, prefix:, :]

    with torch.no_grad(), autocast(device_type="cuda", dtype=torch.bfloat16):
        token_ids = model.decoder.generate(
            patches.to(torch.bfloat16), bos_id=0, eos_id=1, max_len=4)

    inv_vocab = {v: k for k, v in model.vocab.items()}
    words = []
    for t in token_ids[0]:
        w = inv_vocab.get(t.item(), "")
        if w == "<eos>":
            break
        if w not in ("<bos>", "<pad>"):
            words.append(w)
    return " ".join(words)


# ── Main ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-root", type=str,
                        default="/home/jungchun/data/clevr/CLEVR_CoGenT_v1.0")
    parser.add_argument("--n-questions", type=int, default=50)
    parser.add_argument("--alphas", type=str, default="0,0.25,0.5,0.75,1.0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    alphas = [float(a) for a in args.alphas.split(",")]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Alphas: {alphas}")

    model, steervit, vocab, transform = load_model(
        args.checkpoint, args.data_root, device)

    output_dir = Path(args.output_dir) if args.output_dir else \
        Path("outputs/analysis/cogent_zeroshot")
    output_dir.mkdir(parents=True, exist_ok=True)
    tee_stdout(output_dir)

    # Load CoGenT-B validation set
    dataset = CLEVRVQADataset(args.data_root, "valB", transform)
    print(f"CoGenT-B val: {len(dataset)} questions")

    # Find questions with unseen combos
    rng = random.Random(args.seed)
    unseen_questions = []
    for i, q in enumerate(dataset.questions):
        program = q.get("program", [])
        color, shape = find_unseen_combo_in_program(program)
        if color and shape:
            unseen_questions.append({
                "idx": i, "question": q["question"], "answer": q["answer"],
                "color": color, "shape": shape,
            })

    print(f"Questions with unseen combos: {len(unseen_questions)}")
    rng.shuffle(unseen_questions)
    unseen_questions = unseen_questions[:args.n_questions]

    num_layers = len(steervit.vision_model.trunk.blocks)
    results = {f"alpha={a}": {"correct": 0, "total": 0, "details": []}
               for a in alphas}

    for qi, qinfo in enumerate(unseen_questions):
        idx = qinfo["idx"]
        question = qinfo["question"]
        gt_answer = qinfo["answer"]
        color, shape = qinfo["color"], qinfo["shape"]

        basis = make_basis_questions(question, color, shape)
        if basis is None:
            continue
        q1, q2, q3, safe_color = basis
        image = dataset[idx]["image"].unsqueeze(0).to(device)

        print(f"\nQ{qi+1}/{len(unseen_questions)}: {question} → {gt_answer}")
        print(f"  Unseen: {color} {shape} | basis color: {safe_color}")

        # Extract activations for all 4 questions
        act_target = get_layer_activations(steervit, image, question)
        act_q1 = get_layer_activations(steervit, image, q1)
        act_q2 = get_layer_activations(steervit, image, q2)
        act_q3 = get_layer_activations(steervit, image, q3)

        # Composed activations (α=1)
        act_composed = {l: act_q1[l] + act_q2[l] - act_q3[l]
                        for l in range(num_layers)}

        for alpha in alphas:
            key = f"alpha={alpha}"
            if alpha == 0.0:
                # Direct: use target question
                with torch.no_grad(), autocast(device_type="cuda", dtype=torch.bfloat16):
                    preds = model.generate(image, [question])
                pred = preds[0]
            else:
                # Interpolate: (1-α)*target + α*composed
                interpolated = {
                    l: (1 - alpha) * act_target[l] + alpha * act_composed[l]
                    for l in range(num_layers)
                }
                pred = decode_with_activations(
                    model, steervit, image, interpolated, question, device)

            correct = pred.strip().lower() == gt_answer.strip().lower()
            results[key]["correct"] += int(correct)
            results[key]["total"] += 1
            results[key]["details"].append({
                "question": question, "answer": gt_answer,
                "pred": pred, "correct": correct,
            })
            mark = "✓" if correct else "✗"
            print(f"  α={alpha}: {pred} {mark}")

    # Summary
    print(f"\n{'='*60}")
    print(f"Results ({len(unseen_questions)} unseen-combo questions from CoGenT-B)")
    print(f"{'='*60}")
    for alpha in alphas:
        key = f"alpha={alpha}"
        r = results[key]
        if r["total"] > 0:
            acc = r["correct"] / r["total"]
            print(f"  α={alpha:<5}: {r['correct']}/{r['total']} = {acc:.1%}")

    save_path = output_dir / "zeroshot_alpha_sweep.json"
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {save_path}")


if __name__ == "__main__":
    main()
