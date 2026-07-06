"""Binding subspace interchange intervention.

Tests whether binding-aligned geometry in SteerViT causally mediates
answer prediction, not just correlates with it.

Conditions:
  C1: Described-attr interchange — answer should flip
  C2: Same-binding query change  — answer should NOT flip
  C3: Random direction control    — answer should NOT flip
  Layer sweep: C1 across all layers
  Cross-query: d_bind from Q1, test on Q2

Usage:
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python scripts/analysis/binding_interchange.py \
        --checkpoint outputs/model/clevr_dinov2_decoder1l_scratch_s42/best.pt
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

import numpy as np
import torch
from torch.amp import autocast

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from model import CrossAttnViT
from data.clevr import CLEVRVQADataset, ANSWER_TO_IDX, IDX_TO_ANSWER
from tasks.decoder import build_clevr_decoder_vocab, VQADecoder, DecoderModel

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ATTR_KEYS = ("color", "shape", "material", "size")


# ── Model loading (from run_cogent_patching.py pattern) ──────────

def load_model(ckpt_path, device):
    from omegaconf import OmegaConf

    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    vocab = build_clevr_decoder_vocab()

    if "config" not in ckpt:
        cross_attn_layers = [1, 3, 5, 7, 9, 11]
        steervit = CrossAttnViT.from_config(
            "vit_base_patch14_dinov2.lvd142m", device=device,
            cross_attn_layers=cross_attn_layers, resolution=336,
        )
        if "steervit_trainable_state" in ckpt:
            steervit.load_state_dict(ckpt["steervit_trainable_state"], strict=False)
        dec_sd = ckpt.get("decoder_state_dict", {})
        layer_indices = {int(k.split(".")[1]) for k in dec_sd if k.startswith("layers.")}
        num_layers = len(layer_indices) if layer_indices else 2
        decoder = VQADecoder(
            vocab_size=len(vocab), visual_dim=steervit.visual_dim,
            d_model=512, nhead=8, num_layers=num_layers, max_len=8,
        )
        if dec_sd:
            decoder.load_state_dict(dec_sd, strict=False)
        model = DecoderModel(steervit, decoder, vocab, use_steering=True)
        model.load_state_dict(ckpt.get("model_state_dict", {}), strict=False)
    else:
        cfg = OmegaConf.create(ckpt["config"])
        steervit = CrossAttnViT.from_config(
            cfg.model.backbone_name, device=device,
            cross_attn_layers=list(cfg.model.cross_attn_layers),
            resolution=cfg.model.resolution,
            pretrained=cfg.model.get("pretrained", True),
            use_gate=cfg.model.get("use_gate", True),
        )
        dec_cfg = cfg.task.get("decoder", {})
        decoder = VQADecoder(
            vocab_size=len(vocab), visual_dim=steervit.visual_dim,
            d_model=dec_cfg.get("d_model", 512), nhead=dec_cfg.get("nhead", 8),
            num_layers=dec_cfg.get("num_layers", 2), max_len=dec_cfg.get("max_len", 8),
        )
        model = DecoderModel(steervit, decoder, vocab,
                             use_steering=cfg.model.get("use_steering", True))
        model.load_state_dict(ckpt["model_state_dict"], strict=False)

    model = model.to(device)
    model.eval()
    decoder._head_type = "decoder"
    decoder._feature_pool = "cls"
    decoder._vocab_offset = 3
    transform = steervit.get_transforms()
    print(f"Loaded: {Path(ckpt_path).parent.name}")
    return steervit, decoder, vocab, transform


# ── Scene graph utilities ────────────────────────────────────────

def parse_binding_target(question):
    """Extract described attributes from question text.

    E.g. "What is the material of the red cube?" → {"color": "red", "shape": "cube"}
    """
    # Normalize
    q = question.lower().strip().rstrip("?")

    # Common CLEVR attribute words
    colors = {"gray", "red", "blue", "green", "brown", "purple", "cyan", "yellow"}
    shapes = {"cube", "sphere", "cylinder", "block", "ball"}
    materials = {"metal", "metallic", "rubber", "matte", "shiny"}
    sizes = {"large", "big", "small", "tiny"}

    material_map = {"metallic": "metal", "shiny": "metal", "matte": "rubber"}
    shape_map = {"block": "cube", "ball": "sphere"}
    size_map = {"big": "large", "tiny": "small"}

    attrs = {}
    for word in q.split():
        if word in colors:
            attrs["color"] = word
        elif word in shapes:
            attrs["shape"] = shape_map.get(word, word)
        elif word in materials:
            attrs["material"] = material_map.get(word, word)
        elif word in sizes:
            attrs["size"] = size_map.get(word, word)
    return attrs


def scene_has_object(scene, attrs):
    """Check if scene contains an object matching ALL given attrs."""
    for obj in scene["objects"]:
        if all(obj.get(k) == v for k, v in attrs.items()):
            return True
    return False


def detect_queried_attr(question):
    """Detect what attribute is being asked about.

    "What is the material of ..." → "material"
    "What color is ..." → "color"
    """
    q = question.lower()
    for attr in ATTR_KEYS:
        if attr in q:
            return attr
    if "how big" in q:
        return "size"
    return None


# ── Activation extraction ────────────────────────────────────────

@torch.no_grad()
def extract_layer_features(steervit, images, questions, layer, device):
    """Extract mean-pooled patch features at a specific layer (raw, no LayerNorm).

    Returns: (B, D) tensor on CPU.
    """
    captured = {}

    def make_hook(li):
        def fn(module, inp, output):
            out = output[0] if isinstance(output, tuple) else output
            captured[li] = out.detach().float()
        return fn

    blk = steervit.vision_model.trunk.blocks[layer]
    hook = blk.register_forward_hook(make_hook(layer))

    with autocast(device_type="cuda", dtype=torch.bfloat16):
        steervit.forward(images.to(device), questions)

    hook.remove()

    prefix = steervit.vision_model.trunk.num_prefix_tokens
    patches = captured[layer][:, prefix:, :]  # raw, no norm
    return patches.mean(dim=1).cpu()  # (B, D)


@torch.no_grad()
def extract_all_layers(steervit, images, questions, device):
    """Extract mean-pooled patch features at ALL layers.

    Returns: {layer: (B, D)} on CPU.
    """
    layer_out = {}
    hooks = []
    blocks = steervit.vision_model.trunk.blocks

    for idx, blk in enumerate(blocks):
        def make_hook(li):
            def fn(module, inp, output):
                out = output[0] if isinstance(output, tuple) else output
                layer_out[li] = out.detach()
            return fn
        hooks.append(blk.register_forward_hook(make_hook(idx)))

    with autocast(device_type="cuda", dtype=torch.bfloat16):
        steervit.forward(images.to(device), questions)

    for h in hooks:
        h.remove()

    prefix = steervit.vision_model.trunk.num_prefix_tokens
    feats = {}
    for l in range(len(blocks)):
        patches = layer_out[l].float()[:, prefix:, :]  # raw, no norm
        feats[l] = patches.mean(dim=1).cpu()

    return feats


# ── Core: binding direction & intervention ───────────────────────

def compute_binding_direction(steervit, dataset, pos_indices, neg_indices,
                              question, layer, device, batch_size=32):
    """Compute binding direction via contrastive mean difference.

    Returns: (D,) unit vector on CPU.
    """
    def extract_batch(indices):
        all_feats = []
        for start in range(0, len(indices), batch_size):
            end = min(start + batch_size, len(indices))
            imgs = torch.stack([dataset[i]["image"] for i in indices[start:end]])
            qs = [question] * (end - start)
            feats = extract_layer_features(steervit, imgs, qs, layer, device)
            all_feats.append(feats)
        return torch.cat(all_feats, dim=0)

    pos_feats = extract_batch(pos_indices)  # (N_pos, D)
    neg_feats = extract_batch(neg_indices)  # (N_neg, D)

    d = pos_feats.mean(0) - neg_feats.mean(0)
    d = d / d.norm()
    return d


def get_model_answer(steervit, decoder, vocab, image, question, device):
    """Run model and return predicted answer string."""
    prefix = steervit.vision_model.trunk.num_prefix_tokens
    img = image.unsqueeze(0).to(device)
    with torch.no_grad():
        with autocast(device_type="cuda", dtype=torch.bfloat16):
            feats = steervit.forward(img, [question])
        patches = feats[:, prefix:, :].float()
        bos_id = vocab["<bos>"]
        eos_id = vocab["<eos>"]
        token_ids = decoder.generate(patches, bos_id=bos_id, eos_id=eos_id,
                                     max_len=4)
    inv_vocab = {v: k for k, v in vocab.items()}
    words = []
    for t in token_ids[0]:
        w = inv_vocab.get(t.item(), "")
        if w == "<eos>":
            break
        if w not in ("<bos>", "<pad>"):
            words.append(w)
    return " ".join(words)


def run_interchange_single(steervit, decoder, vocab, image_s, question_s,
                           image_t, question_t, d_bind, layer, device):
    """Run subspace interchange on a single source/target pair.

    Replaces source's projection onto d_bind with target's projection.

    Returns:
        original_answer: str
        intervened_answer: str
    """
    prefix = steervit.vision_model.trunk.num_prefix_tokens
    d = d_bind.to(device)  # (D,)

    # 1. Get target activation at layer
    target_feat = extract_layer_features(
        steervit, image_t.unsqueeze(0), [question_t], layer, device)  # (1, D) on CPU
    target_proj = (target_feat[0] @ d_bind).item()  # both on CPU

    # 2. Original answer (no intervention)
    original_answer = get_model_answer(steervit, decoder, vocab,
                                       image_s, question_s, device)

    # 3. Intervened forward: hook at target layer
    def interchange_hook(module, inp, output):
        x = output[0] if isinstance(output, tuple) else output
        x = x.clone().float()
        # Mean-pool raw patches to get scalar projection
        patches = x[:, prefix:, :]
        source_pooled = patches.mean(dim=1)  # (B, D)
        source_proj = (source_pooled[0] @ d).item()
        proj_diff = target_proj - source_proj
        # Add the difference along d_bind to all patch tokens
        x[:, prefix:, :] = x[:, prefix:, :] + proj_diff * d.unsqueeze(0).unsqueeze(0)
        if isinstance(output, tuple):
            return (x,) + output[1:]
        return x

    blk = steervit.vision_model.trunk.blocks[layer]
    hook = blk.register_forward_hook(interchange_hook)

    intervened_answer = get_model_answer(steervit, decoder, vocab,
                                         image_s, question_s, device)
    hook.remove()

    return original_answer, intervened_answer


# ── Sampling ─────────────────────────────────────────────────────

def collect_binding_pairs(dataset, scenes, question, n_pairs, rng):
    """Collect binding-positive and binding-negative scene indices.

    Returns:
        pos_indices: list of dataset indices where scene has binding target
        neg_indices: list of dataset indices where scene does NOT have binding target
    """
    attrs = parse_binding_target(question)
    if not attrs:
        raise ValueError(f"Cannot parse binding target from: {question}")

    seen_images = set()
    pos, neg = [], []

    indices = list(range(len(dataset)))
    rng.shuffle(indices)

    for idx in indices:
        q_data = dataset.questions[idx]
        fname = q_data["image_filename"]
        if fname in seen_images:
            continue
        scene = scenes.get(fname)
        if scene is None:
            continue
        n_obj = len(scene["objects"])
        if n_obj < 3 or n_obj > 6:
            continue
        seen_images.add(fname)

        if scene_has_object(scene, attrs):
            if len(pos) < n_pairs:
                pos.append(idx)
        else:
            if len(neg) < n_pairs:
                neg.append(idx)

        if len(pos) >= n_pairs and len(neg) >= n_pairs:
            break

    return pos, neg


def find_alternative_question(dataset, scenes, source_question, source_attrs,
                              rng, n_needed=100):
    """Find a question with a different binding target for C1.

    E.g. source asks about "red cube" → find question about "blue cube".
    Returns (question_str, target_attrs, matching_indices).
    """
    queried = detect_queried_attr(source_question)
    # Change the described attributes but keep queried attribute the same
    # Look through dataset for questions with different binding targets
    candidates = {}
    for idx in range(len(dataset)):
        q_data = dataset.questions[idx]
        q_text = q_data["question"]
        q_attrs = parse_binding_target(q_text)
        q_queried = detect_queried_attr(q_text)

        # Same queried attribute, different binding target
        if q_queried != queried or not q_attrs:
            continue
        if q_attrs == source_attrs:
            continue

        key = tuple(sorted(q_attrs.items()))
        if key not in candidates:
            candidates[key] = {"question": q_text, "attrs": q_attrs, "indices": []}
        candidates[key]["indices"].append(idx)

    # Pick candidate with enough matching scenes
    best = None
    for key, cand in candidates.items():
        # Check how many scenes actually have this binding target
        count = 0
        for idx in cand["indices"][:500]:
            fname = dataset.questions[idx]["image_filename"]
            scene = scenes.get(fname)
            if scene and scene_has_object(scene, cand["attrs"]):
                count += 1
        if count >= n_needed and (best is None or count > best[1]):
            best = (cand, count)

    if best is None:
        return None, None, None
    cand = best[0]
    return cand["question"], cand["attrs"], cand["indices"]


# ── Conditions ───────────────────────────────────────────────────

def run_condition_c1(steervit, decoder, vocab, dataset, scenes,
                     source_q, target_q, source_attrs, target_attrs,
                     d_bind, layer, n_pairs, rng, device):
    """C1: Described-attr interchange. Answer should flip."""
    source_pos, _ = collect_binding_pairs(dataset, scenes, source_q, n_pairs, rng)
    target_pos, _ = collect_binding_pairs(dataset, scenes, target_q, n_pairs, rng)

    n = min(len(source_pos), len(target_pos), n_pairs)
    flips, total = 0, 0

    for i in range(n):
        s_img = dataset[source_pos[i]]["image"]
        t_img = dataset[target_pos[i]]["image"]
        target_answer = get_model_answer(steervit, decoder, vocab,
                                         t_img, target_q, device)

        orig, interv = run_interchange_single(
            steervit, decoder, vocab, s_img, source_q,
            t_img, target_q, d_bind, layer, device)

        total += 1
        if interv == target_answer and interv != orig:
            flips += 1

        if (i + 1) % 20 == 0:
            print(f"  C1: {i+1}/{n}, flips={flips}/{total} "
                  f"({flips/total*100:.1f}%)", flush=True)

    return {"flip_rate": flips / max(total, 1), "n": total, "flips": flips}


def run_condition_c2(steervit, decoder, vocab, dataset, scenes,
                     source_q, alt_q, source_attrs,
                     d_bind, layer, n_pairs, rng, device):
    """C2: Same binding target, different queried attr. Answer should NOT flip."""
    source_pos, _ = collect_binding_pairs(dataset, scenes, source_q, n_pairs, rng)

    n = min(len(source_pos), n_pairs)
    flips, total = 0, 0

    for i in range(n):
        s_img = dataset[source_pos[i]]["image"]
        orig, interv = run_interchange_single(
            steervit, decoder, vocab, s_img, source_q,
            s_img, alt_q, d_bind, layer, device)

        total += 1
        if interv != orig:
            flips += 1

        if (i + 1) % 20 == 0:
            print(f"  C2: {i+1}/{n}, flips={flips}/{total} "
                  f"({flips/total*100:.1f}%)", flush=True)

    return {"flip_rate": flips / max(total, 1), "n": total, "flips": flips}


def run_condition_c3(steervit, decoder, vocab, dataset, scenes,
                     source_q, target_q, d_bind,
                     layer, n_pairs, rng, device):
    """C3: Random direction control. Answer should NOT flip."""
    # Random unit direction with same dim as d_bind
    d_random = torch.randn_like(d_bind)
    d_random = d_random / d_random.norm()

    source_pos, _ = collect_binding_pairs(dataset, scenes, source_q, n_pairs, rng)
    target_pos, _ = collect_binding_pairs(dataset, scenes, target_q, n_pairs, rng)

    n = min(len(source_pos), len(target_pos), n_pairs)
    flips, total = 0, 0

    for i in range(n):
        s_img = dataset[source_pos[i]]["image"]
        t_img = dataset[target_pos[i]]["image"]

        orig, interv = run_interchange_single(
            steervit, decoder, vocab, s_img, source_q,
            t_img, target_q, d_random, layer, device)

        total += 1
        if interv != orig:
            flips += 1

        if (i + 1) % 20 == 0:
            print(f"  C3: {i+1}/{n}, flips={flips}/{total} "
                  f"({flips/total*100:.1f}%)", flush=True)

    return {"flip_rate": flips / max(total, 1), "n": total, "flips": flips}


def run_layer_sweep(steervit, decoder, vocab, dataset, scenes,
                    source_q, target_q, source_attrs, target_attrs,
                    n_pairs, n_dir_samples, rng, device, batch_size):
    """Run C1 across all layers, computing d_bind per layer."""
    num_layers = len(steervit.vision_model.trunk.blocks)
    results = {}

    for layer in range(num_layers):
        print(f"\n  Layer {layer}:", flush=True)

        # Compute d_bind at this layer
        source_pos, source_neg = collect_binding_pairs(
            dataset, scenes, source_q, n_dir_samples, rng)
        d_bind = compute_binding_direction(
            steervit, dataset, source_pos[:n_dir_samples],
            source_neg[:n_dir_samples],
            source_q, layer, device, batch_size)

        # Run C1 at this layer
        res = run_condition_c1(
            steervit, decoder, vocab, dataset, scenes,
            source_q, target_q, source_attrs, target_attrs,
            d_bind, layer, n_pairs, rng, device)
        results[layer] = res
        print(f"  L{layer}: flip_rate={res['flip_rate']:.3f}", flush=True)

    return results


# ── Plotting ─────────────────────────────────────────────────────

def plot_layer_profile(layer_results, gca_layers, output_path, title):
    fig, ax = plt.subplots(figsize=(8, 4))
    layers = sorted(layer_results.keys())
    rates = [layer_results[l]["flip_rate"] for l in layers]

    ax.plot(layers, rates, "o-", color="#d62728", linewidth=2, markersize=6)

    for gl in gca_layers:
        ax.axvline(gl, color="gray", linestyle="--", alpha=0.3, linewidth=0.8)

    ax.set_xlabel("Layer", fontsize=13)
    ax.set_ylabel("Answer Flip Rate", fontsize=13)
    ax.set_title(title, fontsize=14)
    ax.set_xticks(layers)
    ax.set_xticklabels([f"L{l}" for l in layers], fontsize=9)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(axis="y", alpha=0.2)

    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


# ── Main ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-root", type=str,
                        default="/home/jungchun/data/clevr/CLEVR_v1.0")
    parser.add_argument("--target-layer", type=int, default=7)
    parser.add_argument("--num-pairs", type=int, default=100)
    parser.add_argument("--num-direction-samples", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--skip-layer-sweep", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    steervit, decoder, vocab, transform = load_model(args.checkpoint, device)
    dataset = CLEVRVQADataset(args.data_root, "val", transform)

    scenes_path = Path(args.data_root) / "scenes" / "CLEVR_val_scenes.json"
    with open(scenes_path) as f:
        scene_list = json.load(f)["scenes"]
    scenes = {s["image_filename"]: s for s in scene_list}

    ckpt_dir = Path(args.checkpoint).parent
    model_name = ckpt_dir.name.replace("_s42", "")
    output_dir = Path(args.output_dir) if args.output_dir else \
        Path("outputs/analysis/binding_interchange") / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    gca_layers = [i for i, blk in enumerate(steervit.vision_model.trunk.blocks)
                  if getattr(blk, "gated_cross_attn", None) is not None]

    # ── Setup: pick source/target questions ──
    source_q = "What is the material of the red cube?"
    source_attrs = parse_binding_target(source_q)
    target_q = "What is the material of the blue sphere?"
    target_attrs = parse_binding_target(target_q)
    print(f"Source question: {source_q}")
    print(f"Source binding target: {source_attrs}")
    print(f"Target question: {target_q}")
    print(f"Target binding target: {target_attrs}")

    # C2 alt question: same binding, different queried attr
    queried = detect_queried_attr(source_q)
    alt_queried = "color" if queried != "color" else "shape"
    alt_q = source_q.replace(queried, alt_queried)
    # Simple heuristic — may need adjustment for complex questions
    if alt_queried not in alt_q.lower():
        alt_q = f"What {alt_queried} is the {' '.join(source_attrs.values())}?"
    print(f"C2 alt question: {alt_q}")

    # ── Compute binding direction at target layer ──
    print(f"\nComputing d_bind at L{args.target_layer}...")
    source_pos, source_neg = collect_binding_pairs(
        dataset, scenes, source_q, args.num_direction_samples, rng)
    print(f"  Binding-positive: {len(source_pos)}, negative: {len(source_neg)}")

    d_bind = compute_binding_direction(
        steervit, dataset,
        source_pos[:args.num_direction_samples],
        source_neg[:args.num_direction_samples],
        source_q, args.target_layer, device, args.batch_size)
    print(f"  d_bind norm: {d_bind.norm():.4f}, dim: {d_bind.shape}")

    results = {"model": model_name, "source_q": source_q, "target_q": target_q,
               "source_attrs": source_attrs, "target_attrs": target_attrs,
               "target_layer": args.target_layer, "conditions": {}}

    # ── C1: Described-attr interchange ──
    print(f"\n{'='*60}")
    print("C1: Described-attr interchange (should flip)")
    print(f"{'='*60}")
    c1 = run_condition_c1(steervit, decoder, vocab, dataset, scenes,
                          source_q, target_q, source_attrs, target_attrs,
                          d_bind, args.target_layer, args.num_pairs, rng, device)
    results["conditions"]["C1"] = c1
    print(f"C1 flip rate: {c1['flip_rate']:.3f} ({c1['flips']}/{c1['n']})")

    # ── C2: Same-binding query change ──
    print(f"\n{'='*60}")
    print("C2: Same-binding query change (should NOT flip)")
    print(f"{'='*60}")
    c2 = run_condition_c2(steervit, decoder, vocab, dataset, scenes,
                          source_q, alt_q, source_attrs,
                          d_bind, args.target_layer, args.num_pairs, rng, device)
    results["conditions"]["C2"] = c2
    print(f"C2 flip rate: {c2['flip_rate']:.3f} ({c2['flips']}/{c2['n']})")

    # ── C3: Random direction ──
    print(f"\n{'='*60}")
    print("C3: Random direction control (should NOT flip)")
    print(f"{'='*60}")
    c3 = run_condition_c3(steervit, decoder, vocab, dataset, scenes,
                          source_q, target_q, d_bind,
                          args.target_layer, args.num_pairs, rng, device)
    results["conditions"]["C3"] = c3
    print(f"C3 flip rate: {c3['flip_rate']:.3f} ({c3['flips']}/{c3['n']})")

    # ── Layer sweep ──
    if not args.skip_layer_sweep:
        print(f"\n{'='*60}")
        print("Layer sweep: C1 across all layers")
        print(f"{'='*60}")
        layer_results = run_layer_sweep(
            steervit, decoder, vocab, dataset, scenes,
            source_q, target_q, source_attrs, target_attrs,
            min(args.num_pairs, 50), args.num_direction_samples,
            rng, device, args.batch_size)
        results["layer_sweep"] = {str(k): v for k, v in layer_results.items()}

        plot_layer_profile(
            layer_results, gca_layers,
            output_dir / "layer_profile.png",
            f"Binding Interchange — {model_name}")

    # ── Cross-query generalization ──
    print(f"\n{'='*60}")
    print("Cross-query: d_bind from source_q, test on alt_q")
    print(f"{'='*60}")
    # Use same d_bind (computed from source_q) but test with alt_q
    cross_pos, _ = collect_binding_pairs(dataset, scenes, source_q,
                                         args.num_pairs, rng)
    target_pos2, _ = collect_binding_pairs(dataset, scenes, target_q,
                                            args.num_pairs, rng)
    n = min(len(cross_pos), len(target_pos2), args.num_pairs)
    cross_flips, cross_total = 0, 0
    for i in range(n):
        s_img = dataset[cross_pos[i]]["image"]
        t_img = dataset[target_pos2[i]]["image"]
        orig, interv = run_interchange_single(
            steervit, decoder, vocab, s_img, alt_q,
            t_img, alt_q, d_bind, args.target_layer, device)
        cross_total += 1
        t_ans = get_model_answer(steervit, decoder, vocab, t_img, alt_q, device)
        if interv == t_ans and interv != orig:
            cross_flips += 1
        if (i + 1) % 20 == 0:
            print(f"  Cross: {i+1}/{n}, flips={cross_flips}/{cross_total}", flush=True)

    results["conditions"]["cross_query"] = {
        "flip_rate": cross_flips / max(cross_total, 1),
        "n": cross_total, "flips": cross_flips,
    }
    print(f"Cross-query flip rate: {cross_flips/max(cross_total,1):.3f}")

    # ── Summary ──
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for cond, res in results["conditions"].items():
        print(f"  {cond}: flip_rate={res['flip_rate']:.3f} ({res['flips']}/{res['n']})")

    with open(output_dir / "binding_interchange_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {output_dir / 'binding_interchange_results.json'}")


if __name__ == "__main__":
    main()
