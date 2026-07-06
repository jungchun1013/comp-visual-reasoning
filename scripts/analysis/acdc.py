"""ACDC (Automatic Circuit DisCovery) for SteerViT.

Reference: Conmy et al., NeurIPS 2023
"Towards Automated Circuit Discovery for Mechanistic Interpretability"

Iterates through edges in reverse topological order, greedily removes
edges whose removal causes KL divergence increase below threshold τ.

Usage:
    PYTHONPATH=src python scripts/analysis/acdc.py \
        --checkpoint outputs/model/clevr_dinov2_decoder1l_scratch_s42/best.pt \
        --category color --tau 0.05
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from model import CrossAttnViT
from data.clevr import CLEVRVQADataset
from tasks.decoder import build_clevr_decoder_vocab, VQADecoder, DecoderModel
from analysis.patching_sampling import (
    build_corruption_index, collect_corruption_samples,
    collect_visual_corruption_samples,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── Model loading (from path_patching.py) ────────────────────────

def load_model(ckpt_path, device):
    from omegaconf import OmegaConf
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    vocab = build_clevr_decoder_vocab()
    if "config" not in ckpt:
        steervit = CrossAttnViT.from_config(
            "vit_base_patch14_dinov2.lvd142m", device=device,
            cross_attn_layers=[1, 3, 5, 7, 9, 11], resolution=336)
        if "steervit_trainable_state" in ckpt:
            steervit.load_state_dict(ckpt["steervit_trainable_state"], strict=False)
        dec_sd = ckpt.get("decoder_state_dict", {})
        layer_indices = {int(k.split(".")[1]) for k in dec_sd if k.startswith("layers.")}
        decoder = VQADecoder(
            vocab_size=len(vocab), visual_dim=steervit.visual_dim,
            d_model=512, nhead=8, num_layers=len(layer_indices) or 2, max_len=8)
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
            use_gate=cfg.model.get("use_gate", True))
        dec_cfg = cfg.task.get("decoder", {})
        decoder = VQADecoder(
            vocab_size=len(vocab), visual_dim=steervit.visual_dim,
            d_model=dec_cfg.get("d_model", 512), nhead=dec_cfg.get("nhead", 8),
            num_layers=dec_cfg.get("num_layers", 2), max_len=dec_cfg.get("max_len", 8))
        model = DecoderModel(steervit, decoder, vocab,
                             use_steering=cfg.model.get("use_steering", True))
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model = model.to(device).eval()
    transform = steervit.get_transforms()
    print(f"Loaded: {Path(ckpt_path).parent.name}")
    return steervit, decoder, vocab, transform


# ── ACDC Discoverer ──────────────────────────────────────────────

class ACDCDiscoverer:
    """Automatic Circuit Discovery for SteerViT.

    Computational graph nodes:
      - ('sa', layer, head)   — SA attention head output
      - ('gca', layer, head)  — GCA cross-attention head output
      - ('mlp', layer)        — MLP output
      - ('resid_post', layer) — residual stream after block
      - ('logit',)            — decoder output

    Edges connect each node to all downstream nodes through the
    residual stream. We operate at node level (not edge level) for
    tractability: for each receiver node, test removing each sender.
    """

    def __init__(self, steervit, decoder, vocab):
        self.steervit = steervit
        self.decoder = decoder
        self.vocab = vocab
        self.blocks = steervit.vision_model.trunk.blocks
        self.num_layers = len(self.blocks)
        self.num_prefix = steervit.vision_model.trunk.num_prefix_tokens
        self.bos_id = vocab["<bos>"]

        # SA info
        self.sa_num_heads = self.blocks[0].attn.num_heads
        self.sa_head_dim = self.blocks[0].attn.head_dim

        # GCA info
        self.gca_layers = []
        for idx, blk in enumerate(self.blocks):
            if getattr(blk, "gated_cross_attn", None) is not None:
                self.gca_layers.append(idx)
        self.gca_set = set(self.gca_layers)
        if self.gca_layers:
            gca0 = self.blocks[self.gca_layers[0]].gated_cross_attn
            self.gca_num_heads = gca0.cross_attn.num_heads
            self.gca_head_dim = gca0.cross_attn.head_dim

    def _get_logits(self, features):
        """Get full logit vector from features."""
        patches = features[:, self.num_prefix:, :]
        bos = torch.full((1, 1), self.bos_id, dtype=torch.long,
                         device=features.device)
        logits = self.decoder(bos, patches)
        return logits[0, 0, :]  # (vocab_size,)

    def _record_all(self, images, questions):
        """Record all component outputs in one forward pass.

        Returns dict with:
            sa_pre[layer]: input to attn.proj (B, N, D)
            gca_pre[layer]: input to cross_attn.to_out (B, N, D)
            sa_out[layer]: output of attn (B, N, D)
            gca_contrib[layer]: output - input of GCA (B, N, D)
            features: final features
        """
        sa_pre, gca_pre, sa_out, gca_contrib = {}, {}, {}, {}
        hooks = []

        for idx, blk in enumerate(self.blocks):
            def make_sa_pre(li):
                def fn(m, inp, out):
                    sa_pre[li] = inp[0].detach().clone()
                return fn
            hooks.append(blk.attn.proj.register_forward_hook(make_sa_pre(idx)))

            def make_sa_out(li):
                def fn(m, inp, out):
                    sa_out[li] = out.detach().clone()
                return fn
            hooks.append(blk.attn.register_forward_hook(make_sa_out(idx)))

            if idx in self.gca_set:
                gca = blk.gated_cross_attn

                def make_gca_pre(li):
                    def fn(m, inp, out):
                        gca_pre[li] = inp[0].detach().clone()
                    return fn
                hooks.append(gca.cross_attn.to_out.register_forward_hook(make_gca_pre(idx)))

                def make_gca_contrib(li):
                    def fn(m, inp, out):
                        gca_contrib[li] = (out - inp[0]).detach().clone()
                    return fn
                hooks.append(gca.register_forward_hook(make_gca_contrib(idx)))

        with torch.no_grad(), autocast(device_type="cuda", dtype=torch.bfloat16):
            features = self.steervit.forward(images, questions)

        for h in hooks:
            h.remove()

        return {
            "sa_pre": sa_pre, "gca_pre": gca_pre,
            "sa_out": sa_out, "gca_contrib": gca_contrib,
            "features": features,
        }

    def _patched_forward(self, images, questions,
                         removed_edges, clean_cache, corrupt_cache):
        """Forward pass with specific edges removed.

        For each removed edge (sender_type, sender_layer, sender_head,
        receiver_type, receiver_layer):
            At the receiver, replace the sender's contribution with
            its corrupt-run value.

        Simplified approach (node-level, not edge-level):
            For removed node: replace its output with corrupt version.
            This is equivalent to removing ALL outgoing edges from that node.
        """
        hooks = []

        # Collect which SA heads and GCA heads should be corrupted
        corrupt_sa_heads = set()   # (layer, head)
        corrupt_gca_heads = set()  # (layer, head)

        for edge in removed_edges:
            stype, slayer, shead = edge
            if stype == "sa":
                corrupt_sa_heads.add((slayer, shead))
            elif stype == "gca":
                corrupt_gca_heads.add((slayer, shead))

        # Hook: replace corrupt SA heads' pre-proj dims with corrupt values
        for (layer, head) in corrupt_sa_heads:
            hd = self.sa_head_dim
            hs, he = head * hd, (head + 1) * hd
            corrupt_vals = corrupt_cache["sa_pre"][layer]

            def make_hook(h_s, h_e, c_vals):
                def fn(module, inp):
                    x = inp[0].clone()
                    x[:, :, h_s:h_e] = c_vals[:, :, h_s:h_e]
                    return (x,)
                return fn
            hooks.append(
                self.blocks[layer].attn.proj.register_forward_pre_hook(
                    make_hook(hs, he, corrupt_vals)))

        # Hook: replace corrupt GCA heads' pre-proj dims with corrupt values
        for (layer, head) in corrupt_gca_heads:
            hd = self.gca_head_dim
            hs, he = head * hd, (head + 1) * hd
            corrupt_vals = corrupt_cache["gca_pre"][layer]
            gca = self.blocks[layer].gated_cross_attn

            def make_hook(h_s, h_e, c_vals):
                def fn(module, inp):
                    x = inp[0].clone()
                    x[:, :, h_s:h_e] = c_vals[:, :, h_s:h_e]
                    return (x,)
                return fn
            hooks.append(
                gca.cross_attn.to_out.register_forward_pre_hook(
                    make_hook(hs, he, corrupt_vals)))

        with torch.no_grad(), autocast(device_type="cuda", dtype=torch.bfloat16):
            features = self.steervit.forward(images, questions)

        for h in hooks:
            h.remove()

        return features

    def _kl_divergence(self, logits_p, logits_q):
        """KL(p || q) where p = full model, q = subgraph."""
        p = F.softmax(logits_p, dim=-1)
        log_p = F.log_softmax(logits_p, dim=-1)
        log_q = F.log_softmax(logits_q, dim=-1)
        return (p * (log_p - log_q)).sum().item()

    def run_acdc(self, samples, tau, device, visual_mode=False):
        """Run ACDC algorithm.

        Args:
            samples: list of (image, question, corrupt_q, answer_idx) or
                     (clean_img, corrupt_img, question, answer_idx)
            tau: threshold for edge removal
            device: torch device
            visual_mode: if True, samples are visual corruption format

        Returns:
            removed_nodes: set of (type, layer, head) that were removed
            log: list of (node, delta_kl, removed_bool) per step
        """
        # Build list of all nodes in reverse topological order
        # Order: SA_L11 heads → GCA_L11 heads → SA_L10 → ... → SA_L0 → GCA_L1
        all_nodes = []
        for l in range(self.num_layers - 1, -1, -1):
            # SA heads at this layer
            for h in range(self.sa_num_heads):
                all_nodes.append(("sa", l, h))
            # GCA heads at this layer (if exists)
            if l in self.gca_set:
                for h in range(self.gca_num_heads):
                    all_nodes.append(("gca", l, h))

        print(f"Total nodes: {len(all_nodes)}")
        print(f"Threshold τ: {tau}")
        print(f"Samples: {len(samples)}")

        removed = set()  # nodes removed from circuit
        log = []

        for ni, node in enumerate(all_nodes):
            ntype, nlayer, nhead = node
            node_name = f"{ntype.upper()}_L{nlayer}_H{nhead}"

            # Try removing this node
            candidate = removed | {node}

            # Compute KL divergence with and without this node
            kl_with = 0.0
            kl_without = 0.0

            for si, sample in enumerate(samples):
                if visual_mode:
                    clean_img, corrupt_img, question, answer_idx = sample
                    clean_batch = clean_img.unsqueeze(0).to(device)
                    corrupt_batch = corrupt_img.unsqueeze(0).to(device)
                    clean_qs, corrupt_qs = [question], [question]
                else:
                    image, question, corrupt_q, answer_idx = sample
                    img = image.unsqueeze(0).to(device)
                    clean_batch, corrupt_batch = img, img
                    clean_qs, corrupt_qs = [question], [corrupt_q]

                # Record clean and corrupt activations
                clean_cache = self._record_all(clean_batch, clean_qs)
                corrupt_cache = self._record_all(corrupt_batch, corrupt_qs)

                # Full model logits (reference)
                ref_logits = self._get_logits(clean_cache["features"])

                # Current subgraph (with current removed set)
                feats_current = self._patched_forward(
                    clean_batch, clean_qs, removed, clean_cache, corrupt_cache)
                logits_current = self._get_logits(feats_current)
                kl_with += self._kl_divergence(ref_logits, logits_current)

                # Candidate subgraph (with node also removed)
                feats_candidate = self._patched_forward(
                    clean_batch, clean_qs, candidate, clean_cache, corrupt_cache)
                logits_candidate = self._get_logits(feats_candidate)
                kl_without += self._kl_divergence(ref_logits, logits_candidate)

            # Average KL
            n = len(samples)
            kl_with /= n
            kl_without /= n
            delta_kl = kl_without - kl_with

            if delta_kl < tau:
                removed.add(node)
                log.append((node_name, delta_kl, True))
                status = "REMOVED"
            else:
                log.append((node_name, delta_kl, False))
                status = "KEPT"

            if (ni + 1) % 10 == 0 or status == "KEPT":
                print(f"  [{ni+1}/{len(all_nodes)}] {node_name}: "
                      f"ΔKL={delta_kl:+.4f} → {status}  "
                      f"(circuit: {len(all_nodes) - len(removed)} nodes)")

        circuit_nodes = [n for n in all_nodes if n not in removed]
        print(f"\nDone. Circuit: {len(circuit_nodes)}/{len(all_nodes)} nodes "
              f"({len(removed)} removed)")

        return removed, circuit_nodes, log


# ── Main ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-root", type=str,
                        default="/home/jungchun/data/clevr/CLEVR_v1.0")
    parser.add_argument("--category", type=str, default="color")
    parser.add_argument("--tau", type=float, default=0.05)
    parser.add_argument("--n-samples", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--visual-pairs", type=str, default=None)
    parser.add_argument("--category-all", action="store_true",
                        help="Aggregate samples from all 4 fine attribute categories")
    parser.add_argument("--visual-pairs-dir", type=str,
                        default="outputs/analysis/visual_corruptions",
                        help="Directory containing per-type visual corruption pairs")
    args = parser.parse_args()

    import random
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    steervit, decoder, vocab, transform = load_model(args.checkpoint, device)
    dataset = CLEVRVQADataset(args.data_root, "val", transform)

    output_dir = Path(args.output_dir) if args.output_dir else \
        Path("outputs/analysis/acdc")
    output_dir.mkdir(parents=True, exist_ok=True)

    discoverer = ACDCDiscoverer(steervit, decoder, vocab)
    print(f"SA: {discoverer.sa_num_heads} heads, "
          f"GCA: {discoverer.gca_num_heads} heads")
    print(f"GCA layers: {discoverer.gca_layers}")

    # Load samples
    FINE_ATTRS = ["color", "material", "size", "shape"]
    FINE_QUERIES = ["what_color", "what_material", "what_size", "what_shape"]

    visual_mode = False
    if args.visual_pairs:
        # Single visual pairs file
        samples = collect_visual_corruption_samples(
            dataset, args.visual_pairs, args.n_samples, transform)
        visual_mode = True
        with open(args.visual_pairs) as f:
            visual_ctype = json.load(f)[0].get("corruption_type", "unknown")
        category_name = f"visual_{visual_ctype}"

    elif args.category_all:
        # Aggregate from all 4 fine categories
        corruption_index = build_corruption_index(dataset)
        per_cat = max(1, args.n_samples // 4)

        if args.category.startswith("visual"):
            # Visual all: mix from 4 visual pairs
            visual_mode = True
            samples = []
            for attr in FINE_ATTRS:
                pairs_path = Path(args.visual_pairs_dir) / attr / "pairs.json"
                if pairs_path.exists():
                    s = collect_visual_corruption_samples(
                        dataset, str(pairs_path), per_cat, transform)
                    samples.extend(s)
                    print(f"  {attr}: {len(s)} visual samples")
            category_name = "visual_all"

        elif args.category.startswith("what_"):
            # Queried all: mix from 4 query types
            samples = []
            for qcat in FINE_QUERIES:
                s = collect_corruption_samples(
                    dataset, corruption_index, qcat, per_cat)
                samples.extend(s)
                print(f"  {qcat}: {len(s)} samples")
            category_name = "queried_all"

        else:
            # Described all: mix from 4 attribute types
            samples = []
            for attr in FINE_ATTRS:
                s = collect_corruption_samples(
                    dataset, corruption_index, attr, per_cat)
                samples.extend(s)
                print(f"  {attr}: {len(s)} samples")
            category_name = "described_all"

        import random as _rnd
        _rnd.shuffle(samples)
        samples = samples[:args.n_samples]

    else:
        # Single category
        corruption_index = build_corruption_index(dataset)
        samples = collect_corruption_samples(
            dataset, corruption_index, args.category, args.n_samples)
        category_name = args.category

    print(f"Category: {category_name}, Samples: {len(samples)}")

    t0 = time.time()
    removed, circuit, log = discoverer.run_acdc(
        samples, args.tau, device, visual_mode)
    elapsed = time.time() - t0

    # Save results
    result = {
        "category": category_name,
        "tau": args.tau,
        "n_samples": len(samples),
        "elapsed_sec": round(elapsed, 1),
        "total_nodes": len(circuit) + len(removed),
        "circuit_nodes": len(circuit),
        "removed_nodes": len(removed),
        "circuit": [(t, l, h) for t, l, h in circuit],
        "removed": [(t, l, h) for t, l, h in removed],
        "log": [(name, round(dkl, 6), kept) for name, dkl, kept in log],
    }

    out_path = output_dir / f"acdc_{category_name}_tau{args.tau}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved: {out_path}")

    # Print circuit summary
    print(f"\n{'='*60}")
    print(f"CIRCUIT ({category_name}, τ={args.tau})")
    print(f"{'='*60}")
    print(f"Nodes in circuit ({len(circuit)}):")
    for ntype, nlayer, nhead in sorted(circuit, key=lambda x: (x[1], x[0], x[2])):
        name = f"{ntype.upper()}_L{nlayer}_H{nhead}"
        # Find its delta_kl from log
        dkl = next((d for n, d, k in log if n == name), 0)
        print(f"  {name}: ΔKL={dkl:+.4f}")

    print(f"\nElapsed: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
