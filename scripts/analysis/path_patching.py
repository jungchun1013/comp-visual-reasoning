"""Path patching for circuit discovery in SteerViT.

Reference: "Interpretability in the Wild" (IOI paper, Wang et al. 2022)

Path patching isolates the DIRECT causal effect of a head h on a downstream
component R. Unlike activation patching (which measures total effect), path
patching only allows corruption to flow through residual connections and MLPs,
NOT through other attention heads.

Algorithm (path_patch h → R):
  1. Run clean forward, cache all component outputs
  2. Run corrupt forward, cache h's corrupt activation
  3. Run clean forward again, but:
     - Replace h's output with its corrupt version
     - Freeze all attention heads between h and R at clean values
     - MLPs recompute naturally (part of the "direct path")
  4. Measure effect on output

Phase 1: path_patch(each head → Logits)
  → Finds heads that DIRECTLY affect the output (cf. IOI Name Mover Heads)

Phase 2: path_patch(each head → Phase1 top heads' Q/K/V)
  → Finds heads that DIRECTLY influence Phase1 heads (cf. IOI S-Inhibition Heads)

Usage:
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python scripts/analysis/path_patching.py \
        --checkpoint outputs/model/clevr_dinov2_decoder1l_scratch_s42/best.pt \
        --phase 1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.amp import autocast

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from model import CrossAttnViT
from data.clevr import CLEVRVQADataset
from tasks.decoder import build_clevr_decoder_vocab, VQADecoder, DecoderModel
from analysis.patching_sampling import build_corruption_index, collect_corruption_samples

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


# ── Model loading ─────────────────────────────────────────────────

def load_model(ckpt_path, device):
    from omegaconf import OmegaConf
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    vocab = build_clevr_decoder_vocab()

    if "config" not in ckpt:
        cross_attn_layers = [1, 3, 5, 7, 9, 11]
        steervit = CrossAttnViT.from_config(
            "vit_base_patch14_dinov2.lvd142m", device=device,
            cross_attn_layers=cross_attn_layers, resolution=336)
        if "steervit_trainable_state" in ckpt:
            steervit.load_state_dict(ckpt["steervit_trainable_state"], strict=False)
        dec_sd = ckpt.get("decoder_state_dict", {})
        layer_indices = {int(k.split(".")[1]) for k in dec_sd if k.startswith("layers.")}
        num_layers = len(layer_indices) if layer_indices else 2
        decoder = VQADecoder(
            vocab_size=len(vocab), visual_dim=steervit.visual_dim,
            d_model=512, nhead=8, num_layers=num_layers, max_len=8)
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

    model = model.to(device)
    model.eval()
    decoder._head_type = "decoder"
    decoder._feature_pool = "cls"
    decoder._vocab_offset = 3
    transform = steervit.get_transforms()
    print(f"Loaded: {Path(ckpt_path).parent.name}")
    return steervit, decoder, vocab, transform


# ── PathPatcher ───────────────────────────────────────────────────

class PathPatcher:
    """Path patching following IOI paper methodology.

    Block order in SteerViT:
        GCA(x, text) → x + SA(LN1(x)) → x + MLP(LN2(x))

    Path patching corrupt sender h, freeze all downstream attention heads
    at clean values, let MLPs recompute. Corruption only flows through
    the residual stream + MLPs (direct path).
    """

    def __init__(self, steervit, decoder, vocab):
        self.steervit = steervit
        self.decoder = decoder
        self.vocab = vocab
        self.blocks = steervit.vision_model.trunk.blocks
        self.num_layers = len(self.blocks)
        self.num_prefix = steervit.vision_model.trunk.num_prefix_tokens
        self.bos_id = vocab["<bos>"]

        self.sa_num_heads = self.blocks[0].attn.num_heads
        self.sa_head_dim = self.blocks[0].attn.head_dim

        self.gca_layers = []
        for idx, blk in enumerate(self.blocks):
            if getattr(blk, "gated_cross_attn", None) is not None:
                self.gca_layers.append(idx)
        if self.gca_layers:
            gca0 = self.blocks[self.gca_layers[0]].gated_cross_attn
            self.gca_num_heads = gca0.cross_attn.num_heads
            self.gca_head_dim = gca0.cross_attn.head_dim
        else:
            self.gca_num_heads = self.gca_head_dim = 0

    def to_token_id(self, answer_idx):
        return answer_idx + 3

    def _get_logit(self, features, token_id):
        patches = features[:, self.num_prefix:, :]
        bos = torch.full((1, 1), self.bos_id, dtype=torch.long,
                         device=features.device)
        logits = self.decoder(bos, patches)
        return logits[0, 0, token_id].item()

    def _record_components(self, images, questions):
        """Record all component outputs in one forward pass.

        Returns:
            sa_outputs:    {layer: tensor} — output of blk.attn (added to residual)
            gca_contribs:  {layer: tensor} — GCA contribution (output - input)
            sa_pre_proj:   {layer: tensor} — input to attn.proj (for per-head patching)
            gca_pre_proj:  {layer: tensor} — input to cross_attn.to_out
            sa_qkv:        {layer: tensor} — output of attn.qkv (for Q/K/V freezing)
            features:      final visual features
        """
        sa_outputs, gca_contribs = {}, {}
        sa_pre_proj, gca_pre_proj = {}, {}
        sa_qkv = {}
        hooks = []

        for idx, blk in enumerate(self.blocks):
            # SA full output (what gets added to residual)
            def make_sa_out_hook(li):
                def fn(module, inp, output):
                    sa_outputs[li] = output.detach().clone()
                return fn
            hooks.append(blk.attn.register_forward_hook(make_sa_out_hook(idx)))

            # SA pre-projection (input to proj, for per-head patching)
            def make_sa_pre_hook(li):
                def fn(module, inp, output):
                    sa_pre_proj[li] = inp[0].detach().clone()
                return fn
            hooks.append(blk.attn.proj.register_forward_hook(make_sa_pre_hook(idx)))

            # SA QKV output (for Q/K/V freezing in Phase 2)
            def make_qkv_hook(li):
                def fn(module, inp, output):
                    sa_qkv[li] = output.detach().clone()
                return fn
            hooks.append(blk.attn.qkv.register_forward_hook(make_qkv_hook(idx)))

            # GCA contribution + pre-proj
            if idx in self.gca_layers:
                gca = blk.gated_cross_attn

                def make_gca_contrib_hook(li):
                    def fn(module, inp, output):
                        gca_contribs[li] = (output - inp[0]).detach().clone()
                    return fn
                hooks.append(gca.register_forward_hook(make_gca_contrib_hook(idx)))

                def make_gca_pre_hook(li):
                    def fn(module, inp, output):
                        gca_pre_proj[li] = inp[0].detach().clone()
                    return fn
                hooks.append(
                    gca.cross_attn.to_out.register_forward_hook(make_gca_pre_hook(idx)))

        with torch.no_grad():
            with autocast(device_type="cuda", dtype=torch.bfloat16):
                features = self.steervit.forward(images, questions)

        for h in hooks:
            h.remove()

        return sa_outputs, gca_contribs, sa_pre_proj, gca_pre_proj, sa_qkv, features

    def _path_patch_forward(self, images, questions,
                            target_type, target_layer, target_head,
                            restore_sa_pre, restore_gca_pre,
                            freeze_sa_out, freeze_gca_contrib,
                            freeze_up_to=None):
        """Denoising path patching: corrupt run with one head restored to clean.

        Corrupt run → restore target head to clean → freeze all downstream
        attention heads at CORRUPT values → MLPs recompute.

        Only the direct path from the restored head propagates through
        residual + MLPs. Other attention heads stay corrupt.

        Args:
            images, questions: corrupt input
            target_type/layer/head: the head to restore (clean activation)
            restore_sa_pre/gca_pre: CLEAN pre-proj values (for restoring target)
            freeze_sa_out: CORRUPT SA outputs (for freezing downstream)
            freeze_gca_contrib: CORRUPT GCA contributions (for freezing downstream)
            freeze_up_to: if None, freeze all layers after target (Phase 1).
                          if int, freeze only between target and freeze_up_to.
        """
        hooks = []

        # 1. Restore the target head to clean (replace per-head dims)
        if target_type == "sa":
            hd = self.sa_head_dim
            hs, he = target_head * hd, (target_head + 1) * hd

            def restore_hook(module, inp):
                x = inp[0].clone()
                x[:, :, hs:he] = restore_sa_pre[target_layer][:, :, hs:he]
                return (x,)
            hooks.append(
                self.blocks[target_layer].attn.proj.register_forward_pre_hook(restore_hook))
        else:  # gca
            hd = self.gca_head_dim
            hs, he = target_head * hd, (target_head + 1) * hd
            gca = self.blocks[target_layer].gated_cross_attn

            def restore_hook(module, inp):
                x = inp[0].clone()
                x[:, :, hs:he] = restore_gca_pre[target_layer][:, :, hs:he]
                return (x,)
            hooks.append(
                gca.cross_attn.to_out.register_forward_pre_hook(restore_hook))

        # 2. Freeze downstream attention heads at CORRUPT values
        # Block order: GCA → SA → MLP
        max_freeze = freeze_up_to if freeze_up_to is not None else self.num_layers

        for l in range(self.num_layers):
            should_freeze_sa = False
            should_freeze_gca = False

            if target_type == "sa":
                if target_layer < l < max_freeze:
                    should_freeze_sa = True
                    should_freeze_gca = True
            else:  # target is GCA
                if l == target_layer and l < max_freeze:
                    should_freeze_sa = True  # SA at same layer comes after GCA
                elif target_layer < l < max_freeze:
                    should_freeze_sa = True
                    should_freeze_gca = True

            # Freeze SA at corrupt output
            if should_freeze_sa and l in freeze_sa_out:
                cached = freeze_sa_out[l]

                def make_freeze_sa(c):
                    def fn(module, inp, output):
                        return c
                    return fn
                hooks.append(
                    self.blocks[l].attn.register_forward_hook(make_freeze_sa(cached)))

            # Freeze GCA at corrupt contribution
            if should_freeze_gca and l in self.gca_layers and l in freeze_gca_contrib:
                cached_c = freeze_gca_contrib[l]

                def make_freeze_gca(cc):
                    def fn(module, inp, output):
                        return inp[0] + cc
                    return fn
                hooks.append(
                    self.blocks[l].gated_cross_attn.register_forward_hook(
                        make_freeze_gca(cached_c)))

        # 3. Run modified forward (on corrupt input)
        with torch.no_grad():
            with autocast(device_type="cuda", dtype=torch.bfloat16):
                features = self.steervit.forward(images, questions)

        for h in hooks:
            h.remove()
        return features

    # ── Phase 1: head → Logits ────────────────────────────────────

    def run_phase1_sample(self, clean_imgs, clean_qs, corrupt_imgs, corrupt_qs,
                          token_id, direction="denoising"):
        """Phase 1: path patch each head → Logits.

        denoising: corrupt run → restore one clean head → freeze downstream
                   at corrupt. Positive = head directly helps.
        noising:   clean run → corrupt one head → freeze downstream
                   at clean. Negative = head directly helps. (IOI paper style)

        Returns: sa_heatmap (n_layers, sa_heads), gca_heatmap (n_gca, gca_heads)
        """
        # Cache both clean and corrupt components
        clean_sa_out, clean_gca_contrib, clean_sa_pre, clean_gca_pre, _, clean_feats = \
            self._record_components(clean_imgs, clean_qs)
        corrupt_sa_out, corrupt_gca_contrib, corrupt_sa_pre, corrupt_gca_pre, _, corrupt_feats = \
            self._record_components(corrupt_imgs, corrupt_qs)

        if direction == "denoising":
            # Run on corrupt, restore target to clean, freeze downstream at corrupt
            run_imgs, run_qs = corrupt_imgs, corrupt_qs
            restore_sa_pre, restore_gca_pre = clean_sa_pre, clean_gca_pre
            freeze_sa_out, freeze_gca_contrib = corrupt_sa_out, corrupt_gca_contrib
            baseline_logit = self._get_logit(corrupt_feats, token_id)
        else:  # noising
            # Run on clean, corrupt target, freeze downstream at clean
            run_imgs, run_qs = clean_imgs, clean_qs
            restore_sa_pre, restore_gca_pre = corrupt_sa_pre, corrupt_gca_pre
            freeze_sa_out, freeze_gca_contrib = clean_sa_out, clean_gca_contrib
            baseline_logit = self._get_logit(clean_feats, token_id)

        # Path patch each SA head → Logits
        sa_heatmap = np.zeros((self.num_layers, self.sa_num_heads))
        for layer in range(self.num_layers):
            for head in range(self.sa_num_heads):
                feats = self._path_patch_forward(
                    run_imgs, run_qs, "sa", layer, head,
                    restore_sa_pre, restore_gca_pre,
                    freeze_sa_out, freeze_gca_contrib)
                sa_heatmap[layer, head] = self._get_logit(feats, token_id) - baseline_logit

        # Path patch each GCA head → Logits
        gca_heatmap = np.zeros((len(self.gca_layers), self.gca_num_heads))
        for gi, layer in enumerate(self.gca_layers):
            for head in range(self.gca_num_heads):
                feats = self._path_patch_forward(
                    run_imgs, run_qs, "gca", layer, head,
                    restore_sa_pre, restore_gca_pre,
                    freeze_sa_out, freeze_gca_contrib)
                gca_heatmap[gi, head] = self._get_logit(feats, token_id) - baseline_logit

        return sa_heatmap, gca_heatmap, baseline_logit

    # ── Phase 2: head → receiver's Q/K/V ──────────────────────────

    def _path_patch_forward_qkv(self, images, questions,
                                target_type, target_layer, target_head,
                                restore_sa_pre, restore_gca_pre,
                                freeze_sa_out, freeze_gca_contrib,
                                receiver_layer, receiver_head, qkv_component,
                                baseline_qkv):
        """Phase 2: path patch with per-head Q/K/V freezing at receiver.

        At the receiver SA layer: freeze ALL heads' Q/K/V at baseline,
        EXCEPT the specified receiver head's specified component —
        that one gets the (potentially corrupted) recomputed value.

        QKV layout: (B, N, 3 * num_heads * head_dim)
          Q_all = [:, :, 0*dim : 1*dim]     where dim = num_heads * head_dim
          K_all = [:, :, 1*dim : 2*dim]
          V_all = [:, :, 2*dim : 3*dim]
        Per-head slice within Q_all: head h occupies [h*hd : (h+1)*hd]
        """
        hooks = []
        hd = self.sa_head_dim
        n_heads = self.sa_num_heads
        dim = n_heads * hd

        # 1. Restore the target (sender) head
        if target_type == "sa":
            hs, he = target_head * self.sa_head_dim, (target_head + 1) * self.sa_head_dim
            def restore_hook(module, inp):
                x = inp[0].clone()
                x[:, :, hs:he] = restore_sa_pre[target_layer][:, :, hs:he]
                return (x,)
            hooks.append(
                self.blocks[target_layer].attn.proj.register_forward_pre_hook(restore_hook))
        else:
            hs, he = target_head * self.gca_head_dim, (target_head + 1) * self.gca_head_dim
            gca = self.blocks[target_layer].gated_cross_attn
            def restore_hook(module, inp):
                x = inp[0].clone()
                x[:, :, hs:he] = restore_gca_pre[target_layer][:, :, hs:he]
                return (x,)
            hooks.append(
                gca.cross_attn.to_out.register_forward_pre_hook(restore_hook))

        # 2. Freeze attention heads between target and receiver
        for l in range(self.num_layers):
            should_freeze_sa = False
            should_freeze_gca = False

            if target_type == "sa":
                if target_layer < l < receiver_layer:
                    should_freeze_sa = True
                    should_freeze_gca = True
            else:
                if l == target_layer and l < receiver_layer:
                    should_freeze_sa = True
                elif target_layer < l < receiver_layer:
                    should_freeze_sa = True
                    should_freeze_gca = True

            if should_freeze_sa and l in freeze_sa_out:
                cached = freeze_sa_out[l]
                def make_freeze_sa(c):
                    def fn(module, inp, output):
                        return c
                    return fn
                hooks.append(
                    self.blocks[l].attn.register_forward_hook(make_freeze_sa(cached)))

            if should_freeze_gca and l in self.gca_layers and l in freeze_gca_contrib:
                cached_c = freeze_gca_contrib[l]
                def make_freeze_gca(cc):
                    def fn(module, inp, output):
                        return inp[0] + cc
                    return fn
                hooks.append(
                    self.blocks[l].gated_cross_attn.register_forward_hook(
                        make_freeze_gca(cached_c)))

        # 3. At receiver SA layer: per-head Q/K/V freezing
        # Freeze entire QKV at baseline, then UN-freeze only the
        # receiver head's specified component (let corruption through).
        qkv_baseline = baseline_qkv[receiver_layer]
        comp_idx = {"q": 0, "k": 1, "v": 2}[qkv_component]
        rh_start = receiver_head * hd
        rh_end = (receiver_head + 1) * hd

        def make_qkv_hook(baseline, comp_i, d, rhs, rhe):
            def fn(module, inp, output):
                # Start from baseline (all frozen)
                out = baseline.clone()
                # Un-freeze: let the receiver head's target component
                # use the recomputed (corrupted) value
                offset = comp_i * d
                out[:, :, offset + rhs:offset + rhe] = \
                    output[:, :, offset + rhs:offset + rhe]
                return out
            return fn
        hooks.append(
            self.blocks[receiver_layer].attn.qkv.register_forward_hook(
                make_qkv_hook(qkv_baseline, comp_idx, dim, rh_start, rh_end)))

        # 4. Run modified forward
        with torch.no_grad():
            with autocast(device_type="cuda", dtype=torch.bfloat16):
                features = self.steervit.forward(images, questions)

        for h in hooks:
            h.remove()
        return features

    def run_phase2_sample(self, clean_imgs, clean_qs, corrupt_imgs, corrupt_qs,
                          token_id, receiver_layer, receiver_head, qkv_component,
                          direction="denoising"):
        """Phase 2 with per-head Q/K/V separation.

        For each upstream head h, path patch h → receiver_head's Q (or K or V).
        Freeze between h and receiver. At receiver, freeze all heads' QKV at
        baseline except receiver_head's specified component.

        Args:
            receiver_layer: SA layer index
            receiver_head: specific head index within that layer
            qkv_component: "q", "k", or "v"
        """
        clean_sa_out, clean_gca_contrib, clean_sa_pre, clean_gca_pre, clean_qkv, clean_feats = \
            self._record_components(clean_imgs, clean_qs)
        corrupt_sa_out, corrupt_gca_contrib, corrupt_sa_pre, corrupt_gca_pre, corrupt_qkv, corrupt_feats = \
            self._record_components(corrupt_imgs, corrupt_qs)

        if direction == "denoising":
            run_imgs, run_qs = corrupt_imgs, corrupt_qs
            restore_sa_pre, restore_gca_pre = clean_sa_pre, clean_gca_pre
            freeze_sa_out, freeze_gca_contrib = corrupt_sa_out, corrupt_gca_contrib
            baseline_qkv = corrupt_qkv  # freeze Q/K/V at corrupt baseline
            baseline_logit = self._get_logit(corrupt_feats, token_id)
        else:
            run_imgs, run_qs = clean_imgs, clean_qs
            restore_sa_pre, restore_gca_pre = corrupt_sa_pre, corrupt_gca_pre
            freeze_sa_out, freeze_gca_contrib = clean_sa_out, clean_gca_contrib
            baseline_qkv = clean_qkv
            baseline_logit = self._get_logit(clean_feats, token_id)

        sa_heatmap = np.zeros((self.num_layers, self.sa_num_heads))
        for layer in range(self.num_layers):
            if layer >= receiver_layer:
                continue
            for head in range(self.sa_num_heads):
                feats = self._path_patch_forward_qkv(
                    run_imgs, run_qs, "sa", layer, head,
                    restore_sa_pre, restore_gca_pre,
                    freeze_sa_out, freeze_gca_contrib,
                    receiver_layer, receiver_head, qkv_component, baseline_qkv)
                sa_heatmap[layer, head] = self._get_logit(feats, token_id) - baseline_logit

        gca_heatmap = np.zeros((len(self.gca_layers), self.gca_num_heads))
        for gi, gca_layer in enumerate(self.gca_layers):
            if gca_layer >= receiver_layer:
                continue
            for head in range(self.gca_num_heads):
                feats = self._path_patch_forward_qkv(
                    run_imgs, run_qs, "gca", gca_layer, head,
                    restore_sa_pre, restore_gca_pre,
                    freeze_sa_out, freeze_gca_contrib,
                    receiver_layer, receiver_head, qkv_component, baseline_qkv)
                gca_heatmap[gi, head] = self._get_logit(feats, token_id) - baseline_logit

        return sa_heatmap, gca_heatmap, baseline_logit

    # ── Phase 2 layer-level: head → SA_layer's Q/K/V (all heads) ──

    def run_phase2_layer_sample(self, clean_imgs, clean_qs, corrupt_imgs, corrupt_qs,
                                token_id, receiver_layer, qkv_component,
                                direction="denoising"):
        """Phase 2 layer-level: path patch each head → entire SA layer's Q/K/V.

        Freeze 2 of 3 QKV components for the ENTIRE layer at baseline,
        let corruption enter through the specified component for ALL heads.
        Useful when the whole SA layer is important (e.g. SA_L11).
        """
        clean_sa_out, clean_gca_contrib, clean_sa_pre, clean_gca_pre, clean_qkv, clean_feats = \
            self._record_components(clean_imgs, clean_qs)
        corrupt_sa_out, corrupt_gca_contrib, corrupt_sa_pre, corrupt_gca_pre, corrupt_qkv, corrupt_feats = \
            self._record_components(corrupt_imgs, corrupt_qs)

        if direction == "denoising":
            run_imgs, run_qs = corrupt_imgs, corrupt_qs
            restore_sa_pre, restore_gca_pre = clean_sa_pre, clean_gca_pre
            freeze_sa_out, freeze_gca_contrib = corrupt_sa_out, corrupt_gca_contrib
            baseline_qkv = corrupt_qkv
            baseline_logit = self._get_logit(corrupt_feats, token_id)
        else:
            run_imgs, run_qs = clean_imgs, clean_qs
            restore_sa_pre, restore_gca_pre = corrupt_sa_pre, corrupt_gca_pre
            freeze_sa_out, freeze_gca_contrib = clean_sa_out, clean_gca_contrib
            baseline_qkv = clean_qkv
            baseline_logit = self._get_logit(clean_feats, token_id)

        hd = self.sa_head_dim
        n_heads = self.sa_num_heads
        dim = n_heads * hd
        comp_idx = {"q": 0, "k": 1, "v": 2}[qkv_component]

        def make_layer_qkv_hook(baseline, comp_i, d):
            def fn(module, inp, output):
                out = baseline.clone()
                # Un-freeze the entire target component (all heads)
                out[:, :, comp_i * d:(comp_i + 1) * d] = \
                    output[:, :, comp_i * d:(comp_i + 1) * d]
                return out
            return fn

        sa_heatmap = np.zeros((self.num_layers, self.sa_num_heads))
        for layer in range(self.num_layers):
            if layer >= receiver_layer:
                continue
            for head in range(self.sa_num_heads):
                hooks = []

                # 1. Restore target head
                if True:  # SA sender
                    hs, he = head * self.sa_head_dim, (head + 1) * self.sa_head_dim
                    def make_restore(rsp, tl, hs_, he_):
                        def fn(module, inp):
                            x = inp[0].clone()
                            x[:, :, hs_:he_] = rsp[tl][:, :, hs_:he_]
                            return (x,)
                        return fn
                    hooks.append(
                        self.blocks[layer].attn.proj.register_forward_pre_hook(
                            make_restore(restore_sa_pre, layer, hs, he)))

                # 2. Freeze between
                for l in range(self.num_layers):
                    if layer < l < receiver_layer:
                        if l in freeze_sa_out:
                            cached = freeze_sa_out[l]
                            def make_f(c):
                                def fn(m, i, o): return c
                                return fn
                            hooks.append(self.blocks[l].attn.register_forward_hook(make_f(cached)))
                        if l in self.gca_layers and l in freeze_gca_contrib:
                            cc = freeze_gca_contrib[l]
                            def make_fg(c):
                                def fn(m, i, o): return i[0] + c
                                return fn
                            hooks.append(self.blocks[l].gated_cross_attn.register_forward_hook(make_fg(cc)))

                # 3. Layer-level QKV freeze
                hooks.append(
                    self.blocks[receiver_layer].attn.qkv.register_forward_hook(
                        make_layer_qkv_hook(baseline_qkv[receiver_layer], comp_idx, dim)))

                with torch.no_grad():
                    with autocast(device_type="cuda", dtype=torch.bfloat16):
                        feats = self.steervit.forward(run_imgs, run_qs)
                for h in hooks:
                    h.remove()
                sa_heatmap[layer, head] = self._get_logit(feats, token_id) - baseline_logit

        gca_heatmap = np.zeros((len(self.gca_layers), self.gca_num_heads))
        for gi, gca_layer in enumerate(self.gca_layers):
            if gca_layer >= receiver_layer:
                continue
            for head in range(self.gca_num_heads):
                hooks = []

                # 1. Restore GCA target head
                hs, he = head * self.gca_head_dim, (head + 1) * self.gca_head_dim
                gca = self.blocks[gca_layer].gated_cross_attn
                def make_gca_restore(rgp, tl, hs_, he_):
                    def fn(module, inp):
                        x = inp[0].clone()
                        x[:, :, hs_:he_] = rgp[tl][:, :, hs_:he_]
                        return (x,)
                    return fn
                hooks.append(
                    gca.cross_attn.to_out.register_forward_pre_hook(
                        make_gca_restore(restore_gca_pre, gca_layer, hs, he)))

                # 2. Freeze between (SA at same layer + all between)
                for l in range(self.num_layers):
                    should_freeze_sa = (l == gca_layer and l < receiver_layer) or \
                                       (gca_layer < l < receiver_layer)
                    should_freeze_gca = gca_layer < l < receiver_layer

                    if should_freeze_sa and l in freeze_sa_out:
                        cached = freeze_sa_out[l]
                        def make_f(c):
                            def fn(m, i, o): return c
                            return fn
                        hooks.append(self.blocks[l].attn.register_forward_hook(make_f(cached)))
                    if should_freeze_gca and l in self.gca_layers and l in freeze_gca_contrib:
                        cc = freeze_gca_contrib[l]
                        def make_fg(c):
                            def fn(m, i, o): return i[0] + c
                            return fn
                        hooks.append(self.blocks[l].gated_cross_attn.register_forward_hook(make_fg(cc)))

                # 3. Layer-level QKV freeze
                hooks.append(
                    self.blocks[receiver_layer].attn.qkv.register_forward_hook(
                        make_layer_qkv_hook(baseline_qkv[receiver_layer], comp_idx, dim)))

                with torch.no_grad():
                    with autocast(device_type="cuda", dtype=torch.bfloat16):
                        feats = self.steervit.forward(run_imgs, run_qs)
                for h in hooks:
                    h.remove()
                gca_heatmap[gi, head] = self._get_logit(feats, token_id) - baseline_logit

        return sa_heatmap, gca_heatmap, baseline_logit

    # ── Phase 3: head → GCA receiver ─────────────────────────────

    def run_phase3_layer_sample(self, clean_imgs, clean_qs, corrupt_imgs, corrupt_qs,
                                token_id, receiver_layer, direction="denoising"):
        """Phase 3 layer-level: path patch each head → GCA receiver layer.

        Same as Phase 1 but freezes only between sender and receiver_layer.
        At receiver GCA and beyond: recompute normally.
        No Q/K/V separation — GCA's K/V come from text, only Q from visual residual.
        """
        clean_sa_out, clean_gca_contrib, clean_sa_pre, clean_gca_pre, _, clean_feats = \
            self._record_components(clean_imgs, clean_qs)
        corrupt_sa_out, corrupt_gca_contrib, corrupt_sa_pre, corrupt_gca_pre, _, corrupt_feats = \
            self._record_components(corrupt_imgs, corrupt_qs)

        if direction == "denoising":
            run_imgs, run_qs = corrupt_imgs, corrupt_qs
            restore_sa_pre, restore_gca_pre = clean_sa_pre, clean_gca_pre
            freeze_sa_out, freeze_gca_contrib = corrupt_sa_out, corrupt_gca_contrib
            baseline_logit = self._get_logit(corrupt_feats, token_id)
        else:
            run_imgs, run_qs = clean_imgs, clean_qs
            restore_sa_pre, restore_gca_pre = corrupt_sa_pre, corrupt_gca_pre
            freeze_sa_out, freeze_gca_contrib = clean_sa_out, clean_gca_contrib
            baseline_logit = self._get_logit(clean_feats, token_id)

        sa_heatmap = np.zeros((self.num_layers, self.sa_num_heads))
        for layer in range(self.num_layers):
            if layer >= receiver_layer:
                continue
            for head in range(self.sa_num_heads):
                feats = self._path_patch_forward(
                    run_imgs, run_qs, "sa", layer, head,
                    restore_sa_pre, restore_gca_pre,
                    freeze_sa_out, freeze_gca_contrib,
                    freeze_up_to=receiver_layer)
                sa_heatmap[layer, head] = self._get_logit(feats, token_id) - baseline_logit

        gca_heatmap = np.zeros((len(self.gca_layers), self.gca_num_heads))
        for gi, gca_layer in enumerate(self.gca_layers):
            if gca_layer >= receiver_layer:
                continue
            for head in range(self.gca_num_heads):
                feats = self._path_patch_forward(
                    run_imgs, run_qs, "gca", gca_layer, head,
                    restore_sa_pre, restore_gca_pre,
                    freeze_sa_out, freeze_gca_contrib,
                    freeze_up_to=receiver_layer)
                gca_heatmap[gi, head] = self._get_logit(feats, token_id) - baseline_logit

        return sa_heatmap, gca_heatmap, baseline_logit

    def run_phase3_perhead_sample(self, clean_imgs, clean_qs, corrupt_imgs, corrupt_qs,
                                  token_id, receiver_layer, receiver_head,
                                  direction="denoising"):
        """Phase 3 per-head: path patch each head → specific GCA receiver head.

        At GCA receiver: freeze all heads' pre-proj at baseline,
        un-freeze only receiver_head. Isolates the contribution of one GCA head.
        """
        clean_sa_out, clean_gca_contrib, clean_sa_pre, clean_gca_pre, _, clean_feats = \
            self._record_components(clean_imgs, clean_qs)
        corrupt_sa_out, corrupt_gca_contrib, corrupt_sa_pre, corrupt_gca_pre, _, corrupt_feats = \
            self._record_components(corrupt_imgs, corrupt_qs)

        if direction == "denoising":
            run_imgs, run_qs = corrupt_imgs, corrupt_qs
            restore_sa_pre, restore_gca_pre = clean_sa_pre, clean_gca_pre
            freeze_sa_out, freeze_gca_contrib = corrupt_sa_out, corrupt_gca_contrib
            baseline_gca_pre = corrupt_gca_pre
            baseline_logit = self._get_logit(corrupt_feats, token_id)
        else:
            run_imgs, run_qs = clean_imgs, clean_qs
            restore_sa_pre, restore_gca_pre = corrupt_sa_pre, corrupt_gca_pre
            freeze_sa_out, freeze_gca_contrib = clean_sa_out, clean_gca_contrib
            baseline_gca_pre = clean_gca_pre
            baseline_logit = self._get_logit(clean_feats, token_id)

        gca_hd = self.gca_head_dim
        rhs = receiver_head * gca_hd
        rhe = (receiver_head + 1) * gca_hd
        baseline_pre_proj = baseline_gca_pre[receiver_layer]
        gca_module = self.blocks[receiver_layer].gated_cross_attn

        def make_gca_perhead_hook(bl, rhs_, rhe_):
            def fn(module, inp):
                x = bl.clone()
                x[:, :, rhs_:rhe_] = inp[0][:, :, rhs_:rhe_]
                return (x,)
            return fn

        # Sweep SA senders
        sa_heatmap = np.zeros((self.num_layers, self.sa_num_heads))
        for layer in range(self.num_layers):
            if layer >= receiver_layer:
                continue
            for head in range(self.sa_num_heads):
                hooks = []

                # Restore SA sender
                hs, he = head * self.sa_head_dim, (head + 1) * self.sa_head_dim
                def make_restore(rsp, tl, hs_, he_):
                    def fn(module, inp):
                        x = inp[0].clone()
                        x[:, :, hs_:he_] = rsp[tl][:, :, hs_:he_]
                        return (x,)
                    return fn
                hooks.append(
                    self.blocks[layer].attn.proj.register_forward_pre_hook(
                        make_restore(restore_sa_pre, layer, hs, he)))

                # Freeze between sender and receiver
                for l in range(self.num_layers):
                    if layer < l < receiver_layer:
                        if l in freeze_sa_out:
                            cached = freeze_sa_out[l]
                            def make_f(c):
                                def fn(m, i, o): return c
                                return fn
                            hooks.append(self.blocks[l].attn.register_forward_hook(make_f(cached)))
                        if l in self.gca_layers and l in freeze_gca_contrib:
                            cc = freeze_gca_contrib[l]
                            def make_fg(c):
                                def fn(m, i, o): return i[0] + c
                                return fn
                            hooks.append(self.blocks[l].gated_cross_attn.register_forward_hook(make_fg(cc)))

                # GCA per-head hook at receiver
                hooks.append(
                    gca_module.cross_attn.to_out.register_forward_pre_hook(
                        make_gca_perhead_hook(baseline_pre_proj, rhs, rhe)))

                with torch.no_grad():
                    with autocast(device_type="cuda", dtype=torch.bfloat16):
                        feats = self.steervit.forward(run_imgs, run_qs)
                for h in hooks:
                    h.remove()
                sa_heatmap[layer, head] = self._get_logit(feats, token_id) - baseline_logit

        # Sweep GCA senders
        gca_heatmap = np.zeros((len(self.gca_layers), self.gca_num_heads))
        for gi, gca_layer in enumerate(self.gca_layers):
            if gca_layer >= receiver_layer:
                continue
            for head in range(self.gca_num_heads):
                hooks = []

                # Restore GCA sender
                hs, he = head * self.gca_head_dim, (head + 1) * self.gca_head_dim
                gca_src = self.blocks[gca_layer].gated_cross_attn
                def make_gca_restore(rgp, tl, hs_, he_):
                    def fn(module, inp):
                        x = inp[0].clone()
                        x[:, :, hs_:he_] = rgp[tl][:, :, hs_:he_]
                        return (x,)
                    return fn
                hooks.append(
                    gca_src.cross_attn.to_out.register_forward_pre_hook(
                        make_gca_restore(restore_gca_pre, gca_layer, hs, he)))

                # Freeze between (SA at same layer + all between)
                for l in range(self.num_layers):
                    should_freeze_sa = (l == gca_layer and l < receiver_layer) or \
                                       (gca_layer < l < receiver_layer)
                    should_freeze_gca = gca_layer < l < receiver_layer

                    if should_freeze_sa and l in freeze_sa_out:
                        cached = freeze_sa_out[l]
                        def make_f(c):
                            def fn(m, i, o): return c
                            return fn
                        hooks.append(self.blocks[l].attn.register_forward_hook(make_f(cached)))
                    if should_freeze_gca and l in self.gca_layers and l in freeze_gca_contrib:
                        cc = freeze_gca_contrib[l]
                        def make_fg(c):
                            def fn(m, i, o): return i[0] + c
                            return fn
                        hooks.append(self.blocks[l].gated_cross_attn.register_forward_hook(make_fg(cc)))

                # GCA per-head hook at receiver
                hooks.append(
                    gca_module.cross_attn.to_out.register_forward_pre_hook(
                        make_gca_perhead_hook(baseline_pre_proj, rhs, rhe)))

                with torch.no_grad():
                    with autocast(device_type="cuda", dtype=torch.bfloat16):
                        feats = self.steervit.forward(run_imgs, run_qs)
                for h in hooks:
                    h.remove()
                gca_heatmap[gi, head] = self._get_logit(feats, token_id) - baseline_logit

        return sa_heatmap, gca_heatmap, baseline_logit


# ── Runners ───────────────────────────────────────────────────────

def _unpack_sample(sample, device, visual_mode):
    """Unpack a sample into (clean_imgs, clean_qs, corrupt_imgs, corrupt_qs, answer_idx).

    Text mode:   (image, question, corrupt_question, answer_idx)
    Visual mode: (clean_image, corrupt_image, question, answer_idx)
    """
    if visual_mode:
        clean_img, corrupt_img, question, answer_idx = sample
        return (clean_img.unsqueeze(0).to(device), [question],
                corrupt_img.unsqueeze(0).to(device), [question], answer_idx)
    else:
        image, question, corrupt_q, answer_idx = sample
        img = image.unsqueeze(0).to(device)
        return (img, [question], img, [corrupt_q], answer_idx)


def run_phase1(patcher, samples, device, direction="denoising", label="",
               visual_mode=False):
    """Aggregate phase 1 across samples."""
    sa_accum, gca_accum = [], []
    for i, sample in enumerate(samples):
        clean_imgs, clean_qs, corrupt_imgs, corrupt_qs, answer_idx = \
            _unpack_sample(sample, device, visual_mode)
        token_id = patcher.to_token_id(answer_idx)
        sa_hm, gca_hm, _ = patcher.run_phase1_sample(
            clean_imgs, clean_qs, corrupt_imgs, corrupt_qs, token_id, direction)
        sa_accum.append(sa_hm)
        gca_accum.append(gca_hm)
        if (i + 1) % 10 == 0:
            print(f"    {label} {i+1}/{len(samples)}", flush=True)

    return {
        "sa_mean": np.mean(sa_accum, axis=0).tolist(),
        "gca_mean": np.mean(gca_accum, axis=0).tolist(),
        "sa_std": np.std(sa_accum, axis=0).tolist(),
        "gca_std": np.std(gca_accum, axis=0).tolist(),
        "n": len(samples),
    }


def run_phase2(patcher, samples, device, receiver_layer, receiver_head,
               qkv_component, direction="denoising", label="",
               visual_mode=False):
    """Aggregate phase 2 across samples for one receiver head + Q/K/V."""
    sa_accum, gca_accum = [], []
    for i, sample in enumerate(samples):
        clean_imgs, clean_qs, corrupt_imgs, corrupt_qs, answer_idx = \
            _unpack_sample(sample, device, visual_mode)
        token_id = patcher.to_token_id(answer_idx)
        sa_hm, gca_hm, _ = patcher.run_phase2_sample(
            clean_imgs, clean_qs, corrupt_imgs, corrupt_qs, token_id,
            receiver_layer, receiver_head, qkv_component, direction)
        sa_accum.append(sa_hm)
        gca_accum.append(gca_hm)
        if (i + 1) % 10 == 0:
            print(f"    {label} {i+1}/{len(samples)}", flush=True)

    return {
        "sa_mean": np.mean(sa_accum, axis=0).tolist(),
        "gca_mean": np.mean(gca_accum, axis=0).tolist(),
        "n": len(samples),
        "receiver_layer": receiver_layer,
        "receiver_head": receiver_head,
        "qkv_component": qkv_component,
    }


def run_phase2_layer(patcher, samples, device, receiver_layer, qkv_component,
                     direction="denoising", label="",
                     visual_mode=False):
    """Aggregate layer-level phase 2 across samples."""
    sa_accum, gca_accum = [], []
    for i, sample in enumerate(samples):
        clean_imgs, clean_qs, corrupt_imgs, corrupt_qs, answer_idx = \
            _unpack_sample(sample, device, visual_mode)
        token_id = patcher.to_token_id(answer_idx)
        sa_hm, gca_hm, _ = patcher.run_phase2_layer_sample(
            clean_imgs, clean_qs, corrupt_imgs, corrupt_qs, token_id,
            receiver_layer, qkv_component, direction)
        sa_accum.append(sa_hm)
        gca_accum.append(gca_hm)
        if (i + 1) % 10 == 0:
            print(f"    {label} {i+1}/{len(samples)}", flush=True)

    return {
        "sa_mean": np.mean(sa_accum, axis=0).tolist(),
        "gca_mean": np.mean(gca_accum, axis=0).tolist(),
        "n": len(samples),
        "receiver_layer": receiver_layer,
        "qkv_component": qkv_component,
    }


def run_phase3_layer(patcher, samples, device, receiver_layer,
                     direction="denoising", label="",
                     visual_mode=False):
    """Aggregate phase 3 layer-level across samples."""
    sa_accum, gca_accum = [], []
    for i, sample in enumerate(samples):
        clean_imgs, clean_qs, corrupt_imgs, corrupt_qs, answer_idx = \
            _unpack_sample(sample, device, visual_mode)
        token_id = patcher.to_token_id(answer_idx)
        sa_hm, gca_hm, _ = patcher.run_phase3_layer_sample(
            clean_imgs, clean_qs, corrupt_imgs, corrupt_qs, token_id,
            receiver_layer, direction)
        sa_accum.append(sa_hm)
        gca_accum.append(gca_hm)
        if (i + 1) % 10 == 0:
            print(f"    {label} {i+1}/{len(samples)}", flush=True)

    return {
        "sa_mean": np.mean(sa_accum, axis=0).tolist(),
        "gca_mean": np.mean(gca_accum, axis=0).tolist(),
        "n": len(samples),
        "receiver_layer": receiver_layer,
        "receiver_type": "gca",
    }


def run_phase3_perhead(patcher, samples, device, receiver_layer, receiver_head,
                       direction="denoising", label="",
                       visual_mode=False):
    """Aggregate phase 3 per-head across samples."""
    sa_accum, gca_accum = [], []
    for i, sample in enumerate(samples):
        clean_imgs, clean_qs, corrupt_imgs, corrupt_qs, answer_idx = \
            _unpack_sample(sample, device, visual_mode)
        token_id = patcher.to_token_id(answer_idx)
        sa_hm, gca_hm, _ = patcher.run_phase3_perhead_sample(
            clean_imgs, clean_qs, corrupt_imgs, corrupt_qs, token_id,
            receiver_layer, receiver_head, direction)
        sa_accum.append(sa_hm)
        gca_accum.append(gca_hm)
        if (i + 1) % 10 == 0:
            print(f"    {label} {i+1}/{len(samples)}", flush=True)

    return {
        "sa_mean": np.mean(sa_accum, axis=0).tolist(),
        "gca_mean": np.mean(gca_accum, axis=0).tolist(),
        "n": len(samples),
        "receiver_layer": receiver_layer,
        "receiver_head": receiver_head,
        "receiver_type": "gca",
    }


def get_top_heads(data, gca_layers, top_k=5):
    """Extract top-k heads by absolute effect."""
    sa = np.array(data["sa_mean"])
    gca = np.array(data["gca_mean"])
    heads = []
    for l in range(sa.shape[0]):
        for h in range(sa.shape[1]):
            heads.append(("sa", l, h, sa[l, h]))
    for gi, l in enumerate(gca_layers):
        for h in range(gca.shape[1]):
            heads.append(("gca", l, h, gca[gi, h]))
    heads.sort(key=lambda x: abs(x[3]), reverse=True)
    return heads[:top_k]


# ── Plotting ──────────────────────────────────────────────────────

def plot_heatmap(data, gca_layers, sa_heads, gca_heads, output_path, title):
    """Plot interleaved SA+GCA heatmap."""
    gca_set = set(gca_layers)
    max_heads = max(sa_heads, gca_heads)

    col_labels, col_is_gca = [], []
    for l in range(12):
        if l in gca_set:
            col_labels.append(f"GCA{l}")
            col_is_gca.append(True)
        col_labels.append(f"SA{l}")
        col_is_gca.append(False)
    n_cols = len(col_labels)

    combined = np.full((n_cols, max_heads), np.nan)
    sa = np.array(data["sa_mean"])
    gca = np.array(data["gca_mean"])
    gca_idx, ci = 0, 0
    for l in range(12):
        if l in gca_set:
            combined[ci, :gca_heads] = gca[gca_idx]
            gca_idx += 1
            ci += 1
        combined[ci, :sa_heads] = sa[l]
        ci += 1

    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    masked = np.ma.array(combined, mask=np.isnan(combined))
    vmax = np.nanmax(np.abs(combined)) or 1
    im = ax.imshow(masked.T, aspect="auto", cmap="RdBu_r",
                   vmin=-vmax, vmax=vmax, origin="lower")
    ax.set_xlabel("Module")
    ax.set_ylabel("Head Index")
    ax.set_title(title)
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(col_labels, rotation=60, ha="right", fontsize=7)
    ax.set_yticks(range(max_heads))
    for ci_idx in range(n_cols):
        if col_is_gca[ci_idx]:
            rect = Rectangle((ci_idx - 0.5, -0.5), 1, max_heads,
                              linewidth=1.5, edgecolor="red",
                              facecolor="none", linestyle="--")
            ax.add_patch(rect)
    plt.colorbar(im, ax=ax, label=r"$\Delta$ logit (positive = important)")
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-root", type=str,
                        default="/home/jungchun/data/clevr/CLEVR_v1.0")
    parser.add_argument("--phase", type=str, default="1",
                        choices=["1", "2", "3", "trace"],
                        help="1: head→Logits, 2: head→SA QKV, 3: head→GCA, "
                             "trace: auto-pick receivers from Phase 2")
    parser.add_argument("--n-samples", type=int, default=50)
    parser.add_argument("--category", type=str, default="color",
                        help="Corruption category: color, material, size, shape, etc.")
    parser.add_argument("--top-k", type=int, default=5,
                        help="Phase 2: trace top-k heads from Phase 1")
    parser.add_argument("--direction", type=str, default="denoising",
                        choices=["denoising", "noising"],
                        help="denoising: restore clean head in corrupt run. "
                             "noising: corrupt head in clean run (IOI paper style).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--phase1-json", type=str, default=None)
    parser.add_argument("--phase2-json", type=str, default=None,
                        help="Phase 2 JSON for auto-detecting receivers")
    parser.add_argument("--layer-level", action="store_true",
                        help="Phase 2/3: layer-level instead of per-head")
    parser.add_argument("--receiver-layer", type=int, default=None,
                        help="Phase 2/3: receiver layer (default: auto from prior phase)")
    parser.add_argument("--source-phase2", type=str, default="k",
                        choices=["k", "v", "q"],
                        help="trace mode: which Phase 2 channel to pick receivers from")
    parser.add_argument("--visual-pairs", type=str, default=None,
                        help="Path to visual corruption pairs.json. "
                             "Enables visual corruption mode (different images, same question).")
    args = parser.parse_args()

    import random
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    steervit, decoder, vocab, transform = load_model(args.checkpoint, device)
    dataset = CLEVRVQADataset(args.data_root, "val", transform)

    ckpt_dir = Path(args.checkpoint).parent
    model_name = ckpt_dir.name.replace("_s42", "")
    output_dir = Path(args.output_dir) if args.output_dir else \
        Path("outputs/analysis/path_patching") / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    patcher = PathPatcher(steervit, decoder, vocab)
    print(f"SA: {patcher.sa_num_heads} heads, GCA: {patcher.gca_num_heads} heads")
    print(f"GCA layers: {patcher.gca_layers}")

    # ── Sample loading ──
    visual_mode = False
    if args.visual_pairs:
        from analysis.patching_sampling import collect_visual_corruption_samples
        samples = collect_visual_corruption_samples(
            dataset, args.visual_pairs, args.n_samples, transform)
        visual_mode = True
        with open(args.visual_pairs) as _f:
            visual_ctype = json.load(_f)[0].get("corruption_type", "unknown")
        args.category = f"visual_{visual_ctype}"
        print(f"\nVisual corruption mode: {args.visual_pairs}")
        print(f"Samples ({args.category}): {len(samples)}")
    else:
        corruption_index = build_corruption_index(dataset)
        samples = collect_corruption_samples(
            dataset, corruption_index, args.category, args.n_samples)
        print(f"Samples ({args.category}): {len(samples)}")

    if args.phase == "1":
        print(f"\n{'='*60}")
        print(f"PHASE 1: Path Patch each head → Logits ({args.category})")
        print(f"{'='*60}")

        data = run_phase1(patcher, samples, device, args.direction, args.category,
                          visual_mode=visual_mode)
        data["gca_layers"] = patcher.gca_layers
        data["sa_num_heads"] = patcher.sa_num_heads
        data["gca_num_heads"] = patcher.gca_num_heads
        data["category"] = args.category

        # Top heads (most negative = most important for direct effect)
        top = get_top_heads(data, patcher.gca_layers, 10)
        print(f"\nTop-10 heads (direct effect on logits, {args.category}):")
        for ht, l, h, v in top:
            print(f"  {ht.upper()}_L{l}_H{h}: {v:+.4f}")

        out_path = output_dir / f"phase1_{args.category}.json"
        with open(out_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Saved: {out_path}")

        plot_heatmap(data, patcher.gca_layers, patcher.sa_num_heads,
                     patcher.gca_num_heads,
                     output_dir / f"phase1_{args.category}.png",
                     f"Path Patching Phase 1: head → Logits ({args.category})")

    elif args.phase == "2":
        phase1_path = Path(args.phase1_json) if args.phase1_json else \
            output_dir / f"phase1_{args.category}.json"
        with open(phase1_path) as f:
            phase1_data = json.load(f)

        top = get_top_heads(phase1_data, phase1_data["gca_layers"], args.top_k)

        if args.layer_level:
            # Layer-level: h → SA_receiver_layer's Q/K/V (all heads at once)
            recv_layer = args.receiver_layer if args.receiver_layer is not None else \
                top[0][1]

            print(f"\n{'='*60}")
            print(f"PHASE 2 (layer-level): head → SA_L{recv_layer} Q/K/V")
            print(f"{'='*60}")

            for qkv in ["q", "k", "v"]:
                print(f"\n  SA_L{recv_layer}.{qkv.upper()}:")

                data = run_phase2_layer(patcher, samples, device, recv_layer, qkv,
                                        args.direction, label=f"→SA_L{recv_layer}.{qkv.upper()}",
                                        visual_mode=visual_mode)
                data["gca_layers"] = patcher.gca_layers
                data["sa_num_heads"] = patcher.sa_num_heads
                data["gca_num_heads"] = patcher.gca_num_heads

                up_top = get_top_heads(data, patcher.gca_layers, 5)
                for ut, ul, uh, uv in up_top:
                    print(f"    {ut.upper()}_L{ul}_H{uh}: {uv:+.4f}")

                out_path = output_dir / f"phase2_{args.category}_SA_L{recv_layer}_layer_{qkv}.json"
                with open(out_path, "w") as f:
                    json.dump(data, f, indent=2)
                print(f"  Saved: {out_path}")

                plot_heatmap(data, patcher.gca_layers, patcher.sa_num_heads,
                             patcher.gca_num_heads,
                             output_dir / f"phase2_{args.category}_SA_L{recv_layer}_layer_{qkv}.png",
                             f"Phase 2: head → SA_L{recv_layer}.{qkv.upper()} layer-level ({args.category})")
        else:
            # Per-head: h → specific receiver head's Q/K/V
            print(f"\n{'='*60}")
            print(f"PHASE 2 (per-head): head → top receivers' Q/K/V")
            print(f"{'='*60}")

            for ht, rl, rh, importance in top:
                recv_name = f"{ht.upper()}_L{rl}_H{rh}"
                if ht != "sa":
                    print(f"\n  Skipping {recv_name} (GCA receiver, no Q/K/V separation)")
                    continue

                for qkv in ["q", "k", "v"]:
                    print(f"\n  Receiver: {recv_name}.{qkv.upper()} (Phase 1 effect={importance:+.4f})")

                    data = run_phase2(patcher, samples, device, rl, rh, qkv,
                                      args.direction, label=f"→{recv_name}.{qkv.upper()}",
                                      visual_mode=visual_mode)
                    data["gca_layers"] = patcher.gca_layers
                    data["receiver_name"] = f"{recv_name}.{qkv.upper()}"

                    up_top = get_top_heads(data, patcher.gca_layers, 5)
                    for ut, ul, uh, uv in up_top:
                        print(f"    {ut.upper()}_L{ul}_H{uh}: {uv:+.4f}")

                    out_path = output_dir / f"phase2_{args.category}_{recv_name}_{qkv}.json"
                    with open(out_path, "w") as f:
                        json.dump(data, f, indent=2)

                    plot_heatmap(data, patcher.gca_layers, patcher.sa_num_heads,
                                 patcher.gca_num_heads,
                                 output_dir / f"phase2_{args.category}_{recv_name}_{qkv}.png",
                                 f"Phase 2: head → {recv_name}.{qkv.upper()} ({args.category})")

    elif args.phase == "3":
        # Phase 3: path patch upstream heads → GCA receiver
        recv_layer = args.receiver_layer if args.receiver_layer is not None else 9
        assert recv_layer in patcher.gca_layers, \
            f"Receiver layer {recv_layer} is not a GCA layer. GCA layers: {patcher.gca_layers}"

        if args.layer_level:
            print(f"\n{'='*60}")
            print(f"PHASE 3 (layer-level): head → GCA_L{recv_layer}")
            print(f"{'='*60}")

            data = run_phase3_layer(patcher, samples, device, recv_layer,
                                    args.direction, label=f"→GCA_L{recv_layer}",
                                    visual_mode=visual_mode)
            data["gca_layers"] = patcher.gca_layers
            data["sa_num_heads"] = patcher.sa_num_heads
            data["gca_num_heads"] = patcher.gca_num_heads

            top = get_top_heads(data, patcher.gca_layers, 10)
            print(f"\nTop-10 heads (direct effect on GCA_L{recv_layer}):")
            for ht, l, h, v in top:
                print(f"  {ht.upper()}_L{l}_H{h}: {v:+.4f}")

            out_path = output_dir / f"phase3_{args.category}_GCA_L{recv_layer}_layer.json"
            with open(out_path, "w") as f:
                json.dump(data, f, indent=2)
            print(f"Saved: {out_path}")

            plot_heatmap(data, patcher.gca_layers, patcher.sa_num_heads,
                         patcher.gca_num_heads,
                         output_dir / f"phase3_{args.category}_GCA_L{recv_layer}_layer.png",
                         f"Phase 3: head → GCA_L{recv_layer} layer-level ({args.category})")
        else:
            # Per-head: find top GCA heads from Phase 2 K→SA_L11 results
            phase2_path = Path(args.phase2_json) if args.phase2_json else \
                output_dir / f"phase2_{args.category}_SA_L11_layer_k.json"

            if not phase2_path.exists():
                print(f"Phase 2 K results not found: {phase2_path}")
                print("Run Phase 2 layer-level first or specify --phase2-json")
                sys.exit(1)

            with open(phase2_path) as f:
                phase2_data = json.load(f)

            # Extract top GCA heads at receiver layer
            gca_mean = np.array(phase2_data["gca_mean"])
            p2_gca_layers = phase2_data["gca_layers"]
            gi = p2_gca_layers.index(recv_layer)
            gca_heads_ranked = [(h, float(gca_mean[gi, h]))
                                for h in range(gca_mean.shape[1])]
            gca_heads_ranked.sort(key=lambda x: abs(x[1]), reverse=True)
            top_gca = gca_heads_ranked[:args.top_k]

            print(f"\n{'='*60}")
            print(f"PHASE 3 (per-head): head → GCA_L{recv_layer} top heads")
            print(f"Top-{args.top_k} from Phase 2 K→SA_L11:")
            for gh, gv in top_gca:
                print(f"  GCA_L{recv_layer}_H{gh}: {gv:+.4f}")
            print(f"{'='*60}")

            for gca_head, importance in top_gca:
                recv_name = f"GCA_L{recv_layer}_H{gca_head}"
                print(f"\n  Receiver: {recv_name} (Phase 2 K effect={importance:+.4f})")

                data = run_phase3_perhead(
                    patcher, samples, device, recv_layer, gca_head,
                    args.direction, label=f"→{recv_name}",
                    visual_mode=visual_mode)
                data["gca_layers"] = patcher.gca_layers
                data["sa_num_heads"] = patcher.sa_num_heads
                data["gca_num_heads"] = patcher.gca_num_heads

                up_top = get_top_heads(data, patcher.gca_layers, 5)
                for ut, ul, uh, uv in up_top:
                    print(f"    {ut.upper()}_L{ul}_H{uh}: {uv:+.4f}")

                out_path = output_dir / f"phase3_{args.category}_{recv_name}.json"
                with open(out_path, "w") as f:
                    json.dump(data, f, indent=2)
                print(f"  Saved: {out_path}")

                plot_heatmap(data, patcher.gca_layers, patcher.sa_num_heads,
                             patcher.gca_num_heads,
                             output_dir / f"phase3_{args.category}_{recv_name}.png",
                             f"Phase 3: head → {recv_name} ({args.category})")

    elif args.phase == "trace":
        # Iterative trace: auto-pick top-k receivers from Phase 2 results
        qkv_src = args.source_phase2
        phase2_path = Path(args.phase2_json) if args.phase2_json else \
            output_dir / f"phase2_{args.category}_SA_L11_layer_{qkv_src}.json"

        if not phase2_path.exists():
            print(f"Phase 2 {qkv_src.upper()} results not found: {phase2_path}")
            print("Run Phase 2 layer-level first.")
            sys.exit(1)

        with open(phase2_path) as f:
            phase2_data = json.load(f)

        # Rank ALL heads (SA + GCA) by absolute effect
        receivers = get_top_heads(phase2_data, phase2_data["gca_layers"], args.top_k)

        print(f"\n{'='*60}")
        print(f"TRACE: upstream of Phase 2 {qkv_src.upper()}→SA_L11 top-{args.top_k}")
        print(f"Source: {phase2_path.name}")
        print(f"{'='*60}")
        for ht, rl, rh, imp in receivers:
            print(f"  {ht.upper()}_L{rl}_H{rh}: {imp:+.4f}")
        print()

        gca_set = set(patcher.gca_layers)

        for ht, rl, rh, importance in receivers:
            recv_name = f"{ht.upper()}_L{rl}_H{rh}"

            if ht == "sa":
                # SA receiver → Phase 2 per-head approach (Q/K/V separation)
                for qkv in ["q", "k", "v"]:
                    print(f"\n  {recv_name}.{qkv.upper()} (Ph2 {qkv_src.upper()} effect={importance:+.4f})")

                    data = run_phase2(patcher, samples, device, rl, rh, qkv,
                                      args.direction, label=f"→{recv_name}.{qkv.upper()}",
                                      visual_mode=visual_mode)
                    data["gca_layers"] = patcher.gca_layers
                    data["sa_num_heads"] = patcher.sa_num_heads
                    data["gca_num_heads"] = patcher.gca_num_heads
                    data["receiver_name"] = f"{recv_name}.{qkv.upper()}"

                    up_top = get_top_heads(data, patcher.gca_layers, 5)
                    for ut, ul, uh, uv in up_top:
                        print(f"    {ut.upper()}_L{ul}_H{uh}: {uv:+.4f}")

                    out_path = output_dir / f"trace_{args.category}_{recv_name}_{qkv}.json"
                    with open(out_path, "w") as f:
                        json.dump(data, f, indent=2)
                    print(f"  Saved: {out_path}")

                    plot_heatmap(data, patcher.gca_layers, patcher.sa_num_heads,
                                 patcher.gca_num_heads,
                                 output_dir / f"trace_{args.category}_{recv_name}_{qkv}.png",
                                 f"Trace: head → {recv_name}.{qkv.upper()} ({args.category})")

            elif ht == "gca" and rl in gca_set:
                # GCA receiver → Phase 3 per-head approach (no Q/K/V separation)
                print(f"\n  {recv_name} (Ph2 {qkv_src.upper()} effect={importance:+.4f})")

                data = run_phase3_perhead(
                    patcher, samples, device, rl, rh,
                    args.direction, label=f"→{recv_name}",
                    visual_mode=visual_mode)
                data["gca_layers"] = patcher.gca_layers
                data["sa_num_heads"] = patcher.sa_num_heads
                data["gca_num_heads"] = patcher.gca_num_heads

                up_top = get_top_heads(data, patcher.gca_layers, 5)
                for ut, ul, uh, uv in up_top:
                    print(f"    {ut.upper()}_L{ul}_H{uh}: {uv:+.4f}")

                out_path = output_dir / f"trace_{args.category}_{recv_name}.json"
                with open(out_path, "w") as f:
                    json.dump(data, f, indent=2)
                print(f"  Saved: {out_path}")

                plot_heatmap(data, patcher.gca_layers, patcher.sa_num_heads,
                             patcher.gca_num_heads,
                             output_dir / f"trace_{args.category}_{recv_name}.png",
                             f"Trace: head → {recv_name} ({args.category})")
            else:
                print(f"\n  Skipping {recv_name} (unsupported receiver type)")

    print("\nDone.")


if __name__ == "__main__":
    main()
