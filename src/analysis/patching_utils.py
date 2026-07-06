"""Shared patching infrastructure: HeadPatcher, ComponentPatcher.

Supports both decoder and classifier heads — auto-detected from head metadata.
"""

from __future__ import annotations

import torch
import numpy as np

# load_model not needed — use checkpoint loading pattern from main codebase


class HeadPatcher:
    """Per-head activation patching via hooking attn.proj input."""

    def __init__(self, steervit, head, vocab=None):
        self.steervit = steervit
        self.head = head
        self.head_type = getattr(head, "_head_type", "decoder")
        self.feature_pool = getattr(head, "_feature_pool", "cls")
        self._vocab_offset = getattr(head, "_vocab_offset", 3)
        self.vocab = vocab
        if self.head_type == "decoder":
            self.bos_id = vocab["<bos>"]
        self._cached_q_enc = None
        self.num_prefix = steervit.vision_model.trunk.num_prefix_tokens
        self.blocks = steervit.vision_model.trunk.blocks
        self.num_layers = len(self.blocks)
        self.sa_num_heads = self.blocks[0].attn.num_heads
        self.sa_head_dim = self.blocks[0].attn.head_dim
        # GCA info
        self.gca_layers = []
        for idx, blk in enumerate(self.blocks):
            gca = getattr(blk, "gated_cross_attn", None)
            if gca is not None:
                self.gca_layers.append(idx)
        if self.gca_layers:
            gca0 = self.blocks[self.gca_layers[0]].gated_cross_attn
            self.gca_num_heads = gca0.cross_attn.num_heads
            self.gca_head_dim = gca0.cross_attn.head_dim
        else:
            self.gca_num_heads = 0
            self.gca_head_dim = 0

    def _cache_question(self, questions):
        """Pre-compute question CLS encoding for classifier head."""
        if self.head_type == "classifier":
            with torch.no_grad():
                tok = self.steervit.tokenizer(
                    questions, padding=True, truncation=True,
                    max_length=512, return_tensors="pt",
                )
                tok = {k: v.to(self.steervit.text_model.device) for k, v in tok.items()}
                self._cached_q_enc = self.steervit.text_model(**tok).last_hidden_state[:, 0, :]

    def to_token_id(self, answer_idx: int) -> int:
        """Convert dataset answer_idx to model token id."""
        return answer_idx + self._vocab_offset

    def _pool_visual(self, features):
        """Pool visual features according to head's feature_pool setting."""
        if self.feature_pool == "cls":
            return features[:, 0, :]
        else:  # mean
            return features[:, self.num_prefix:, :].mean(dim=1)

    def _get_logit(self, features, correct_token_id):
        if self.head_type == "decoder":
            patches = features[:, self.num_prefix:, :]
            bos = torch.full((1, 1), self.bos_id, dtype=torch.long, device=features.device)
            logits = self.head(bos, patches)
            return logits[0, 0, correct_token_id].item()
        else:
            vis = self._pool_visual(features)
            logits = self.head(vis, self._cached_q_enc)
            return logits[0, correct_token_id].item()

    def _record_sa_pre_proj(self, images, questions):
        """Record self-attention pre-projection outputs at all layers."""
        sa_pre_proj = {}
        hooks = []

        for idx, blk in enumerate(self.blocks):
            def make_hook(layer_idx):
                def hook_fn(module, inp, output):
                    sa_pre_proj[layer_idx] = inp[0].detach().clone()
                return hook_fn
            hooks.append(blk.attn.proj.register_forward_hook(make_hook(idx)))

        with torch.no_grad():
            features = self.steervit.forward(images, questions)

        for h in hooks:
            h.remove()

        return sa_pre_proj, features

    def _patched_forward_sa_head(self, images, questions, target_layer, target_head,
                                  clean_pre_proj):
        """Forward with one SA head's pre-proj output restored from clean run."""
        hd = self.sa_head_dim
        h_start = target_head * hd
        h_end = h_start + hd

        def replace_head_hook(module, inp, output):
            modified = inp[0].clone()
            modified[:, :, h_start:h_end] = clean_pre_proj[:, :, h_start:h_end]
            return module.weight @ modified.transpose(-1, -2)

        def pre_hook(module, inp):
            x = inp[0].clone()
            x[:, :, h_start:h_end] = clean_pre_proj[:, :, h_start:h_end]
            return (x,)

        hook = self.blocks[target_layer].attn.proj.register_forward_pre_hook(pre_hook)
        with torch.no_grad():
            features = self.steervit.forward(images, questions)
        hook.remove()

        return features

    def run_sa_denoising(self, clean_imgs, clean_qs, corrupt_imgs, corrupt_qs,
                          correct_token_id, device):
        """Run per-SA-head denoising.

        Returns:
            heatmap: (num_layers, sa_num_heads) restoration values
        """
        self._cache_question(clean_qs)
        clean_pre_proj, clean_features = self._record_sa_pre_proj(clean_imgs, clean_qs)
        clean_logit = self._get_logit(clean_features, correct_token_id)

        _, corrupt_features = self._record_sa_pre_proj(corrupt_imgs, corrupt_qs)
        corrupt_logit = self._get_logit(corrupt_features, correct_token_id)

        heatmap = np.zeros((self.num_layers, self.sa_num_heads))
        for layer_idx in range(self.num_layers):
            for head_idx in range(self.sa_num_heads):
                patched = self._patched_forward_sa_head(
                    corrupt_imgs, corrupt_qs, layer_idx, head_idx,
                    clean_pre_proj[layer_idx],
                )
                patched_logit = self._get_logit(patched, correct_token_id)
                heatmap[layer_idx, head_idx] = patched_logit - corrupt_logit

        return heatmap, clean_logit, corrupt_logit

    # ── GCA head patching ──────────────────────────────────────────────

    def _record_gca_pre_proj(self, images, questions):
        """Record GCA pre-projection outputs (input to cross_attn.to_out)."""
        gca_pre_proj = {}
        hooks = []
        for layer_idx in self.gca_layers:
            gca = self.blocks[layer_idx].gated_cross_attn
            def make_hook(li):
                def hook_fn(module, inp, output):
                    gca_pre_proj[li] = inp[0].detach().clone()
                return hook_fn
            hooks.append(gca.cross_attn.to_out.register_forward_hook(make_hook(layer_idx)))

        with torch.no_grad():
            features = self.steervit.forward(images, questions)
        for h in hooks:
            h.remove()
        return gca_pre_proj, features

    def _patched_forward_gca_head(self, images, questions, target_layer, target_head,
                                   clean_pre_proj):
        """Forward with one GCA head's pre-proj output restored from clean run."""
        hd = self.gca_head_dim
        h_start = target_head * hd
        h_end = h_start + hd

        gca = self.blocks[target_layer].gated_cross_attn

        def pre_hook(module, inp):
            x = inp[0].clone()
            x[:, :, h_start:h_end] = clean_pre_proj[:, :, h_start:h_end]
            return (x,)

        hook = gca.cross_attn.to_out.register_forward_pre_hook(pre_hook)
        with torch.no_grad():
            features = self.steervit.forward(images, questions)
        hook.remove()
        return features

    def run_gca_denoising(self, clean_imgs, clean_qs, corrupt_imgs, corrupt_qs,
                           correct_token_id, device):
        """Per-GCA-head denoising. Returns (num_gca_layers, gca_num_heads) heatmap."""
        self._cache_question(clean_qs)
        clean_pre, clean_features = self._record_gca_pre_proj(clean_imgs, clean_qs)
        clean_logit = self._get_logit(clean_features, correct_token_id)

        _, corrupt_features = self._record_gca_pre_proj(corrupt_imgs, corrupt_qs)
        corrupt_logit = self._get_logit(corrupt_features, correct_token_id)

        n_gca = len(self.gca_layers)
        heatmap = np.zeros((n_gca, self.gca_num_heads))
        for gi, layer_idx in enumerate(self.gca_layers):
            for head_idx in range(self.gca_num_heads):
                patched = self._patched_forward_gca_head(
                    corrupt_imgs, corrupt_qs, layer_idx, head_idx,
                    clean_pre[layer_idx],
                )
                patched_logit = self._get_logit(patched, correct_token_id)
                heatmap[gi, head_idx] = patched_logit - corrupt_logit

        return heatmap, clean_logit, corrupt_logit


class ComponentPatcher:
    """Decomposed patching: SA contribution vs GCA contribution."""

    def __init__(self, steervit, head, vocab=None):
        self.steervit = steervit
        self.head = head
        self.head_type = getattr(head, "_head_type", "decoder")
        self.feature_pool = getattr(head, "_feature_pool", "cls")
        self._vocab_offset = getattr(head, "_vocab_offset", 3)
        self.vocab = vocab
        if self.head_type == "decoder":
            self.bos_id = vocab["<bos>"]
        self._cached_q_enc = None
        self.num_prefix = steervit.vision_model.trunk.num_prefix_tokens
        self.blocks = steervit.vision_model.trunk.blocks
        self.num_layers = len(self.blocks)
        self.gca_layers = [
            i for i, blk in enumerate(self.blocks)
            if getattr(blk, "gated_cross_attn", None) is not None
        ]

    def _cache_question(self, questions):
        if self.head_type == "classifier":
            with torch.no_grad():
                tok = self.steervit.tokenizer(
                    questions, padding=True, truncation=True,
                    max_length=512, return_tensors="pt",
                )
                tok = {k: v.to(self.steervit.text_model.device) for k, v in tok.items()}
                self._cached_q_enc = self.steervit.text_model(**tok).last_hidden_state[:, 0, :]

    def to_token_id(self, answer_idx: int) -> int:
        """Convert dataset answer_idx to model token id."""
        return answer_idx + self._vocab_offset

    def _pool_visual(self, features):
        if self.feature_pool == "cls":
            return features[:, 0, :]
        else:
            return features[:, self.num_prefix:, :].mean(dim=1)

    def _get_logit(self, features, correct_token_id):
        if self.head_type == "decoder":
            patches = features[:, self.num_prefix:, :]
            bos = torch.full((1, 1), self.bos_id, dtype=torch.long,
                             device=features.device)
            logits = self.head(bos, patches)
            return logits[0, 0, correct_token_id].item()
        else:
            vis = self._pool_visual(features)
            logits = self.head(vis, self._cached_q_enc)
            return logits[0, correct_token_id].item()

    def _record_components(self, images, questions):
        """Record per-layer SA outputs and GCA contributions."""
        sa_outputs = {}
        gca_contribs = {}
        hooks = []

        for idx, blk in enumerate(self.blocks):
            def make_sa_hook(li):
                def fn(module, inp, output):
                    sa_outputs[li] = output.detach().clone()
                return fn
            hooks.append(blk.attn.register_forward_hook(make_sa_hook(idx)))

        for idx in self.gca_layers:
            gca = self.blocks[idx].gated_cross_attn
            def make_gca_hook(li):
                def fn(module, inp, output):
                    gca_contribs[li] = (output - inp[0]).detach().clone()
                return fn
            hooks.append(gca.register_forward_hook(make_gca_hook(idx)))

        with torch.no_grad():
            features = self.steervit.forward(images, questions)

        for h in hooks:
            h.remove()
        return sa_outputs, gca_contribs, features

    def _patched_forward_sa(self, images, questions, layer, clean_sa):
        """Corrupt run, restore clean SA output at one layer."""
        def hook(module, inp, output):
            return clean_sa
        h = self.blocks[layer].attn.register_forward_hook(hook)
        with torch.no_grad():
            feats = self.steervit.forward(images, questions)
        h.remove()
        return feats

    def _patched_forward_gca(self, images, questions, layer, clean_gca_contrib):
        """Corrupt run, restore clean GCA contribution at one layer."""
        gca = self.blocks[layer].gated_cross_attn
        def hook(module, inp, output):
            return inp[0] + clean_gca_contrib
        h = gca.register_forward_hook(hook)
        with torch.no_grad():
            feats = self.steervit.forward(images, questions)
        h.remove()
        return feats

    def _patched_forward_sa_gca(self, images, questions, layer,
                                 clean_sa, clean_gca_contrib):
        """Corrupt run, restore both SA and GCA at one layer."""
        gca = self.blocks[layer].gated_cross_attn
        def gca_hook(module, inp, output):
            return inp[0] + clean_gca_contrib
        def sa_hook(module, inp, output):
            return clean_sa
        h1 = gca.register_forward_hook(gca_hook)
        h2 = self.blocks[layer].attn.register_forward_hook(sa_hook)
        with torch.no_grad():
            feats = self.steervit.forward(images, questions)
        h1.remove()
        h2.remove()
        return feats

    def run_text_denoising(self, img_batch, clean_q, corrupt_q,
                            correct_token_id, device):
        """Component-decomposed text denoising at every layer."""
        self._cache_question([clean_q])
        clean_sa, clean_gca, clean_feats = self._record_components(
            img_batch, [clean_q])
        _, _, corrupt_feats = self._record_components(
            img_batch, [corrupt_q])
        corrupt_logit = self._get_logit(corrupt_feats, correct_token_id)

        sa_deltas = {}
        gca_deltas = {}
        sagca_deltas = {}

        for l in range(self.num_layers):
            f = self._patched_forward_sa(img_batch, [corrupt_q], l, clean_sa[l])
            sa_deltas[l] = self._get_logit(f, correct_token_id) - corrupt_logit

        for l in self.gca_layers:
            f = self._patched_forward_gca(img_batch, [corrupt_q], l, clean_gca[l])
            gca_deltas[l] = self._get_logit(f, correct_token_id) - corrupt_logit

        for l in self.gca_layers:
            f = self._patched_forward_sa_gca(
                img_batch, [corrupt_q], l, clean_sa[l], clean_gca[l])
            sagca_deltas[l] = self._get_logit(f, correct_token_id) - corrupt_logit

        return sa_deltas, gca_deltas, sagca_deltas

    def run_text_noising(self, img_batch, clean_q, corrupt_q,
                          correct_token_id, device):
        """Component-decomposed text noising at every layer.

        Clean run → inject corrupt SA/GCA at each layer.
        Deltas are negative (degradation from injecting corrupt component).
        """
        self._cache_question([clean_q])
        _, _, clean_feats = self._record_components(img_batch, [clean_q])
        corrupt_sa, corrupt_gca, _ = self._record_components(
            img_batch, [corrupt_q])
        clean_logit = self._get_logit(clean_feats, correct_token_id)

        sa_deltas = {}
        gca_deltas = {}
        sagca_deltas = {}

        for l in range(self.num_layers):
            f = self._patched_forward_sa(img_batch, [clean_q], l, corrupt_sa[l])
            sa_deltas[l] = self._get_logit(f, correct_token_id) - clean_logit

        for l in self.gca_layers:
            f = self._patched_forward_gca(img_batch, [clean_q], l, corrupt_gca[l])
            gca_deltas[l] = self._get_logit(f, correct_token_id) - clean_logit

        for l in self.gca_layers:
            f = self._patched_forward_sa_gca(
                img_batch, [clean_q], l, corrupt_sa[l], corrupt_gca[l])
            sagca_deltas[l] = self._get_logit(f, correct_token_id) - clean_logit

        return sa_deltas, gca_deltas, sagca_deltas
