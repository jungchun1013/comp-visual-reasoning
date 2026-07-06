"""Interventional activation patching for SteerViT.

Replaces activations at each layer and measures the causal effect
on output logits (logit difference). Supports both decoder and classifier heads.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CausalPatcher:
    """Interventional activation patching: replace layer activations
    and measure logit[correct] change.

    Supports both decoder and classifier heads via head_type parameter.
    """

    def __init__(self, steervit: nn.Module, head: nn.Module,
                 vocab: dict[str, int] | None = None):
        self.steervit = steervit
        self.head = head
        self.head_type = getattr(head, "_head_type", "decoder")
        self.feature_pool = getattr(head, "_feature_pool", "cls")
        self.vocab = vocab
        if self.head_type == "decoder":
            self.bos_id = vocab["<bos>"]
        self._vocab_offset = getattr(head, "_vocab_offset", 3)
        self._cached_q_enc = None
        self.num_prefix = steervit.vision_model.trunk.num_prefix_tokens
        self.num_layers = len(steervit.vision_model.trunk.blocks)

    def to_token_id(self, answer_idx: int) -> int:
        """Convert dataset answer_idx to model token id."""
        return answer_idx + self._vocab_offset

    def _record_activations(
        self, images: torch.Tensor, questions: list[str] | None,
    ) -> tuple[dict[int, torch.Tensor], torch.Tensor]:
        """Forward pass recording all layer activations (block outputs).

        Returns:
            activations: {layer_idx: (B, N_tokens, D)} block outputs
            features: (B, N_tokens, D) final output from SteerViT
        """
        activations: dict[int, torch.Tensor] = {}
        hooks = []
        for idx, blk in enumerate(self.steervit.vision_model.trunk.blocks):
            def make_hook(layer_idx):
                def hook_fn(module, input, output):
                    activations[layer_idx] = output[0].detach().clone()
                return hook_fn
            hooks.append(blk.register_forward_hook(make_hook(idx)))

        with torch.no_grad():
            features = self.steervit.forward(images, questions)

        for h in hooks:
            h.remove()

        return activations, features

    def _record_activations_with_inputs(
        self, images: torch.Tensor, questions: list[str] | None,
    ) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor], torch.Tensor]:
        """Forward pass recording block inputs AND outputs.

        Returns:
            inputs: {layer_idx: (B, N_tokens, D)} block inputs
            outputs: {layer_idx: (B, N_tokens, D)} block outputs
            features: final SteerViT output
        """
        inputs: dict[int, torch.Tensor] = {}
        outputs: dict[int, torch.Tensor] = {}
        hooks = []
        for idx, blk in enumerate(self.steervit.vision_model.trunk.blocks):
            def make_hook(layer_idx):
                def hook_fn(module, inp, output):
                    inputs[layer_idx] = inp[0][0].detach().clone()  # x before block
                    outputs[layer_idx] = output[0].detach().clone()  # x after block
                return hook_fn
            hooks.append(blk.register_forward_hook(make_hook(idx)))

        with torch.no_grad():
            features = self.steervit.forward(images, questions)

        for h in hooks:
            h.remove()

        return inputs, outputs, features

    def _patched_forward(
        self,
        images: torch.Tensor,
        questions: list[str],
        target_layer: int,
        corrupted_x: torch.Tensor,
    ) -> torch.Tensor:
        """Forward with activation replacement at target_layer.

        Replaces visual features (output[0]) at the target layer with
        corrupted_x, while keeping text_feats and attn_mask from the
        clean forward (original question conditioning).

        Returns:
            features: (B, N_tokens, D) final output
        """
        def replace_hook(module, input, output):
            # Keep text_feats and attn_mask from clean run,
            # replace only the visual features
            return (corrupted_x, output[1], output[2])

        hook = self.steervit.vision_model.trunk.blocks[target_layer].register_forward_hook(
            replace_hook
        )
        with torch.no_grad():
            features = self.steervit.forward(images, questions)
        hook.remove()

        return features

    def _patched_forward_contribution(
        self,
        images: torch.Tensor,
        questions: list[str],
        target_layer: int,
        clean_input: torch.Tensor,
        clean_output: torch.Tensor,
    ) -> torch.Tensor:
        """Forward with only the target layer's CONTRIBUTION replaced.

        Instead of replacing the entire residual stream, computes:
          patched_x = corrupt_input + clean_contribution
        where clean_contribution = clean_output - clean_input.

        This preserves the residual stream from all other layers while
        swapping only what the target block added.
        """
        clean_contribution = clean_output - clean_input

        def replace_contribution_hook(module, inp, output):
            corrupt_x_in = inp[0][0]   # block input: x from residual stream
            corrupt_x_out = output[0]  # block output
            # Replace this block's contribution with the clean version
            restored_x = corrupt_x_in + clean_contribution
            return (restored_x, output[1], output[2])

        hook = self.steervit.vision_model.trunk.blocks[target_layer].register_forward_hook(
            replace_contribution_hook
        )
        with torch.no_grad():
            features = self.steervit.forward(images, questions)
        hook.remove()

        return features

    def _patched_forward_local(
        self,
        images: torch.Tensor,
        questions: list[str],
        target_layer: int,
        corrupted_x: torch.Tensor,
        patch_indices: list[int],
    ) -> torch.Tensor:
        """Forward with activation replacement at specific patch positions only.

        Replaces only the patches at patch_indices with values from corrupted_x,
        keeping all other patches (including CLS) from the clean forward.
        """
        prefix = self.num_prefix

        def replace_hook(module, input, output):
            x_clean = output[0].clone()
            for pidx in patch_indices:
                x_clean[0, prefix + pidx, :] = corrupted_x[0, prefix + pidx, :]
            return (x_clean, output[1], output[2])

        hook = self.steervit.vision_model.trunk.blocks[target_layer].register_forward_hook(
            replace_hook
        )
        with torch.no_grad():
            features = self.steervit.forward(images, questions)
        hook.remove()

        return features

    def _cache_question(self, question: str):
        """Pre-compute question CLS encoding for classifier head."""
        if self.head_type == "classifier":
            with torch.no_grad():
                tok = self.steervit.tokenizer(
                    [question], padding=True, truncation=True,
                    max_length=512, return_tensors="pt",
                )
                tok = {k: v.to(self.steervit.text_model.device) for k, v in tok.items()}
                self._cached_q_enc = self.steervit.text_model(**tok).last_hidden_state[:, 0, :]

    def _pool_visual(self, features):
        """Pool visual features according to head's feature_pool setting."""
        if self.feature_pool == "cls":
            return features[:, 0, :]
        else:  # mean
            return features[:, self.num_prefix:, :].mean(dim=1)

    def _get_logit(self, features: torch.Tensor, correct_token_id: int) -> float:
        """Get logit for the correct answer token."""
        if self.head_type == "decoder":
            patches = features[:, self.num_prefix:, :]
            bos = torch.full(
                (features.size(0), 1), self.bos_id,
                dtype=torch.long, device=features.device,
            )
            logits = self.head(bos, patches)  # (B, 1, vocab_size)
            return logits[0, 0, correct_token_id].item()
        else:
            vis = self._pool_visual(features)
            logits = self.head(vis, self._cached_q_enc)  # (B, num_answers)
            return logits[0, correct_token_id].item()

    def run_single(
        self,
        image: torch.Tensor,
        question: str,
        corrupted_question: str,
        correct_token_id: int,
        device: torch.device,
    ) -> tuple[dict[int, float], float]:
        """Run interventional patching for one image-question pair (degradation).

        Injects corrupted block CONTRIBUTION into clean forward pass.

        Returns:
            deltas: {layer_idx: Δ_logit} where Δ = patched - clean.
                    Negative means patching with corrupted contribution hurts.
            clean_logit: The clean run logit[correct] for reference.
        """
        self._cache_question(question)
        image_batch = image.unsqueeze(0).to(device)

        # Step 1: Clean run
        _, _, clean_features = self._record_activations_with_inputs(image_batch, [question])
        clean_logit = self._get_logit(clean_features, correct_token_id)

        # Step 2: Corrupted run — record inputs and outputs
        corrupt_inputs, corrupt_outputs, _ = \
            self._record_activations_with_inputs(image_batch, [corrupted_question])

        # Step 3: Noising — replace each block's contribution with corrupt version
        deltas = {}
        for layer_idx in range(self.num_layers):
            patched_features = self._patched_forward_contribution(
                image_batch, [question], layer_idx,
                corrupt_inputs[layer_idx], corrupt_outputs[layer_idx],
            )
            patched_logit = self._get_logit(patched_features, correct_token_id)
            deltas[layer_idx] = patched_logit - clean_logit

        return deltas, clean_logit

    def run_text_denoising(
        self,
        image: torch.Tensor,
        question: str,
        corrupted_question: str,
        correct_token_id: int,
        device: torch.device,
    ) -> tuple[dict[int, float], float, float]:
        """Text denoising: corrupt question input, restore clean block contribution.

        Returns:
            restorations: {layer_idx: Δ_logit} positive = recovery.
            clean_logit, corrupt_logit.
        """
        self._cache_question(question)
        image_batch = image.unsqueeze(0).to(device)

        # Clean run
        clean_inputs, clean_outputs, clean_features = \
            self._record_activations_with_inputs(image_batch, [question])
        clean_logit = self._get_logit(clean_features, correct_token_id)

        # Corrupt run
        _, corrupt_features = self._record_activations(image_batch, [corrupted_question])
        corrupt_logit = self._get_logit(corrupt_features, correct_token_id)

        # Restore clean contribution at each layer
        restorations = {}
        for layer_idx in range(self.num_layers):
            patched_features = self._patched_forward_contribution(
                image_batch, [corrupted_question], layer_idx,
                clean_inputs[layer_idx], clean_outputs[layer_idx],
            )
            patched_logit = self._get_logit(patched_features, correct_token_id)
            restorations[layer_idx] = patched_logit - corrupt_logit

        return restorations, clean_logit, corrupt_logit

    def run_text_noising(
        self,
        image: torch.Tensor,
        question: str,
        corrupted_question: str,
        correct_token_id: int,
        device: torch.device,
    ) -> tuple[dict[int, float], float, float]:
        """Text noising: clean question input, inject corrupt block contribution.

        Returns:
            deltas: {layer_idx: Δ_logit} negative = degradation.
            clean_logit, corrupt_logit.
        """
        self._cache_question(question)
        image_batch = image.unsqueeze(0).to(device)

        # Clean run — record contributions
        clean_inputs, clean_outputs, clean_features = \
            self._record_activations_with_inputs(image_batch, [question])
        clean_logit = self._get_logit(clean_features, correct_token_id)

        # Corrupt run — record contributions
        corrupt_inputs, corrupt_outputs, corrupt_features = \
            self._record_activations_with_inputs(image_batch, [corrupted_question])
        corrupt_logit = self._get_logit(corrupt_features, correct_token_id)

        # Inject corrupt contribution into clean forward at each layer
        deltas = {}
        for layer_idx in range(self.num_layers):
            patched_features = self._patched_forward_contribution(
                image_batch, [question], layer_idx,
                corrupt_inputs[layer_idx], corrupt_outputs[layer_idx],
            )
            patched_logit = self._get_logit(patched_features, correct_token_id)
            deltas[layer_idx] = patched_logit - clean_logit

        return deltas, clean_logit, corrupt_logit

    def run_vision_noising(
        self,
        clean_image: torch.Tensor,
        corrupt_image: torch.Tensor,
        question: str,
        correct_token_id: int,
        device: torch.device,
    ) -> tuple[dict[int, float], float]:
        """Vision noising: clean image input, inject corrupt block contribution.

        Returns:
            deltas: {layer_idx: Δ_logit} negative = degradation.
            clean_logit.
        """
        self._cache_question(question)
        clean_batch = clean_image.unsqueeze(0).to(device)
        corrupt_batch = corrupt_image.unsqueeze(0).to(device)

        # Clean run
        _, _, clean_features = self._record_activations_with_inputs(clean_batch, [question])
        clean_logit = self._get_logit(clean_features, correct_token_id)

        # Corrupt run — record contributions
        corrupt_inputs, corrupt_outputs, _ = \
            self._record_activations_with_inputs(corrupt_batch, [question])

        # Inject corrupt contribution into clean forward
        deltas = {}
        for layer_idx in range(self.num_layers):
            patched_features = self._patched_forward_contribution(
                clean_batch, [question], layer_idx,
                corrupt_inputs[layer_idx], corrupt_outputs[layer_idx],
            )
            patched_logit = self._get_logit(patched_features, correct_token_id)
            deltas[layer_idx] = patched_logit - clean_logit

        return deltas, clean_logit

    def run_restoration(
        self,
        clean_image: torch.Tensor,
        corrupt_image: torch.Tensor,
        question: str,
        correct_token_id: int,
        device: torch.device,
        patch_indices: list[int] | None = None,
    ) -> tuple[dict[int, float], float, float]:
        """Run restoration patching (NOTICE-style) with SIP image pairs.

        Runs corrupt image, then restores clean activations at each layer
        to measure which layers carry the information that distinguishes
        the two images.

        Args:
            clean_image: (3, H, W) clean image tensor.
            corrupt_image: (3, H, W) SIP-corrupted image (one attribute changed).
            question: Question string (same for both images).
            correct_token_id: Decoder vocab index of the CLEAN image's correct answer.
            device: Torch device.
            patch_indices: If provided, only restore these patch positions (localized).
                          If None, restore all patches (full layer replacement).

        Returns:
            restorations: {layer_idx: restoration_logit - corrupt_logit}
                          Positive = restoring clean activation recovers correct answer.
            clean_logit: logit[correct] from clean run.
            corrupt_logit: logit[correct] from corrupt run.
        """
        self._cache_question(question)
        clean_batch = clean_image.unsqueeze(0).to(device)
        corrupt_batch = corrupt_image.unsqueeze(0).to(device)

        # Step 1: Clean run — record block inputs AND outputs
        clean_inputs, clean_outputs, clean_features = \
            self._record_activations_with_inputs(clean_batch, [question])
        clean_logit = self._get_logit(clean_features, correct_token_id)

        # Step 2: Corrupt run — baseline (model sees wrong image)
        _, corrupt_features = self._record_activations(corrupt_batch, [question])
        corrupt_logit = self._get_logit(corrupt_features, correct_token_id)

        # Step 3: Restoration — replace each block's CONTRIBUTION with clean version
        # contribution = block_output - block_input (what this block added to residual stream)
        restorations = {}
        for layer_idx in range(self.num_layers):
            patched_features = self._patched_forward_contribution(
                corrupt_batch, [question], layer_idx,
                clean_inputs[layer_idx], clean_outputs[layer_idx],
            )
            patched_logit = self._get_logit(patched_features, correct_token_id)
            restorations[layer_idx] = patched_logit - corrupt_logit

        return restorations, clean_logit, corrupt_logit

    # ── Progressive / cumulative patching ─────────────────────────────

    def _patched_forward_progressive(
        self,
        images: torch.Tensor,
        questions: list[str],
        layers_to_patch: list[int],
        alt_inputs: dict[int, torch.Tensor],
        alt_outputs: dict[int, torch.Tensor],
    ) -> torch.Tensor:
        """Forward with contributions replaced at ALL layers in layers_to_patch."""
        hooks = []
        for layer_idx in layers_to_patch:
            contrib = alt_outputs[layer_idx] - alt_inputs[layer_idx]

            def make_hook(c):
                def hook_fn(module, inp, output):
                    x_in = inp[0][0]
                    return (x_in + c, output[1], output[2])
                return hook_fn

            hooks.append(
                self.steervit.vision_model.trunk.blocks[layer_idx]
                .register_forward_hook(make_hook(contrib))
            )
        with torch.no_grad():
            features = self.steervit.forward(images, questions)
        for h in hooks:
            h.remove()
        return features

    def run_progressive_noising(
        self,
        image: torch.Tensor,
        question: str,
        corrupted_question: str,
        correct_token_id: int,
        device: torch.device,
    ) -> tuple[dict[int, float], float]:
        """Progressive noising: cumulatively inject corrupt contributions L0..Lk.

        Returns:
            deltas: {k: Δlogit} for k=0..num_layers-1, where at step k
                    layers 0..k have corrupt contributions injected.
            clean_logit.
        """
        self._cache_question(question)
        image_batch = image.unsqueeze(0).to(device)

        _, _, clean_features = self._record_activations_with_inputs(
            image_batch, [question])
        clean_logit = self._get_logit(clean_features, correct_token_id)

        corrupt_inputs, corrupt_outputs, _ = \
            self._record_activations_with_inputs(image_batch, [corrupted_question])

        deltas = {}
        for k in range(self.num_layers):
            patched_features = self._patched_forward_progressive(
                image_batch, [question],
                list(range(k + 1)),
                corrupt_inputs, corrupt_outputs,
            )
            patched_logit = self._get_logit(patched_features, correct_token_id)
            deltas[k] = patched_logit - clean_logit

        return deltas, clean_logit

    def run_progressive_denoising(
        self,
        image: torch.Tensor,
        question: str,
        corrupted_question: str,
        correct_token_id: int,
        device: torch.device,
    ) -> tuple[dict[int, float], float, float]:
        """Progressive denoising: cumulatively restore clean contributions L0..Lk.

        Returns:
            restorations: {k: Δlogit} for k=0..num_layers-1, where at step k
                          layers 0..k have clean contributions restored.
            clean_logit, corrupt_logit.
        """
        self._cache_question(question)
        image_batch = image.unsqueeze(0).to(device)

        clean_inputs, clean_outputs, clean_features = \
            self._record_activations_with_inputs(image_batch, [question])
        clean_logit = self._get_logit(clean_features, correct_token_id)

        _, corrupt_features = self._record_activations(
            image_batch, [corrupted_question])
        corrupt_logit = self._get_logit(corrupt_features, correct_token_id)

        restorations = {}
        for k in range(self.num_layers):
            patched_features = self._patched_forward_progressive(
                image_batch, [corrupted_question],
                list(range(k + 1)),
                clean_inputs, clean_outputs,
            )
            patched_logit = self._get_logit(patched_features, correct_token_id)
            restorations[k] = patched_logit - corrupt_logit

        return restorations, clean_logit, corrupt_logit

    # ── Progressive patching including decoder layers ──────────────

    @property
    def num_decoder_layers(self) -> int:
        if self.head_type != "decoder":
            return 0
        return len(self.head.layers)

    @property
    def total_layers(self) -> int:
        """Encoder layers + decoder layers."""
        return self.num_layers + self.num_decoder_layers

    def _get_logit_from_decoder_out(self, decoder_out: torch.Tensor,
                                     correct_token_id: int) -> float:
        """Get logit from decoder hidden state (before output_proj)."""
        logits = self.head.output_proj(decoder_out)
        return logits[0, 0, correct_token_id].item()

    def _record_decoder_activations(
        self, features: torch.Tensor, correct_token_id: int,
    ) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor], float]:
        """Record decoder layer inputs/outputs."""
        patches = features[:, self.num_prefix:, :]
        memory = self.head.visual_proj(patches)
        bos = torch.full((features.size(0), 1), self.bos_id,
                         dtype=torch.long, device=features.device)
        tgt = self.head.token_embedding(bos) + self.head.pos_embedding(
            torch.arange(1, device=features.device))

        dec_inputs = {}
        dec_outputs = {}
        out = tgt
        for i, layer in enumerate(self.head.layers):
            dec_inputs[i] = out.detach().clone()
            out = layer(out, memory, tgt_mask=None)
            dec_outputs[i] = out.detach().clone()

        logit = self._get_logit_from_decoder_out(out, correct_token_id)
        return dec_inputs, dec_outputs, logit

    def run_progressive_noising_with_decoder(
        self,
        image: torch.Tensor,
        question: str,
        corrupted_question: str,
        correct_token_id: int,
        device: torch.device,
    ) -> tuple[dict[int, float], float]:
        """Progressive noising across encoder + decoder layers.

        Keys 0..num_layers-1 = encoder, num_layers.. = decoder.
        """
        if self.head_type != "decoder":
            return self.run_progressive_noising(
                image, question, corrupted_question, correct_token_id, device)

        self._cache_question(question)
        image_batch = image.unsqueeze(0).to(device)

        clean_enc_inputs, clean_enc_outputs, clean_features = \
            self._record_activations_with_inputs(image_batch, [question])
        clean_dec_inputs, clean_dec_outputs, clean_logit = \
            self._record_decoder_activations(clean_features, correct_token_id)

        corrupt_enc_inputs, corrupt_enc_outputs, corrupt_features = \
            self._record_activations_with_inputs(image_batch, [corrupted_question])
        corrupt_dec_inputs, corrupt_dec_outputs, _ = \
            self._record_decoder_activations(corrupt_features, correct_token_id)

        deltas = {}
        n_enc = self.num_layers
        n_dec = self.num_decoder_layers

        for k in range(n_enc + n_dec):
            if k < n_enc:
                patched_features = self._patched_forward_progressive(
                    image_batch, [question],
                    list(range(k + 1)),
                    corrupt_enc_inputs, corrupt_enc_outputs,
                )
                _, _, patched_logit = self._record_decoder_activations(
                    patched_features, correct_token_id)
            else:
                patched_features = self._patched_forward_progressive(
                    image_batch, [question],
                    list(range(n_enc)),
                    corrupt_enc_inputs, corrupt_enc_outputs,
                )
                patches = patched_features[:, self.num_prefix:, :]
                memory = self.head.visual_proj(patches)
                bos = torch.full((1, 1), self.bos_id,
                                 dtype=torch.long, device=device)
                tgt = self.head.token_embedding(bos) + self.head.pos_embedding(
                    torch.arange(1, device=device))
                out = tgt
                dec_k = k - n_enc
                for i, layer in enumerate(self.head.layers):
                    if i <= dec_k:
                        contrib = corrupt_dec_outputs[i] - corrupt_dec_inputs[i]
                        out = out + contrib
                    else:
                        out = layer(out, memory, tgt_mask=None)
                patched_logit = self._get_logit_from_decoder_out(
                    out, correct_token_id)

            deltas[k] = patched_logit - clean_logit

        return deltas, clean_logit

    def run_progressive_denoising_with_decoder(
        self,
        image: torch.Tensor,
        question: str,
        corrupted_question: str,
        correct_token_id: int,
        device: torch.device,
    ) -> tuple[dict[int, float], float, float]:
        """Progressive denoising across encoder + decoder layers."""
        if self.head_type != "decoder":
            return self.run_progressive_denoising(
                image, question, corrupted_question, correct_token_id, device)

        self._cache_question(question)
        image_batch = image.unsqueeze(0).to(device)

        clean_enc_inputs, clean_enc_outputs, clean_features = \
            self._record_activations_with_inputs(image_batch, [question])
        clean_dec_inputs, clean_dec_outputs, clean_logit = \
            self._record_decoder_activations(clean_features, correct_token_id)

        _, corrupt_features = self._record_activations(
            image_batch, [corrupted_question])
        _, _, corrupt_logit = \
            self._record_decoder_activations(corrupt_features, correct_token_id)

        restorations = {}
        n_enc = self.num_layers
        n_dec = self.num_decoder_layers

        for k in range(n_enc + n_dec):
            if k < n_enc:
                patched_features = self._patched_forward_progressive(
                    image_batch, [corrupted_question],
                    list(range(k + 1)),
                    clean_enc_inputs, clean_enc_outputs,
                )
                _, _, patched_logit = self._record_decoder_activations(
                    patched_features, correct_token_id)
            else:
                patched_features = self._patched_forward_progressive(
                    image_batch, [corrupted_question],
                    list(range(n_enc)),
                    clean_enc_inputs, clean_enc_outputs,
                )
                patches = patched_features[:, self.num_prefix:, :]
                memory = self.head.visual_proj(patches)
                bos = torch.full((1, 1), self.bos_id,
                                 dtype=torch.long, device=device)
                tgt = self.head.token_embedding(bos) + self.head.pos_embedding(
                    torch.arange(1, device=device))
                out = tgt
                dec_k = k - n_enc
                for i, layer in enumerate(self.head.layers):
                    if i <= dec_k:
                        contrib = clean_dec_outputs[i] - clean_dec_inputs[i]
                        out = out + contrib
                    else:
                        out = layer(out, memory, tgt_mask=None)
                patched_logit = self._get_logit_from_decoder_out(
                    out, correct_token_id)

            restorations[k] = patched_logit - corrupt_logit

        return restorations, clean_logit, corrupt_logit
