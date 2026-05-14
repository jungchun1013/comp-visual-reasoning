"""Lazy cache for frozen RoBERTa text encodings.

Caches per-question RoBERTa outputs to avoid redundant forward passes
across epochs. RoBERTa is ~23% of forward time; caching eliminates it
for previously-seen questions.

Memory: ~80KB per unique question on CPU. Uses LRU eviction when
max_size is reached.
"""

from __future__ import annotations

from collections import OrderedDict

import torch
import torch.nn.functional as F


class TextCache:
    """Lazy LRU cache for frozen text encoder outputs.

    Caches per-question: (cls_token, raw_feats_unpadded, seq_len).
    Stored on CPU to avoid GPU memory pressure. Moved to GPU in batches.

    Args:
        steervit: CrossAttnViT model (uses its tokenizer, text_model, connector).
        max_size: Max cached questions. 0 = unlimited. Default 200k (~16GB CPU).
    """

    def __init__(self, steervit, max_size: int = 200_000):
        self.steervit = steervit
        self.max_size = max_size
        self._cache: OrderedDict[str, tuple[torch.Tensor, torch.Tensor]] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def encode_text(self, questions: list[str]):
        """Cached version of steervit.encode_text().

        Returns:
            text_feats: (B, T, visual_dim) projected text features for GCA.
            attn_mask: (B, num_img_tokens + T) attention mask.
            raw_text_feats: (B, T, text_dim) raw RoBERTa outputs.
        """
        device = self.steervit.text_model.device

        # Split into cached vs uncached
        cached_indices = []
        uncached_indices = []
        for i, q in enumerate(questions):
            if q in self._cache:
                cached_indices.append(i)
                self._cache.move_to_end(q)  # LRU touch
                self.hits += 1
            else:
                uncached_indices.append(i)
                self.misses += 1

        # Compute uncached
        if uncached_indices:
            uncached_qs = [questions[i] for i in uncached_indices]
            with torch.no_grad():
                roberta_dict = self.steervit.tokenizer(
                    uncached_qs, padding=True, truncation=True,
                    max_length=512, return_tensors="pt",
                )
                roberta_dict = {k: v.to(device) for k, v in roberta_dict.items()}
                raw_out = self.steervit.text_model(**roberta_dict).last_hidden_state
                mask = roberta_dict["attention_mask"]

            # Cache per-question (CPU, unpadded)
            for batch_idx, orig_idx in enumerate(uncached_indices):
                q = questions[orig_idx]
                seq_len = mask[batch_idx].sum().item()
                self._cache[q] = (
                    raw_out[batch_idx, :seq_len, :].cpu(),  # raw_feats (seq_len, text_dim)
                    seq_len,
                )
                # LRU eviction
                if self.max_size > 0 and len(self._cache) > self.max_size:
                    self._cache.popitem(last=False)

        # Reconstruct padded batch from cache
        all_feats = []
        all_seq_lens = []
        for q in questions:
            raw_feats_q, seq_len = self._cache[q]
            all_feats.append(raw_feats_q)
            all_seq_lens.append(seq_len)

        max_len = max(all_seq_lens)
        text_dim = all_feats[0].size(-1)
        B = len(questions)

        raw_text_feats = torch.zeros(B, max_len, text_dim, device=device)
        attn_mask_text = torch.zeros(B, max_len, dtype=torch.bool, device=device)

        for i, (feats, sl) in enumerate(zip(all_feats, all_seq_lens)):
            raw_text_feats[i, :sl, :] = feats.to(device)
            attn_mask_text[i, :sl] = True

        # Project through connector (same as steervit.encode_text)
        text_feats = F.normalize(raw_text_feats, dim=-1)
        text_feats = self.steervit.connector(text_feats)

        # Build full attention mask (image prefix + text)
        num_img_tokens = self.steervit.num_img_tokens
        img_mask = torch.ones(B, num_img_tokens, dtype=torch.bool, device=device)
        attn_mask = torch.cat([img_mask, attn_mask_text], dim=-1)

        return text_feats, attn_mask, raw_text_feats

    def get_cls_tokens(self, questions: list[str]) -> torch.Tensor:
        """Get RoBERTa CLS tokens (position 0) for question encoding.

        Returns:
            cls_tokens: (B, text_dim)
        """
        device = self.steervit.text_model.device

        # Ensure all questions are cached
        uncached = [q for q in questions if q not in self._cache]
        if uncached:
            # Batch-encode uncached questions to populate cache
            self.encode_text(uncached)

        # Extract CLS tokens from cache
        cls_list = []
        for q in questions:
            raw_feats_q, _ = self._cache[q]
            cls_list.append(raw_feats_q[0])  # position 0 = CLS

        return torch.stack(cls_list).to(device)

    @property
    def cache_size(self) -> int:
        return len(self._cache)

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
