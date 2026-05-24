"""Mixture-of-Transformers (MoT) for generative VQA.

Early fusion with shared self-attention but modality-specific FFNs.
Image patches and text tokens are interleaved in a single sequence.
Attention is shared (cross-modal), FFN is split (vision FFN vs text FFN).

Reference: "Mixture-of-Transformers: A Sparse and Scalable Architecture
for Multi-Modal Foundation Models" (Meta, 2024)
"""

from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms


def _tokenize(text: str) -> list[str]:
    """Simple word tokenizer for CLEVR questions."""
    text = text.lower().strip().rstrip("?").strip()
    return re.findall(r"[a-z0-9]+", text)


def build_clevr_mot_vocab(data_root: str | Path) -> dict[str, int]:
    """Build word-level vocab from CLEVR training questions + answers."""
    import json
    from data.clevr import CLEVR_ANSWERS

    vocab = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<sep>": 3, "<unk>": 4}
    for a in CLEVR_ANSWERS:
        if a not in vocab:
            vocab[a] = len(vocab)

    q_path = Path(data_root) / "questions" / "CLEVR_train_questions.json"
    with open(q_path) as f:
        q_data = json.load(f)

    word_counter = Counter()
    for q in q_data["questions"]:
        words = _tokenize(q["question"])
        word_counter.update(words)

    for word in sorted(word_counter.keys()):
        if word not in vocab:
            vocab[word] = len(vocab)

    return vocab


class FFN(nn.Module):
    """Simple feedforward network with GELU."""

    def __init__(self, dim: int, mult: int = 4):
        super().__init__()
        inner = dim * mult
        self.net = nn.Sequential(
            nn.Linear(dim, inner),
            nn.GELU(),
            nn.Linear(inner, dim),
        )

    def forward(self, x):
        return self.net(x)


class MoTLayer(nn.Module):
    """Mixture-of-Transformers layer: shared attention + modality-specific FFN."""

    def __init__(self, dim: int, nhead: int, ff_mult: int = 4, dropout: float = 0.0):
        super().__init__()
        # Shared self-attention
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, nhead, dropout=dropout, batch_first=True)

        # Vision-specific FFN
        self.norm2_v = nn.LayerNorm(dim)
        self.ffn_v = FFN(dim, ff_mult)

        # Text-specific FFN
        self.norm2_t = nn.LayerNorm(dim)
        self.ffn_t = FFN(dim, ff_mult)

    def forward(self, x, n_vision, attn_mask=None):
        """Forward pass.

        Args:
            x: (B, S, D) full sequence [vision_tokens | text_tokens]
            n_vision: number of vision tokens (constant per batch)
            attn_mask: (S, S) causal mask for text, bidirectional for vision
        """
        # Shared self-attention (pre-norm)
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, attn_mask=attn_mask)
        x = x + attn_out

        # Split by modality for FFN
        v = x[:, :n_vision, :]  # vision tokens
        t = x[:, n_vision:, :]  # text tokens

        v = v + self.ffn_v(self.norm2_v(v))
        t = t + self.ffn_t(self.norm2_t(t))

        return torch.cat([v, t], dim=1)


class MoTVQAModel(nn.Module):
    """Mixture-of-Transformers for CLEVR VQA.

    Sequence: [image_patches] [question_words] [<sep>] [answer] [<eos>]
    - Image patches: bidirectional attention (can attend to each other freely)
    - Text tokens: causal attention for generation (attend to all image + previous text)
    - Loss: CE on answer+eos tokens only
    """

    def __init__(
        self,
        vocab: dict[str, int],
        resolution: int = 224,
        patch_size: int = 16,
        dim: int = 512,
        depth: int = 12,
        heads: int = 8,
        ff_mult: int = 4,
    ):
        super().__init__()
        self.vocab = vocab
        self.inv_vocab = {v: k for k, v in vocab.items()}
        self.resolution = resolution
        self.patch_size = patch_size
        self.num_patches = (resolution // patch_size) ** 2

        # Image patch embedding
        self.patch_embed = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)
        self.patch_norm = nn.LayerNorm(dim)

        # Text embedding
        self.text_embed = nn.Embedding(len(vocab), dim)

        # Positional embeddings (separate for vision and text)
        self.vision_pos = nn.Parameter(torch.randn(1, self.num_patches, dim) * 0.02)
        max_text_len = 64  # CLEVR questions ~20 words + answer
        self.text_pos = nn.Parameter(torch.randn(1, max_text_len, dim) * 0.02)

        # MoT layers
        self.layers = nn.ModuleList([
            MoTLayer(dim, heads, ff_mult) for _ in range(depth)
        ])
        self.final_norm = nn.LayerNorm(dim)

        # Output head (text tokens only)
        self.output_proj = nn.Linear(dim, len(vocab))

        # Image normalization
        self.register_buffer("img_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("img_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        self.text_cache = None  # compatibility

    def set_text_cache(self, text_cache):
        pass

    def get_transforms(self):
        return transforms.Compose([
            transforms.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            transforms.Resize((self.resolution, self.resolution),
                              interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
        ])

    def _embed_images(self, images):
        """Images → patch embeddings. (B, 3, H, W) → (B, N, D)"""
        images = (images - self.img_mean.to(images.device)) / self.img_std.to(images.device)
        patches = self.patch_embed(images)  # (B, D, H/p, W/p)
        B, D, H, W = patches.shape
        patches = patches.flatten(2).transpose(1, 2)  # (B, N, D)
        patches = self.patch_norm(patches)
        patches = patches + self.vision_pos[:, :patches.size(1), :]
        return patches

    def _embed_text(self, token_ids):
        """Token IDs → text embeddings. (B, T) → (B, T, D)"""
        emb = self.text_embed(token_ids)
        T = token_ids.size(1)
        emb = emb + self.text_pos[:, :T, :]
        return emb

    def _build_attn_mask(self, n_vision, n_text, device):
        """Build attention mask: bidirectional for vision, causal for text.

        Vision tokens attend to all vision tokens freely.
        Text tokens attend to all vision tokens + causally to previous text.
        """
        S = n_vision + n_text
        # Start with all-allowed
        mask = torch.zeros(S, S, device=device)
        neg_inf = torch.finfo(mask.dtype).min

        # Text-to-text: causal (upper triangle blocked)
        text_causal = torch.triu(
            torch.full((n_text, n_text), neg_inf, device=device),
            diagonal=1,
        )
        mask[n_vision:, n_vision:] = text_causal

        # Vision-to-text: vision cannot attend to future text
        # Actually vision is bidirectional among itself and doesn't attend to text at all
        # Wait — in early fusion, vision tokens are part of the prefix context.
        # Let's make it: vision tokens are bidirectional, text tokens are causal
        # Vision can see all vision, text can see all vision + causal text
        # Vision does NOT see text (vision is "prefix")
        mask[:n_vision, n_vision:] = neg_inf

        return mask

    def _encode_questions(self, questions):
        """Tokenize questions to IDs."""
        unk = self.vocab["<unk>"]
        q_ids = []
        for q in questions:
            words = _tokenize(q)
            ids = [self.vocab.get(w, unk) for w in words]
            q_ids.append(ids)
        return q_ids

    def forward(self, images, questions, answer_ids):
        """Forward pass.

        Args:
            images: (B, 3, H, W)
            questions: list[str]
            answer_ids: (B, 4) [bos, answer, eos, pad]

        Returns:
            logits: (B, T_answer, vocab_size) for CE loss on answer tokens
        """
        B = images.size(0)
        device = images.device

        # Embed images
        v_emb = self._embed_images(images)  # (B, N, D)
        n_vision = v_emb.size(1)

        # Build text sequences: [question_words] [<sep>] [answer] [<eos>]
        sep_id = self.vocab["<sep>"]
        q_ids_list = self._encode_questions(questions)

        # Pad text to same length
        text_seqs = []
        for i in range(B):
            ans_token = answer_ids[i, 1].item()
            eos_token = self.vocab["<eos>"]
            seq = q_ids_list[i] + [sep_id, ans_token, eos_token]
            text_seqs.append(seq)

        max_tlen = max(len(s) for s in text_seqs)
        pad_id = self.vocab["<pad>"]
        text_tensor = torch.full((B, max_tlen), pad_id, dtype=torch.long, device=device)
        for i, seq in enumerate(text_seqs):
            text_tensor[i, :len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)

        t_emb = self._embed_text(text_tensor)  # (B, T, D)
        n_text = t_emb.size(1)

        # Concatenate: [vision | text]
        x = torch.cat([v_emb, t_emb], dim=1)  # (B, N+T, D)

        # Attention mask
        attn_mask = self._build_attn_mask(n_vision, n_text, device)

        # Forward through MoT layers
        for layer in self.layers:
            x = layer(x, n_vision, attn_mask=attn_mask)

        x = self.final_norm(x)

        # Get text logits only
        text_out = x[:, n_vision:, :]  # (B, T, D)
        logits = self.output_proj(text_out)  # (B, T, vocab_size)

        # We want logits for predicting the NEXT token (shift by 1)
        # Input: [q0, q1, ..., qN, <sep>, answer, <eos>]
        # Target: [q1, q2, ..., qN, <sep>, answer, <eos>, <pad>]
        # But we only care about answer+eos prediction
        # Return logits[:, :-1] so trainer can compute CE with targets = text_tensor[:, 1:]
        return logits[:, :-1, :]

    @torch.no_grad()
    def generate(self, images, questions, max_len=4):
        """Generate answers autoregressively."""
        B = images.size(0)
        device = images.device

        v_emb = self._embed_images(images)
        n_vision = v_emb.size(1)
        q_ids_list = self._encode_questions(questions)

        sep_id = self.vocab["<sep>"]
        eos_id = self.vocab["<eos>"]

        results = []
        for i in range(B):
            # Build initial text: question + <sep>
            text_ids = q_ids_list[i] + [sep_id]
            text_tensor = torch.tensor(text_ids, dtype=torch.long, device=device).unsqueeze(0)

            for _ in range(max_len):
                t_emb = self._embed_text(text_tensor)
                n_text = t_emb.size(1)
                x = torch.cat([v_emb[i:i+1], t_emb], dim=1)
                attn_mask = self._build_attn_mask(n_vision, n_text, device)

                for layer in self.layers:
                    x = layer(x, n_vision, attn_mask=attn_mask)

                x = self.final_norm(x)
                last_logits = self.output_proj(x[:, -1, :])  # (1, vocab_size)
                next_token = last_logits.argmax(dim=-1)  # (1,)

                text_tensor = torch.cat([text_tensor, next_token.unsqueeze(0)], dim=1)

                if next_token.item() == eos_id:
                    break

            # Decode: tokens after <sep>
            gen = text_tensor.squeeze(0)[len(q_ids_list[i]) + 1:]
            words = []
            for t in gen:
                w = self.inv_vocab.get(t.item(), "")
                if w == "<eos>":
                    break
                if w not in ("<pad>", "<bos>", "<sep>", "<unk>"):
                    words.append(w)
            results.append(" ".join(words) if words else "<unk>")

        return results


def build_mot_model(cfg) -> MoTVQAModel:
    """Build MoT VQA from Hydra config."""
    mot_cfg = cfg.task.get("mot", {})
    vocab = build_clevr_mot_vocab(cfg.data.root)

    model = MoTVQAModel(
        vocab=vocab,
        resolution=cfg.model.get("resolution", 224),
        patch_size=mot_cfg.get("patch_size", 16),
        dim=mot_cfg.get("dim", 512),
        depth=mot_cfg.get("depth", 12),
        heads=mot_cfg.get("heads", 8),
        ff_mult=mot_cfg.get("ff_mult", 4),
    )
    return model
