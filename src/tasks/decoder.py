"""Decoder task: CoCa-style transformer decoder for generative VQA."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from model.crossattention import GatedCrossAttention


class VQADecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, use_text_gca=False):
        super().__init__()
        self.base_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward, batch_first=True,
        )
        self.text_gca = None
        if use_text_gca:
            self.text_gca = GatedCrossAttention(
                layer_idx=-1, dim=d_model,
                ff_mult=2, use_ffn=False,
                head_dim=d_model // nhead, num_heads=nhead,
            )

    def forward(self, tgt, memory, text_feats=None, tgt_mask=None, memory_key_padding_mask=None):
        out = self.base_layer(tgt, memory, tgt_mask=tgt_mask, memory_key_padding_mask=memory_key_padding_mask)
        if self.text_gca is not None and text_feats is not None:
            out = self.text_gca(out, text_feats, attn_mask=None)
        return out


class VQADecoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        visual_dim: int = 768,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 2,
        max_len: int = 8,
        use_text_gca: bool = False,
        text_dim: int = 1024,
    ):
        super().__init__()
        self.d_model = d_model
        self.use_text_gca = use_text_gca
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_len, d_model)
        self.visual_proj = nn.Linear(visual_dim, d_model)
        self.layers = nn.ModuleList([
            VQADecoderLayer(d_model, nhead, d_model * 4, use_text_gca=use_text_gca)
            for _ in range(num_layers)
        ])
        self.output_proj = nn.Linear(d_model, vocab_size)
        self.text_proj = nn.Linear(text_dim, d_model) if use_text_gca else None

    def _run_layers(self, tgt, memory, text_feats=None, tgt_mask=None, memory_key_padding_mask=None):
        out = tgt
        t_proj = None
        if text_feats is not None and self.text_proj is not None:
            t_proj = self.text_proj(text_feats)
        for layer in self.layers:
            out = layer(out, memory, text_feats=t_proj, tgt_mask=tgt_mask,
                        memory_key_padding_mask=memory_key_padding_mask)
        return out

    def forward(self, tgt_ids, visual_patches, text_feats=None, tgt_mask=None, memory_key_padding_mask=None):
        B, T = tgt_ids.shape
        positions = torch.arange(T, device=tgt_ids.device)
        tgt = self.token_embedding(tgt_ids) + self.pos_embedding(positions)
        memory = self.visual_proj(visual_patches)
        if tgt_mask is None:
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(T, device=tgt_ids.device)
        out = self._run_layers(tgt, memory, text_feats=text_feats, tgt_mask=tgt_mask,
                               memory_key_padding_mask=memory_key_padding_mask)
        return self.output_proj(out)

    @torch.no_grad()
    def generate(self, visual_patches, bos_id, eos_id, text_feats=None, max_len=8, memory_key_padding_mask=None):
        B = visual_patches.size(0)
        memory = self.visual_proj(visual_patches)
        generated = torch.full((B, 1), bos_id, dtype=torch.long, device=visual_patches.device)
        for _ in range(max_len):
            T = generated.size(1)
            positions = torch.arange(T, device=generated.device)
            tgt = self.token_embedding(generated) + self.pos_embedding(positions)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(T, device=generated.device)
            out = self._run_layers(tgt, memory, text_feats=text_feats, tgt_mask=tgt_mask,
                                   memory_key_padding_mask=memory_key_padding_mask)
            next_logits = self.output_proj(out[:, -1, :])
            next_token = next_logits.argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            if (next_token == eos_id).all():
                break
        return generated[:, 1:]


class ConcatSelfAttnDecoder(nn.Module):
    """Concatenated self-attention decoder (prefix-LM style).

    Concatenates [vis_patches, text_tokens, answer_tokens] into one sequence
    and processes with self-attention layers using a prefix-LM mask.
    """

    def __init__(
        self,
        vocab_size: int,
        visual_dim: int = 768,
        text_dim: int = 768,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 2,
        max_vis_tokens: int = 577,
        max_text_tokens: int = 64,
        max_ans_len: int = 8,
    ):
        super().__init__()
        self.d_model = d_model
        self.use_text_gca = True  # signals DecoderModel._encode to provide text_feats

        self.visual_proj = nn.Linear(visual_dim, d_model)
        self.text_proj = nn.Linear(text_dim, d_model)
        self.token_embedding = nn.Embedding(vocab_size, d_model)

        self.vis_pos_embed = nn.Parameter(torch.randn(1, max_vis_tokens, d_model) * 0.02)
        self.text_pos_embed = nn.Parameter(torch.randn(1, max_text_tokens, d_model) * 0.02)
        self.ans_pos_embed = nn.Embedding(max_ans_len, d_model)
        self.modality_embed = nn.Embedding(3, d_model)

        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead,
                dim_feedforward=d_model * 4, batch_first=True,
            )
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, vocab_size)

    @staticmethod
    def _build_prefix_lm_mask(n_prefix, n_ans, device):
        """Prefix-LM mask: prefix bidirectional, answer causal, prefix blocked from answer."""
        S = n_prefix + n_ans
        mask = torch.zeros(S, S, device=device)
        neg_inf = torch.finfo(mask.dtype).min
        # Prefix cannot attend to answer tokens
        mask[:n_prefix, n_prefix:] = neg_inf
        # Answer tokens: causal among themselves
        if n_ans > 0:
            ans_causal = torch.triu(
                torch.full((n_ans, n_ans), neg_inf, device=device), diagonal=1,
            )
            mask[n_prefix:, n_prefix:] = ans_causal
        return mask

    def _embed(self, visual_patches, text_feats, ans_ids=None):
        """Embed and concatenate all modalities. Returns (seq, n_vis, n_text, n_ans)."""
        B = visual_patches.size(0)
        device = visual_patches.device
        n_vis = visual_patches.size(1)
        n_text = text_feats.size(1)

        vis_emb = self.visual_proj(visual_patches) + self.vis_pos_embed[:, :n_vis] + self.modality_embed(torch.zeros(1, dtype=torch.long, device=device))
        text_emb = self.text_proj(text_feats) + self.text_pos_embed[:, :n_text] + self.modality_embed(torch.ones(1, dtype=torch.long, device=device))

        if ans_ids is not None:
            T = ans_ids.size(1)
            positions = torch.arange(T, device=device)
            ans_emb = self.token_embedding(ans_ids) + self.ans_pos_embed(positions) + self.modality_embed(torch.full((1,), 2, dtype=torch.long, device=device))
            seq = torch.cat([vis_emb, text_emb, ans_emb], dim=1)
            return seq, n_vis, n_text, T
        else:
            seq = torch.cat([vis_emb, text_emb], dim=1)
            return seq, n_vis, n_text, 0

    def forward(self, tgt_ids, visual_patches, text_feats=None, tgt_mask=None):
        seq, n_vis, n_text, n_ans = self._embed(visual_patches, text_feats, tgt_ids)
        n_prefix = n_vis + n_text
        mask = self._build_prefix_lm_mask(n_prefix, n_ans, seq.device)
        for layer in self.layers:
            seq = layer(seq, src_mask=mask)
        ans_out = self.norm(seq[:, n_prefix:, :])
        return self.output_proj(ans_out)

    @torch.no_grad()
    def generate(self, visual_patches, bos_id, eos_id, text_feats=None, max_len=8):
        B = visual_patches.size(0)
        device = visual_patches.device
        # Build prefix once
        prefix, n_vis, n_text, _ = self._embed(visual_patches, text_feats)
        n_prefix = n_vis + n_text

        generated = torch.full((B, 1), bos_id, dtype=torch.long, device=device)
        for _ in range(max_len):
            T = generated.size(1)
            positions = torch.arange(T, device=device)
            ans_emb = self.token_embedding(generated) + self.ans_pos_embed(positions) + self.modality_embed(torch.full((1,), 2, dtype=torch.long, device=device))
            seq = torch.cat([prefix, ans_emb], dim=1)
            mask = self._build_prefix_lm_mask(n_prefix, T, device)
            for layer in self.layers:
                seq = layer(seq, src_mask=mask)
            next_logits = self.output_proj(self.norm(seq[:, -1, :]))
            next_token = next_logits.argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            if (next_token == eos_id).all():
                break
        return generated[:, 1:]


class DecoderModel(nn.Module):
    """CrossAttnViT + transformer decoder for generative VQA."""

    def __init__(self, steervit, decoder, vocab, use_steering=True):
        super().__init__()
        self.steervit = steervit
        self.decoder = decoder
        self.vocab = vocab
        self.inv_vocab = {v: k for k, v in vocab.items()}
        self.use_steering = use_steering
        self.text_cache = None  # set via set_text_cache()

    def set_text_cache(self, text_cache):
        """Attach a TextCache for lazy RoBERTa caching."""
        self.text_cache = text_cache

    def _encode(self, images, questions):
        prefix = self.steervit.vision_model.trunk.num_prefix_tokens

        if questions is not None and self.use_steering and self.text_cache is not None:
            text_feats_gca, attn_mask, _ = self.text_cache.encode_text(questions)
            feats = self.steervit.forward_with_conditioning(images, text_feats_gca, attn_mask)
        else:
            steer_q = questions if self.use_steering else None
            feats = self.steervit.forward(images, steer_q)

        patches = feats[:, prefix:, :]

        text_feats = None
        if self.decoder.use_text_gca and questions is not None:
            if self.text_cache is not None:
                text_feats, _, _ = self.text_cache.encode_text(questions)
            else:
                text_feats, _, _ = self.steervit.encode_text(questions)
            text_feats = text_feats.detach()

        return patches, text_feats

    def forward(self, images, questions, answer_ids):
        patches, text_feats = self._encode(images, questions)
        logits = self.decoder(answer_ids[:, :-1], patches, text_feats=text_feats)
        return logits

    @torch.no_grad()
    def generate(self, images, questions, max_len=4):
        patches, text_feats = self._encode(images, questions)
        token_ids = self.decoder.generate(
            patches, bos_id=self.vocab["<bos>"], eos_id=self.vocab["<eos>"],
            text_feats=text_feats, max_len=max_len,
        )
        results = []
        for seq in token_ids:
            words = []
            for t in seq:
                w = self.inv_vocab.get(t.item(), "")
                if w == "<eos>":
                    break
                if w not in ("<bos>", "<pad>"):
                    words.append(w)
            results.append(" ".join(words))
        return results


class MirrorDecoderModel(DecoderModel):
    """Mirror variant: vanilla frozen ViT patches are the GCA key/value inside RoBERTa;
    the connector-projected RoBERTa hidden states are the decoder memory."""

    def set_text_cache(self, text_cache):
        """No-op: RoBERTa must run live (text_gca sits inside it)."""
        self.text_cache = None

    def _encode(self, images, questions):
        prefix = self.steervit.vision_model.trunk.num_prefix_tokens
        with torch.no_grad():
            feats = self.steervit.forward(images, None)
        patches = feats[:, prefix:, :]
        mem, mask = self.steervit.encode_text_mirror(questions, patches)
        return mem, None, ~mask

    def forward(self, images, questions, answer_ids):
        mem, text_feats, pad_mask = self._encode(images, questions)
        return self.decoder(answer_ids[:, :-1], mem, text_feats=text_feats,
                            memory_key_padding_mask=pad_mask)

    @torch.no_grad()
    def generate(self, images, questions, max_len=4):
        mem, text_feats, pad_mask = self._encode(images, questions)
        token_ids = self.decoder.generate(
            mem, bos_id=self.vocab["<bos>"], eos_id=self.vocab["<eos>"],
            text_feats=text_feats, max_len=max_len, memory_key_padding_mask=pad_mask,
        )
        results = []
        for seq in token_ids:
            words = []
            for t in seq:
                w = self.inv_vocab.get(t.item(), "")
                if w == "<eos>":
                    break
                if w not in ("<bos>", "<pad>"):
                    words.append(w)
            results.append(" ".join(words))
        return results


def build_clevr_decoder_vocab():
    from data.clevr import CLEVR_ANSWERS
    vocab = {"<bos>": 0, "<eos>": 1, "<pad>": 2}
    for i, a in enumerate(CLEVR_ANSWERS):
        vocab[a] = i + 3
    return vocab


def build_gqa_decoder_vocab(data_root, max_answers=1500):
    import json
    from collections import Counter
    from pathlib import Path
    q_path = Path(data_root) / "questions" / "train_balanced_questions.json"
    with open(q_path) as f:
        q_data = json.load(f)
    counter = Counter(q["answer"] for q in q_data.values())
    top_answers = [a for a, _ in counter.most_common(max_answers)]
    vocab = {"<bos>": 0, "<eos>": 1, "<pad>": 2}
    for i, a in enumerate(top_answers):
        vocab[a] = i + 3
    return vocab


def build_decoder_model(steervit, cfg) -> DecoderModel:
    """Build from Hydra config."""
    dataset = cfg.data.get("dataset", "clevr")
    if dataset == "gqa":
        max_answers = cfg.task.get("max_answers", 1500)
        vocab = build_gqa_decoder_vocab(cfg.data.root, max_answers=max_answers)
    else:
        vocab = build_clevr_decoder_vocab()

    dec_cfg = cfg.task.get("decoder", {})
    use_text_gca = cfg.task.get("use_text_gca", False)
    decoder_type = dec_cfg.get("type", "cross_attention")

    if decoder_type == "concat_self_attention":
        decoder = ConcatSelfAttnDecoder(
            vocab_size=len(vocab),
            visual_dim=steervit.visual_dim,
            text_dim=steervit.visual_dim,
            d_model=dec_cfg.get("d_model", 512),
            nhead=dec_cfg.get("nhead", 8),
            num_layers=dec_cfg.get("num_layers", 2),
            max_vis_tokens=dec_cfg.get("max_vis_tokens", 577),
            max_text_tokens=dec_cfg.get("max_text_tokens", 64),
            max_ans_len=dec_cfg.get("max_len", 8),
        )
    else:
        decoder = VQADecoder(
            vocab_size=len(vocab),
            visual_dim=steervit.visual_dim,
            d_model=dec_cfg.get("d_model", 512),
            nhead=dec_cfg.get("nhead", 8),
            num_layers=dec_cfg.get("num_layers", 2),
            max_len=dec_cfg.get("max_len", 8),
            use_text_gca=use_text_gca,
            text_dim=steervit.visual_dim if use_text_gca else 768,
        )

    model_cls = MirrorDecoderModel if cfg.model.get("mirror", False) else DecoderModel
    model = model_cls(steervit, decoder, vocab,
                      use_steering=cfg.model.get("use_steering", True))

    if cfg.model.get("unfreeze_gca", True):
        for blk in steervit.vision_model.trunk.blocks:
            gca = getattr(blk, "gated_cross_attn", None)
            if gca is not None:
                for p in gca.parameters():
                    p.requires_grad = True
        if getattr(steervit, "text_gca", None) is not None:
            for p in steervit.text_gca.parameters():
                p.requires_grad = True
    if cfg.model.get("unfreeze_connector", True):
        for p in steervit.connector.parameters():
            p.requires_grad = True
    if cfg.model.get("unfreeze_backbone", False):
        for p in steervit.vision_model.parameters():
            p.requires_grad = True

    return model
