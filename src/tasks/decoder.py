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

    def forward(self, tgt, memory, text_feats=None, tgt_mask=None):
        out = self.base_layer(tgt, memory, tgt_mask=tgt_mask)
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

    def _run_layers(self, tgt, memory, text_feats=None, tgt_mask=None):
        out = tgt
        t_proj = None
        if text_feats is not None and self.text_proj is not None:
            t_proj = self.text_proj(text_feats)
        for layer in self.layers:
            out = layer(out, memory, text_feats=t_proj, tgt_mask=tgt_mask)
        return out

    def forward(self, tgt_ids, visual_patches, text_feats=None, tgt_mask=None):
        B, T = tgt_ids.shape
        positions = torch.arange(T, device=tgt_ids.device)
        tgt = self.token_embedding(tgt_ids) + self.pos_embedding(positions)
        memory = self.visual_proj(visual_patches)
        if tgt_mask is None:
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(T, device=tgt_ids.device)
        out = self._run_layers(tgt, memory, text_feats=text_feats, tgt_mask=tgt_mask)
        return self.output_proj(out)

    @torch.no_grad()
    def generate(self, visual_patches, bos_id, eos_id, text_feats=None, max_len=8):
        B = visual_patches.size(0)
        memory = self.visual_proj(visual_patches)
        generated = torch.full((B, 1), bos_id, dtype=torch.long, device=visual_patches.device)
        for _ in range(max_len):
            T = generated.size(1)
            positions = torch.arange(T, device=generated.device)
            tgt = self.token_embedding(generated) + self.pos_embedding(positions)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(T, device=generated.device)
            out = self._run_layers(tgt, memory, text_feats=text_feats, tgt_mask=tgt_mask)
            next_logits = self.output_proj(out[:, -1, :])
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

    model = DecoderModel(steervit, decoder, vocab,
                         use_steering=cfg.model.get("use_steering", True))

    if cfg.model.get("unfreeze_gca", True):
        for blk in steervit.vision_model.trunk.blocks:
            gca = getattr(blk, "gated_cross_attn", None)
            if gca is not None:
                for p in gca.parameters():
                    p.requires_grad = True
    if cfg.model.get("unfreeze_connector", True):
        for p in steervit.connector.parameters():
            p.requires_grad = True

    return model
