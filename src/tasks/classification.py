"""Classification task: VQA head + model wrapper."""

from __future__ import annotations

import torch
import torch.nn as nn


class VQAHead(nn.Module):
    """MLP classifier over concatenated [visual_feat; question_feat]."""

    def __init__(
        self,
        visual_dim: int = 768,
        text_dim: int = 1024,
        num_answers: int = 28,
        hidden_dim: int = 1024,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.visual_proj = nn.Linear(visual_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_answers),
        )

    def forward(self, visual_feat: torch.Tensor, question_feat: torch.Tensor) -> torch.Tensor:
        v = self.visual_proj(visual_feat)
        q = self.text_proj(question_feat)
        combined = torch.cat([v, q], dim=-1)
        return self.classifier(combined)


class ClassificationModel(nn.Module):
    """CrossAttnViT + VQA classification head.

    Supports: steered, unsteered, text_only, oracle modes.
    """

    def __init__(
        self,
        steervit: nn.Module,
        vqa_head: VQAHead,
        use_steering: bool = True,
        feature_pool: str = "cls",
        text_only: bool = False,
    ):
        super().__init__()
        self.steervit = steervit
        self.vqa_head = vqa_head
        self.use_steering = use_steering
        self.feature_pool = feature_pool
        self.text_only = text_only
        self.text_cache = None  # set via set_text_cache()

    def set_text_cache(self, text_cache):
        """Attach a TextCache for lazy RoBERTa caching."""
        self.text_cache = text_cache

    def _get_question_encoding(self, questions: list[str]) -> torch.Tensor:
        if self.text_cache is not None:
            return self.text_cache.get_cls_tokens(questions)
        with torch.no_grad():
            roberta_dict = self.steervit.tokenizer(
                questions, padding=True, truncation=True, max_length=512, return_tensors="pt"
            )
            roberta_dict = {k: v.to(self.steervit.text_model.device) for k, v in roberta_dict.items()}
            text_out = self.steervit.text_model(**roberta_dict).last_hidden_state
            return text_out[:, 0, :]

    def _get_visual_features(self, images: torch.Tensor, prompts: list[str] | None) -> torch.Tensor:
        if self.text_only:
            return torch.zeros(images.size(0), self.steervit.visual_dim, device=images.device)

        if prompts is not None and self.use_steering and self.text_cache is not None:
            text_feats, attn_mask, _ = self.text_cache.encode_text(prompts)
            feats = self.steervit.forward_with_conditioning(images, text_feats, attn_mask)
        else:
            steering_text = prompts if self.use_steering else None
            feats = self.steervit.forward(images, steering_text)

        if self.feature_pool == "cls":
            return feats[:, 0, :]
        else:
            prefix = self.steervit.vision_model.trunk.num_prefix_tokens
            return feats[:, prefix:, :].mean(dim=1)

    def forward(self, images: torch.Tensor, questions: list[str],
                oracle_prompts: list[str] | None = None) -> torch.Tensor:
        q_encoding = self._get_question_encoding(questions)
        steering_prompts = oracle_prompts if oracle_prompts is not None else questions
        vis_features = self._get_visual_features(images, steering_prompts)
        return self.vqa_head(vis_features, q_encoding)


def build_classification_model(steervit, cfg) -> ClassificationModel:
    """Build from Hydra config (cfg.task + cfg.model sections)."""
    vqa_head = VQAHead(
        visual_dim=steervit.visual_dim,
        text_dim=steervit.text_dim,
        num_answers=cfg.task.num_answers,
        hidden_dim=cfg.task.get("hidden_dim", 1024),
        dropout=cfg.task.get("dropout", 0.3),
    )
    model = ClassificationModel(
        steervit=steervit,
        vqa_head=vqa_head,
        use_steering=cfg.model.get("use_steering", True),
        feature_pool=cfg.model.get("feature_pool", "cls"),
        text_only=cfg.model.get("text_only", False),
    )
    if cfg.model.get("unfreeze_gca", False):
        for blk in steervit.vision_model.trunk.blocks:
            gca = getattr(blk, "gated_cross_attn", None)
            if gca is not None:
                for p in gca.parameters():
                    p.requires_grad = True
    if cfg.model.get("unfreeze_connector", False):
        for p in steervit.connector.parameters():
            p.requires_grad = True
    return model
