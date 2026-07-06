"""Unified checkpoint loading for main-format and legacy checkpoints.

Replaces the five near-copies of load_model/load_legacy_model that lived in
scripts/{eval_closure,eval_clevr_math,eval_legacy_humans,run_cogent_patching,
analysis/activation_patching}.py (L3 consolidation — see docs/legacy-reference.md §5).

Format detection: main-format checkpoints carry a "config" key (the full Hydra
config); legacy checkpoints (SteerViT-legacy exp_vqa trainers) do not — they hold
`steervit_trainable_state` + `decoder_state_dict` and hardcode DINOv2-B/14 @336,
GCA at [1,3,5,7,9,11], d_model 512, nhead 8 (num_layers inferred from keys).

Correctness notes vs the old copies:
  - `use_gate`, `condition_type`, `feature_aggregation` from the saved config are
    forwarded to CrossAttnViT.from_config. (The old eval_generalization loader
    dropped `use_gate`, silently rebuilding nogate checkpoints with zero-initialized
    gates — GCA output nulled.)
  - classification checkpoints build a ClassificationModel, not a DecoderModel.
"""
from __future__ import annotations

from pathlib import Path

import torch

LEGACY_BACKBONE = "vit_base_patch14_dinov2.lvd142m"
LEGACY_GCA_LAYERS = [1, 3, 5, 7, 9, 11]
LEGACY_RESOLUTION = 336


def is_legacy(ckpt: dict) -> bool:
    return "config" not in ckpt


def load_any_checkpoint(ckpt_path, device):
    """Load a checkpoint of any known format/task.

    Returns (model, steervit, transform, vocab, task_type, meta) where
      model      full task model (DecoderModel / ClassificationModel / MoTVQAModel), eval mode
      steervit   the CrossAttnViT backbone (None for MoT)
      transform  eval image transform
      vocab      answer vocab dict
      task_type  "decoder" | "classification" | "mot"
      meta       {"epoch": ..., "val_acc": ..., "best_acc": ..., "name": ..., "legacy": bool}
    """
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    meta = {
        "epoch": ckpt.get("epoch", "?"),
        "val_acc": ckpt.get("val_acc", ckpt.get("best_val_acc")),
        "best_acc": ckpt.get("best_acc"),
        "name": Path(ckpt_path).parent.name,
        "legacy": is_legacy(ckpt),
    }
    if is_legacy(ckpt):
        model, steervit, transform, vocab = _load_legacy(ckpt, device)
        return model, steervit, transform, vocab, "decoder", meta
    return (*_load_main(ckpt, device), meta)


def _load_legacy(ckpt, device):
    from model import CrossAttnViT
    from tasks.decoder import build_clevr_decoder_vocab, VQADecoder, DecoderModel

    steervit = CrossAttnViT.from_config(
        LEGACY_BACKBONE, device=device,
        cross_attn_layers=LEGACY_GCA_LAYERS, resolution=LEGACY_RESOLUTION)
    if "steervit_trainable_state" in ckpt:
        steervit.load_state_dict(ckpt["steervit_trainable_state"], strict=False)

    vocab = build_clevr_decoder_vocab()
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
    model = model.to(device)
    model.eval()
    return model, steervit, steervit.get_transforms(), vocab


def _load_main(ckpt, device):
    from omegaconf import OmegaConf

    cfg = OmegaConf.create(ckpt["config"])
    task_type = cfg.task.get("type", cfg.task.get("name", "decoder"))

    if task_type == "mot":
        from tasks.mot_vqa import MoTVQAModel, build_clevr_mot_vocab
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
        ).to(device)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        model.eval()
        return model, None, model.get_transforms(), vocab, task_type

    from model import CrossAttnViT

    fc_kwargs = {}
    for key in ("use_gate", "condition_type", "feature_aggregation"):
        if key in cfg.model:
            fc_kwargs[key] = cfg.model[key]
    steervit = CrossAttnViT.from_config(
        cfg.model.backbone_name, device=device,
        cross_attn_layers=list(cfg.model.cross_attn_layers),
        resolution=cfg.model.resolution,
        pretrained=cfg.model.get("pretrained", True),
        **fc_kwargs)

    model_cfg = OmegaConf.create(
        {"model": cfg.model, "task": cfg.task, "data": cfg.data})
    if task_type == "classification":
        from tasks.classification import build_classification_model
        model = build_classification_model(steervit, model_cfg)
        vocab = getattr(model, "vocab", None) or {}
    else:
        from tasks.decoder import build_decoder_model, build_clevr_decoder_vocab
        model = build_decoder_model(steervit, model_cfg)
        vocab = build_clevr_decoder_vocab()

    model = model.to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()
    return model, steervit, steervit.get_transforms(), vocab, task_type
