"""Smoke tests: model construction + checkpoint loading (no dataset, no GPU needed).

Run from the repo root:
    PYTHONPATH=src python -m pytest tests/test_smoke.py -x -q

Two groups:
  1. Build CrossAttnViT from each backbone used in the paper matrix with
     pretrained=False (no downloads) and forward a random image batch with a
     dummy question through the text pipeline disabled path.
  2. If checkpoints exist under outputs/model/, load one per known format
     through src/model/checkpoint_io.load_any_checkpoint (skipped otherwise —
     e.g. on a fresh public clone).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

BACKBONES = [
    "vit_base_patch14_dinov2.lvd142m",
    "vit_base_patch16_siglip_224",
    "vit_base_patch16_224.augreg_in21k",
    "vit_base_patch16_224.mae",
]

REPO = Path(__file__).resolve().parents[1]
CKPTS = {
    "main-decoder": REPO / "outputs/model/clevr_dinov2_decoder1l_scratch_s42/best.pt",
    "main-cls": REPO / "outputs/model/clevr_dinov2_cls_scratch_s42/best.pt",
}


@pytest.mark.parametrize("backbone", BACKBONES)
def test_build_and_forward(backbone):
    from model import CrossAttnViT

    model = CrossAttnViT.from_config(
        backbone, device=None, cross_attn_layers=[1, 3, 5, 7, 9, 11],
        resolution=224, pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        feats = model.forward(x, None)  # no language conditioning
    assert feats.ndim == 3 and feats.shape[0] == 2
    assert torch.isfinite(feats).all()


@pytest.mark.parametrize("name", sorted(CKPTS))
def test_load_any_checkpoint(name):
    path = CKPTS[name]
    if not path.exists():
        pytest.skip(f"checkpoint not present: {path}")
    from model.checkpoint_io import load_any_checkpoint

    model, _sv, transform, _vocab, task_type, meta = \
        load_any_checkpoint(path, torch.device("cpu"))
    assert task_type in ("decoder", "classification", "mot")
    assert meta["val_acc"] is not None
    assert transform is not None
    assert sum(p.numel() for p in model.parameters()) > 0
