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


# Large backbones added to the matrix (I-JEPA ViT-H/14, DINOv2 ViT-g/14). These
# are NOT 12-block ViT-B, so they exercise whether the ViTBackbone monkeypatch
# generalizes: embed_dim auto-derived from the trunk, GCA attached at
# depth-fraction-matched indices, feature_pool consistent with prefix tokens.
# (name, cross_attn_layers, resolution, feature_pool, embed_dim, has_cls)
LARGE_BACKBONES = [
    ("vit_huge_patch14_gap_224.in1k_ijepa", [3, 8, 14, 20, 25, 31], 224, "mean", 1280, False),
    ("vit_giant_patch14_dinov2.lvd142m", [4, 11, 18, 25, 32, 39], 336, "cls", 1536, True),
]


@pytest.mark.parametrize(
    "name,layers,res,pool,dim,has_cls", LARGE_BACKBONES,
    ids=[b[0].split(".")[0] for b in LARGE_BACKBONES])
def test_large_backbone_build_and_gca(name, layers, res, pool, dim, has_cls):
    from model import CrossAttnViT

    dev = torch.device("cuda") if torch.cuda.is_available() else None
    model = CrossAttnViT.from_config(
        name, device=dev, cross_attn_layers=layers,
        resolution=res, feature_aggregation=pool, pretrained=False)
    trunk = model.vision_model.trunk
    # embed dim is auto-derived from the timm trunk, never configured
    assert trunk.embed_dim == dim
    # GCA attached at exactly the requested block indices, nowhere else
    attached = [i for i, blk in enumerate(trunk.blocks) if blk.gated_cross_attn is not None]
    assert attached == layers
    # feature_pool must match token layout: cls needs a prefix token, mean does not
    assert (trunk.num_prefix_tokens > 0) == has_cls

    x = torch.randn(2, 3, res, res, device=dev)
    with torch.no_grad():
        feats = model.forward(x, None)  # unconditioned: threads all blocks, GCA idle
    assert feats.ndim == 3 and feats.shape[0] == 2 and feats.shape[-1] == dim
    assert torch.isfinite(feats).all()

    # conditioned forward drives GCA at the new embed dim; needs the text encoder
    # weights (roberta-large). Skip that leg if they aren't cached offline.
    try:
        with torch.no_grad():
            cfeats = model.forward(x, ["what color is the large cube", "how many spheres"])
    except (OSError, EnvironmentError) as e:
        pytest.skip(f"text encoder weights unavailable offline: {e}")
    assert cfeats.shape == feats.shape and torch.isfinite(cfeats).all()


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
