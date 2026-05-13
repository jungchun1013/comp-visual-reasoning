"""Standalone evaluation entry point.

Usage:
    python scripts/evaluate.py +experiment=clevr_cls checkpoint=outputs/clevr_cls/best.pt
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

# Add src/ to path for local imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from steervit import SteerViT
from evaluator import evaluate_classification, evaluate_decoder, format_results


def load_steervit(cfg, device):
    """Load SteerViT from config."""
    cross_attn_layers = cfg.model.get("cross_attn_layers", None)
    if cross_attn_layers is not None:
        cross_attn_layers = list(cross_attn_layers)
        backbone = cfg.model.backbone_name
        resolution = cfg.model.get("resolution", 224)
        steervit = SteerViT.from_config(backbone, device=device,
                                         cross_attn_layers=cross_attn_layers,
                                         resolution=resolution)
    else:
        steervit = SteerViT.from_pretrained(cfg.model.checkpoint, device=device)
    return steervit


def build_model(steervit, cfg):
    """Build task-specific model from config."""
    task_type = cfg.task.type
    if task_type == "classification":
        from tasks.classification import build_classification_model
        return build_classification_model(steervit, cfg)
    elif task_type == "decoder":
        from tasks.decoder import build_decoder_model
        return build_decoder_model(steervit, cfg)
    else:
        raise ValueError(f"Unknown task type: {task_type}")


def build_val_loader(cfg, transform):
    """Build validation dataloader."""
    dataset_name = cfg.data.dataset

    if dataset_name == "clevr":
        from data.clevr import CLEVRVQADataset, clevr_collate_fn
        val_dataset = CLEVRVQADataset(
            cfg.data.root, "val", transform,
            use_oracle=cfg.model.get("use_oracle", False),
            max_samples=cfg.data.get("max_val_samples"),
        )
        collate_fn = clevr_collate_fn
    elif dataset_name == "gqa":
        from data.gqa import GQAClsDataset, gqa_cls_collate_fn
        val_dataset = GQAClsDataset(
            cfg.data.root, "val_balanced", transform,
            max_answers=cfg.task.get("num_answers", 1500),
            max_samples=cfg.data.get("max_val_samples"),
        )
        collate_fn = gqa_cls_collate_fn
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return DataLoader(
        val_dataset,
        batch_size=cfg.data.get("val_batch_size", cfg.data.batch_size),
        shuffle=False,
        num_workers=cfg.data.get("num_workers", 8),
        collate_fn=collate_fn,
        pin_memory=True,
    )


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    checkpoint_path = cfg.get("checkpoint")
    if checkpoint_path is None:
        raise ValueError("Must provide checkpoint=<path> on command line")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    steervit = load_steervit(cfg, device)
    transform = steervit.get_transforms()
    model = build_model(steervit, cfg)
    model = model.to(device)

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    print(f"Loaded checkpoint from {checkpoint_path} (epoch {ckpt.get('epoch', '?')})", flush=True)

    val_loader = build_val_loader(cfg, transform)

    task_type = cfg.task.type
    if task_type == "decoder":
        raw = model.module if hasattr(model, "module") else model
        results = evaluate_decoder(model, val_loader, device, raw.vocab)
    else:
        results = evaluate_classification(model, val_loader, device)

    print(format_results(results))

    output_path = Path("eval_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Saved: {output_path}", flush=True)


if __name__ == "__main__":
    main()
