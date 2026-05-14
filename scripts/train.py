"""Unified training entry point.

Usage:
    python scripts/train.py +experiment=clevr_cls
    python scripts/train.py +experiment=clevr_decoder training.lr=5e-5
    python scripts/train.py +experiment=gqa_cls training.epochs=30
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import hydra
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch.amp import GradScaler
from torch.utils.data import DataLoader

# Add src/ to path for local imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from steervit import SteerViT
from trainer import train_one_epoch, get_scheduler
from evaluator import evaluate_classification, evaluate_decoder, format_results

try:
    import wandb
except ImportError:
    wandb = None


def load_steervit(cfg: DictConfig, device: torch.device) -> SteerViT:
    """Load SteerViT from config."""
    cross_attn_layers = cfg.model.get("cross_attn_layers", None)
    if cross_attn_layers is not None:
        cross_attn_layers = list(cross_attn_layers)
        backbone = cfg.model.backbone_name
        resolution = cfg.model.get("resolution", 224)
        print(f"SteerViT from_config: {backbone}, layers={cross_attn_layers}, res={resolution}", flush=True)
        steervit = SteerViT.from_config(backbone, device=device,
                                         cross_attn_layers=cross_attn_layers,
                                         resolution=resolution)
    else:
        checkpoint = cfg.model.checkpoint
        print(f"SteerViT from_pretrained: {checkpoint}", flush=True)
        steervit = SteerViT.from_pretrained(checkpoint, device=device)
    return steervit


def build_model(steervit: SteerViT, cfg: DictConfig):
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


def build_dataloaders(cfg: DictConfig, transform):
    """Build train and val dataloaders from config."""
    dataset_name = cfg.data.dataset

    if dataset_name == "clevr":
        from data.clevr import CLEVRVQADataset, clevr_collate_fn
        use_oracle = cfg.model.get("use_oracle", False)
        train_dataset = CLEVRVQADataset(
            cfg.data.root, "train", transform, use_oracle=use_oracle,
            max_samples=cfg.data.get("max_train_samples"),
        )
        val_dataset = CLEVRVQADataset(
            cfg.data.root, "val", transform, use_oracle=use_oracle,
            max_samples=cfg.data.get("max_val_samples"),
        )
        collate_fn = clevr_collate_fn

    elif dataset_name == "gqa":
        from data.gqa import GQAClsDataset, gqa_cls_collate_fn
        train_dataset = GQAClsDataset(
            cfg.data.root, "train_balanced", transform,
            max_answers=cfg.task.get("num_answers", 1500),
            max_samples=cfg.data.get("max_train_samples"),
        )
        val_dataset = GQAClsDataset(
            cfg.data.root, "val_balanced", transform,
            answer_vocab=train_dataset.answer_vocab,
            max_samples=cfg.data.get("max_val_samples"),
        )
        collate_fn = gqa_cls_collate_fn
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    num_workers = cfg.data.get("num_workers", 8)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.data.get("val_batch_size", cfg.data.batch_size),
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
    )

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}", flush=True)
    return train_loader, val_loader


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    print(f"Config:\n{OmegaConf.to_yaml(cfg)}", flush=True)

    # Output dir (Hydra auto-creates one, but we also save explicitly)
    output_dir = Path(hydra.utils.get_original_cwd()) / "outputs" / (cfg.wandb.get("name") or "default")
    output_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, output_dir / "config.yaml")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    # Seed
    seed = cfg.get("seed", 42)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # WandB
    if wandb is not None and cfg.wandb.get("enabled", True):
        wandb.init(
            project=cfg.wandb.project,
            name=cfg.wandb.get("name"),
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    # Build model
    steervit = load_steervit(cfg, device)
    transform = steervit.get_transforms()
    model = build_model(steervit, cfg)
    model = model.to(device)

    # Lazy text encoding cache (saves ~23% of forward time from epoch 2+)
    if cfg.get("text_cache", True):
        from text_cache import TextCache
        text_cache = TextCache(steervit, max_size=cfg.get("text_cache_size", 200_000))
        raw = model.module if hasattr(model, "module") else model
        raw.set_text_cache(text_cache)
        print("Text cache enabled", flush=True)

    # torch.compile (optional, ~19% speedup on ViT forward)
    if cfg.get("compile", False):
        steervit.vision_model = torch.compile(steervit.vision_model, mode="reduce-overhead")
        print("torch.compile enabled on vision_model", flush=True)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)", flush=True)

    # Data
    train_loader, val_loader = build_dataloaders(cfg, transform)

    # Optimizer & scheduler
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=cfg.training.lr,
        weight_decay=cfg.training.get("weight_decay", 0.05),
    )
    scheduler = get_scheduler(optimizer, cfg)

    # Loss
    task_type = cfg.task.type
    vocab = None
    if task_type == "decoder":
        raw = model.module if hasattr(model, "module") else model
        vocab = raw.vocab
        criterion = nn.CrossEntropyLoss(ignore_index=vocab["<pad>"])
    else:
        criterion = nn.CrossEntropyLoss()

    scaler = GradScaler("cuda", enabled=cfg.training.get("mixed_precision", "bf16") != "none")

    # Training loop
    best_acc = 0.0
    global_step = 0
    save_every = cfg.training.get("save_every", 5)

    for epoch in range(cfg.training.epochs):
        t0 = time.time()

        train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler,
            device, cfg, epoch,
            task_type=task_type, vocab=vocab, global_step=global_step,
        )
        global_step = train_metrics["global_step"]
        scheduler.step()

        elapsed = time.time() - t0
        cache_info = ""
        if cfg.get("text_cache", True) and text_cache.cache_size > 0:
            cache_info = f" | Cache: {text_cache.cache_size:,} keys, {text_cache.hit_rate:.0%} hit"
        print(f"Epoch {epoch} | Loss: {train_metrics['train_loss']:.4f} | "
              f"Acc: {train_metrics['train_acc']:.4f} | Time: {elapsed:.1f}s{cache_info}", flush=True)

        # Save last.pt BEFORE validation (protect against eval interruption)
        ckpt_data = {
            "epoch": epoch,
            "model_state_dict": {
                k: v for k, v in model.state_dict().items()
                if any(p.data_ptr() == v.data_ptr()
                       for p in model.parameters() if p.requires_grad)
            },
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "best_acc": best_acc,
            "config": OmegaConf.to_container(cfg, resolve=True),
        }
        torch.save(ckpt_data, output_dir / "last.pt")

        # Validate
        if task_type == "decoder":
            val_results = evaluate_decoder(model, val_loader, device, vocab)
        else:
            val_results = evaluate_classification(model, val_loader, device)

        val_acc = val_results["accuracy"]
        print(f"Epoch {epoch} | Val acc: {val_acc:.4f}", flush=True)
        print(format_results(val_results), flush=True)

        # Update last.pt with val results + save best/periodic
        ckpt_data["val_acc"] = val_acc
        torch.save(ckpt_data, output_dir / "last.pt")

        if val_acc > best_acc:
            best_acc = val_acc
            ckpt_data["best_acc"] = best_acc
            torch.save(ckpt_data, output_dir / "best.pt")
            print(f"  New best: {best_acc:.4f}", flush=True)

        if (epoch + 1) % save_every == 0:
            torch.save(ckpt_data, output_dir / f"epoch_{epoch}.pt")

        # WandB epoch logging
        if wandb is not None and wandb.run is not None:
            log_data = {
                "train/loss_epoch": train_metrics["train_loss"],
                "train/acc_epoch": train_metrics["train_acc"],
                "val/acc": val_acc,
                "epoch": epoch,
            }
            for k, v in val_results.get("breakdown", {}).items():
                log_data[f"val/acc_{k}"] = v["accuracy"]
            wandb.log(log_data, step=global_step)

        # JSON log
        log_entry = {
            **{k: v for k, v in train_metrics.items() if k != "global_step"},
            "val_acc": val_acc,
            "epoch": epoch,
        }
        with open(output_dir / "train_log.jsonl", "a") as f:
            f.write(json.dumps(log_entry) + "\n")

    print(f"\nDone. Best val acc: {best_acc:.4f}", flush=True)
    if wandb is not None and wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
