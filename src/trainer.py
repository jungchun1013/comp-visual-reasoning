"""Unified training loop with WandB and window-based accuracy logging."""

from __future__ import annotations

import logging
import time

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader

log = logging.getLogger(__name__)

try:
    import wandb
except ImportError:
    wandb = None


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scaler: GradScaler,
    device: torch.device,
    cfg: DictConfig,
    epoch: int,
    task_type: str = "classification",
    vocab: dict | None = None,
    global_step: int = 0,
) -> dict:
    """Train one epoch.

    Args:
        model: The model to train.
        dataloader: Training data loader.
        optimizer: Optimizer.
        criterion: Loss function.
        scaler: GradScaler for mixed precision.
        device: Device.
        cfg: Full Hydra config.
        epoch: Current epoch number.
        task_type: "classification" or "decoder".
        vocab: Token vocabulary (required for decoder task).
        global_step: Step counter across epochs (for WandB x-axis).

    Returns:
        Dict with train_loss, train_acc, and updated global_step.
    """
    model.train()
    # Keep frozen parts in eval mode
    raw = model.module if hasattr(model, "module") else model
    raw.steervit.vision_model.eval()
    raw.steervit.text_model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    total_tokens = 0  # for decoder token accuracy

    # Window accumulators for step-level logging (THE FIX)
    window_loss = 0.0
    window_correct = 0
    window_samples = 0
    window_tokens = 0

    grad_clip = cfg.training.get("grad_clip", 1.0)
    log_every = cfg.training.get("log_every", 100)
    use_amp = cfg.training.get("mixed_precision", "bf16") != "none"
    t_epoch_start = time.time()

    for step, batch in enumerate(dataloader):
        images = batch["image"].to(device)
        questions = batch["question"]
        answers = batch["answer"].to(device)

        optimizer.zero_grad()

        with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
            if task_type == "decoder":
                # Convert answer indices to decoder token sequences
                answer_ids = _answers_to_decoder_ids(answers, vocab)
                logits = model(images, questions, answer_ids)
                targets = answer_ids[:, 1:]
                loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            else:
                oracle_prompts = batch.get("oracle_prompt")
                logits = model(images, questions, oracle_prompts=oracle_prompts)
                loss = criterion(logits, answers)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            max_norm=grad_clip,
        )
        scaler.step(optimizer)
        scaler.update()

        # Compute accuracy
        batch_size = images.size(0)
        batch_loss = loss.item() * batch_size

        if task_type == "decoder":
            with torch.no_grad():
                preds = logits.argmax(dim=-1)
                mask = targets != vocab["<pad>"]
                step_correct = ((preds == targets) & mask).sum().item()
                step_tokens = mask.sum().item()
            total_correct += step_correct
            total_tokens += step_tokens
            window_correct += step_correct
            window_tokens += step_tokens
        else:
            step_correct = (logits.argmax(dim=-1) == answers).sum().item()
            total_correct += step_correct
            total_samples += batch_size
            window_correct += step_correct
            window_samples += batch_size

        total_loss += batch_loss
        window_loss += batch_loss
        global_step += 1

        # Step-level logging with WINDOW accuracy (not running average)
        if (step + 1) % log_every == 0:
            if task_type == "decoder":
                w_acc = window_correct / max(window_tokens, 1)
                w_loss = window_loss / max(window_tokens, 1)
            else:
                w_acc = window_correct / max(window_samples, 1)
                w_loss = window_loss / max(window_samples, 1)

            elapsed = time.time() - t_epoch_start
            mins, secs = divmod(int(elapsed), 60)
            log.info(f"Epoch {epoch} | Step {step+1} | Loss: {w_loss:.4f} | Acc: {w_acc:.4f} | {mins}:{secs:02d}")

            if wandb is not None and wandb.run is not None:
                wandb.log({
                    "train/loss_step": w_loss,
                    "train/acc_step": w_acc,
                    "train/lr": optimizer.param_groups[0]["lr"],
                    "global_step": global_step,
                }, step=global_step)

            # Reset window
            window_loss = 0.0
            window_correct = 0
            window_samples = 0
            window_tokens = 0

    # Epoch-level metrics
    if task_type == "decoder":
        epoch_acc = total_correct / max(total_tokens, 1)
        n = total_tokens
    else:
        epoch_acc = total_correct / max(total_samples, 1)
        n = total_samples

    epoch_loss = total_loss / max(n, 1)

    return {
        "train_loss": epoch_loss,
        "train_acc": epoch_acc,
        "global_step": global_step,
    }


def _answers_to_decoder_ids(answer_indices: torch.Tensor, vocab: dict) -> torch.Tensor:
    """Convert answer class indices to decoder token sequences [BOS, ans, EOS, PAD]."""
    B = answer_indices.size(0)
    bos, eos, pad = vocab["<bos>"], vocab["<eos>"], vocab["<pad>"]
    answer_tokens = answer_indices + 3
    seqs = torch.full((B, 4), pad, dtype=torch.long, device=answer_indices.device)
    seqs[:, 0] = bos
    seqs[:, 1] = answer_tokens
    seqs[:, 2] = eos
    return seqs


def get_scheduler(optimizer, cfg):
    """Create warmup + cosine scheduler."""
    from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

    warmup_epochs = cfg.training.get("warmup_epochs", 2)
    total_epochs = cfg.training.epochs

    if warmup_epochs > 0:
        warmup = LinearLR(optimizer, start_factor=0.01, end_factor=1.0,
                          total_iters=warmup_epochs)
        cosine = CosineAnnealingLR(optimizer, T_max=total_epochs - warmup_epochs)
        return SequentialLR(optimizer, [warmup, cosine], milestones=[warmup_epochs])
    return CosineAnnealingLR(optimizer, T_max=total_epochs)
