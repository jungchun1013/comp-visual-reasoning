"""LLaVA-1.5 + DINOv2 + LoRA training on CLEVR VQA.

Replaces LLaVA's CLIP vision tower with DINOv2 ViT-B.
Trains: projection MLP (768→4096) + LoRA on LLM (r=16).
Freezes: DINOv2, LLM base weights.

Usage:
    CUDA_VISIBLE_DEVICES=1 python scripts/train_llava.py \
        --epochs 16 --batch-size 8 --lr 2e-4
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Dataset
from PIL import Image

from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import timm

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")

CLEVR_ANSWERS = [
    "yes", "no",
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
    "gray", "red", "blue", "green", "brown", "purple", "cyan", "yellow",
    "cube", "sphere", "cylinder",
    "metal", "rubber",
    "large", "small",
]


class CLEVRLLaVADataset(Dataset):
    """CLEVR dataset formatted for LLaVA-style training."""

    def __init__(self, root, split, processor, dinov2_transform):
        self.root = Path(root)
        self.processor = processor
        self.dinov2_transform = dinov2_transform
        self.split = split

        q_file = self.root / "questions" / f"CLEVR_{split}_questions.json"
        with open(q_file) as f:
            self.questions = json.load(f)["questions"]
        self.img_dir = self.root / "images" / split

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        q = self.questions[idx]
        img_path = self.img_dir / q["image_filename"]
        image = Image.open(img_path).convert("RGB")

        # DINOv2 transform for visual features
        dinov2_image = self.dinov2_transform(image)

        question = q["question"]
        answer = str(q["answer"]).lower()

        return {
            "dinov2_image": dinov2_image,
            "question": question,
            "answer": answer,
        }


class LLaVADINOv2Model(nn.Module):
    """LLaVA-1.5-7B with DINOv2 replacing CLIP vision tower."""

    def __init__(self, llava_model_name="llava-hf/llava-1.5-7b-hf",
                 dinov2_name="vit_base_patch14_dinov2.lvd142m",
                 resolution=336, lora_r=16, device="cuda"):
        super().__init__()

        # Load Vicuna-7B (LLaVA-1.5's LLM) with 4-bit quantization (QLoRA)
        llm_name = "lmsys/vicuna-7b-v1.5"
        log.info(f"Loading {llm_name} (4-bit quantized)...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_name,
            quantization_config=bnb_config,
            low_cpu_mem_usage=True,
            device_map={"": device},
        )

        # Load DINOv2
        log.info(f"Loading DINOv2: {dinov2_name}")
        self.dinov2 = timm.create_model(dinov2_name, pretrained=True)
        # Resize pos embedding for target resolution (same as ViTBackbone)
        if resolution != 518:
            self.dinov2.set_input_size(img_size=resolution)
        data_cfg = timm.data.resolve_data_config(self.dinov2.pretrained_cfg)
        data_cfg["input_size"] = (3, resolution, resolution)
        self.dinov2_transform = timm.data.create_transform(**data_cfg, is_training=False)
        self.dinov2_dim = self.dinov2.embed_dim  # 768
        self.num_prefix = self.dinov2.num_prefix_tokens  # 1 (CLS)

        # Freeze DINOv2
        for p in self.dinov2.parameters():
            p.requires_grad = False
        self.dinov2.eval()

        # Projection MLP: DINOv2 768 → LLM 4096
        llm_dim = self.llm.config.hidden_size  # 4096
        self.multi_modal_projector = nn.Sequential(
            nn.Linear(self.dinov2_dim, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim),
        )

        # Freeze LLM, prepare for QLoRA
        for p in self.llm.parameters():
            p.requires_grad = False
        self.llm = prepare_model_for_kbit_training(self.llm)

        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_r * 2,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.llm = get_peft_model(self.llm, lora_config)

        # Move non-quantized parts to device
        self.dinov2 = self.dinov2.to(device)
        self.multi_modal_projector = self.multi_modal_projector.to(device)

        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in self.parameters())
        log.info(f"Trainable: {n_trainable:,} / {n_total:,} ({n_trainable/n_total*100:.2f}%)")

    def get_dinov2_transform(self):
        return self.dinov2_transform

    def to(self, device):
        # Skip quantized LLM (already on device), only move non-quantized parts
        self.dinov2 = self.dinov2.to(device)
        self.multi_modal_projector = self.multi_modal_projector.to(device)
        return self

    @torch.no_grad()
    def encode_vision(self, images):
        """Extract DINOv2 patch features. images: (B, 3, H, W)"""
        self.dinov2.eval()
        feats = self.dinov2.forward_features(images)
        patches = feats[:, self.num_prefix:, :]  # skip CLS
        return patches  # (B, N_patches, 768)

    def forward(self, dinov2_images, input_ids, attention_mask, labels):
        """
        dinov2_images: (B, 3, H, W) — preprocessed for DINOv2
        input_ids: (B, T) — tokenized [USER question ASSISTANT answer]
        attention_mask: (B, T)
        labels: (B, T) — -100 for non-answer tokens
        """
        # Visual features
        with torch.no_grad():
            visual_feats = self.encode_vision(dinov2_images)  # (B, N, 768)
        visual_embeds = self.multi_modal_projector(visual_feats)  # (B, N, 4096)

        # Text embeddings
        text_embeds = self.llm.get_input_embeddings()(input_ids)

        # Find image token positions and replace with visual embeds
        # We use a special approach: prepend visual tokens before text
        B, N_vis, D = visual_embeds.shape
        B, T, _ = text_embeds.shape

        # Concatenate: [visual_tokens, text_tokens]
        inputs_embeds = torch.cat([visual_embeds, text_embeds], dim=1)

        # Extend attention mask and labels
        vis_mask = torch.ones(B, N_vis, device=attention_mask.device, dtype=attention_mask.dtype)
        full_attention_mask = torch.cat([vis_mask, attention_mask], dim=1)

        vis_labels = torch.full((B, N_vis), -100, device=labels.device, dtype=labels.dtype)
        full_labels = torch.cat([vis_labels, labels], dim=1)

        # Forward through LLM
        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=full_attention_mask,
            labels=full_labels,
        )
        return outputs

    @torch.no_grad()
    def generate_answer(self, dinov2_images, input_ids, attention_mask,
                        tokenizer, max_new_tokens=10):
        """Generate answer tokens."""
        visual_feats = self.encode_vision(dinov2_images)
        visual_embeds = self.multi_modal_projector(visual_feats)

        text_embeds = self.llm.get_input_embeddings()(input_ids)

        B, N_vis, D = visual_embeds.shape
        inputs_embeds = torch.cat([visual_embeds, text_embeds], dim=1)

        vis_mask = torch.ones(B, N_vis, device=attention_mask.device, dtype=attention_mask.dtype)
        full_attention_mask = torch.cat([vis_mask, attention_mask], dim=1)

        outputs = self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=full_attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
        # Decode only new tokens
        answers = []
        for seq in outputs:
            text = tokenizer.decode(seq, skip_special_tokens=True).strip().lower()
            answers.append(text)
        return answers


def collate_fn(batch, tokenizer):
    """Collate batch: tokenize questions+answers for causal LM."""
    dinov2_images = torch.stack([b["dinov2_image"] for b in batch])
    questions = [b["question"] for b in batch]
    answers = [b["answer"] for b in batch]

    # Format: "USER: {question}\nASSISTANT: {answer}</s>"
    prompts = [f"USER: {q}\nASSISTANT: " for q in questions]
    full_texts = [f"{p}{a}</s>" for p, a in zip(prompts, answers)]

    # Tokenize full text
    full_enc = tokenizer(full_texts, return_tensors="pt", padding=True,
                         truncation=True, max_length=128)

    # Create labels: -100 for prompt tokens, actual ids for answer tokens
    labels = full_enc["input_ids"].clone()
    prompt_enc = tokenizer(prompts, return_tensors="pt", padding=True,
                           truncation=True, max_length=128)

    for i in range(len(batch)):
        # Length of prompt (without padding)
        prompt_len = prompt_enc["attention_mask"][i].sum().item()
        labels[i, :prompt_len] = -100
    # Mask padding
    labels[full_enc["attention_mask"] == 0] = -100

    return {
        "dinov2_images": dinov2_images,
        "input_ids": full_enc["input_ids"],
        "attention_mask": full_enc["attention_mask"],
        "labels": labels,
        "answers": answers,
    }


def evaluate(model, dataloader, tokenizer, device, max_batches=None):
    """Evaluate accuracy on CLEVR."""
    model.eval()
    correct, total = 0, 0

    for i, batch in enumerate(dataloader):
        if max_batches and i >= max_batches:
            break

        dinov2_images = batch["dinov2_images"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        gt_answers = batch["answers"]

        # For generation, use only the prompt part (before answer)
        prompts = [f"USER: {q}\nASSISTANT: " for q in batch["questions"]]
        prompt_enc = tokenizer(prompts, return_tensors="pt", padding=True,
                               truncation=True, max_length=128).to(device)

        with autocast(device_type="cuda", dtype=torch.bfloat16):
            pred_answers = model.generate_answer(
                dinov2_images, prompt_enc["input_ids"],
                prompt_enc["attention_mask"], tokenizer)

        for pred, gt in zip(pred_answers, gt_answers):
            # Extract first word/number from prediction
            pred_clean = pred.split("ASSISTANT:")[-1].strip().split()[0] if pred else ""
            pred_clean = pred_clean.strip(".,!?").lower()
            if pred_clean == gt.lower():
                correct += 1
            total += 1

    acc = correct / total if total > 0 else 0
    return acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str,
                        default="/home/jungchun/data/clevr/CLEVR_v1.0")
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--val-batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--resolution", type=int, default=336)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    # Output dir
    output_dir = Path(args.output_dir) if args.output_dir else \
        Path(f"outputs/model/clevr_llava_dinov2_lora_s{args.seed}")
    output_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"Output: {output_dir}")

    # Model
    model = LLaVADINOv2Model(
        lora_r=args.lora_r, resolution=args.resolution, device=device)
    dinov2_transform = model.get_dinov2_transform()

    # Tokenizer
    from transformers import LlamaTokenizer
    tokenizer = LlamaTokenizer.from_pretrained("lmsys/vicuna-7b-v1.5")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Dataset
    train_dataset = CLEVRLLaVADataset(args.data_root, "train", None, dinov2_transform)
    val_dataset = CLEVRLLaVADataset(args.data_root, "val", None, dinov2_transform)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True,
        collate_fn=lambda b: collate_fn(b, tokenizer))
    val_loader = DataLoader(
        val_dataset, batch_size=args.val_batch_size, shuffle=False,
        num_workers=4, pin_memory=True,
        collate_fn=lambda b: collate_fn(b, tokenizer))

    if args.eval_only:
        ckpt = torch.load(output_dir / "best.pt", map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        acc = evaluate(model, val_loader, tokenizer, device)
        log.info(f"Val acc: {acc:.4f}")
        return

    # Optimizer — only trainable params
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.05)

    # Warmup + cosine schedule
    total_steps = len(train_loader) * args.epochs // args.grad_accum
    warmup_steps = len(train_loader) * args.warmup_epochs // args.grad_accum

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + __import__("math").cos(__import__("math").pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_acc = 0.0
    log.info(f"Training: {args.epochs} epochs, batch={args.batch_size}, "
             f"grad_accum={args.grad_accum}, lr={args.lr}")

    for epoch in range(args.epochs):
        model.train()
        model.dinov2.eval()  # keep DINOv2 in eval mode
        epoch_loss = 0.0
        epoch_start = time.time()

        optimizer.zero_grad()
        for step, batch in enumerate(train_loader):
            dinov2_images = batch["dinov2_images"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(dinov2_images, input_ids, attention_mask, labels)
                loss = outputs.loss / args.grad_accum

            loss.backward()

            if (step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

            epoch_loss += outputs.loss.item()

            if (step + 1) % 100 == 0:
                avg_loss = epoch_loss / (step + 1)
                elapsed = time.time() - epoch_start
                log.info(f"Epoch {epoch+1} | Step {step+1} | "
                         f"Loss: {avg_loss:.4f} | {elapsed:.0f}s")

        avg_loss = epoch_loss / len(train_loader)
        elapsed = time.time() - epoch_start
        log.info(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | Time: {elapsed:.0f}s")

        # Eval
        log.info("Evaluating...")
        val_acc = evaluate(model, val_loader, tokenizer, device, max_batches=200)
        log.info(f"Epoch {epoch+1} | Val acc: {val_acc:.4f}")

        # Save
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "val_acc": val_acc,
            }, output_dir / "best.pt")
            log.info(f"  New best: {val_acc:.4f}")

        if (epoch + 1) % 5 == 0:
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "val_acc": val_acc,
            }, output_dir / f"epoch_{epoch+1}.pt")

    log.info(f"Done. Best val acc: {best_acc:.4f}")


if __name__ == "__main__":
    main()
