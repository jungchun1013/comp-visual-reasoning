"""Flamingo-style early fusion: DINOv2 + GCA into LLaMA on CLEVR VQA.

Vision features injected into LLaMA via gated cross-attention at
intermediate layers (same GCA module as our ViT architecture, reversed).

Architecture:
    DINOv2 ViT-B (frozen) → connector → GCA into LLaMA layers → answer

Optimizations:
    - DINOv2 features precomputed & cached (each image used ~10x)
    - Llama-3.2-1B in bf16 (no quantization needed, only 2.5GB)
    - Large batch size (1B model fits easily on 46GB A40)
    - SDPA/FlashAttention enabled by default

Usage:
    # Step 1: Precompute DINOv2 features (one-time, ~15 min)
    CUDA_VISIBLE_DEVICES=1 python scripts/train_flamingo_clevr.py --precompute

    # Step 2: Train
    CUDA_VISIBLE_DEVICES=1 python scripts/train_flamingo_clevr.py \
        --epochs 16 --batch-size 32
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Dataset
from PIL import Image

import timm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from model.crossattention import GatedCrossAttention

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")


class DINOv2Cache:
    """Lazy LRU cache for frozen DINOv2 features, inspired by TextCache.

    First encounter: compute DINOv2 features and cache on CPU.
    Subsequent encounters: return cached features (zero compute).

    CLEVR has ~70K unique images but ~700K questions (~10 questions/image).
    With max_size=70000, epoch 1 is cache-miss (cold), epoch 2+ is ~100% hit.

    Memory: each image = (576, 768) fp16 = 884 KB.
    70K images ≈ 58 GB CPU RAM. Set max_size lower if RAM is limited.
    """

    def __init__(self, dinov2, num_prefix, device, max_size=70000):
        self.dinov2 = dinov2
        self.num_prefix = num_prefix
        self.device = device
        self.max_size = max_size
        self._cache = {}  # {image_filename: tensor on CPU}
        self.hits = 0
        self.misses = 0

    @torch.no_grad()
    def get_features(self, images, filenames):
        """Get DINOv2 patch features, using cache when available.

        Args:
            images: (B, 3, H, W) tensor — raw images (for cache misses)
            filenames: list of image filenames (cache keys)
        Returns:
            features: (B, N_patches, dim) fp16 tensor on device
        """
        B = len(filenames)
        results = [None] * B
        uncached = []  # (batch_idx, filename)

        for i, fn in enumerate(filenames):
            if fn in self._cache:
                results[i] = self._cache[fn]
                self.hits += 1
            else:
                uncached.append(i)
                self.misses += 1

        # Compute uncached
        if uncached:
            uncached_imgs = torch.stack([images[i] for i in uncached]).to(self.device)
            self.dinov2.eval()
            feats = self.dinov2.forward_features(uncached_imgs)
            patches = feats[:, self.num_prefix:, :].cpu().half()

            for j, batch_idx in enumerate(uncached):
                fn = filenames[batch_idx]
                results[batch_idx] = patches[j]
                self._cache[fn] = patches[j]

            # LRU-style eviction (simple: pop oldest if over limit)
            if self.max_size > 0:
                while len(self._cache) > self.max_size:
                    oldest = next(iter(self._cache))
                    del self._cache[oldest]

        return torch.stack(results).to(self.device)

    @property
    def cache_size(self):
        return len(self._cache)

    @property
    def hit_rate(self):
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


# ── Dataset ────────────────────────────────────────────────────────

class CLEVRFlamingoDataset(Dataset):
    """CLEVR VQA dataset that returns raw images for DINOv2 + lazy cache."""

    def __init__(self, root, split, dinov2_transform):
        self.root = Path(root)
        self.dinov2_transform = dinov2_transform
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
        return {
            "image": self.dinov2_transform(image),
            "filename": q["image_filename"],
            "question": q["question"],
            "answer": str(q["answer"]).lower(),
        }


# ── Model ──────────────────────────────────────────────────────────

class FlamingoModel(nn.Module):
    """Flamingo-style: DINOv2 features → GCA into LLaMA layers."""

    def __init__(self, llm_name="meta-llama/Llama-3.2-1B",
                 dinov2_dim=768, gca_layers=None, lora_r=16,
                 use_lora=True, freeze_llm=False, device="cuda"):
        super().__init__()
        if gca_layers is None:
            gca_layers = [1, 4, 7, 10, 13, 15]  # every ~3rd for 16-layer LLaMA

        self.gca_layer_indices = gca_layers
        self.device = device
        self.use_lora = use_lora

        # 1. Load LLaMA in bf16 (no quantization — 1B model is tiny)
        log.info(f"Loading {llm_name} (bf16)...")
        from transformers import AutoModelForCausalLM
        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_name, torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            device_map={"": device},
        )
        llm_dim = self.llm.config.hidden_size
        num_llm_layers = self.llm.config.num_hidden_layers

        # 2. Connector: DINOv2 dim → LLaMA dim
        self.connector = nn.Sequential(
            nn.Linear(dinov2_dim, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim),
        ).to(device)

        # 3. GCA layers
        valid_gca = [i for i in gca_layers if i < num_llm_layers]
        self.gca_layer_indices = valid_gca
        log.info(f"GCA layers: {valid_gca} (LLaMA has {num_llm_layers} layers)")

        self.gca_modules = nn.ModuleDict()
        for idx in valid_gca:
            gca = GatedCrossAttention(
                layer_idx=idx, dim=llm_dim, ff_mult=2,
                use_ffn=False, head_dim=llm_dim // 16, num_heads=16,
                use_gate=True,
            )
            self.gca_modules[str(idx)] = gca
        self.gca_modules = self.gca_modules.to(device)

        # 4. LLM trainability: frozen (Flamingo recipe) / LoRA / full fine-tune
        if freeze_llm:
            for p in self.llm.parameters():
                p.requires_grad = False
        elif use_lora:
            from peft import LoraConfig, get_peft_model
            for p in self.llm.parameters():
                p.requires_grad = False
            lora_config = LoraConfig(
                r=lora_r, lora_alpha=lora_r * 2,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.05, bias="none",
                task_type="CAUSAL_LM",
            )
            self.llm = get_peft_model(self.llm, lora_config)
        # If not using LoRA, all LLM params are trainable (full fine-tune)

        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in self.parameters())
        log.info(f"Trainable: {n_trainable:,} / {n_total:,} "
                 f"({n_trainable/n_total*100:.2f}%)")

    def _register_gca_hooks(self, visual_kv):
        """Register forward hooks to inject GCA at specified layers."""
        hooks = []
        for idx in self.gca_layer_indices:
            gca = self.gca_modules[str(idx)]
            try:
                llm_layer = self.llm.base_model.model.model.layers[idx]
            except AttributeError:
                try:
                    llm_layer = self.llm.model.layers[idx]
                except AttributeError:
                    llm_layer = self.llm.model.model.layers[idx]

            def make_hook(gca_module, vis_kv):
                def hook_fn(module, args, output):
                    if isinstance(output, tuple):
                        hidden = output[0]
                    else:
                        hidden = output
                    hidden = gca_module(hidden, vis_kv, attn_mask=None)
                    if isinstance(output, tuple):
                        return (hidden,) + output[1:]
                    return hidden
                return hook_fn

            hook = llm_layer.register_forward_hook(make_hook(gca, visual_kv))
            hooks.append(hook)
        return hooks

    def forward(self, visual_features, input_ids, attention_mask, labels):
        """
        Args:
            visual_features: (B, N_patches, dinov2_dim) — from DINOv2Cache
            input_ids, attention_mask, labels: tokenized text
        """
        # 1. Project visual features (fp32 for connector)
        visual_kv = self.connector(visual_features.float())  # (B, N, llm_dim)

        # 2. Register GCA hooks
        hooks = self._register_gca_hooks(visual_kv)

        # 3. Forward through LLM
        outputs = self.llm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        # 4. Remove hooks
        for h in hooks:
            h.remove()

        return outputs

    @torch.no_grad()
    def generate_answer(self, visual_features, input_ids, attention_mask,
                        tokenizer, max_new_tokens=10):
        visual_kv = self.connector(visual_features.float())
        hooks = self._register_gca_hooks(visual_kv)

        outputs = self.llm.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

        for h in hooks:
            h.remove()

        answers = []
        for seq in outputs:
            text = tokenizer.decode(seq, skip_special_tokens=True).strip().lower()
            answers.append(text)
        return answers


# ── Collate & eval ─────────────────────────────────────────────────

def collate_fn(batch, tokenizer, max_length=64):
    images = torch.stack([b["image"] for b in batch])
    filenames = [b["filename"] for b in batch]
    questions = [b["question"] for b in batch]
    answers = [b["answer"] for b in batch]

    prompts = [f"Question: {q}\nAnswer:" for q in questions]
    full_texts = [f"{p} {a}</s>" for p, a in zip(prompts, answers)]

    full_enc = tokenizer(full_texts, return_tensors="pt", padding=True,
                         truncation=True, max_length=max_length)

    labels = full_enc["input_ids"].clone()
    prompt_enc = tokenizer(prompts, return_tensors="pt", padding=True,
                           truncation=True, max_length=max_length)
    for i in range(len(batch)):
        prompt_len = prompt_enc["attention_mask"][i].sum().item()
        labels[i, :prompt_len] = -100
    labels[full_enc["attention_mask"] == 0] = -100

    return {
        "images": images,
        "filenames": filenames,
        "input_ids": full_enc["input_ids"],
        "attention_mask": full_enc["attention_mask"],
        "labels": labels,
        "questions": questions,
        "answers": answers,
    }


def evaluate(model, dataloader, dinov2_cache, tokenizer, device, max_batches=None):
    model.eval()
    correct, total = 0, 0

    for i, batch in enumerate(dataloader):
        if max_batches and i >= max_batches:
            break

        features = dinov2_cache.get_features(batch["images"], batch["filenames"])
        gt_answers = batch["answers"]

        prompts = [f"Question: {q}\nAnswer:" for q in batch["questions"]]
        prompt_enc = tokenizer(prompts, return_tensors="pt", padding=True,
                               truncation=True, max_length=64).to(device)

        with autocast(device_type="cuda", dtype=torch.bfloat16):
            pred_answers = model.generate_answer(
                features, prompt_enc["input_ids"],
                prompt_enc["attention_mask"], tokenizer)

        for pred, gt in zip(pred_answers, gt_answers):
            # generate_answer lowercases the decode, so match the marker lowercase
            tail = pred.split("answer:")[-1].strip() if pred else ""
            pred_clean = tail.split()[0] if tail else ""
            pred_clean = pred_clean.strip(".,!?").lower()
            if pred_clean == gt.lower():
                correct += 1
            total += 1

    return correct / total if total > 0 else 0


# ── Main ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str,
                        default="/home/jungchun/data/clevr/CLEVR_v1.0")
    parser.add_argument("--llm-name", type=str,
                        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--dinov2-name", type=str,
                        default="vit_base_patch14_dinov2.lvd142m")
    parser.add_argument("--resolution", type=int, default=336)
    parser.add_argument("--gca-layers", type=str, default="1,5,9,13,17,21",
                        help="Comma-separated LLaMA layer indices for GCA")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--no-lora", action="store_true",
                        help="Full fine-tune instead of LoRA")
    parser.add_argument("--freeze-llm", action="store_true",
                        help="Freeze the LLM entirely; train only GCA + connector "
                             "(Flamingo recipe). Overrides LoRA.")
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--val-batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--cache-size", type=int, default=20000,
                        help="Max images in DINOv2 feature cache (20K~17GB)")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gca_layers = [int(x) for x in args.gca_layers.split(",")]

    output_dir = Path(args.output_dir) if args.output_dir else \
        Path(f"outputs/model/clevr_flamingo_dinov2_early_s{args.seed}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Redirect stdout to log file
    import sys as _sys
    log_path = output_dir.parent / f"{output_dir.name}.log"
    log_file = open(log_path, "w", buffering=1)

    class Tee:
        def __init__(self, *streams):
            self.streams = streams
        def write(self, data):
            for s in self.streams:
                s.write(data)
                s.flush()
        def flush(self):
            for s in self.streams:
                s.flush()

    _sys.stdout = Tee(_sys.stdout, log_file)
    _sys.stderr = Tee(_sys.stderr, log_file)

    log.info(f"Output: {output_dir}")
    log.info(f"Log: {log_path}")

    # Load DINOv2 for feature extraction + cache
    log.info(f"Loading DINOv2: {args.dinov2_name}")
    dinov2 = timm.create_model(args.dinov2_name, pretrained=True)
    if args.resolution != 518:
        dinov2.set_input_size(img_size=args.resolution)
    data_cfg = timm.data.resolve_data_config(dinov2.pretrained_cfg)
    data_cfg["input_size"] = (3, args.resolution, args.resolution)
    dinov2_transform = timm.data.create_transform(**data_cfg, is_training=False)
    num_prefix = dinov2.num_prefix_tokens
    dinov2_dim = dinov2.embed_dim
    for p in dinov2.parameters():
        p.requires_grad = False
    dinov2 = dinov2.to(device).eval()

    dinov2_cache = DINOv2Cache(dinov2, num_prefix, device, max_size=args.cache_size)
    log.info(f"DINOv2 dim={dinov2_dim}, patches per image={576}")

    # Model
    model = FlamingoModel(
        llm_name=args.llm_name, dinov2_dim=dinov2_dim,
        gca_layers=gca_layers, lora_r=args.lora_r,
        use_lora=not args.no_lora and not args.freeze_llm,
        freeze_llm=args.freeze_llm, device=device)

    # Tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.llm_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Dataset
    train_dataset = CLEVRFlamingoDataset(args.data_root, "train", dinov2_transform)
    val_dataset = CLEVRFlamingoDataset(args.data_root, "val", dinov2_transform)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=8, pin_memory=True,
        collate_fn=lambda b: collate_fn(b, tokenizer))
    val_loader = DataLoader(
        val_dataset, batch_size=args.val_batch_size, shuffle=False,
        num_workers=8, pin_memory=True,
        collate_fn=lambda b: collate_fn(b, tokenizer))

    # Optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.05)

    total_steps = len(train_loader) * args.epochs // args.grad_accum
    warmup_steps = len(train_loader) * args.warmup_epochs // args.grad_accum

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_acc = 0.0
    log.info(f"Training: {args.epochs} epochs, batch={args.batch_size}, "
             f"grad_accum={args.grad_accum}, lr={args.lr}, "
             f"gca_layers={gca_layers}, "
             f"llm={'frozen' if args.freeze_llm else ('full-ft' if args.no_lora else f'lora_r={args.lora_r}')}")
    log.info(f"Steps/epoch: {len(train_loader)}, total_steps: {total_steps}")

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_start = time.time()

        optimizer.zero_grad()
        for step, batch in enumerate(train_loader):
            features = dinov2_cache.get_features(batch["images"], batch["filenames"])
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            with autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(features, input_ids, attention_mask, labels)
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
                hr = dinov2_cache.hit_rate
                log.info(f"Epoch {epoch+1} | Step {step+1}/{len(train_loader)} | "
                         f"Loss: {avg_loss:.4f} | Cache: {hr:.0%} | {elapsed:.0f}s")

        avg_loss = epoch_loss / len(train_loader)
        elapsed = time.time() - epoch_start
        log.info(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | Time: {elapsed:.0f}s")

        # Eval at final epoch only
        if epoch == args.epochs - 1:
            log.info("Evaluating...")
            val_acc = evaluate(model, val_loader, dinov2_cache, tokenizer, device)
            log.info(f"Epoch {epoch+1} | Val acc: {val_acc:.4f}")

            if val_acc > best_acc:
                best_acc = val_acc

        # Save checkpoint
        ckpt = {
            "epoch": epoch + 1,
            "model_state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
            "gca_layers": gca_layers,
            "llm_name": args.llm_name,
            "freeze_llm": args.freeze_llm,
            "use_lora": not args.no_lora and not args.freeze_llm,
        }
        torch.save(ckpt, output_dir / "last.pt")

        if (epoch + 1) % 5 == 0:
            torch.save(ckpt, output_dir / f"epoch_{epoch+1}.pt")

    log.info(f"Done. Best val acc: {best_acc:.4f}")


if __name__ == "__main__":
    main()
