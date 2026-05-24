"""Transfusion early-fusion model for generative VQA.

Uses transfusion-pytorch for unified image+text processing in a single transformer.
Image patches and text tokens are interleaved and processed together (early fusion),
unlike the GCA approach where a frozen ViT is conditioned via cross-attention (late fusion).
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import transforms

from transfusion_pytorch import Transfusion


def build_clevr_text_vocab(data_root: str | Path) -> dict[str, int]:
    """Build word-level vocab from CLEVR training questions + answers.

    Includes special tokens and all 28 CLEVR answer words.
    Question words are extracted from the training set.
    """
    import json

    # Special tokens
    vocab = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<sep>": 3, "<unk>": 4}

    # CLEVR answers (same as decoder vocab)
    from data.clevr import CLEVR_ANSWERS
    for a in CLEVR_ANSWERS:
        if a not in vocab:
            vocab[a] = len(vocab)

    # Extract question words from training set
    q_path = Path(data_root) / "questions" / "CLEVR_train_questions.json"
    with open(q_path) as f:
        q_data = json.load(f)

    word_counter = Counter()
    for q in q_data["questions"]:
        words = _tokenize(q["question"])
        word_counter.update(words)

    # Add all words that appear (CLEVR has ~90 unique words)
    for word in sorted(word_counter.keys()):
        if word not in vocab:
            vocab[word] = len(vocab)

    return vocab


def _tokenize(text: str) -> list[str]:
    """Simple word tokenizer for CLEVR questions."""
    text = text.lower().strip().rstrip("?").strip()
    return re.findall(r"[a-z0-9]+", text)


def _encode_text(words: list[str], vocab: dict[str, int]) -> list[int]:
    """Encode word list to token IDs."""
    unk = vocab["<unk>"]
    return [vocab.get(w, unk) for w in words]


class TransfusionVQAModel(nn.Module):
    """Early-fusion VQA model using Transfusion.

    Architecture:
        Image → patch embedding → continuous latents
        Question → word tokenizer → discrete tokens
        Answer → word tokenizer → discrete tokens
        Sequence: [image_patches] [question_tokens] [<sep>] [answer_tokens] [<eos>]
        Training: causal LM loss on text tokens (question + answer)
        Generation: autoregressive after [<sep>]
    """

    def __init__(
        self,
        vocab: dict[str, int],
        resolution: int = 224,
        patch_size: int = 16,
        dim: int = 512,
        depth: int = 12,
        heads: int = 8,
    ):
        super().__init__()
        self.vocab = vocab
        self.inv_vocab = {v: k for k, v in vocab.items()}
        self.resolution = resolution
        self.patch_size = patch_size
        self.num_patches = (resolution // patch_size) ** 2
        self.patch_dim = 3 * patch_size * patch_size  # raw pixel patches

        # Image normalization (standard ImageNet)
        self.register_buffer("img_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("img_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        # Transfusion model
        self.transfusion = Transfusion(
            num_text_tokens=len(vocab),
            dim_latent=(self.patch_dim,),
            modality_default_shape=((self.num_patches,),),
            modality_num_dim=(1,),
            transformer=dict(dim=dim, depth=depth, heads=heads),
            flow_loss_weight=0.0,      # No image reconstruction loss
            text_loss_weight=1.0,      # Only text prediction
            velocity_consistency_loss_weight=0.0,
        )

        self.text_cache = None  # compatibility with trainer

    def set_text_cache(self, text_cache):
        """No-op for compatibility."""
        pass

    def get_transforms(self):
        """Image transforms — simple resize + to tensor (no backbone-specific normalization)."""
        return transforms.Compose([
            transforms.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            transforms.Resize((self.resolution, self.resolution),
                              interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
        ])

    def _images_to_patches(self, images: torch.Tensor) -> torch.Tensor:
        """Convert images to patch sequences.

        Args:
            images: (B, 3, H, W) normalized images.

        Returns:
            patches: (B, num_patches, patch_dim) flattened pixel patches.
        """
        B, C, H, W = images.shape
        p = self.patch_size
        # Normalize
        images = (images - self.img_mean.to(images.device)) / self.img_std.to(images.device)
        # Reshape to patches: (B, 3, H/p, p, W/p, p) → (B, H/p*W/p, 3*p*p)
        patches = images.unfold(2, p, p).unfold(3, p, p)  # (B, C, nH, nW, p, p)
        patches = patches.contiguous().view(B, C, -1, p, p)  # (B, C, num_patches, p, p)
        patches = patches.permute(0, 2, 1, 3, 4)  # (B, num_patches, C, p, p)
        patches = patches.reshape(B, -1, self.patch_dim)  # (B, num_patches, patch_dim)
        return patches

    def _encode_questions(self, questions: list[str]) -> tuple[list[list[int]], list[list[int]]]:
        """Tokenize questions into word IDs.

        Returns:
            q_ids: list of token ID lists (variable length per question)
        """
        q_ids = []
        for q in questions:
            words = _tokenize(q)
            ids = _encode_text(words, self.vocab)
            q_ids.append(ids)
        return q_ids

    def forward(self, images, questions, answer_ids):
        """Forward pass with mixed image+text sequences.

        Args:
            images: (B, 3, H, W) images.
            questions: list[str] of length B.
            answer_ids: (B, 4) token IDs [bos, answer, eos, pad] (from trainer).

        Returns:
            loss: scalar loss (text prediction).
        """
        B = images.size(0)
        device = images.device
        patches = self._images_to_patches(images)  # (B, num_patches, patch_dim)

        # Build per-sample modality sequences
        sep_id = self.vocab["<sep>"]
        eos_id = self.vocab["<eos>"]
        pad_id = self.vocab["<pad>"]

        q_ids_list = self._encode_questions(questions)

        samples = []
        for i in range(B):
            # Answer: extract actual answer token (skip bos/eos/pad)
            ans_token = answer_ids[i, 1].item()  # answer_ids is [bos, ans, eos, pad]

            # Build sequence: [image_patches, question_tokens, <sep>, answer, <eos>]
            q_tokens = torch.tensor(q_ids_list[i], dtype=torch.long, device=device)
            answer_seq = torch.tensor([sep_id, ans_token, eos_id], dtype=torch.long, device=device)
            text_tokens = torch.cat([q_tokens, answer_seq])

            sample = [patches[i], text_tokens]  # [Float(num_patches, dim), Int(T)]
            samples.append(sample)

        loss = self.transfusion.forward(samples, return_loss=True)
        return loss

    @torch.no_grad()
    def generate(self, images, questions, max_len=4):
        """Generate answers autoregressively.

        Args:
            images: (B, 3, H, W) images.
            questions: list[str] of length B.
            max_len: max answer tokens to generate.

        Returns:
            list[str]: predicted answer strings.
        """
        B = images.size(0)
        device = images.device
        patches = self._images_to_patches(images)
        q_ids_list = self._encode_questions(questions)

        sep_id = self.vocab["<sep>"]
        eos_id = self.vocab["<eos>"]

        results = []
        for i in range(B):
            q_tokens = torch.tensor(q_ids_list[i], dtype=torch.long, device=device)
            sep_token = torch.tensor([sep_id], dtype=torch.long, device=device)
            prompt_text = torch.cat([q_tokens, sep_token])

            # Build prompt: [image, question + sep]
            prompt = [patches[i], prompt_text.unsqueeze(0)]  # need batch dim for text

            # Use forward_text-based generation for speed
            # Encode the full prompt through transfusion, then generate text
            # For simplicity, use generate_text_only with the text part
            # (image context requires the full sample() which is slow)

            # Simple approach: just use forward in a loop
            generated = prompt_text.unsqueeze(0)  # (1, T)
            img_sample = [patches[i:i+1]]  # keep as batch

            for _ in range(max_len):
                # Build sample with image + current text
                sample = [patches[i], generated.squeeze(0)]
                # Get logits (no loss)
                logits = self.transfusion.forward(
                    [sample],
                    return_loss=False,
                    return_embed=False,
                )
                # logits shape: depends on implementation
                # For text tokens, get the last text position
                if isinstance(logits, tuple):
                    logits = logits[0]

                # Get last token logits and sample
                next_logits = logits[0, -1, :len(self.vocab)]
                next_token = next_logits.argmax(dim=-1)

                generated = torch.cat([generated, next_token.view(1, 1)], dim=1)

                if next_token.item() == eos_id:
                    break

            # Decode generated tokens (after sep)
            gen_tokens = generated.squeeze(0)[len(q_ids_list[i]) + 1:]  # skip q + sep
            words = []
            for t in gen_tokens:
                w = self.inv_vocab.get(t.item(), "")
                if w == "<eos>":
                    break
                if w not in ("<pad>", "<bos>", "<sep>", "<unk>"):
                    words.append(w)
            results.append(" ".join(words) if words else "<unk>")

        return results


def build_transfusion_model(cfg) -> TransfusionVQAModel:
    """Build TransfusionVQA from Hydra config."""
    tf_cfg = cfg.task.get("transfusion", {})

    data_root = cfg.data.root
    vocab = build_clevr_text_vocab(data_root)

    model = TransfusionVQAModel(
        vocab=vocab,
        resolution=cfg.model.get("resolution", 224),
        patch_size=tf_cfg.get("patch_size", 16),
        dim=tf_cfg.get("dim", 512),
        depth=tf_cfg.get("depth", 12),
        heads=tf_cfg.get("heads", 8),
    )

    return model
