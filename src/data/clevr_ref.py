"""CLEVR-Ref+ dataset for referential segmentation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class CLEVRRefDataset(Dataset):
    """CLEVR-Ref+ referential segmentation dataset.

    Each sample: image + referring expression → patch-level mask GT.

    Args:
        root: Path to clevr_ref+_1.0 directory.
        split: "train" or "val".
        transform: Image transform (from model).
        patch_size: ViT patch size (14 for DINOv2, 16 for SigLIP).
        resolution: Image resolution after transform.
        max_samples: Limit samples (for debugging).
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        transform=None,
        patch_size: int = 14,
        resolution: int = 336,
        max_samples: int | None = None,
    ):
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.patch_size = patch_size
        self.resolution = resolution
        self.n_patches = resolution // patch_size  # grid size per side

        self.image_dir = self.root / "images" / split
        self.mask_dir = self.root / "masks" / split

        # Load referring expressions
        refexp_path = self.root / "refexps" / f"clevr_ref+_{split}_refexps.json"
        with open(refexp_path) as f:
            refexp_data = json.load(f)
        self.refexps = refexp_data["refexps"]

        if max_samples is not None:
            self.refexps = self.refexps[:max_samples]

    def __len__(self) -> int:
        return len(self.refexps)

    def _patchify_mask(self, mask_path: str) -> torch.Tensor:
        """Load pixel mask, resize to resolution, patchify to n×n grid.

        Returns normalized patch-level GT (sums to 1).
        """
        mask = Image.open(mask_path).convert("L")
        # Resize to model resolution
        mask = mask.resize((self.resolution, self.resolution), Image.NEAREST)
        mask = np.array(mask).astype(np.float32) / 255.0  # 0-1

        n = self.n_patches
        ps = self.patch_size
        # Reshape to (n, ps, n, ps) and average over patch pixels
        patch_gt = mask.reshape(n, ps, n, ps).mean(axis=(1, 3))  # (n, n)
        patch_gt = patch_gt.flatten()  # (n*n,)

        # Normalize to probability distribution
        total = patch_gt.sum()
        if total > 0:
            patch_gt = patch_gt / total
        else:
            # Empty mask: uniform distribution
            patch_gt = np.ones_like(patch_gt) / len(patch_gt)

        return torch.from_numpy(patch_gt)

    def __getitem__(self, idx: int) -> dict:
        ref = self.refexps[idx]

        # Image
        img_filename = ref["image_filename"]
        img_path = self.image_dir / img_filename
        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        # Referring expression
        text = ref["refexp"]

        # Mask (patch-level GT)
        mask_filename = ref.get("mask_filename")
        if mask_filename:
            mask_path = self.mask_dir / mask_filename
        else:
            # Construct mask path from image + refexp index
            img_stem = Path(img_filename).stem
            ref_idx = ref.get("refexp_index", idx)
            mask_path = self.mask_dir / f"{img_stem}_mask_{ref_idx}.png"

        patch_gt = self._patchify_mask(str(mask_path))

        return {
            "image": image,
            "text": text,
            "patch_mask_gt": patch_gt,
        }


def clevr_ref_collate_fn(batch: list[dict]) -> dict:
    return {
        "image": torch.stack([b["image"] for b in batch]),
        "text": [b["text"] for b in batch],
        "patch_mask_gt": torch.stack([b["patch_mask_gt"] for b in batch]),
    }
