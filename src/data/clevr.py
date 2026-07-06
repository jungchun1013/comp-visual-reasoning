"""CLEVR VQA dataset."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

from data.clevr_programs import (
    coarse_question_type,
    compute_program_depth,
    extract_oracle_prompt,
)

# Fixed CLEVR answer vocabulary (28 classes)
CLEVR_ANSWERS = [
    "yes", "no",
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
    "gray", "red", "blue", "green", "brown", "purple", "cyan", "yellow",
    "cube", "sphere", "cylinder",
    "metal", "rubber",
    "large", "small",
]

ANSWER_TO_IDX = {a: i for i, a in enumerate(CLEVR_ANSWERS)}
IDX_TO_ANSWER = {i: a for i, a in enumerate(CLEVR_ANSWERS)}


class CLEVRVQADataset(Dataset):
    """CLEVR VQA dataset.

    Args:
        root: Path to CLEVR_v1.0 directory.
        split: One of "train", "val", "test".
        transform: Image transform.
        use_oracle: If True, load scene files and compute oracle steering prompts.
        max_samples: Limit samples (for debugging).
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        transform=None,
        use_oracle: bool = False,
        max_samples: int | None = None,
        questions_file: str | Path | None = None,
        image_dir: str | Path | None = None,
    ):
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.use_oracle = use_oracle
        self.image_dir = Path(image_dir) if image_dir else self.root / "images" / split

        q_path = Path(questions_file) if questions_file else \
            self.root / "questions" / f"CLEVR_{split}_questions.json"
        with open(q_path) as f:
            q_data = json.load(f)
        self.questions = q_data["questions"]

        if max_samples is not None:
            self.questions = self.questions[:max_samples]

        self.scenes = {}
        if use_oracle and split != "test":
            scene_path = self.root / "scenes" / f"CLEVR_{split}_scenes.json"
            with open(scene_path) as f:
                scene_data = json.load(f)
            for s in scene_data["scenes"]:
                self.scenes[s["image_index"]] = s

    def __len__(self) -> int:
        return len(self.questions)

    def __getitem__(self, idx: int) -> dict:
        q = self.questions[idx]

        img_path = self.image_dir / q["image_filename"]
        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        answer_str = q.get("answer", "")
        answer_idx = ANSWER_TO_IDX.get(answer_str, -1)

        program = q.get("program", [])
        program_depth = compute_program_depth(program)
        question_type = coarse_question_type(program)

        item = {
            "image": image,
            "question": q["question"],
            "answer": answer_idx,
            "program_depth": program_depth,
            "question_type": question_type,
            "image_index": q["image_index"],
        }

        if self.use_oracle:
            scene = self.scenes.get(q["image_index"], {})
            item["oracle_prompt"] = extract_oracle_prompt(scene, program)

        return item


def clevr_collate_fn(batch: list[dict]) -> dict:
    """Custom collate that keeps questions as list[str]."""
    return {
        "image": torch.stack([b["image"] for b in batch]),
        "question": [b["question"] for b in batch],
        "answer": torch.tensor([b["answer"] for b in batch], dtype=torch.long),
        "program_depth": torch.tensor([b["program_depth"] for b in batch], dtype=torch.long),
        "question_type": [b["question_type"] for b in batch],
        "image_index": torch.tensor([b["image_index"] for b in batch], dtype=torch.long),
        **({"oracle_prompt": [b["oracle_prompt"] for b in batch]} if "oracle_prompt" in batch[0] else {}),
    }
