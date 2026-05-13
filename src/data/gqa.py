"""GQA dataset for classification-based VQA."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset


class GQAClsDataset(Dataset):
    """GQA dataset with answers mapped to class indices.

    Args:
        root: Path to GQA directory.
        split: One of "train_balanced", "val_balanced".
        transform: Image transform.
        answer_vocab: Dict mapping answer string -> class index.
            If None, build from this split's data.
        max_answers: Number of top answers to keep.
        max_samples: Limit samples for debugging.
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train_balanced",
        transform=None,
        answer_vocab: dict[str, int] | None = None,
        max_answers: int = 1500,
        max_samples: int | None = None,
    ):
        self.root = Path(root)
        self.transform = transform
        self.image_dir = self.root / "images"

        q_path = self.root / "questions" / f"{split}_questions.json"
        with open(q_path) as f:
            q_data = json.load(f)

        self.questions = []
        for qid, q in q_data.items():
            q["question_id"] = qid
            self.questions.append(q)

        if answer_vocab is None:
            counter = Counter(q["answer"] for q in self.questions)
            top_answers = [a for a, _ in counter.most_common(max_answers)]
            self.answer_vocab = {a: i for i, a in enumerate(top_answers)}
        else:
            self.answer_vocab = answer_vocab

        self.num_answers = len(self.answer_vocab)

        self.questions = [
            q for q in self.questions if q["answer"] in self.answer_vocab
        ]

        if max_samples is not None:
            self.questions = self.questions[:max_samples]

    def __len__(self) -> int:
        return len(self.questions)

    def __getitem__(self, idx: int) -> dict:
        q = self.questions[idx]

        img_path = self.image_dir / q["imageId"]
        if not img_path.exists():
            for ext in [".jpg", ".png", ".jpeg"]:
                candidate = img_path.with_suffix(ext)
                if candidate.exists():
                    img_path = candidate
                    break

        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        answer_idx = self.answer_vocab[q["answer"]]

        return {
            "image": image,
            "question": q["question"],
            "answer": answer_idx,
            "question_id": q["question_id"],
            "image_id": q["imageId"],
            "answer_text": q["answer"],
            "types": q.get("types", {}),
        }


def gqa_cls_collate_fn(batch: list[dict]) -> dict:
    return {
        "image": torch.stack([b["image"] for b in batch]),
        "question": [b["question"] for b in batch],
        "answer": torch.tensor([b["answer"] for b in batch], dtype=torch.long),
        "question_id": [b["question_id"] for b in batch],
        "answer_text": [b["answer_text"] for b in batch],
        "types": [b["types"] for b in batch],
    }
