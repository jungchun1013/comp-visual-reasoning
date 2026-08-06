"""Corruption-based sampling for patching experiments.

Builds on top of utils/sampling.py (generic) with CLEVR corruption-specific
key_fn, metadata_fn, and collect_samples.

Usage:
    from analysis.patching_sampling import (
        build_corruption_index, collect_corruption_samples, CATEGORIES,
    )

    index = build_corruption_index(dataset)
    samples = collect_corruption_samples(dataset, index, "color", n=50)
    # samples = [(image_tensor, question, corrupt_question, answer_idx), ...]
"""

from __future__ import annotations

import random

from analysis.sampling import build_index, print_index_summary


# ── Category definitions ─────────────────────────────────────────
# (group_name, category_key)
# group_name determines which plot this category appears in.
# category_key is used to look up samples in the index.

CATEGORIES = [
    # Coarse — 4 types
    ("coarse", "attribute"),
    ("coarse", "spatial"),
    ("coarse", "attribute_query"),
    ("coarse", "quantifier"),
    # Fine Attribute — 4 types
    ("fine_attribute", "color"),
    ("fine_attribute", "material"),
    ("fine_attribute", "size"),
    ("fine_attribute", "shape"),
    # Fine Attribute Query — 4 types
    ("fine_attribute_query", "what_color"),
    ("fine_attribute_query", "what_material"),
    ("fine_attribute_query", "what_size"),
    ("fine_attribute_query", "what_shape"),
    # Anchor swap — anchor-side described-attribute corruption
    ("anchor_swap", "anchor_swap"),
]


# ── Key function ─────────────────────────────────────────────────

def _corruption_key_fn(sample):
    """key_fn for build_index: group by corruption type.

    Input: dict with at least "question" key (from metadata_fn).
    Output: list of string keys (both coarse and fine).
    """
    from analysis.clevr_corruptions import generate_corruptions

    corruptions = generate_corruptions(sample["question"])
    keys = set()
    for c in corruptions:
        keys.add(c["type"])       # coarse key
        keys.add(c["fine_type"])  # fine key
    return list(keys)


def _clevr_metadata_fn(dataset, idx):
    """Read question text only, skip image loading."""
    return {"question": dataset.questions[idx]["question"]}


# ── Public API ───────────────────────────────────────────────────

def build_corruption_index(dataset):
    """Build corruption index for a CLEVR dataset. Scans once, text-only.

    Returns:
        {key: [idx, ...]} where key is coarse type or fine type string.
    """
    print("Building corruption index (text-only scan)...", flush=True)
    index = build_index(dataset, _corruption_key_fn,
                        metadata_fn=_clevr_metadata_fn)
    print_index_summary(index)
    return index


def collect_corruption_samples(dataset, index, category_key, n_samples):
    """Sample n items for a corruption category, apply random matching swap.

    Args:
        dataset: CLEVRVQADataset
        index: from build_corruption_index()
        category_key: e.g. "attribute" (coarse) or "color" (fine)
        n_samples: number of samples

    Returns:
        list of (image_tensor, question, corrupted_question, answer_idx)

    Algorithm:
        1. sample_from_index(index, key, n) — uniform random from eligible pool
        2. For each sampled idx, load image, regenerate corruptions,
           filter by key, random.choice one swap
    """
    from analysis.clevr_corruptions import generate_corruptions
    from analysis.sampling import sample_from_index

    chosen_indices = sample_from_index(index, category_key, n_samples)

    samples = []
    for idx in chosen_indices:
        sample = dataset[idx]
        corruptions = generate_corruptions(sample["question"])
        matching = [
            c for c in corruptions
            if c["type"] == category_key or c["fine_type"] == category_key
        ]
        chosen = random.choice(matching)
        samples.append((
            sample["image"],
            sample["question"],
            chosen["corrupted_question"],
            sample["answer"],
        ))
    return samples


# ── Anchor-swap sampling ────────────────────────────────────────

def collect_anchor_swap_samples(dataset, n_samples, families=None):
    """Sample anchor_swap corruptions from the attr_query_same families.

    Restricts to questions whose ``question_family_index`` is one of
    ``families`` (default: RETRIEVAL_CATEGORIES["attr_query_same"] =
    [53, 59, 55, 57, 61, 60]), keeps only questions for which
    ``generate_anchor_swap`` yields ≥1 valid swap, then draws n_samples and
    picks one swap per question.

    Args:
        dataset: CLEVRVQADataset
        n_samples: number of samples
        families: iterable of question_family_index ids (default attr_query_same)

    Returns:
        list of (image_tensor, question, corrupted_question, answer_idx)
    """
    from analysis.clevr_corruptions import generate_anchor_swap
    from data.clevr_sampling import RETRIEVAL_CATEGORIES, build_family_index

    if families is None:
        families = RETRIEVAL_CATEGORIES["attr_query_same"]
    families = set(families)

    index = build_family_index(dataset)
    eligible = []
    for fam in families:
        for idx in index.get(f"F{fam}", []):
            q = dataset.questions[idx]
            swaps = generate_anchor_swap(q["question"], q.get("program", []))
            if swaps:
                eligible.append((idx, swaps))
    print(f"anchor_swap: {len(eligible)} eligible questions "
          f"in families {sorted(families)}", flush=True)

    random.shuffle(eligible)
    chosen = eligible[:n_samples]

    samples = []
    for idx, swaps in chosen:
        sample = dataset[idx]
        chosen_swap = random.choice(swaps)
        samples.append((
            sample["image"],
            sample["question"],
            chosen_swap["corrupted_question"],
            sample["answer"],
        ))
    return samples


# ── Visual corruption sampling ──────────────────────────────────

def collect_visual_corruption_samples(dataset, pairs_json, n_samples, transform=None):
    """Sample visual corruption pairs (corrupt image, same question).

    Args:
        dataset: CLEVRVQADataset (for loading clean images and answer mapping)
        pairs_json: path to rendered pairs JSON from render_visual_corruptions.py
        n_samples: number of samples
        transform: image transform (same as dataset's)

    Returns:
        list of (clean_image_tensor, corrupt_image_tensor, question, answer_idx)
        Note: question is the SAME for both — corruption is in the image.
    """
    import json
    from pathlib import Path
    from PIL import Image

    pairs_path = Path(pairs_json)
    with open(pairs_path) as f:
        pairs = json.load(f)

    img_dir = pairs_path.parent / "images"
    orig_img_dir = dataset.image_dir

    if transform is None:
        transform = dataset.transform

    random.shuffle(pairs)
    selected = pairs[:n_samples]

    # Build answer vocab mapping
    from data.clevr import ANSWER_TO_IDX
    vocab = ANSWER_TO_IDX

    samples = []
    for pair in selected:
        question = pair["question"]
        answer = pair["answer"]
        if answer not in vocab:
            continue

        answer_idx = vocab[answer]

        # Load clean image
        clean_path = orig_img_dir / pair["original_image"]
        clean_img = Image.open(clean_path).convert("RGB")
        if transform:
            clean_img = transform(clean_img)

        # Load corrupt image
        corrupt_path = img_dir / pair["corrupt_image"]
        corrupt_img = Image.open(corrupt_path).convert("RGB")
        if transform:
            corrupt_img = transform(corrupt_img)

        samples.append((clean_img, corrupt_img, question, answer_idx))

    return samples
