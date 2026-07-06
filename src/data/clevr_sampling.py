"""CLEVR family-balanced query sampling.

Ported from SteerViT-legacy/exp_vqa/analysis/utils/{sampling,retrieval_sampling}.py

Usage:
    from data.clevr_sampling import RETRIEVAL_CATEGORIES, build_family_index, sample_queries

    index = build_family_index(dataset)
    queries = sample_queries(dataset, index, "attr_query_direct", n_total=72)
"""

from __future__ import annotations

import random

# ── Category → family mapping (from EXPERIMENT_SETUP.md §7) ─────
# {category_name: [family_ids]} — sorted by depth (simplest first), ≤6

RETRIEVAL_CATEGORIES = {
    "attr_query_direct": [86, 87, 88, 89],
    "attr_query_same": [53, 59, 55, 57, 61, 60],
    "attr_query_spatial": [76, 74, 75, 77, 80, 81],
    "exist_direct": [85],
    "exist_same": [36, 38, 39, 37, 47, 44],
    "exist_spatial": [73, 26, 79],
    "count_direct": [84],
    "count_same": [40, 41, 43, 42, 48, 49],
    "count_spatial": [72, 78, 25],
    "attr_compare_direct": [9, 11, 12, 10],
    "attr_compare_spatial": [23, 13, 18, 20, 14, 16],
}


# ── Generic index/sample utilities ──────────────────────────────

def build_family_index(dataset):
    """Build {family_key: [dataset_idx, ...]} from dataset questions.

    Text-only scan (no image loading). Returns e.g. {"F86": [0, 5, 12, ...], ...}
    """
    index: dict[str, list[int]] = {}
    for idx in range(len(dataset)):
        fam = dataset.questions[idx].get("question_family_index", -1)
        key = f"F{fam}"
        index.setdefault(key, []).append(idx)
    print(f"Family index: {len(index)} families, {len(dataset)} questions", flush=True)
    return index


def sample_queries(dataset, index, category, n_total=72, rng=None,
                   scenes=None, min_obj=3, max_obj=5):
    """Sample n_total queries for a category, evenly across families.

    Algorithm:
        n_per_family = n_total // n_families
        remainder distributed to first families (+1 each)
        If scenes provided, filter to queries whose scene has min_obj..max_obj objects.
        Oversamples 5x then filters (same as legacy run_retrieval_by_category.py).

    Returns:
        list of dicts: {dataset_idx, family, question, answer, program}
    """
    if rng is None:
        rng = random.Random(42)

    families = RETRIEVAL_CATEGORIES.get(category, [])
    if not families:
        return []

    # Oversample if filtering by scene object count
    oversample = 5 if scenes is not None else 1

    n_families = len(families)
    base = n_total * oversample // n_families
    remainder = (n_total * oversample) % n_families

    raw_queries = []
    for i, fam in enumerate(families):
        n = base + (1 if i < remainder else 0)
        pool = index.get(f"F{fam}", [])
        sampled = rng.sample(pool, min(n, len(pool)))
        for idx in sampled:
            q_data = dataset.questions[idx]
            raw_queries.append({
                "dataset_idx": idx,
                "family": fam,
                "question": q_data["question"],
                "answer": q_data.get("answer", "?"),
                "program": q_data.get("program", []),
            })

    # Filter by object count if scenes provided
    if scenes is not None:
        filtered = []
        for q in raw_queries:
            fname = dataset.questions[q["dataset_idx"]]["image_filename"]
            scene = scenes.get(fname)
            if scene and min_obj <= len(scene["objects"]) <= max_obj:
                filtered.append(q)
        raw_queries = filtered

    # Take n_total, balanced across families
    per_family_target = {}
    base = n_total // n_families
    remainder = n_total % n_families
    for i, fam in enumerate(families):
        per_family_target[fam] = base + (1 if i < remainder else 0)

    queries = []
    family_count = {fam: 0 for fam in families}
    for q in raw_queries:
        fam = q["family"]
        if family_count[fam] < per_family_target[fam]:
            queries.append(q)
            family_count[fam] += 1
        if len(queries) >= n_total:
            break

    for fam in families:
        print(f"    F{fam}: {family_count[fam]} queries")
    return queries
