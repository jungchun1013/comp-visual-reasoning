"""Generic balanced sampling from a dataset.

Scan a dataset once, group samples by arbitrary criteria, then sample
N items per group. Used by all analysis experiments (patching, retrieval,
probing, embedding viz).

Usage:
    index = build_index(dataset, key_fn)
    samples = sample_from_index(dataset, index, key="color", n=50)
"""

from __future__ import annotations

import random
from typing import Callable


def build_index(dataset, key_fn: Callable,
                metadata_fn: Callable | None = None) -> dict[str, list[int]]:
    """Scan dataset once, group sample indices by key_fn.

    Args:
        dataset: any indexable dataset.
        key_fn: function(item) → list of string keys this sample belongs to.
                A sample can belong to multiple groups.
                Return [] if the sample doesn't belong to any group.
        metadata_fn: optional function(dataset, idx) → item to pass to key_fn.
                     Use this to avoid expensive dataset[idx] (e.g. image loading).
                     If None, uses dataset[idx].

    Returns:
        {key: [idx, idx, ...]} — each key maps to list of dataset indices.

    Algorithm:
        Linear scan O(N). For each index, get metadata via metadata_fn (fast)
        or dataset[idx] (slow). Call key_fn to get group keys.
        Append index to each group's list.
    """
    index: dict[str, list[int]] = {}
    for idx in range(len(dataset)):
        item = metadata_fn(dataset, idx) if metadata_fn else dataset[idx]
        keys = key_fn(item)
        for k in keys:
            index.setdefault(k, []).append(idx)
    return index


def sample_from_index(
    index: dict[str, list[int]],
    key: str,
    n: int,
) -> list[int]:
    """Uniformly sample n indices from a group in the index.

    Args:
        index: from build_index()
        key: group key to sample from
        n: number of samples

    Returns:
        list of dataset indices (length = min(n, pool_size))

    Algorithm:
        random.sample(index[key], n). If pool < n, returns all available.
    """
    pool = index.get(key, [])
    return random.sample(pool, min(n, len(pool)))


def print_index_summary(index: dict[str, list[int]]) -> None:
    """Print the size of each group in the index."""
    for key in sorted(index.keys()):
        print(f"  {str(key):40s}: {len(index[key]):6d} eligible", flush=True)
