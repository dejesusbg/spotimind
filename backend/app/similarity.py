"""Cosine nearest neighbors and the brain-vs-audio overlap metric."""

from __future__ import annotations

import numpy as np


def cosine_matrix(emb: np.ndarray) -> np.ndarray:
    """Pairwise cosine similarity. Rows are re-normalized defensively, so un-normalized input is fine."""
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    unit = emb / np.maximum(norms, 1e-12)
    return unit @ unit.T


def top_k(sims: np.ndarray, i: int, k: int) -> list[tuple[int, float]]:
    """Top-k (index, score) for row i of a similarity matrix, excluding i itself. Ties break by index."""
    row = sims[i].astype(np.float64).copy()
    row[i] = -np.inf
    k = max(0, min(k, len(row) - 1))
    order = np.lexsort((np.arange(len(row)), -row))[:k]
    return [(int(j), float(row[j])) for j in order]


def overlap(a: list[int], b: list[int]) -> dict:
    """Shared items between two neighbor lists and their Jaccard index."""
    sa, sb = set(a), set(b)
    union = sa | sb
    shared = sa & sb
    return {
        "shared": len(shared),
        "shared_ids": sorted(shared),
        "jaccard": len(shared) / len(union) if union else 0.0,
    }
