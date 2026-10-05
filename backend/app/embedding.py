"""Song embeddings from parcel timelines (brain) and pooled audio features (baseline).

The Colab notebook (cell [C7]) carries an identical copy of these functions so it stays
self-contained; docs/DATA_CONTRACT.md is the spec both follow.
"""

from __future__ import annotations

import numpy as np

EPS = 1e-6


def time_bins(x: np.ndarray, n_bins: int) -> np.ndarray:
    """[T, K] -> [n_bins, K]: split time into n_bins near-equal chunks and average each."""
    if x.shape[0] < n_bins:
        raise ValueError(f"timeline has {x.shape[0]} rows, need at least {n_bins}")
    return np.stack([chunk.mean(0) for chunk in np.array_split(x, n_bins, axis=0)])


def raw_brain_blocks(timeline: np.ndarray, netmat: np.ndarray, n_bins: int) -> dict[str, np.ndarray]:
    """Un-normalized feature blocks for one song. timeline [T, P], netmat [P, K] (parcel -> network mean)."""
    return {
        "mean": timeline.mean(0),
        "std": timeline.std(0),
        "shape": time_bins(timeline @ netmat, n_bins).reshape(-1),
    }


def zscore_columns(x: np.ndarray) -> np.ndarray:
    """Catalog-level z-score: each feature standardized across songs (rows)."""
    return (x - x.mean(0)) / np.maximum(x.std(0), EPS)


def l2_rows(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), EPS)


def brain_embeddings(
    timelines: list[np.ndarray], netmat: np.ndarray, n_bins: int, weights: dict[str, float]
) -> tuple[np.ndarray, list[dict]]:
    """[N, D] brain embeddings: z-score per feature across the catalog, L2 per block, weight, concat, L2."""
    per_song = [raw_brain_blocks(t, netmat, n_bins) for t in timelines]
    blocks, info = [], []
    for name in ("mean", "std", "shape"):
        x = np.stack([s[name] for s in per_song]).astype(np.float64)
        blocks.append(weights[name] * l2_rows(zscore_columns(x)))
        info.append({"name": name, "dim": int(x.shape[1]), "weight": float(weights[name])})
    return l2_rows(np.concatenate(blocks, axis=1)).astype(np.float32), info


def audio_embeddings(vectors: list[np.ndarray]) -> np.ndarray:
    """[N, D] baseline embeddings: catalog z-score, then L2 rows."""
    return l2_rows(zscore_columns(np.stack(vectors).astype(np.float64))).astype(np.float32)
