import numpy as np
import pytest

from app.embedding import audio_embeddings, brain_embeddings, time_bins
from app.similarity import cosine_matrix, overlap, top_k


def test_cosine_matrix_known_values():
    emb = np.array([[1, 0], [0, 2], [1, 1]], float)
    sims = cosine_matrix(emb)
    assert sims[0, 1] == pytest.approx(0.0)
    assert sims[0, 2] == pytest.approx(1 / np.sqrt(2))
    assert np.allclose(np.diag(sims), 1.0)


def test_top_k_excludes_self_and_sorts():
    emb = np.array([[1, 0], [0.9, 0.1], [0, 1], [0.5, 0.5]], float)
    res = top_k(cosine_matrix(emb), 0, 10)
    ids = [j for j, _ in res]
    assert 0 not in ids
    assert ids == [1, 3, 2]  # k is clipped to N-1
    scores = [s for _, s in res]
    assert scores == sorted(scores, reverse=True)


def test_top_k_ties_break_by_index():
    sims = np.array([[1, 0.5, 0.5, 0.5]] * 4)
    assert [j for j, _ in top_k(sims, 0, 3)] == [1, 2, 3]


def test_overlap_and_jaccard():
    ov = overlap([1, 2, 3, 4], [3, 4, 5, 6])
    assert ov["shared"] == 2 and ov["shared_ids"] == [3, 4]
    assert ov["jaccard"] == pytest.approx(2 / 6)
    assert overlap([], [])["jaccard"] == 0.0
    assert overlap([1, 2], [1, 2])["jaccard"] == 1.0


def test_time_bins_shape_and_values():
    x = np.arange(20, dtype=float).reshape(10, 2)
    b = time_bins(x, 5)
    assert b.shape == (5, 2)
    assert np.allclose(b[:, 0], [1, 5, 9, 13, 17])
    with pytest.raises(ValueError):
        time_bins(x, 11)


def test_brain_embeddings_normalized_and_dims():
    rng = np.random.default_rng(0)
    P, K = 6, 2
    netmat = np.zeros((P, K))
    netmat[:3, 0] = 1 / 3
    netmat[3:, 1] = 1 / 3
    tls = [rng.standard_normal((rng.integers(20, 40), P)) for _ in range(5)]
    emb, info = brain_embeddings(tls, netmat, 4, {"mean": 1, "std": 1, "shape": 1})
    assert emb.shape == (5, P + P + 4 * K)
    assert [b["dim"] for b in info] == [P, P, 8]
    assert np.allclose(np.linalg.norm(emb, axis=1), 1, atol=1e-5)
    assert emb.dtype == np.float32


def test_identical_songs_have_similarity_one():
    rng = np.random.default_rng(1)
    netmat = np.full((4, 1), 0.25)
    base = rng.standard_normal((30, 4))
    tls = [base, base.copy(), rng.standard_normal((30, 4)), rng.standard_normal((25, 4))]
    emb, _ = brain_embeddings(tls, netmat, 3, {"mean": 1, "std": 1, "shape": 1})
    assert cosine_matrix(emb)[0, 1] == pytest.approx(1.0, abs=1e-5)


def test_audio_embeddings_unit_rows():
    rng = np.random.default_rng(2)
    emb = audio_embeddings([rng.standard_normal(16) for _ in range(4)])
    assert np.allclose(np.linalg.norm(emb, axis=1), 1, atol=1e-5)
