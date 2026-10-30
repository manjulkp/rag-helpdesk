from __future__ import annotations

import numpy as np
import pytest

faiss = pytest.importorskip("faiss")

from scripts.faiss_lab import ann_recall, normalise  # noqa: E402


def random_unit_vectors(n: int, d: int = 384, seed: int = 0) -> np.ndarray:
    return normalise(np.random.default_rng(seed).standard_normal((n, d)))


def flat_top10(xb: np.ndarray, xq: np.ndarray) -> np.ndarray:
    index = faiss.IndexFlatIP(xb.shape[1])
    index.add(xb)
    return index.search(xq, 10)[1]


# a. the ANN recall function: 1.0 when results match exact search, 0.5 when half do
def test_ann_recall_full_and_half() -> None:
    truth = np.array([list(range(10)), list(range(10, 20))])
    assert ann_recall(truth, truth) == 1.0
    half = np.array([list(range(5)) + [90, 91, 92, 93, 94], list(range(10, 15)) + [-1] * 5])
    assert ann_recall(half, truth) == 0.5


# b. HNSW with efSearch=256 on 200 random normalised vectors returns the same top 10 as Flat
def test_hnsw_with_large_ef_matches_flat() -> None:
    xb, xq = random_unit_vectors(200), random_unit_vectors(20, seed=1)
    index = faiss.IndexHNSWFlat(xb.shape[1], 16, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efSearch = 256
    index.add(xb)
    assert np.array_equal(index.search(xq, 10)[1], flat_top10(xb, xq))


# c. an IVF index with nprobe = nlist returns the same results as Flat (every cluster searched)
def test_ivf_searching_every_cluster_is_exact() -> None:
    xb, xq = random_unit_vectors(400), random_unit_vectors(20, seed=1)
    quantizer = faiss.IndexFlatIP(xb.shape[1])
    index = faiss.IndexIVFFlat(quantizer, xb.shape[1], 8, faiss.METRIC_INNER_PRODUCT)
    index.train(xb)
    index.add(xb)
    index.nprobe = 8
    assert np.array_equal(index.search(xq, 10)[1], flat_top10(xb, xq))


def test_l2_and_inner_product_rank_unit_vectors_the_same() -> None:
    xb, xq = random_unit_vectors(300), random_unit_vectors(10, seed=1)
    l2 = faiss.IndexFlatL2(xb.shape[1])
    l2.add(xb)
    # |a - b|^2 = 2 - 2 a.b for unit vectors: same order, so the same top 10
    assert np.array_equal(l2.search(xq, 10)[1], flat_top10(xb, xq))
