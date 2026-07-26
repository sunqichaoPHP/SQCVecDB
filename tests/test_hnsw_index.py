import numpy as np
import pytest

from sqcvecdb.index.flat import FlatIndex
from sqcvecdb.index.hnsw import HNSWIndex


def test_add_and_len():
    index = HNSWIndex(dim=4, metric="l2", M=4, ef_construction=20, ef_search=20)
    vectors = np.random.rand(30, 4).astype(np.float32)
    index.add(list(range(30)), vectors)
    assert len(index) == 30


def test_add_duplicate_id_raises():
    index = HNSWIndex(dim=2, metric="l2", M=4)
    index.add([1], np.random.rand(1, 2).astype(np.float32))
    with pytest.raises(ValueError):
        index.add([1], np.random.rand(1, 2).astype(np.float32))


def test_search_on_empty_index_returns_empty():
    index = HNSWIndex(dim=2, metric="l2", M=4)
    assert index.search(np.array([0, 0], dtype=np.float32), top_k=5) == []


def test_search_finds_exact_nearest_in_small_dataset():
    index = HNSWIndex(dim=2, metric="l2", M=8, ef_construction=50, ef_search=50)
    vectors = np.array([[0, 0], [1, 1], [5, 5], [10, 10]], dtype=np.float32)
    index.add([10, 11, 12, 13], vectors)

    result = index.search(np.array([0.1, 0.1], dtype=np.float32), top_k=2)
    ids = [r[0] for r in result]
    assert ids[0] == 10


def test_remove_is_soft_delete_and_excluded_from_search():
    index = HNSWIndex(dim=2, metric="l2", M=8, ef_construction=50, ef_search=50)
    vectors = np.array([[0, 0], [1, 1], [5, 5]], dtype=np.float32)
    index.add([1, 2, 3], vectors)
    index.remove([1])

    assert len(index) == 2
    assert index.get_vector(1) is None
    result = index.search(np.array([0, 0], dtype=np.float32), top_k=3)
    ids = [r[0] for r in result]
    assert 1 not in ids


def test_recall_against_flat_index_is_reasonably_high():
    rng = np.random.default_rng(0)
    dim = 16
    n = 300
    vectors = rng.random((n, dim)).astype(np.float32)
    ids = list(range(n))

    flat = FlatIndex(dim=dim, metric="l2")
    flat.add(ids, vectors)

    hnsw = HNSWIndex(dim=dim, metric="l2", M=16, ef_construction=100, ef_search=64)
    hnsw.add(ids, vectors)

    queries = rng.random((20, dim)).astype(np.float32)
    top_k = 10
    hits = 0
    total = 0
    for query in queries:
        exact_ids = {vec_id for vec_id, _ in flat.search(query, top_k)}
        approx_ids = {vec_id for vec_id, _ in hnsw.search(query, top_k)}
        hits += len(exact_ids & approx_ids)
        total += top_k

    recall = hits / total
    assert recall > 0.7
