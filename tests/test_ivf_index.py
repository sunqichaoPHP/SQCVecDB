import numpy as np
import pytest

from sqcvecdb.index.flat import FlatIndex
from sqcvecdb.index.ivf import IVFIndex


def test_add_triggers_training_and_len():
    index = IVFIndex(dim=4, metric="l2", nlist=4, nprobe=2)
    vectors = np.random.rand(20, 4).astype(np.float32)
    index.add(list(range(20)), vectors)

    assert index.is_trained
    assert len(index) == 20


def test_add_duplicate_id_raises():
    index = IVFIndex(dim=2, metric="l2", nlist=2, nprobe=1)
    index.add([1, 2], np.random.rand(2, 2).astype(np.float32))
    with pytest.raises(ValueError):
        index.add([1], np.random.rand(1, 2).astype(np.float32))


def test_search_before_training_returns_empty():
    index = IVFIndex(dim=2, metric="l2", nlist=2, nprobe=1)
    assert index.search(np.array([0, 0], dtype=np.float32), top_k=5) == []


def test_remove_then_search():
    index = IVFIndex(dim=2, metric="l2", nlist=2, nprobe=2)
    vectors = np.array([[0, 0], [1, 1], [5, 5], [6, 6]], dtype=np.float32)
    index.add([1, 2, 3, 4], vectors)
    index.remove([1])

    assert len(index) == 3
    result = index.search(np.array([0, 0], dtype=np.float32), top_k=4)
    ids = [r[0] for r in result]
    assert 1 not in ids
    assert set(ids) == {2, 3, 4}


def test_recall_against_flat_index_is_reasonably_high():
    rng = np.random.default_rng(0)
    dim = 16
    n = 500
    vectors = rng.random((n, dim)).astype(np.float32)
    ids = list(range(n))

    flat = FlatIndex(dim=dim, metric="l2")
    flat.add(ids, vectors)

    ivf = IVFIndex(dim=dim, metric="l2", nlist=20, nprobe=10)
    ivf.add(ids, vectors)

    queries = rng.random((20, dim)).astype(np.float32)
    top_k = 10
    hits = 0
    total = 0
    for query in queries:
        exact_ids = {vec_id for vec_id, _ in flat.search(query, top_k)}
        approx_ids = {vec_id for vec_id, _ in ivf.search(query, top_k)}
        hits += len(exact_ids & approx_ids)
        total += top_k

    recall = hits / total
    assert recall > 0.6  # nprobe 覆盖了一半的桶，召回率应该明显好于随机
