import numpy as np
import pytest

from xiangliang.index.flat import FlatIndex


def make_index(metric="l2", dim=4):
    return FlatIndex(dim=dim, metric=metric)


def test_add_and_len():
    index = make_index()
    index.add([1, 2, 3], np.random.rand(3, 4).astype(np.float32))
    assert len(index) == 3


def test_add_duplicate_id_raises():
    index = make_index()
    index.add([1], np.random.rand(1, 4).astype(np.float32))
    with pytest.raises(ValueError):
        index.add([1], np.random.rand(1, 4).astype(np.float32))


def test_search_l2_returns_exact_nearest():
    index = make_index(metric="l2", dim=2)
    vectors = np.array([[0, 0], [1, 1], [5, 5], [10, 10]], dtype=np.float32)
    index.add([10, 11, 12, 13], vectors)

    result = index.search(np.array([0.1, 0.1], dtype=np.float32), top_k=2)
    ids = [r[0] for r in result]
    assert ids == [10, 11]


def test_search_respects_candidate_ids():
    index = make_index(metric="l2", dim=2)
    vectors = np.array([[0, 0], [1, 1], [5, 5]], dtype=np.float32)
    index.add([1, 2, 3], vectors)

    result = index.search(np.array([0, 0], dtype=np.float32), top_k=3, candidate_ids=[2, 3])
    ids = [r[0] for r in result]
    assert ids == [2, 3]
    assert 1 not in ids


def test_remove_then_search():
    index = make_index(metric="l2", dim=2)
    vectors = np.array([[0, 0], [1, 1], [5, 5]], dtype=np.float32)
    index.add([1, 2, 3], vectors)
    index.remove([1])

    assert len(index) == 2
    result = index.search(np.array([0, 0], dtype=np.float32), top_k=3)
    ids = [r[0] for r in result]
    assert 1 not in ids
    assert set(ids) == {2, 3}


def test_snapshot_roundtrip():
    index = make_index(metric="cosine", dim=3)
    vectors = np.random.rand(5, 3).astype(np.float32)
    ids = [100, 101, 102, 103, 104]
    index.add(ids, vectors)

    snap_ids, snap_vectors = index.export()
    restored = FlatIndex(dim=3, metric="cosine")
    restored.add(snap_ids, snap_vectors)

    assert len(restored) == len(index)
    for vec_id in ids:
        assert restored.get_vector(vec_id) is not None
