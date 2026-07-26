import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

from xiangliang.collection import Collection


@pytest.fixture
def tmp_dir():
    path = Path(tempfile.mkdtemp())
    yield path
    shutil.rmtree(path, ignore_errors=True)


@pytest.mark.parametrize(
    "index_type,index_params",
    [
        ("flat", {}),
        ("ivf", {"nlist": 4, "nprobe": 2}),
        ("hnsw", {"M": 8, "ef_construction": 50, "ef_search": 50}),
    ],
)
def test_collection_works_with_all_index_types(index_type, index_params):
    col = Collection(dim=8, metric="l2", index_type=index_type, index_params=index_params)
    rng = np.random.default_rng(1)
    vectors = rng.random((30, 8)).astype(np.float32)
    col.insert_many(
        ids=list(range(30)),
        vectors=vectors,
        metadatas=[{"tag": "even" if i % 2 == 0 else "odd"} for i in range(30)],
    )

    assert len(col) == 30
    results = col.search(vectors[0], top_k=5)
    assert len(results) == 5

    filtered = col.search(vectors[0], top_k=5, filter={"tag": "even"})
    assert all(vec_id % 2 == 0 for vec_id, _ in filtered)


@pytest.mark.parametrize(
    "index_type,index_params",
    [
        ("flat", {}),
        ("ivf", {"nlist": 4, "nprobe": 2}),
        ("hnsw", {"M": 8, "ef_construction": 50, "ef_search": 50}),
    ],
)
def test_collection_save_and_load_roundtrip(tmp_dir, index_type, index_params):
    col = Collection(dim=6, metric="l2", index_type=index_type, index_params=index_params)
    rng = np.random.default_rng(2)
    vectors = rng.random((15, 6)).astype(np.float32)
    col.insert_many(
        ids=list(range(15)),
        vectors=vectors,
        metadatas=[{"tag": str(i % 3)} for i in range(15)],
    )
    col.save(tmp_dir)

    loaded = Collection.load(tmp_dir)
    assert loaded.index_type == index_type
    assert len(loaded) == 15

    results = loaded.search(vectors[0], top_k=3)
    assert len(results) == 3
