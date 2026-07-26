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


def test_insert_and_search():
    col = Collection(dim=3, metric="l2")
    col.insert(1, [0, 0, 0], metadata={"category": "a"})
    col.insert(2, [1, 1, 1], metadata={"category": "b"})

    result = col.search([0, 0, 0], top_k=1)
    assert result[0][0] == 1


def test_insert_many_and_metadata_filter():
    col = Collection(dim=2, metric="l2")
    col.insert_many(
        ids=[1, 2, 3],
        vectors=np.array([[0, 0], [1, 1], [2, 2]], dtype=np.float32),
        metadatas=[{"tag": "x"}, {"tag": "y"}, {"tag": "x"}],
    )

    result = col.search([0, 0], top_k=3, filter={"tag": "x"})
    ids = [r[0] for r in result]
    assert set(ids) == {1, 3}


def test_delete_and_get():
    col = Collection(dim=2, metric="l2")
    col.insert(1, [1, 2], metadata={"name": "foo"})
    assert col.get(1) is not None

    col.delete([1])
    assert col.get(1) is None
    assert len(col) == 0


def test_save_and_load_roundtrip(tmp_dir):
    col = Collection(dim=2, metric="cosine")
    col.insert_many(
        ids=[1, 2, 3],
        vectors=np.array([[1, 0], [0, 1], [1, 1]], dtype=np.float32),
        metadatas=[{"tag": "x"}, {"tag": "y"}, {"tag": "x"}],
    )
    col.save(tmp_dir)

    loaded = Collection.load(tmp_dir)
    assert len(loaded) == 3
    assert loaded.get(2)[1] == {"tag": "y"}

    result = loaded.search([1, 0], top_k=1, filter={"tag": "x"})
    assert result[0][0] == 1
