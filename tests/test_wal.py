import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

from sqcvecdb.storage.wal import WALReader, WALWriter, truncate_wal


@pytest.fixture
def tmp_dir():
    path = Path(tempfile.mkdtemp())
    yield path
    shutil.rmtree(path, ignore_errors=True)


def test_wal_writer_and_reader_roundtrip(tmp_dir):
    wal_path = tmp_dir / "test.wal"
    with WALWriter(wal_path) as writer:
        writer.log_insert(1, np.array([0.1, 0.2], dtype=np.float32), {"tag": "a"})
        writer.log_insert(2, np.array([0.3, 0.4], dtype=np.float32), {"tag": "b"})
        writer.log_delete(1)

    reader = WALReader(wal_path)
    records = reader.replay()
    assert len(records) == 3
    assert records[0]["op"] == "insert"
    assert records[0]["id"] == 1
    # 使用近似比较，因为 float32 转 JSON 再转回来有精度损失
    assert np.allclose(records[0]["vector"], [0.1, 0.2], atol=1e-5)
    assert records[0]["metadata"] == {"tag": "a"}
    assert records[2]["op"] == "delete"
    assert records[2]["id"] == 1


def test_wal_reader_replay_from_offset(tmp_dir):
    wal_path = tmp_dir / "test.wal"
    with WALWriter(wal_path) as writer:
        writer.log_insert(1, np.array([0.1], dtype=np.float32), {})
        writer.log_insert(2, np.array([0.2], dtype=np.float32), {})
        writer.log_insert(3, np.array([0.3], dtype=np.float32), {})

    reader = WALReader(wal_path)
    records = reader.replay(start_offset=1)
    assert len(records) == 2
    assert records[0]["id"] == 2
    assert records[1]["id"] == 3


def test_truncate_wal(tmp_dir):
    wal_path = tmp_dir / "test.wal"
    with WALWriter(wal_path) as writer:
        writer.log_insert(1, np.array([0.1], dtype=np.float32), {})
        writer.log_insert(2, np.array([0.2], dtype=np.float32), {})
        writer.log_insert(3, np.array([0.3], dtype=np.float32), {})

    truncate_wal(wal_path, keep_offset=1)
    reader = WALReader(wal_path)
    records = reader.replay()
    assert len(records) == 2
    assert records[0]["id"] == 2


def test_truncate_wal_clears_all_if_offset_exceeds_length(tmp_dir):
    wal_path = tmp_dir / "test.wal"
    with WALWriter(wal_path) as writer:
        writer.log_insert(1, np.array([0.1], dtype=np.float32), {})

    truncate_wal(wal_path, keep_offset=10)
    assert wal_path.read_text() == ""
