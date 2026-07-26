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


def test_collection_with_wal_basic_operations(tmp_dir):
    col = Collection(dim=2, metric="l2", data_dir=tmp_dir, enable_wal=True)
    col.insert(1, [0.1, 0.2], metadata={"tag": "a"})
    col.insert(2, [0.3, 0.4], metadata={"tag": "b"})
    assert len(col) == 2

    col.delete([1])
    assert len(col) == 1
    assert col.get(1) is None

    col.close()


def test_collection_crash_recovery_from_wal(tmp_dir):
    # 第一阶段：写入数据（模拟进程运行中）
    col = Collection(dim=2, metric="l2", data_dir=tmp_dir, enable_wal=True)
    col.insert(1, [0.1, 0.2], metadata={"tag": "a"})
    col.insert(2, [0.3, 0.4], metadata={"tag": "b"})
    col.insert(3, [0.5, 0.6], metadata={"tag": "c"})
    col.delete([2])
    # 模拟崩溃：不调用 close()，直接删除 col
    del col

    # 第二阶段：重启并加载（从 WAL 恢复）
    recovered = Collection(dim=2, metric="l2", data_dir=tmp_dir, enable_wal=True)
    assert len(recovered) == 2  # id=1 和 id=3，id=2 被删了
    assert recovered.get(1) is not None
    assert recovered.get(2) is None
    assert recovered.get(3) is not None
    assert recovered.get(1)[1] == {"tag": "a"}

    recovered.close()


def test_collection_checkpoint_and_compaction(tmp_dir):
    col = Collection(dim=2, metric="l2", data_dir=tmp_dir, enable_wal=True)
    for i in range(100):
        col.insert(i, [float(i), float(i)], metadata={"id": i})

    wal_path = tmp_dir / "wal.log"
    assert wal_path.exists()
    wal_lines_before = len(wal_path.read_text().strip().split("\n"))
    assert wal_lines_before == 100

    # 手动触发 checkpoint
    col.checkpoint()

    # WAL 应该被清空（因为快照包含了所有数据）
    wal_lines_after = len(wal_path.read_text().strip().split("\n")) if wal_path.read_text().strip() else 0
    assert wal_lines_after == 0

    # 快照应该正确保存
    snapshot_path = tmp_dir / "vectors.npz"
    assert snapshot_path.exists()

    # 重新加载，数据应该完整
    col.close()
    col2 = Collection(dim=2, metric="l2", data_dir=tmp_dir, enable_wal=True)
    assert len(col2) == 100
    col2.close()


def test_collection_auto_checkpoint(tmp_dir):
    # 设置很小的阈值，触发自动 checkpoint
    col = Collection(
        dim=2, metric="l2", data_dir=tmp_dir, enable_wal=True, auto_checkpoint_threshold=10
    )
    for i in range(15):
        col.insert(i, [float(i), float(i)], metadata={})

    # 应该已经自动触发了 checkpoint（10条后触发，然后又写了5条）
    wal_path = tmp_dir / "wal.log"
    wal_lines = len(wal_path.read_text().strip().split("\n")) if wal_path.read_text().strip() else 0
    assert wal_lines < 15  # 应该已经compaction过，WAL行数少于总写入次数

    col.close()


def test_collection_backward_compatibility_without_wal(tmp_dir):
    # 不启用 WAL，行为应该和 Phase 1-2 一致
    col = Collection(dim=2, metric="l2")
    col.insert(1, [0.1, 0.2], metadata={"tag": "a"})
    col.save(tmp_dir)

    col2 = Collection.load(tmp_dir, enable_wal=False)
    assert len(col2) == 1
    assert col2.get(1)[1] == {"tag": "a"}


def test_collection_load_with_wal_enabled(tmp_dir):
    # 先用不启用 WAL 的方式保存
    col = Collection(dim=2, metric="l2")
    col.insert(1, [0.1, 0.2], metadata={"tag": "a"})
    col.save(tmp_dir)

    # 然后用启用 WAL 的方式加载
    col2 = Collection.load(tmp_dir, enable_wal=True)
    col2.insert(2, [0.3, 0.4], metadata={"tag": "b"})
    col2.close()

    # WAL 应该记录了 id=2 的插入
    wal_path = tmp_dir / "wal.log"
    assert wal_path.exists()

    # 重启，应该能从快照+WAL恢复两条数据
    col3 = Collection(dim=2, metric="l2", data_dir=tmp_dir, enable_wal=True)
    assert len(col3) == 2
    col3.close()
