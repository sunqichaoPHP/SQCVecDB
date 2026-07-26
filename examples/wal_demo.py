"""WAL 崩溃恢复示例。

演示如何使用 WAL（Write-Ahead Log）保证数据持久化和崩溃恢复。

运行方式（在项目根目录下）：
    pip install -e ".[dev]"
    python examples/wal_demo.py
"""

import shutil
import tempfile
from pathlib import Path

import numpy as np

from sqcvecdb import Collection


def main() -> None:
    data_dir = Path(tempfile.mkdtemp())
    print(f"数据目录: {data_dir}")

    try:
        # 第一阶段：启用 WAL，插入数据
        print("\n[第一阶段] 启用 WAL，写入 100 条数据...")
        col = Collection(dim=4, metric="l2", data_dir=data_dir, enable_wal=True)
        rng = np.random.default_rng(seed=42)
        for i in range(100):
            col.insert(i, rng.random(4).astype(np.float32), metadata={"id": i})
        print(f"  已插入 {len(col)} 条数据")

        # 手动触发一次 checkpoint（快照 + compaction）
        print("  手动触发 checkpoint...")
        col.checkpoint()
        wal_path = data_dir / "wal.log"
        wal_size_after_checkpoint = wal_path.stat().st_size if wal_path.exists() else 0
        print(f"  checkpoint 后 WAL 大小: {wal_size_after_checkpoint} bytes")

        # 继续写入数据（这些会记录在新的 WAL 中）
        print("  继续写入 50 条数据...")
        for i in range(100, 150):
            col.insert(i, rng.random(4).astype(np.float32), metadata={"id": i})
        print(f"  当前共 {len(col)} 条数据")
        wal_size_after_more_inserts = wal_path.stat().st_size if wal_path.exists() else 0
        print(f"  写入 50 条后 WAL 大小: {wal_size_after_more_inserts} bytes")

        # 模拟崩溃：不调用 close()，直接删除 col 对象
        print("\n[模拟崩溃] 进程异常终止，未调用 close()...")
        del col

        # 第二阶段：重启，从快照 + WAL 恢复
        print("\n[第二阶段] 重启进程，从快照 + WAL 恢复...")
        recovered = Collection(dim=4, metric="l2", data_dir=data_dir, enable_wal=True)
        print(f"  恢复后共 {len(recovered)} 条数据")
        assert len(recovered) == 150, "崩溃恢复失败，数据丢失！"

        # 验证数据完整性
        for i in range(150):
            record = recovered.get(i)
            assert record is not None, f"id={i} 丢失！"
            assert record[1]["id"] == i, f"id={i} 的元数据不一致！"

        print("  ✓ 所有数据恢复成功，元数据完整")

        # 继续正常工作
        print("\n[继续工作] 删除部分数据并再次 checkpoint...")
        for i in range(0, 50):
            recovered.delete([i])
        print(f"  删除 50 条后剩余 {len(recovered)} 条")
        recovered.checkpoint()
        print("  checkpoint 完成")

        recovered.close()
        print("\n✓ WAL 示例运行成功")

    finally:
        shutil.rmtree(data_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
