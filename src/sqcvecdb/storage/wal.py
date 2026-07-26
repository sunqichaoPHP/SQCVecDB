"""Write-Ahead Log (WAL) 实现。

WAL 用于保证持久化和崩溃恢复：所有写操作（insert/delete）先追加写入日志，
然后再更新内存索引。进程崩溃后，可以从最新快照开始，回放 WAL 日志恢复到
崩溃前的状态。

文件格式：
- wal.log：每行一个 JSON 记录，格式为 {"op": "insert"/"delete", ...}
- 顺序追加写入，每条记录后立即 flush 保证持久化

Compaction：
- 做快照时记录当前 WAL 的行号（offset）
- 快照完成后可以安全地截断 offset 之前的日志（因为快照已经包含这些状态）
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TextIO

import numpy as np


class WALWriter:
    """WAL 写入器，顺序追加日志记录。"""

    def __init__(self, wal_path: Path) -> None:
        self.wal_path = wal_path
        self._file: TextIO | None = None

    def open(self) -> None:
        """打开 WAL 文件，追加模式。"""
        if self._file is None:
            self.wal_path.parent.mkdir(parents=True, exist_ok=True)
            self._file = self.wal_path.open("a", encoding="utf-8")

    def close(self) -> None:
        """关闭 WAL 文件。"""
        if self._file is not None:
            self._file.close()
            self._file = None

    def log_insert(self, vec_id: int, vector: np.ndarray, metadata: dict[str, Any]) -> None:
        """记录 insert 操作。"""
        if self._file is None:
            raise RuntimeError("WAL 未打开，请先调用 open()")
        record = {
            "op": "insert",
            "id": int(vec_id),
            "vector": vector.tolist(),
            "metadata": metadata,
        }
        self._file.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._file.flush()

    def log_delete(self, vec_id: int) -> None:
        """记录 delete 操作。"""
        if self._file is None:
            raise RuntimeError("WAL 未打开，请先调用 open()")
        record = {"op": "delete", "id": int(vec_id)}
        self._file.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._file.flush()

    def __enter__(self) -> "WALWriter":
        self.open()
        return self

    def __exit__(self, *args) -> None:
        self.close()


class WALReader:
    """WAL 读取器，用于回放日志恢复状态。"""

    def __init__(self, wal_path: Path) -> None:
        self.wal_path = wal_path

    def replay(self, start_offset: int = 0) -> list[dict[str, Any]]:
        """从 start_offset 行开始读取 WAL，返回所有记录。

        Args:
            start_offset: 起始行号（0-indexed），用于跳过已经包含在快照中的记录。

        Returns:
            记录列表，每条记录是一个 dict，包含 op/id/vector/metadata 字段。
        """
        if not self.wal_path.exists():
            return []

        records = []
        with self.wal_path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f):
                if line_no < start_offset:
                    continue
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    records.append(record)
                except json.JSONDecodeError:
                    # 日志损坏，停止回放（避免读到不完整的记录）
                    break
        return records

    def count_lines(self) -> int:
        """返回 WAL 文件的总行数。"""
        if not self.wal_path.exists():
            return 0
        with self.wal_path.open("r", encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())


def truncate_wal(wal_path: Path, keep_offset: int) -> None:
    """截断 WAL，保留 keep_offset 行之后的记录。

    用于 compaction：快照完成后，可以安全地删除快照之前的日志。
    """
    if not wal_path.exists() or keep_offset == 0:
        return

    with wal_path.open("r", encoding="utf-8") as f:
        lines = f.readlines()

    if keep_offset >= len(lines):
        # 全部已包含在快照中，清空 WAL
        wal_path.write_text("", encoding="utf-8")
    else:
        # 保留 keep_offset 之后的行
        with wal_path.open("w", encoding="utf-8") as f:
            f.writelines(lines[keep_offset:])
