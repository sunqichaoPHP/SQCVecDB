"""快照持久化 + WAL offset 管理。

Phase 3 版本：支持增量快照 + WAL。快照时记录当前 WAL 的行号（offset），
重启时先加载快照，再从 offset 位置开始回放 WAL，实现崩溃恢复。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

_VECTORS_FILE = "vectors.npz"
_METADATA_FILE = "metadata.json"
_CONFIG_FILE = "config.json"
_WAL_OFFSET_FILE = "wal_offset.txt"


def save_snapshot(
    dir_path: str | Path,
    dim: int,
    metric: str,
    ids: list[int],
    vectors: np.ndarray,
    metadata: dict[int, dict[str, Any]],
    index_type: str = "flat",
    index_params: dict[str, Any] | None = None,
    wal_offset: int = 0,
) -> None:
    """将 collection 的当前状态整体写入 dir_path 目录。

    Args:
        wal_offset: 快照对应的 WAL 行号，用于 compaction（截断 offset 之前的日志）。
    """
    dir_path = Path(dir_path)
    dir_path.mkdir(parents=True, exist_ok=True)

    ids_array = np.asarray(ids, dtype=np.int64)
    np.savez(dir_path / _VECTORS_FILE, ids=ids_array, vectors=vectors.astype(np.float32))

    # json 的 key 必须是字符串，落盘时把 id 转成 str，加载时再转回 int
    metadata_serializable = {str(vec_id): meta for vec_id, meta in metadata.items()}
    (dir_path / _METADATA_FILE).write_text(
        json.dumps(metadata_serializable, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    config = {
        "dim": dim,
        "metric": metric,
        "index_type": index_type,
        "index_params": index_params or {},
    }
    (dir_path / _CONFIG_FILE).write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    (dir_path / _WAL_OFFSET_FILE).write_text(str(wal_offset), encoding="utf-8")


def load_snapshot(dir_path: str | Path) -> dict[str, Any]:
    """从 dir_path 目录加载快照，返回 dim/metric/ids/vectors/metadata。"""
    dir_path = Path(dir_path)
    vectors_path = dir_path / _VECTORS_FILE
    config_path = dir_path / _CONFIG_FILE
    metadata_path = dir_path / _METADATA_FILE

    if not vectors_path.exists() or not config_path.exists():
        raise FileNotFoundError(f"目录 {dir_path} 下没有找到有效的快照文件")

    config = json.loads(config_path.read_text(encoding="utf-8"))

    with np.load(vectors_path) as npz:
        ids = npz["ids"].tolist()
        vectors = npz["vectors"]

    metadata: dict[int, dict[str, Any]] = {}
    if metadata_path.exists():
        raw_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata = {int(vec_id): meta for vec_id, meta in raw_metadata.items()}

    wal_offset = 0
    offset_path = dir_path / _WAL_OFFSET_FILE
    if offset_path.exists():
        wal_offset = int(offset_path.read_text(encoding="utf-8").strip())

    return {
        "dim": config["dim"],
        "metric": config["metric"],
        "index_type": config.get("index_type", "flat"),
        "index_params": config.get("index_params", {}),
        "ids": ids,
        "vectors": vectors,
        "metadata": metadata,
        "wal_offset": wal_offset,
    }
