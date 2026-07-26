"""Collection：面向用户的核心入口，组合"索引 + 元数据存储 + 持久化"。

用法示例见 examples/quickstart.py。

Phase 3 新增：WAL（Write-Ahead Log）支持，启用后可保证崩溃恢复。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np

from xiangliang.distance import SUPPORTED_METRICS
from xiangliang.index.base import BaseIndex
from xiangliang.index.flat import FlatIndex
from xiangliang.index.hnsw import HNSWIndex
from xiangliang.index.ivf import IVFIndex
from xiangliang.storage.persistence import load_snapshot, save_snapshot
from xiangliang.storage.wal import WALReader, WALWriter, truncate_wal

_INDEX_REGISTRY: dict[str, type[BaseIndex]] = {
    "flat": FlatIndex,
    "ivf": IVFIndex,
    "hnsw": HNSWIndex,
}


def _create_index(index_type: str, dim: int, metric: str, index_params: dict[str, Any]) -> BaseIndex:
    index_cls = _INDEX_REGISTRY.get(index_type)
    if index_cls is None:
        raise ValueError(f"不支持的索引类型: {index_type!r}，可选值为 {tuple(_INDEX_REGISTRY)}")
    return index_cls(dim, metric, **index_params)


class Collection:
    """一个 collection 对应一组同维度的向量 + 各自的标量元数据。

    Phase 3 新增 WAL 支持：
    - 启用 WAL 后，所有写操作（insert/delete）先写日志再更新内存
    - 进程崩溃后，可以从快照 + WAL 回放恢复
    - 调用 checkpoint() 做快照并截断已持久化的 WAL
    """

    def __init__(
        self,
        dim: int,
        metric: str = "l2",
        index_type: str = "flat",
        index_params: dict[str, Any] | None = None,
        data_dir: str | Path | None = None,
        enable_wal: bool = False,
        auto_checkpoint_threshold: int = 10000,
    ) -> None:
        """创建或加载一个 collection。

        Args:
            dim: 向量维度。
            metric: 距离度量，可选 "l2" / "cosine" / "ip"。
            index_type: 索引类型，可选 "flat" / "ivf" / "hnsw"。
            index_params: 索引参数，例如 {"nlist": 100, "nprobe": 8}。
            data_dir: 数据目录，如果存在快照/WAL 会自动加载。如果启用 WAL 则必须指定。
            enable_wal: 是否启用 WAL（崩溃恢复）。
            auto_checkpoint_threshold: WAL 行数超过此阈值时自动触发快照+compaction。
        """
        if metric not in SUPPORTED_METRICS:
            raise ValueError(f"不支持的距离度量: {metric!r}，可选值为 {SUPPORTED_METRICS}")
        if enable_wal and data_dir is None:
            raise ValueError("启用 WAL 时必须指定 data_dir")

        self.dim = dim
        self.metric = metric
        self.index_type = index_type
        self.index_params = dict(index_params or {})
        self.data_dir = Path(data_dir) if data_dir else None
        self.enable_wal = enable_wal
        self.auto_checkpoint_threshold = auto_checkpoint_threshold

        self._index = _create_index(index_type, dim, metric, self.index_params)
        self._metadata: dict[int, dict[str, Any]] = {}
        self._wal_writer: WALWriter | None = None

        # 如果 data_dir 存在，尝试加载快照 + 回放 WAL
        if self.data_dir and self.data_dir.exists():
            self._load_from_disk()

        # 如果启用 WAL，打开 WAL 文件
        if self.enable_wal:
            assert self.data_dir is not None
            wal_path = self.data_dir / "wal.log"
            self._wal_writer = WALWriter(wal_path)
            self._wal_writer.open()

    def _load_from_disk(self) -> None:
        """从 data_dir 加载快照和 WAL（如果存在）。"""
        assert self.data_dir is not None
        config_path = self.data_dir / "config.json"
        wal_path = self.data_dir / "wal.log"

        # 情况 1：有快照，先加载快照再回放 WAL
        if config_path.exists():
            state = load_snapshot(self.data_dir)
            # 验证配置一致性
            if state["dim"] != self.dim or state["metric"] != self.metric:
                raise ValueError(
                    f"快照配置不匹配：期望 dim={self.dim}, metric={self.metric}，"
                    f"但快照是 dim={state['dim']}, metric={state['metric']}"
                )

            # 加载快照数据
            if state["ids"]:
                self._index.add(state["ids"], state["vectors"])
            self._metadata = state["metadata"]

            # 回放 WAL（从快照对应的 offset 之后开始）
            if wal_path.exists():
                reader = WALReader(wal_path)
                records = reader.replay(start_offset=state["wal_offset"])
                self._replay_wal_records(records)

        # 情况 2：没有快照，只有 WAL（可能是刚启动 WAL，还没做过快照）
        elif wal_path.exists():
            reader = WALReader(wal_path)
            records = reader.replay(start_offset=0)
            self._replay_wal_records(records)

    def _replay_wal_records(self, records: list[dict]) -> None:
        """回放 WAL 记录，恢复状态。"""
        for record in records:
            if record["op"] == "insert":
                vec_id = record["id"]
                vector = np.array(record["vector"], dtype=np.float32)
                metadata = record["metadata"]
                # 直接更新索引和元数据，不再写 WAL（避免重复）
                self._index.add([vec_id], vector.reshape(1, -1))
                self._metadata[vec_id] = metadata
            elif record["op"] == "delete":
                vec_id = record["id"]
                self._index.remove([vec_id])
                self._metadata.pop(vec_id, None)

    def insert(self, vec_id: int, vector: np.ndarray, metadata: dict[str, Any] | None = None) -> None:
        """插入单条向量。"""
        self.insert_many([vec_id], np.asarray(vector).reshape(1, -1), [metadata or {}])

    def insert_many(
        self,
        ids: Iterable[int],
        vectors: np.ndarray,
        metadatas: Iterable[dict[str, Any]] | None = None,
    ) -> None:
        """批量插入向量及其元数据。"""
        ids = list(ids)
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        metadatas = list(metadatas) if metadatas is not None else [{} for _ in ids]
        if len(metadatas) != len(ids):
            raise ValueError("metadatas 数量必须与 ids 一致")

        # 先写 WAL（如果启用）
        if self._wal_writer is not None:
            for vec_id, vector, metadata in zip(ids, vectors, metadatas):
                self._wal_writer.log_insert(vec_id, vector, metadata)

        # 再更新内存索引
        self._index.add(ids, vectors)
        for vec_id, meta in zip(ids, metadatas):
            self._metadata[vec_id] = meta

        # 检查是否需要自动 checkpoint
        self._maybe_auto_checkpoint()

    def delete(self, ids: Iterable[int]) -> None:
        """按 id 删除向量及其元数据。"""
        ids = list(ids)

        # 先写 WAL（如果启用）
        if self._wal_writer is not None:
            for vec_id in ids:
                self._wal_writer.log_delete(vec_id)

        # 再更新内存索引
        self._index.remove(ids)
        for vec_id in ids:
            self._metadata.pop(vec_id, None)

        # 检查是否需要自动 checkpoint
        self._maybe_auto_checkpoint()

    def get(self, vec_id: int) -> tuple[np.ndarray, dict[str, Any]] | None:
        """按 id 查询向量和元数据，不存在返回 None。"""
        vector = self._index.get_vector(vec_id)
        if vector is None:
            return None
        return vector, self._metadata.get(vec_id, {})

    def search(
        self,
        query: np.ndarray,
        top_k: int = 10,
        filter: dict[str, Any] | None = None,
    ) -> list[tuple[int, float]]:
        """检索最相似的 top_k 条向量，可选按元数据 filter 做 pre-filter。"""
        candidate_ids = self._apply_filter(filter) if filter else None
        return self._index.search(np.asarray(query), top_k, candidate_ids)

    def _apply_filter(self, filter: dict[str, Any]) -> list[int]:
        """极简的等值过滤：要求 metadata 中所有 key 都精确匹配。"""
        return [
            vec_id
            for vec_id, meta in self._metadata.items()
            if all(meta.get(key) == value for key, value in filter.items())
        ]

    def __len__(self) -> int:
        return len(self._index)

    def close(self) -> None:
        """关闭 WAL 文件。启用 WAL 时，程序退出前应调用此方法。"""
        if self._wal_writer is not None:
            self._wal_writer.close()
            self._wal_writer = None

    def checkpoint(self) -> None:
        """手动触发快照 + compaction。

        做快照时会记录当前 WAL 的行号，然后截断 WAL 中已包含在快照里的部分。
        """
        if self.data_dir is None:
            raise RuntimeError("未指定 data_dir，无法执行 checkpoint")

        wal_path = self.data_dir / "wal.log"
        reader = WALReader(wal_path)
        wal_offset = reader.count_lines()

        ids, vectors = self._index.export()

        # 截断 WAL（保留 offset 之后的记录，但当前 offset 已经是文件末尾，所以清空）
        truncate_wal(wal_path, wal_offset)

        # 保存快照，wal_offset=0（因为 WAL 已经被截断清空，后续写入从 0 开始）
        save_snapshot(
            self.data_dir,
            self.dim,
            self.metric,
            ids,
            vectors,
            self._metadata,
            index_type=self.index_type,
            index_params=self.index_params,
            wal_offset=0,
        )

        # 重要：checkpoint 后需要重新打开 WAL writer，因为 truncate 操作会影响文件句柄
        if self._wal_writer is not None:
            self._wal_writer.close()
            self._wal_writer.open()

    def _maybe_auto_checkpoint(self) -> None:
        """检查 WAL 行数，超过阈值时自动触发 checkpoint。"""
        if not self.enable_wal or self.data_dir is None:
            return
        wal_path = self.data_dir / "wal.log"
        if not wal_path.exists():
            return
        reader = WALReader(wal_path)
        if reader.count_lines() >= self.auto_checkpoint_threshold:
            self.checkpoint()

    def save(self, dir_path: str | Path | None = None) -> None:
        """将 collection 持久化到磁盘目录。

        Args:
            dir_path: 目标目录。如果启用了 WAL，则忽略此参数，直接对当前 data_dir 做快照。
                如果未启用 WAL，则必须指定 dir_path，做全量快照。

        注意：
        - 启用 WAL 时，save() 等价于 checkpoint()（增量快照+compaction）
        - 未启用 WAL 时，save() 是全量快照，索引内部结构（IVF 聚类中心、HNSW 图）
          不会被保存，load() 时会重建索引
        """
        if self.enable_wal:
            # 启用 WAL 时，直接做 checkpoint
            self.checkpoint()
        else:
            # 未启用 WAL 时，做全量快照到指定目录
            if dir_path is None:
                raise ValueError("未启用 WAL 时，save() 必须指定 dir_path")
            ids, vectors = self._index.export()
            save_snapshot(
                dir_path,
                self.dim,
                self.metric,
                ids,
                vectors,
                self._metadata,
                index_type=self.index_type,
                index_params=self.index_params,
            )

    @classmethod
    def load(cls, dir_path: str | Path, enable_wal: bool = False) -> "Collection":
        """从磁盘目录恢复 collection（会重建索引）。

        Args:
            dir_path: 数据目录。
            enable_wal: 是否启用 WAL。启用后，后续写操作会记录到 WAL。

        Returns:
            加载后的 Collection 实例。
        """
        state = load_snapshot(dir_path)
        # 构造时不传 data_dir，避免自动加载（我们手动加载）
        collection = cls(
            state["dim"],
            state["metric"],
            index_type=state["index_type"],
            index_params=state["index_params"],
            data_dir=None,
            enable_wal=False,
        )
        # 手动加载快照数据
        if state["ids"]:
            collection._index.add(state["ids"], state["vectors"])
        collection._metadata = state["metadata"]

        # 回放 WAL（如果存在）
        wal_path = Path(dir_path) / "wal.log"
        if wal_path.exists():
            reader = WALReader(wal_path)
            records = reader.replay(start_offset=state["wal_offset"])
            collection._replay_wal_records(records)

        # 最后再启用 WAL（如果需要）
        if enable_wal:
            collection.enable_wal = True
            collection.data_dir = Path(dir_path)
            wal_writer = WALWriter(wal_path)
            wal_writer.open()
            collection._wal_writer = wal_writer

        return collection
