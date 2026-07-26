"""索引层的统一接口定义。

后续新增 IVF / HNSW 等索引实现时，都应遵循这个接口，方便在 Collection 中
按需切换索引类型。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable

import numpy as np


class BaseIndex(ABC):
    """向量索引的抽象基类。"""

    def __init__(self, dim: int, metric: str) -> None:
        self.dim = dim
        self.metric = metric

    @abstractmethod
    def add(self, ids: Iterable[int], vectors: np.ndarray) -> None:
        """批量插入向量。"""

    @abstractmethod
    def remove(self, ids: Iterable[int]) -> None:
        """按 id 删除向量。"""

    @abstractmethod
    def search(
        self,
        query: np.ndarray,
        top_k: int,
        candidate_ids: Iterable[int] | None = None,
    ) -> list[tuple[int, float]]:
        """检索最相似的 top_k 个向量。

        Args:
            query: 形状为 (dim,) 的查询向量。
            top_k: 返回的结果数量。
            candidate_ids: 可选的候选 id 子集，用于配合元数据 pre-filter。
                为 None 时表示在全量数据中检索。

        Returns:
            按相似度从高到低排序的 (id, distance) 列表。
        """

    @abstractmethod
    def get_vector(self, vec_id: int) -> np.ndarray | None:
        """按 id 查询原始向量，不存在时返回 None。"""

    @abstractmethod
    def export(self) -> tuple[list[int], np.ndarray]:
        """导出索引中所有 (id, 向量)，用于持久化。

        注意：这里只导出原始向量数据，不包括索引内部结构（如 IVF 的聚类中心、
        HNSW 的图结构）。重新加载时会基于导出的向量重建索引，这是 Phase 1/2
        的简化处理，后续 Phase 3 会改进为真正的增量持久化。
        """

    @abstractmethod
    def __len__(self) -> int:
        """当前索引中的向量数量。"""
