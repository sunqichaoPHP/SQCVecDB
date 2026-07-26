"""Flat（暴力搜索）索引实现。

这是最简单的索引：不做任何近似，检索时和全部（或候选子集）向量逐一计算
距离，保证 100% 召回率，作为其他 ANN 索引的正确性基线（ground truth）。
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

from sqcvecdb.distance import pairwise_distance
from sqcvecdb.index.base import BaseIndex


class FlatIndex(BaseIndex):
    """暴力搜索索引，O(n) 检索复杂度。"""

    def __init__(self, dim: int, metric: str) -> None:
        super().__init__(dim, metric)
        self._vectors = np.empty((0, dim), dtype=np.float32)
        self._ids: list[int] = []
        self._id_to_row: dict[int, int] = {}

    def add(self, ids: Iterable[int], vectors: np.ndarray) -> None:
        ids = list(ids)
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        if vectors.shape[0] != len(ids):
            raise ValueError("ids 数量与 vectors 行数不一致")
        if vectors.shape[1] != self.dim:
            raise ValueError(f"向量维度不匹配，期望 {self.dim}，实际 {vectors.shape[1]}")
        for vec_id in ids:
            if vec_id in self._id_to_row:
                raise ValueError(f"id={vec_id} 已存在，如需更新请先 remove 再 add")

        start_row = len(self._ids)
        self._vectors = np.vstack([self._vectors, vectors])
        for offset, vec_id in enumerate(ids):
            self._id_to_row[vec_id] = start_row + offset
            self._ids.append(vec_id)

    def remove(self, ids: Iterable[int]) -> None:
        for vec_id in ids:
            row = self._id_to_row.pop(vec_id, None)
            if row is None:
                continue
            last_row = len(self._ids) - 1
            last_id = self._ids[last_row]
            if row != last_row:
                # 用最后一行覆盖被删除的行，避免整体搬移，O(1) 删除
                self._vectors[row] = self._vectors[last_row]
                self._ids[row] = last_id
                self._id_to_row[last_id] = row
            self._ids.pop()
            self._vectors = self._vectors[:-1]

    def search(
        self,
        query: np.ndarray,
        top_k: int,
        candidate_ids: Iterable[int] | None = None,
    ) -> list[tuple[int, float]]:
        if len(self._ids) == 0 or top_k <= 0:
            return []

        query = np.asarray(query, dtype=np.float32)

        if candidate_ids is not None:
            ids = [cid for cid in candidate_ids if cid in self._id_to_row]
            if not ids:
                return []
            rows = [self._id_to_row[cid] for cid in ids]
            matrix = self._vectors[rows]
        else:
            ids = self._ids
            matrix = self._vectors

        distances = pairwise_distance(query, matrix, self.metric)
        k = min(top_k, len(ids))
        if k < len(ids):
            top_idx = np.argpartition(distances, k - 1)[:k]
        else:
            top_idx = np.arange(len(ids))
        top_idx = top_idx[np.argsort(distances[top_idx])]
        return [(ids[i], float(distances[i])) for i in top_idx]

    def get_vector(self, vec_id: int) -> np.ndarray | None:
        row = self._id_to_row.get(vec_id)
        if row is None:
            return None
        return self._vectors[row].copy()

    def __len__(self) -> int:
        return len(self._ids)

    def export(self) -> tuple[list[int], np.ndarray]:
        return list(self._ids), self._vectors.copy()
