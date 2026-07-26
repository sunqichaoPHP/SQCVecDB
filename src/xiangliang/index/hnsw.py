"""HNSW（Hierarchical Navigable Small World）索引实现。

参考 Malkov & Yashunin (2016)
《Efficient and robust approximate nearest neighbor search using
Hierarchical Navigable Small World graphs》。

这里实现的是核心算法的简化版本：
- 每个节点插入时随机分配一个层数（层数越高，该层的节点越稀疏，
  起到"高速公路"的作用，让检索能快速跳到目标区域附近）
- 插入时自顶向下贪心查找入口点，在目标层及以下用 ef_construction
  做 best-first search 来建立邻居连接
- 检索时同样自顶向下逐层收窄，最后在第 0 层用 ef_search 做
  best-first search 得到最终候选集

为了保持代码可读性，这里没有实现论文中更复杂的"启发式邻居选择"
（neighbor selection heuristic，会兼顾多样性），而是用"直接取最近的
M 个"这种朴素策略——召回率会略低于工业级实现（如 hnswlib），但足够
用来理解 HNSW 的核心思想。

局限：
- remove() 采用软删除（保留节点维持图连通性，但从结果里过滤掉），
  这是业界常见做法，因为硬删除会破坏导航结构。
- search() 的 candidate_ids 预过滤是在 ef 个候选里再筛选，如果符合
  条件的向量都不在 best-first search 探索到的范围内，可能会漏召回。
  这是图索引配合标量过滤的通用难题，Milvus/Qdrant 等也需要专门的
  post-filter/迭代扩大 ef 等策略来缓解。
"""

from __future__ import annotations

import heapq
import math
from typing import Iterable

import numpy as np

from xiangliang.distance import distance_one
from xiangliang.index.base import BaseIndex


class HNSWIndex(BaseIndex):
    """分层可导航小世界图索引。"""

    def __init__(
        self,
        dim: int,
        metric: str,
        M: int = 16,
        ef_construction: int = 200,
        ef_search: int = 50,
        seed: int = 42,
    ) -> None:
        super().__init__(dim, metric)
        if M < 2:
            raise ValueError("M 必须 >= 2")
        self.M = M
        self.M_max0 = 2 * M
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self._level_mult = 1.0 / math.log(M)
        self._rng = np.random.default_rng(seed)

        self._vectors: dict[int, np.ndarray] = {}
        self._levels: dict[int, int] = {}
        # graph[id][layer] = 邻居 id 集合
        self._graph: dict[int, dict[int, set[int]]] = {}
        self._deleted: set[int] = set()

        self._entry_point: int | None = None
        self._max_level: int = -1

    # ---- 内部工具 ----

    def _random_level(self) -> int:
        return int(-math.log(self._rng.random()) * self._level_mult)

    def _dist(self, vec: np.ndarray, other_id: int) -> float:
        return distance_one(vec, self._vectors[other_id], self.metric)

    def _search_layer(
        self, query: np.ndarray, entry_points: list[int], ef: int, layer: int
    ) -> list[tuple[float, int]]:
        """在指定层做 best-first search，返回最多 ef 个 (dist, id)。"""
        visited = set(entry_points)
        candidates: list[tuple[float, int]] = []
        result: list[tuple[float, int]] = []  # 存 (-dist, id)，堆顶是当前最差的

        for ep in entry_points:
            d = self._dist(query, ep)
            heapq.heappush(candidates, (d, ep))
            heapq.heappush(result, (-d, ep))

        while candidates:
            cur_dist, cur_id = heapq.heappop(candidates)
            worst_dist = -result[0][0]
            if cur_dist > worst_dist and len(result) >= ef:
                break

            for neighbor_id in self._graph.get(cur_id, {}).get(layer, ()):
                if neighbor_id in visited:
                    continue
                visited.add(neighbor_id)
                d = self._dist(query, neighbor_id)
                worst_dist = -result[0][0]
                if len(result) < ef or d < worst_dist:
                    heapq.heappush(candidates, (d, neighbor_id))
                    heapq.heappush(result, (-d, neighbor_id))
                    if len(result) > ef:
                        heapq.heappop(result)

        return [(-neg_d, node_id) for neg_d, node_id in result]

    @staticmethod
    def _select_neighbors(candidates: list[tuple[float, int]], m: int) -> list[int]:
        """朴素邻居选择：直接取距离最近的 m 个（简化版，非论文的多样性启发式）。"""
        ordered = sorted(candidates, key=lambda item: item[0])
        return [node_id for _, node_id in ordered[:m]]

    def _shrink_neighbors(self, node_id: int, layer: int, m_max: int) -> None:
        neighbors = self._graph[node_id].get(layer, set())
        if len(neighbors) <= m_max:
            return
        vec = self._vectors[node_id]
        scored = [(self._dist(vec, nb), nb) for nb in neighbors]
        self._graph[node_id][layer] = set(self._select_neighbors(scored, m_max))

    def _connect(self, node_id: int, neighbor_ids: list[int], layer: int) -> None:
        m_max = self.M_max0 if layer == 0 else self.M
        self._graph[node_id].setdefault(layer, set()).update(neighbor_ids)
        self._shrink_neighbors(node_id, layer, m_max)
        for neighbor_id in neighbor_ids:
            self._graph[neighbor_id].setdefault(layer, set()).add(node_id)
            self._shrink_neighbors(neighbor_id, layer, m_max)

    def _insert_one(self, vec_id: int, vector: np.ndarray) -> None:
        self._vectors[vec_id] = vector
        self._graph[vec_id] = {}
        level = self._random_level()
        self._levels[vec_id] = level

        if self._entry_point is None:
            self._entry_point = vec_id
            self._max_level = level
            return

        ep = self._entry_point
        # 从最高层贪心下降到 level+1 层，每层只找 1 个最近点作为下一层入口
        for layer in range(self._max_level, level, -1):
            nearest = self._search_layer(vector, [ep], ef=1, layer=layer)
            if nearest:
                ep = nearest[0][1]

        entry_points = [ep]
        for layer in range(min(self._max_level, level), -1, -1):
            candidates = self._search_layer(vector, entry_points, ef=self.ef_construction, layer=layer)
            neighbor_ids = self._select_neighbors(candidates, self.M)
            self._connect(vec_id, neighbor_ids, layer)
            entry_points = [node_id for _, node_id in candidates] or entry_points

        if level > self._max_level:
            self._max_level = level
            self._entry_point = vec_id

    # ---- BaseIndex 接口 ----

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
            if vec_id in self._vectors:
                raise ValueError(f"id={vec_id} 已存在，如需更新请先 remove 再 add")

        for vec_id, vector in zip(ids, vectors):
            self._insert_one(vec_id, vector)

    def remove(self, ids: Iterable[int]) -> None:
        # HNSW 的硬删除会破坏图的导航结构，这里采用业界常用的"软删除"：
        # 保留节点用于维持图连通性，但从检索结果 / get_vector 中排除。
        for vec_id in ids:
            if vec_id in self._vectors:
                self._deleted.add(vec_id)

    def search(
        self,
        query: np.ndarray,
        top_k: int,
        candidate_ids: Iterable[int] | None = None,
    ) -> list[tuple[int, float]]:
        if self._entry_point is None or top_k <= 0:
            return []

        query = np.asarray(query, dtype=np.float32)
        candidate_filter = set(candidate_ids) if candidate_ids is not None else None

        ep = self._entry_point
        for layer in range(self._max_level, 0, -1):
            nearest = self._search_layer(query, [ep], ef=1, layer=layer)
            if nearest:
                ep = nearest[0][1]

        # ef 至少要覆盖 top_k，也不能小于配置的 ef_search
        ef = max(self.ef_search, top_k)
        results = self._search_layer(query, [ep], ef=ef, layer=0)
        results.sort(key=lambda item: item[0])

        output: list[tuple[int, float]] = []
        for dist, node_id in results:
            if node_id in self._deleted:
                continue
            if candidate_filter is not None and node_id not in candidate_filter:
                continue
            output.append((node_id, dist))
            if len(output) >= top_k:
                break
        return output

    def get_vector(self, vec_id: int) -> np.ndarray | None:
        if vec_id in self._deleted:
            return None
        vector = self._vectors.get(vec_id)
        return None if vector is None else vector.copy()

    def export(self) -> tuple[list[int], np.ndarray]:
        ids = [vec_id for vec_id in self._vectors if vec_id not in self._deleted]
        if not ids:
            return [], np.empty((0, self.dim), dtype=np.float32)
        vectors = np.vstack([self._vectors[vec_id] for vec_id in ids])
        return ids, vectors

    def __len__(self) -> int:
        return len(self._vectors) - len(self._deleted)
