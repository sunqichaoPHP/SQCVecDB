"""IVF（Inverted File，倒排文件）索引实现。

核心思路：先用 k-means 把向量空间划分成 nlist 个"桶"（Voronoi cell），
每个向量归属到离它最近的聚类中心所在的桶。检索时只需要在离 query 最近的
nprobe 个桶里做暴力搜索，而不是全库扫描，从而用可控的召回率损失换取
检索速度的大幅提升。

与 Faiss 的用法类似：需要先 train（用一批样本学习聚类中心），之后才能
add。为了让 IVFIndex 兼容 BaseIndex 的 add() 接口，这里做了简化：
第一次调用 add() 时，如果还没训练过，就直接用这批数据训练。
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

from xiangliang.distance import pairwise_distance
from xiangliang.index.base import BaseIndex


def _kmeans(vectors: np.ndarray, n_clusters: int, n_iter: int = 20, seed: int = 42) -> np.ndarray:
    """极简版 Lloyd's k-means，返回形状为 (n_clusters, dim) 的聚类中心。

    教学目的，没有做 k-means++ 初始化优化和空簇处理的边界情况打磨，
    仅保证在常规数据下能收敛出可用的聚类中心。
    """
    rng = np.random.default_rng(seed)
    n_samples = vectors.shape[0]
    init_idx = rng.choice(n_samples, size=n_clusters, replace=False)
    centroids = vectors[init_idx].copy()

    for _ in range(n_iter):
        # 用欧氏距离分配样本到最近的中心（k-means 的划分标准始终是欧氏距离，
        # 与索引本身使用的检索 metric 无关）
        diffs = vectors[:, None, :] - centroids[None, :, :]
        distances = np.einsum("ijk,ijk->ij", diffs, diffs)
        assignments = np.argmin(distances, axis=1)

        new_centroids = centroids.copy()
        for cluster_id in range(n_clusters):
            members = vectors[assignments == cluster_id]
            if len(members) > 0:
                new_centroids[cluster_id] = members.mean(axis=0)
            # 空簇保留原中心，避免除零

        if np.allclose(new_centroids, centroids):
            centroids = new_centroids
            break
        centroids = new_centroids

    return centroids


class IVFIndex(BaseIndex):
    """基于倒排桶的近似最近邻索引。"""

    def __init__(self, dim: int, metric: str, nlist: int = 100, nprobe: int = 8) -> None:
        super().__init__(dim, metric)
        self.nlist = nlist
        self.nprobe = nprobe

        self._centroids: np.ndarray | None = None  # (nlist, dim)
        # 每个桶维护自己的 ids / vectors，方便桶内暴力搜索
        self._bucket_ids: list[list[int]] = []
        self._bucket_vectors: list[np.ndarray] = []
        self._id_to_bucket: dict[int, int] = {}
        self._id_to_row: dict[int, int] = {}  # id -> 在所属桶内的行号

    @property
    def is_trained(self) -> bool:
        return self._centroids is not None

    def train(self, vectors: np.ndarray) -> None:
        """用一批样本训练聚类中心，建立空的倒排桶。"""
        vectors = np.asarray(vectors, dtype=np.float32)
        nlist = min(self.nlist, vectors.shape[0])
        if nlist < 1:
            raise ValueError("训练数据至少需要 1 条向量")
        self.nlist = nlist
        self._centroids = _kmeans(vectors, nlist)
        self._bucket_ids = [[] for _ in range(nlist)]
        self._bucket_vectors = [np.empty((0, self.dim), dtype=np.float32) for _ in range(nlist)]

    def _nearest_bucket(self, vector: np.ndarray) -> int:
        distances = pairwise_distance(vector, self._centroids, "l2")
        return int(np.argmin(distances))

    def add(self, ids: Iterable[int], vectors: np.ndarray) -> None:
        ids = list(ids)
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        if vectors.shape[0] != len(ids):
            raise ValueError("ids 数量与 vectors 行数不一致")
        if vectors.shape[1] != self.dim:
            raise ValueError(f"向量维度不匹配，期望 {self.dim}，实际 {vectors.shape[1]}")

        if not self.is_trained:
            # 简化处理：首次 add 时，直接用这批数据训练聚类中心
            self.train(vectors)

        for vec_id in ids:
            if vec_id in self._id_to_bucket:
                raise ValueError(f"id={vec_id} 已存在，如需更新请先 remove 再 add")

        for vec_id, vector in zip(ids, vectors):
            bucket_id = self._nearest_bucket(vector)
            row = len(self._bucket_ids[bucket_id])
            self._bucket_ids[bucket_id].append(vec_id)
            self._bucket_vectors[bucket_id] = np.vstack([self._bucket_vectors[bucket_id], vector[None, :]])
            self._id_to_bucket[vec_id] = bucket_id
            self._id_to_row[vec_id] = row

    def remove(self, ids: Iterable[int]) -> None:
        for vec_id in ids:
            bucket_id = self._id_to_bucket.pop(vec_id, None)
            if bucket_id is None:
                continue
            row = self._id_to_row.pop(vec_id)
            bucket_ids = self._bucket_ids[bucket_id]
            bucket_vectors = self._bucket_vectors[bucket_id]

            last_row = len(bucket_ids) - 1
            last_id = bucket_ids[last_row]
            if row != last_row:
                bucket_vectors[row] = bucket_vectors[last_row]
                bucket_ids[row] = last_id
                self._id_to_row[last_id] = row
            bucket_ids.pop()
            self._bucket_vectors[bucket_id] = bucket_vectors[:-1]

    def search(
        self,
        query: np.ndarray,
        top_k: int,
        candidate_ids: Iterable[int] | None = None,
    ) -> list[tuple[int, float]]:
        if not self.is_trained or top_k <= 0:
            return []

        query = np.asarray(query, dtype=np.float32)
        candidate_filter = set(candidate_ids) if candidate_ids is not None else None

        nprobe = min(self.nprobe, self.nlist)
        centroid_distances = pairwise_distance(query, self._centroids, "l2")
        probe_buckets = np.argpartition(centroid_distances, nprobe - 1)[:nprobe]

        all_ids: list[int] = []
        matrices = []
        for bucket_id in probe_buckets:
            bucket_ids = self._bucket_ids[bucket_id]
            if not bucket_ids:
                continue
            if candidate_filter is not None:
                keep_rows = [i for i, vec_id in enumerate(bucket_ids) if vec_id in candidate_filter]
                if not keep_rows:
                    continue
                all_ids.extend(bucket_ids[i] for i in keep_rows)
                matrices.append(self._bucket_vectors[bucket_id][keep_rows])
            else:
                all_ids.extend(bucket_ids)
                matrices.append(self._bucket_vectors[bucket_id])

        if not all_ids:
            return []

        matrix = np.vstack(matrices)
        distances = pairwise_distance(query, matrix, self.metric)
        k = min(top_k, len(all_ids))
        top_idx = np.argpartition(distances, k - 1)[:k] if k < len(all_ids) else np.arange(len(all_ids))
        top_idx = top_idx[np.argsort(distances[top_idx])]
        return [(all_ids[i], float(distances[i])) for i in top_idx]

    def get_vector(self, vec_id: int) -> np.ndarray | None:
        bucket_id = self._id_to_bucket.get(vec_id)
        if bucket_id is None:
            return None
        row = self._id_to_row[vec_id]
        return self._bucket_vectors[bucket_id][row].copy()

    def export(self) -> tuple[list[int], np.ndarray]:
        all_ids: list[int] = []
        matrices = []
        for bucket_ids, bucket_vectors in zip(self._bucket_ids, self._bucket_vectors):
            if bucket_ids:
                all_ids.extend(bucket_ids)
                matrices.append(bucket_vectors)
        if not matrices:
            return [], np.empty((0, self.dim), dtype=np.float32)
        return all_ids, np.vstack(matrices)

    def __len__(self) -> int:
        return len(self._id_to_bucket)
