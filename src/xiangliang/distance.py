"""距离/相似度度量函数。

约定：所有函数返回的是"越小越相似"的分数（距离语义），方便索引层统一按
升序取 top_k。对于内积、余弦这类"越大越相似"的度量，内部会转换成负值或
1 - similarity 的形式，保持对外接口一致。
"""

from __future__ import annotations

import numpy as np

# 支持的度量名称
METRIC_L2 = "l2"
METRIC_COSINE = "cosine"
METRIC_IP = "ip"  # inner product

SUPPORTED_METRICS = (METRIC_L2, METRIC_COSINE, METRIC_IP)


def _check_metric(metric: str) -> None:
    if metric not in SUPPORTED_METRICS:
        raise ValueError(
            f"不支持的距离度量: {metric!r}，可选值为 {SUPPORTED_METRICS}"
        )


def pairwise_distance(query: np.ndarray, matrix: np.ndarray, metric: str) -> np.ndarray:
    """计算单条 query 向量到 matrix 中每一行的距离。

    Args:
        query: 形状为 (dim,) 的一维向量。
        matrix: 形状为 (n, dim) 的向量矩阵。
        metric: 距离度量，取值见 SUPPORTED_METRICS。

    Returns:
        形状为 (n,) 的距离数组，值越小表示越相似。
    """
    _check_metric(metric)
    if matrix.shape[0] == 0:
        return np.empty((0,), dtype=np.float32)

    if metric == METRIC_L2:
        diff = matrix - query
        return np.einsum("ij,ij->i", diff, diff)

    if metric == METRIC_IP:
        # 内积越大越相似，取负数转换为"越小越相似"
        return -(matrix @ query)

    if metric == METRIC_COSINE:
        matrix_norm = np.linalg.norm(matrix, axis=1)
        query_norm = np.linalg.norm(query)
        denom = matrix_norm * query_norm
        # 避免除零：范数为 0 的向量视为与任何向量的余弦相似度为 0
        with np.errstate(divide="ignore", invalid="ignore"):
            cosine_sim = np.where(denom > 0, (matrix @ query) / denom, 0.0)
        return 1.0 - cosine_sim

    raise AssertionError("unreachable")  # pragma: no cover


def distance_one(vec_a: np.ndarray, vec_b: np.ndarray, metric: str) -> float:
    """计算两个向量之间的距离（标量版本），主要给 HNSW 这类逐点比较的
    图索引使用，避免每次比较都构造一个 (1, dim) 矩阵。
    """
    _check_metric(metric)
    if metric == METRIC_L2:
        diff = vec_a - vec_b
        return float(np.dot(diff, diff))

    if metric == METRIC_IP:
        return float(-np.dot(vec_a, vec_b))

    if metric == METRIC_COSINE:
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        denom = norm_a * norm_b
        if denom == 0:
            return 1.0
        return float(1.0 - np.dot(vec_a, vec_b) / denom)

    raise AssertionError("unreachable")  # pragma: no cover
