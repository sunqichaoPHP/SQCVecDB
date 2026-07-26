"""对比 Flat / IVF / HNSW 三种索引的 recall@k 与检索 QPS。

运行方式（在项目根目录下）：
    pip install -e ".[dev]"
    python examples/benchmark.py
"""

from __future__ import annotations

import time

import numpy as np

from sqcvecdb.index.flat import FlatIndex
from sqcvecdb.index.hnsw import HNSWIndex
from sqcvecdb.index.ivf import IVFIndex

DIM = 32
N_VECTORS = 5000
N_QUERIES = 200
TOP_K = 10


def build_dataset(seed: int = 0):
    rng = np.random.default_rng(seed)
    vectors = rng.random((N_VECTORS, DIM)).astype(np.float32)
    queries = rng.random((N_QUERIES, DIM)).astype(np.float32)
    ids = list(range(N_VECTORS))
    return ids, vectors, queries


def recall_at_k(ground_truth: list[set[int]], approx_results: list[list[tuple[int, float]]], k: int) -> float:
    hits = 0
    for truth, approx in zip(ground_truth, approx_results):
        approx_ids = {vec_id for vec_id, _ in approx}
        hits += len(truth & approx_ids)
    return hits / (len(ground_truth) * k)


def benchmark_index(name: str, index, ids, vectors, queries, ground_truth) -> None:
    build_start = time.perf_counter()
    index.add(ids, vectors)
    build_time = time.perf_counter() - build_start

    search_start = time.perf_counter()
    results = [index.search(q, top_k=TOP_K) for q in queries]
    search_time = time.perf_counter() - search_start

    recall = recall_at_k(ground_truth, results, TOP_K)
    qps = len(queries) / search_time

    print(f"{name:>12} | build={build_time:6.3f}s | search={search_time:6.3f}s | "
          f"qps={qps:8.1f} | recall@{TOP_K}={recall:.3f}")


def main() -> None:
    ids, vectors, queries = build_dataset()

    flat = FlatIndex(dim=DIM, metric="l2")
    flat.add(ids, vectors)
    ground_truth = [
        {vec_id for vec_id, _ in flat.search(q, top_k=TOP_K)} for q in queries
    ]

    print(f"数据集: {N_VECTORS} 条 {DIM} 维向量, {N_QUERIES} 条查询, top_k={TOP_K}\n")

    benchmark_index("Flat", FlatIndex(dim=DIM, metric="l2"), ids, vectors, queries, ground_truth)
    benchmark_index(
        "IVF",
        IVFIndex(dim=DIM, metric="l2", nlist=100, nprobe=8),
        ids, vectors, queries, ground_truth,
    )
    benchmark_index(
        "HNSW",
        HNSWIndex(dim=DIM, metric="l2", M=16, ef_construction=100, ef_search=64),
        ids, vectors, queries, ground_truth,
    )


if __name__ == "__main__":
    main()
