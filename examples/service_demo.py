#!/usr/bin/env python3
"""
sqcvecdb REST API 服务演示

展示如何启动服务、创建 collection、插入/查询向量
"""

import requests
import time
import numpy as np
from pathlib import Path


BASE_URL = "http://127.0.0.1:8000"


def main():
    print("="*60)
    print("sqcvecdb REST API 服务演示")
    print("="*60)
    print()
    print("⚠️  此演示脚本假设 API 服务已启动")
    print("启动方式: uvicorn sqcvecdb.service:app --reload")
    print()

    # 1. 健康检查
    print("[1] 健康检查")
    try:
        resp = requests.get(f"{BASE_URL}/health")
        print(f"  ✓ 服务状态: {resp.json()['status']}")
    except requests.exceptions.ConnectionError:
        print("  ✗ 无法连接到服务，请先启动 API")
        return

    # 2. 创建 collection
    print()
    print("[2] 创建 collection")
    collections_config = [
        {
            "name": "flat_collection",
            "config": {
                "dim": 4,
                "metric": "l2",
                "index_type": "flat",
                "enable_wal": False,
            },
        },
        {
            "name": "hnsw_collection",
            "config": {
                "dim": 4,
                "metric": "l2",
                "index_type": "hnsw",
                "index_params": {"M": 16, "ef_search": 64},
                "enable_wal": True,
            },
        },
    ]

    for col_cfg in collections_config:
        name = col_cfg["name"]
        resp = requests.post(
            f"{BASE_URL}/collections",
            params={"name": name},
            json=col_cfg["config"],
        )
        if resp.status_code == 200:
            print(f"  ✓ 创建 collection '{name}'")
        else:
            print(f"  ✗ 创建失败: {resp.json()}")

    # 3. 列出所有 collection
    print()
    print("[3] 列出所有 collection")
    resp = requests.get(f"{BASE_URL}/collections")
    for col in resp.json():
        print(f"  - {col['name']}: dim={col['dim']}, metric={col['metric']}, "
              f"index_type={col['index_type']}, items={col['num_items']}")

    # 4. 插入向量
    print()
    print("[4] 批量插入向量 (flat_collection)")
    rng = np.random.RandomState(42)
    records = [
        {
            "id": i,
            "vector": rng.random(4).astype(np.float32).tolist(),
            "metadata": {"label": f"item_{i}", "category": "A" if i % 2 == 0 else "B"},
        }
        for i in range(100)
    ]

    resp = requests.post(
        f"{BASE_URL}/collections/flat_collection/insert_many",
        json={"records": records},
    )
    if resp.status_code == 200:
        print(f"  ✓ {resp.json()['message']}")
    else:
        print(f"  ✗ 插入失败: {resp.json()}")

    # 5. 查询向量
    print()
    print("[5] 查询向量")
    query_vector = rng.random(4).astype(np.float32).tolist()
    resp = requests.post(
        f"{BASE_URL}/collections/flat_collection/search",
        json={"query": query_vector, "top_k": 5},
    )
    if resp.status_code == 200:
        results = resp.json()["results"]
        print(f"  查询耗时: {results[0]['distance'] if results else 'N/A':.2f}ms")
        print("  Top-5 结果:")
        for i, r in enumerate(results[:5]):
            print(f"    {i+1}. id={r['id']}, distance={r['distance']:.4f}, "
                  f"metadata={r['metadata']}")
    else:
        print(f"  ✗ 查询失败: {resp.json()}")

    # 6. 带过滤条件的查询
    print()
    print("[6] 带元数据过滤的查询")
    resp = requests.post(
        f"{BASE_URL}/collections/flat_collection/search",
        json={
            "query": query_vector,
            "top_k": 5,
            "filter": {"category": "A"},
        },
    )
    if resp.status_code == 200:
        results = resp.json()["results"]
        print(f"  找到 {len(results)} 条 category=A 的结果")
        for r in results[:3]:
            print(f"    - id={r['id']}, metadata={r['metadata']}")
    else:
        print(f"  ✗ 查询失败: {resp.json()}")

    # 7. 向 HNSW collection 插入数据并演示 checkpoint
    print()
    print("[7] HNSW 索引 + WAL 演示")
    records_hnsw = [
        {
            "id": i,
            "vector": rng.random(4).astype(np.float32).tolist(),
            "metadata": {"index": i},
        }
        for i in range(50)
    ]

    resp = requests.post(
        f"{BASE_URL}/collections/hnsw_collection/insert_many",
        json={"records": records_hnsw},
    )
    if resp.status_code == 200:
        print(f"  ✓ HNSW collection 插入 50 条向量")

    # 手动 checkpoint
    resp = requests.post(f"{BASE_URL}/collections/hnsw_collection/checkpoint")
    if resp.status_code == 200:
        print(f"  ✓ 手动触发 checkpoint")
    else:
        print(f"  ✗ checkpoint 失败: {resp.json()}")

    # 8. 获取 collection 统计
    print()
    print("[8] Collection 统计信息")
    resp = requests.get(f"{BASE_URL}/collections/flat_collection/stats")
    if resp.status_code == 200:
        stats = resp.json()
        print(f"  flat_collection:")
        print(f"    - 维度: {stats['dim']}")
        print(f"    - 度量: {stats['metric']}")
        print(f"    - 索引类型: {stats['index_type']}")
        print(f"    - 向量数: {stats['num_items']}")
        print(f"    - WAL 启用: {stats['enable_wal']}")

    # 9. 删除操作
    print()
    print("[9] 删除向量")
    resp = requests.delete(
        f"{BASE_URL}/collections/flat_collection/delete",
        json={"id": 0},
    )
    if resp.status_code == 200:
        print(f"  ✓ 删除向量 id=0")

    resp = requests.get(f"{BASE_URL}/collections/flat_collection/stats")
    print(f"  删除后向量数: {resp.json()['num_items']}")

    print()
    print("="*60)
    print("✓ 演示完成")
    print("="*60)


if __name__ == "__main__":
    main()
